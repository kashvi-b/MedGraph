"""Chunking. Pure functions — no model, no network, no database.

Chunk size is bounded by the embedding model's window. bge-small-en-v1.5
truncates at 512 word pieces WITHOUT warning, so an oversized chunk silently
loses its tail — invisible corpus corruption in a system whose premise is
traceable citations.

A word ceiling alone is not sufficient, and this module originally got that
wrong: a 350-word ceiling was assumed safe from a ~1.3 tokens/word rule of
thumb, but clinical text with numbers, units and hyphenates runs much higher,
and two chunks in a real corpus reached 521 and 557 tokens. Word count is
therefore only a cheap first pass; when a real tokenizer is supplied via
`count_tokens`, the token ceiling is enforced directly and is authoritative.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_WORDS = 300          # cheap pre-filter, deliberately below the token ceiling
MAX_TOKENS = 500         # hard ceiling, 12 tokens of headroom under the model's 512
OVERLAP_WORDS = 50
MIN_WORDS = 20

_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_WS = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Chunk:
    source_id: str
    title: str
    url: str | None
    text: str
    ordinal: int


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _split_long(sentences: list[str]) -> list[list[str]]:
    """Pack sentences into windows of <= MAX_WORDS with OVERLAP_WORDS carryover."""
    out: list[list[str]] = []
    current: list[str] = []
    count = 0

    for sentence in sentences:
        words = len(sentence.split())
        if words >= MAX_WORDS:
            # A single sentence over the window: hard-split it rather than truncate.
            if current:
                out.append(current)
                current, count = [], 0
            tokens = sentence.split()
            for i in range(0, len(tokens), MAX_WORDS):
                out.append([" ".join(tokens[i:i + MAX_WORDS])])
            continue

        if count + words > MAX_WORDS and current:
            out.append(current)
            carry: list[str] = []
            carried = 0
            for s in reversed(current):
                sw = len(s.split())
                if carried + sw > OVERLAP_WORDS:
                    break
                carry.insert(0, s)
                carried += sw
            current, count = list(carry), carried

        current.append(sentence)
        count += words

    if current:
        out.append(current)
    return out


def _split_to_token_limit(text: str, count_tokens, max_tokens: int) -> list[str]:
    """Recursively halve on sentence boundaries until every part fits.

    Only called when a real tokenizer is supplied. Splitting on sentences rather
    than a hard token cut keeps each part independently meaningful.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    sentences = [x.strip() for x in _SENT.split(text) if x.strip()]
    if len(sentences) < 2:
        # A single sentence over the ceiling: cut on words, since there is no
        # smaller natural boundary left.
        words = text.split()
        mid = max(1, len(words) // 2)
        left, right = " ".join(words[:mid]), " ".join(words[mid:])
    else:
        mid = len(sentences) // 2
        left = " ".join(sentences[:mid])
        right = " ".join(sentences[mid:])

    return (_split_to_token_limit(left, count_tokens, max_tokens)
            + _split_to_token_limit(right, count_tokens, max_tokens))


def chunk_document(source_id: str, title: str, text: str,
                   url: str | None = None, count_tokens=None) -> list[Chunk]:
    """Split on paragraph boundaries first, then pack sentences into windows.

    Paragraph-first matters for policy documents: a coverage criterion and its
    conditions live in one paragraph, and splitting mid-criterion produces chunks
    that retrieve well but cannot actually answer anything.

    Invariant: no input text is ever discarded. Fragments below MIN_WORDS are
    carried forward and merged into the next window rather than dropped — an
    earlier version dropped them, which silently removed every document whose
    paragraphs were all short. Small chunks retrieve poorly; missing chunks
    cannot be retrieved at all.

    Pass `count_tokens` (e.g. a tokenizer's encode-and-length) to enforce the
    real token ceiling. Without it the word ceiling is a conservative proxy, and
    the caller accepts the risk of silent truncation on token-dense text.
    """
    text = normalize(text)
    if not text:
        return []

    windows: list[str] = []
    for paragraph in _PARA.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = [s.strip() for s in _SENT.split(paragraph) if s.strip()]
        if not sentences:
            continue
        windows.extend(" ".join(w).strip() for w in _split_long(sentences))

    if count_tokens is not None:
        windows = [part for w in windows
                   for part in _split_to_token_limit(w, count_tokens, MAX_TOKENS)]

    chunks: list[Chunk] = []
    pending = ""

    def flush(final: bool = False) -> None:
        nonlocal pending
        if not pending:
            return
        # A trailing runt merges BACKWARD into the previous chunk when that keeps
        # it inside the window. Only when there is no previous chunk does it stand
        # alone — losing it is not an option, but neither is emitting a fragment
        # too small to retrieve on.
        if (final and chunks and len(pending.split()) < MIN_WORDS
                and len(chunks[-1].text.split()) + len(pending.split()) <= MAX_WORDS):
            prev = chunks[-1]
            chunks[-1] = Chunk(prev.source_id, prev.title, prev.url,
                               f"{prev.text} {pending}".strip(), prev.ordinal)
            pending = ""
            return
        chunks.append(Chunk(source_id, title, url, pending, len(chunks)))
        pending = ""

    for window in windows:
        if not window:
            continue
        if not pending:
            pending = window
        elif len(pending.split()) < MIN_WORDS and \
                len(pending.split()) + len(window.split()) <= MAX_WORDS:
            # previous fragment is too small to stand alone — merge forward
            pending = f"{pending} {window}".strip()
        else:
            flush()
            pending = window

        if len(pending.split()) >= MIN_WORDS:
            flush()

    flush(final=True)   # fold or emit the tail — never discard it
    return chunks
