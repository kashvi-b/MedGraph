"""Chunking is where silent corpus corruption would happen, so it is tested hard."""
import pytest

from medgraph.chunking import MAX_WORDS, MIN_WORDS, chunk_document, normalize


def words(text: str) -> int:
    return len(text.split())


def test_no_chunk_exceeds_the_embedding_window():
    """bge-small truncates at 512 word pieces WITHOUT warning. A chunk over the
    window loses its tail silently — invisible corpus corruption."""
    body = " ".join(f"Sentence number {i} about coverage criteria." for i in range(400))
    chunks = chunk_document("SRC", "t", body)
    assert chunks
    assert all(words(c.text) <= MAX_WORDS for c in chunks)


def test_paragraphs_are_not_merged_across_boundaries():
    text = "First paragraph about metformin dosing guidance here.\n\nSecond paragraph about statin coverage criteria here."
    chunks = chunk_document("SRC", "t", text)
    joined = " ".join(c.text for c in chunks)
    assert "metformin" in joined and "statin" in joined


def test_single_oversized_sentence_is_split_not_dropped():
    giant = "word " * (MAX_WORDS * 2)
    chunks = chunk_document("SRC", "t", giant.strip())
    assert len(chunks) >= 2
    assert all(words(c.text) <= MAX_WORDS for c in chunks)


def test_tiny_trailing_fragment_is_folded_into_the_previous_chunk():
    """Folded backward, not emitted as a runt and not discarded."""
    text = " ".join(["Real sentence about coverage."] * 30) + "\n\nOk."
    chunks = chunk_document("SRC", "t", text)
    assert all(words(c.text) >= MIN_WORDS for c in chunks)
    assert chunks[-1].text.endswith("Ok."), "trailing fragment was lost"


def test_ordinals_are_sequential_and_ids_are_stable():
    text = "\n\n".join(" ".join(["Sentence about policy criteria."] * 30) for _ in range(3))
    chunks = chunk_document("CMS-1", "t", text, url="http://x")
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.source_id == "CMS-1" and c.url == "http://x" for c in chunks)


def test_empty_and_whitespace_documents_produce_nothing():
    assert chunk_document("SRC", "t", "") == []
    assert chunk_document("SRC", "t", "   \n\n  ") == []


def test_normalize_collapses_whitespace_without_losing_paragraphs():
    assert normalize("a  \t b\r\n\r\n\r\nc") == "a b\n\nc"


def test_overlap_preserves_context_between_adjacent_chunks():
    sentences = [f"Criterion {i} requires prior authorization documentation." for i in range(120)]
    chunks = chunk_document("SRC", "t", " ".join(sentences))
    assert len(chunks) >= 2
    first_tail = set(chunks[0].text.split()[-30:])
    second_head = set(chunks[1].text.split()[:60])
    assert first_tail & second_head, "adjacent chunks share no overlap"


def test_short_paragraph_document_is_not_silently_dropped():
    """Regression: every paragraph below MIN_WORDS produced ZERO chunks, so the
    whole document vanished from the corpus with no error."""
    text = "First paragraph about metformin dosing.\n\nSecond paragraph about statin coverage."
    chunks = chunk_document("SRC", "t", text)
    assert chunks, "document produced no chunks at all"
    joined = " ".join(c.text for c in chunks)
    assert "metformin" in joined and "statin" in joined


def test_a_single_short_document_still_produces_one_chunk():
    chunks = chunk_document("SRC", "t", "Metformin treats type 2 diabetes.")
    assert len(chunks) == 1
    assert "Metformin" in chunks[0].text


def test_no_input_words_are_lost():
    paragraphs = [" ".join([f"p{p}w{w}" for w in range(40)]) for p in range(6)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_document("SRC", "t", text)
    produced = " ".join(c.text for c in chunks)
    for p in range(6):
        assert f"p{p}w0" in produced and f"p{p}w39" in produced


def test_token_ceiling_is_enforced_when_a_tokenizer_is_supplied():
    """Regression: a 350-word ceiling was assumed safe from a ~1.3 tokens/word
    rule of thumb. On real clinical text two chunks reached 521 and 557 tokens
    and were silently truncated by the encoder. Word count is a proxy; the
    tokenizer is authoritative."""
    from medgraph.chunking import MAX_TOKENS

    # A pathological counter: every word costs two tokens, as token-dense text
    # with numbers, units and hyphenates effectively does.
    def dense(text: str) -> int:
        return len(text.split()) * 2

    body = " ".join(f"LDL-C 130 mg/dL measured at visit {i}." for i in range(400))
    chunks = chunk_document("SRC", "t", body, count_tokens=dense)
    assert chunks
    assert all(dense(c.text) <= MAX_TOKENS for c in chunks), \
        "a chunk exceeded the token ceiling the encoder will silently truncate"


def test_token_splitting_preserves_all_content():
    def dense(text: str) -> int:
        return len(text.split()) * 2

    sentences = [f"Marker{i} appears exactly once here." for i in range(150)]
    chunks = chunk_document("SRC", "t", " ".join(sentences), count_tokens=dense)
    produced = " ".join(c.text for c in chunks)
    for i in range(150):
        assert f"Marker{i}" in produced, f"Marker{i} was lost during token splitting"


def test_word_ceiling_still_applies_without_a_tokenizer():
    body = " ".join(f"Sentence number {i} about coverage criteria." for i in range(400))
    chunks = chunk_document("SRC", "t", body)
    assert all(len(c.text.split()) <= MAX_WORDS for c in chunks)
