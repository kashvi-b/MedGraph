"""PHI scrubbing and the clinical-advice boundary.

Runs BEFORE anything reaches a model provider. There is no output-side pass:
the corpus is public documents, so a grounded answer has no identifier to leak.
Deliberately conservative: over-redaction is recoverable, a leaked identifier is not.
"""
from __future__ import annotations

import re

# Ordered most-specific first so an MRN is not partially eaten by the phone rule.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("MRN", re.compile(r"\b(?:MRN|mrn)[:\s#]*([A-Z0-9]{6,12})\b")),
    ("NPI", re.compile(r"\b\d{10}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("DOB", re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b")),
]

# Every trigger is anchored to the FIRST PERSON. An earlier version matched a
# bare "prescribe", which refused "What are anticoagulants prescribed for?" — a
# general question about a drug class, not a request for personal advice. The
# evaluation set caught it. Over-refusal is a cheaper failure than answering a
# dosing question, but it is still a failure: a system that declines legitimate
# policy questions is not usable.
_ADVICE_TRIGGERS = (
    "should i take", "should i stop", "should i keep taking",
    "what dose should i", "what dosage should i", "how much should i take",
    "am i having", "do i have", "have i got",
    "is it safe for me", "is it safe for my",
    "prescribe me", "prescribe for me", "diagnose me",
    "my child's dose", "dose for my", "dosage for my",
    "what should i do about my",
)


def scrub(text: str) -> tuple[str, list[str]]:
    """Redact identifiers. Returns the scrubbed text and the labels removed."""
    found: list[str] = []
    out = text
    for label, pattern in _PATTERNS:
        out, n = pattern.subn(f"[{label}_REDACTED]", out)
        if n:
            found.append(label)
    return out, found


def seeks_individual_advice(text: str) -> bool:
    """True when the query asks for a personal clinical decision.

    MedGraph answers questions ABOUT policy and guideline documents. It does not
    advise an individual patient, and the graph refuses rather than hedging.
    """
    lowered = text.lower()
    return any(trigger in lowered for trigger in _ADVICE_TRIGGERS)
