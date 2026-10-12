"""Graph state. LangGraph threads one typed dict through every node."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from .schemas import (
    Answer,
    Citation,
    Classification,
    Extraction,
    RunTrace,
    VerificationResult,
)


class MedGraphState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    run_id: str
    query: str
    scrubbed_query: str
    redacted: list[str]
    classification: Classification
    extraction: Extraction
    citations: list[Citation]
    draft_answer: str
    used_citations: list[Citation]
    hallucinated_source_ids: list[str]
    verification: VerificationResult
    answer: Answer
    trace: RunTrace
    error: str | None
