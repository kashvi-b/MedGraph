"""Test doubles. They let every graph path run with no API key, no database,
and without the ~1GB torch install."""
from __future__ import annotations

from medgraph.schemas import (
    Citation,
    Classification,
    Extraction,
    LLMCall,
    QueryIntent,
    Synthesis,
    VerificationResult,
)


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]


class FakeRetriever:
    def __init__(self, citations: list[Citation] | None = None) -> None:
        self.citations = citations if citations is not None else [
            Citation(source_id="CMS-NCD-280.1", title="CGM coverage",
                     snippet="Covered for insulin-treated diabetes.", score=0.81),
            Citation(source_id="DAILYMED-METFORMIN", title="Metformin label",
                     snippet="Contraindicated in severe renal impairment.", score=0.74),
        ]

    def search(self, embedding: list[float]) -> list[Citation]:
        return list(self.citations)


def make_invoke(
    *,
    intent: QueryIntent = QueryIntent.COVERAGE,
    cited_ids: list[str] | None = None,
    grounded: bool = True,
    answer_text: str = "CMS covers CGMs for insulin-treated diabetes.",
):
    """Build a fake `invoke` returning canned, schema-valid objects per node."""
    calls: list[str] = []

    def invoke(node: str, schema: type, messages: list):
        calls.append(node)
        usage = LLMCall(node=node, provider="fake", model="fake",
                        input_tokens=100, output_tokens=20, latency_ms=1)
        if node == "classify":
            return Classification(intent=intent, confidence=0.9,
                                  reasoning="canned"), usage
        if node == "extract":
            return Extraction(entities=[], normalized_query="normalized"), usage
        if node == "synthesize":
            ids = ["CMS-NCD-280.1"] if cited_ids is None else cited_ids
            return Synthesis(answer=answer_text, cited_source_ids=ids), usage
        if node == "verify":
            return VerificationResult(
                grounded=grounded,
                unsupported_claims=[] if grounded else ["fabricated threshold"],
                reasoning="canned",
            ), usage
        raise AssertionError(f"unexpected node {node}")

    invoke.calls = calls
    return invoke
