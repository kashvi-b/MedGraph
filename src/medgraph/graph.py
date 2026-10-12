"""The LangGraph state machine.

    scrub ─┬─(seeks individual advice)───────────────────────────▶ refuse ─▶ END
           └─▶ classify ─┬─(out_of_scope)──────────────────────▶ refuse ─▶ END
                         └─▶ extract ─▶ retrieve ─┬─(no hits)──▶ refuse ─▶ END
                                                └─▶ synthesize ─┬─(no sources)─▶ refuse
                                                                └─▶ verify ─┬─(ungrounded)─▶ refuse
                                                                            └─▶ END

Every edge out of `classify`, `retrieve`, `synthesize` and `verify` is conditional,
so declining is a first-class path rather than an error case.

On checkpointing: the graph compiles with `MemorySaver`, which is a process-local
dict. That is enough to inspect and resume a thread inside one process, and NOT
enough to survive a crash or a redeploy — durable resumability needs a persistent
checkpointer (e.g. the Postgres saver) and is not implemented.

The retriever and embedder are injected rather than imported at module scope so
the graph can be built and exercised without a database or a 1GB torch install.
"""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from . import guardrails, prompts
from .config import RETRIEVAL_K
from .schemas import (
    Answer,
    Citation,
    Classification,
    Extraction,
    QueryIntent,
    RunTrace,
    Synthesis,
    VerificationResult,
)
from .state import MedGraphState

MAX_SNIPPET = 1200


def _trace(state: MedGraphState, node: str) -> RunTrace:
    trace = state.get("trace") or RunTrace(run_id=state.get("run_id", "local"))
    trace.nodes_visited.append(node)
    return trace


def _format_sources(citations: list[Citation]) -> str:
    return "\n\n".join(
        f"[{c.source_id}] {c.title}\n{c.snippet[:MAX_SNIPPET]}" for c in citations
    )


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def scrub_node(state: MedGraphState) -> dict[str, Any]:
    """Redact identifiers BEFORE any provider call. First node for that reason.

    The individual-advice check also runs here rather than after classification.
    It used to sit on the edge out of `classify`, which meant a question asking
    for a personal clinical decision still made an LLM call first — and when that
    call failed (a provider returned invalid JSON on exactly such a question), the
    run raised instead of refusing. A safety guardrail must not depend on a model
    call succeeding, and refusing before the call also costs nothing.
    """
    cleaned, removed = guardrails.scrub(state["query"])
    trace = _trace(state, "scrub")
    out: dict[str, Any] = {
        "scrubbed_query": cleaned,
        "redacted": removed,
        "trace": trace,
    }
    if guardrails.seeks_individual_advice(cleaned):
        out["error"] = ("this asks for a decision about an individual's care, which "
                        "this system does not answer")
    return out


def make_classify_node(invoke: Callable) -> Callable:
    def classify_node(state: MedGraphState) -> dict[str, Any]:
        trace = _trace(state, "classify")
        parsed, call = invoke(
            "classify",
            Classification,
            [
                SystemMessage(content=prompts.CLASSIFY),
                HumanMessage(content=state["scrubbed_query"]),
            ],
        )
        trace.calls.append(call)
        return {"classification": parsed, "trace": trace}

    return classify_node


def make_extract_node(invoke: Callable) -> Callable:
    def extract_node(state: MedGraphState) -> dict[str, Any]:
        trace = _trace(state, "extract")
        parsed, call = invoke(
            "extract",
            Extraction,
            [
                SystemMessage(content=prompts.EXTRACT),
                HumanMessage(content=state["scrubbed_query"]),
            ],
        )
        trace.calls.append(call)
        return {"extraction": parsed, "trace": trace}

    return extract_node


def make_retrieve_node(retriever: Any, embedder: Any) -> Callable:
    def retrieve_node(state: MedGraphState) -> dict[str, Any]:
        trace = _trace(state, "retrieve")
        extraction = state.get("extraction")
        query = extraction.normalized_query if extraction else state["scrubbed_query"]
        vector = embedder.embed_query(query)
        citations = retriever.search(vector)[:RETRIEVAL_K]
        return {"citations": citations, "trace": trace}

    return retrieve_node


def make_synthesize_node(invoke: Callable) -> Callable:
    def synthesize_node(state: MedGraphState) -> dict[str, Any]:
        trace = _trace(state, "synthesize")
        citations: list[Citation] = state.get("citations") or []
        parsed, call = invoke(
            "synthesize",
            Synthesis,
            [
                SystemMessage(content=prompts.SYNTHESIZE),
                HumanMessage(
                    content=(
                        f"Question:\n{state['scrubbed_query']}\n\n"
                        f"Sources:\n{_format_sources(citations)}"
                    )
                ),
            ],
        )
        trace.calls.append(call)

        # Fabrication guard: keep only ids the retriever actually returned. A model
        # inventing a source_id is the single failure this system exists to prevent.
        by_id = {c.source_id: c for c in citations}
        used = [by_id[sid] for sid in dict.fromkeys(parsed.cited_source_ids) if sid in by_id]
        hallucinated = [sid for sid in parsed.cited_source_ids if sid not in by_id]

        return {
            "draft_answer": parsed.answer,
            "used_citations": used,
            "hallucinated_source_ids": hallucinated,
            "trace": trace,
        }

    return synthesize_node


def make_verify_node(invoke: Callable) -> Callable:
    def verify_node(state: MedGraphState) -> dict[str, Any]:
        trace = _trace(state, "verify")
        used: list[Citation] = state.get("used_citations") or []
        parsed, call = invoke(
            "verify",
            VerificationResult,
            [
                SystemMessage(content=prompts.VERIFY),
                HumanMessage(
                    content=(
                        f"Question:\n{state['scrubbed_query']}\n\n"
                        f"Drafted answer:\n{state.get('draft_answer', '')}\n\n"
                        f"Sources:\n{_format_sources(used)}"
                    )
                ),
            ],
        )
        trace.calls.append(call)

        if not parsed.grounded:
            reason = "; ".join(parsed.unsupported_claims) or parsed.reasoning
            return {
                "verification": parsed,
                "error": f"answer failed grounding check: {reason}",
                "trace": trace,
            }

        return {
            "verification": parsed,
            "answer": Answer(answer=state["draft_answer"], citations=used),
            "trace": trace,
        }

    return verify_node


def refuse_node(state: MedGraphState) -> dict[str, Any]:
    trace = _trace(state, "refuse")
    reason = state.get("error") or "the corpus does not support an answer to this question"
    return {
        "answer": Answer(answer="", citations=[], refused=True, refusal_reason=reason),
        "trace": trace,
    }


# --------------------------------------------------------------------------
# Conditional edges
# --------------------------------------------------------------------------

def _after_scrub(state: MedGraphState) -> str:
    """Refuse an individual-advice question before spending a provider call."""
    return "refuse" if state.get("error") else "classify"


def _after_classify(state: MedGraphState) -> str:
    cls = state.get("classification")
    if cls is not None and cls.intent == QueryIntent.OUT_OF_SCOPE:
        return "refuse"
    return "extract"


def _after_retrieve(state: MedGraphState) -> str:
    return "synthesize" if state.get("citations") else "refuse"


def _after_synthesize(state: MedGraphState) -> str:
    return "verify" if state.get("used_citations") else "refuse"


def _after_verify(state: MedGraphState) -> str:
    return "refuse" if state.get("error") else END


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def build_graph(
    *,
    retriever: Any = None,
    embedder: Any = None,
    invoke: Callable | None = None,
    checkpointer: Any = None,
):
    """Compile the graph. Dependencies default to the real implementations."""
    if invoke is None:
        from .providers import invoke as invoke  # noqa: PLC0414
    if retriever is None:
        from .retrieval import Retriever

        retriever = Retriever()
    if embedder is None:
        from .embeddings import LocalEmbeddings

        embedder = LocalEmbeddings()

    g = StateGraph(MedGraphState)
    g.add_node("scrub", scrub_node)
    g.add_node("classify", make_classify_node(invoke))
    g.add_node("extract", make_extract_node(invoke))
    g.add_node("retrieve", make_retrieve_node(retriever, embedder))
    g.add_node("synthesize", make_synthesize_node(invoke))
    g.add_node("verify", make_verify_node(invoke))
    g.add_node("refuse", refuse_node)

    g.set_entry_point("scrub")
    g.add_conditional_edges("scrub", _after_scrub,
                            {"classify": "classify", "refuse": "refuse"})
    g.add_conditional_edges("classify", _after_classify,
                            {"extract": "extract", "refuse": "refuse"})
    g.add_edge("extract", "retrieve")
    g.add_conditional_edges("retrieve", _after_retrieve,
                            {"synthesize": "synthesize", "refuse": "refuse"})
    g.add_conditional_edges("synthesize", _after_synthesize,
                            {"verify": "verify", "refuse": "refuse"})
    g.add_conditional_edges("verify", _after_verify, {END: END, "refuse": "refuse"})
    g.add_edge("refuse", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())
