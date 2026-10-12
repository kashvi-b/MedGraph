import pytest
"""Every path through the graph, exercised end to end with fakes."""
from medgraph.graph import build_graph
from medgraph.schemas import Citation, QueryIntent
from tests.fakes import FakeEmbedder, FakeRetriever, make_invoke


def run(query: str, **kw):
    invoke = make_invoke(**kw)
    graph = build_graph(retriever=kw.pop("retriever", None) or FakeRetriever(),
                        embedder=FakeEmbedder(), invoke=invoke)
    state = graph.invoke({"query": query, "run_id": "t"},
                         config={"configurable": {"thread_id": "t"}})
    return state, invoke


def test_happy_path_produces_grounded_answer():
    state, invoke = run("Does Medicare cover continuous glucose monitors?")
    answer = state["answer"]
    assert not answer.refused
    assert answer.citations and answer.citations[0].source_id == "CMS-NCD-280.1"
    assert invoke.calls == ["classify", "extract", "synthesize", "verify"]
    assert state["trace"].nodes_visited == [
        "scrub", "classify", "extract", "retrieve", "synthesize", "verify"
    ]


def test_out_of_scope_refuses_before_retrieval():
    state, invoke = run("What is the capital of France?",
                        intent=QueryIntent.OUT_OF_SCOPE)
    assert state["answer"].refused
    assert invoke.calls == ["classify"]          # short-circuits: no wasted tokens
    assert "retrieve" not in state["trace"].nodes_visited


def test_individual_advice_refuses_before_any_provider_call():
    """The guardrail runs in scrub, not on the edge out of classify.

    Regression: it used to fire after classification, so an advice question still
    made an LLM call — and a provider returning invalid JSON on exactly such a
    question raised instead of refusing. A safety guardrail must not depend on a
    model call succeeding."""
    state, invoke = run("Should I take 1000mg of metformin tonight?",
                        intent=QueryIntent.DRUG_INFO)
    assert state["answer"].refused
    assert invoke.calls == [], "an advice question must not reach a provider"
    assert state["trace"].nodes_visited == ["scrub", "refuse"]


def test_advice_refusal_states_the_reason():
    state, _ = run("Do I have diabetes? My fasting sugar was 130.")
    assert "individual" in state["answer"].refusal_reason


def test_no_retrieval_hits_refuses_instead_of_answering():
    invoke = make_invoke()
    graph = build_graph(retriever=FakeRetriever(citations=[]),
                        embedder=FakeEmbedder(), invoke=invoke)
    state = graph.invoke({"query": "Is quantum tunnelling covered by Medicare?",
                          "run_id": "t"}, config={"configurable": {"thread_id": "t"}})
    assert state["answer"].refused
    assert "synthesize" not in invoke.calls


def test_fabricated_source_id_is_dropped():
    state, _ = run("Does Medicare cover CGMs?",
                   cited_ids=["CMS-NCD-280.1", "TOTALLY-MADE-UP"])
    assert state["hallucinated_source_ids"] == ["TOTALLY-MADE-UP"]
    ids = [c.source_id for c in state["answer"].citations]
    assert ids == ["CMS-NCD-280.1"]


def test_answer_citing_only_fabricated_sources_refuses():
    state, invoke = run("Does Medicare cover CGMs?", cited_ids=["NOT-REAL"])
    assert state["answer"].refused
    assert "verify" not in invoke.calls      # never reaches the verifier


def test_ungrounded_answer_is_refused_by_verifier():
    state, invoke = run("Does Medicare cover CGMs?", grounded=False)
    assert state["answer"].refused
    assert "fabricated threshold" in state["answer"].refusal_reason
    assert invoke.calls == ["classify", "extract", "synthesize", "verify"]


def test_phi_is_redacted_before_any_model_call():
    state, _ = run("Patient MRN 884213, DOB 04/11/1972 — is their knee MRI covered?")
    assert "884213" not in state["scrubbed_query"]
    assert "04/11/1972" not in state["scrubbed_query"]
    assert {"MRN", "DOB"} <= set(state["redacted"])


def test_token_usage_is_accumulated_for_budget_enforcement():
    state, _ = run("Does Medicare cover CGMs?")
    trace = state["trace"]
    assert len(trace.calls) == 4
    assert trace.input_tokens == 400
    assert trace.output_tokens == 80
    assert trace.cost_usd == 0.0


# --- provider failover ------------------------------------------------------

def test_provider_failure_falls_through_to_the_next_in_the_chain(monkeypatch):
    """A retired model 404'd mid-evaluation and killed 8 of 20 cases. The chain
    exists so that degrades the run instead of ending it."""
    from medgraph import providers
    from medgraph.schemas import Classification, QueryIntent

    calls = []

    def fake_bind(spec, node, schema):
        calls.append((spec.provider, spec.model))

        class R:
            def with_config(self, _):
                return self

            def invoke(self, _messages):
                if len(calls) == 1:
                    raise RuntimeError("404 NOT_FOUND: model is no longer available")
                return {"parsed": Classification(intent=QueryIntent.COVERAGE,
                                                 confidence=0.9, reasoning="ok"),
                        "raw": None, "parsing_error": None}
        return R()

    monkeypatch.setattr(providers, "_bind", fake_bind)
    parsed, call = providers.invoke("classify", Classification, [])
    assert parsed.intent == QueryIntent.COVERAGE
    assert len(calls) == 2, "should have tried the second entry in the chain"
    assert call.fell_back is True


def test_schema_failure_does_not_fail_over(monkeypatch):
    """A validation error is a prompt or schema defect, not a provider outage.
    Trying another provider would hide a real bug behind a slower failure."""
    from medgraph import providers
    from medgraph.schemas import Classification

    calls = []

    def fake_bind(spec, node, schema):
        calls.append(spec.model)

        class R:
            def with_config(self, _):
                return self

            def invoke(self, _messages):
                raise ValueError("Failed to parse Classification from completion")
        return R()

    monkeypatch.setattr(providers, "_bind", fake_bind)
    with pytest.raises(ValueError):
        providers.invoke("classify", Classification, [])
    assert len(calls) == 1, "must not try another provider on a schema failure"


def test_every_node_has_a_fallback_configured():
    from medgraph.config import NODE_CHAINS
    for node, chain in NODE_CHAINS.items():
        assert len(chain) >= 2, f"{node} has no fallback"
        assert len({c.provider for c in chain}) >= 2, \
            f"{node} never leaves one provider, so a vendor outage still ends the run"
