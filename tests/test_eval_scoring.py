"""The eval's scoring functions. If these are wrong, every number it reports is."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from run_eval import CaseResult, precision_at_k, recall_at_k, summarize  # noqa: E402


def test_chunk_suffix_is_ignored_when_matching_sources():
    """Retrieval returns MEDLINEPLUS-DIABETES#3; the golden set names the document."""
    assert recall_at_k(["MEDLINEPLUS-DIABETES#3"], ["MEDLINEPLUS-DIABETES"]) == 1.0
    assert precision_at_k(["MEDLINEPLUS-DIABETES#3"], ["MEDLINEPLUS-DIABETES"]) == 1.0


def test_precision_and_recall_on_partial_hits():
    retrieved = ["A#0", "B#1", "C#0", "D#0"]
    assert precision_at_k(retrieved, ["A", "B"]) == 0.5
    assert recall_at_k(retrieved, ["A", "B", "Z"]) == 2 / 3


def test_empty_expectations_are_perfect_recall_not_zero():
    """A case that must refuse has no expected sources; it must not be scored 0."""
    assert recall_at_k([], []) == 1.0


def test_no_retrieval_is_zero_precision():
    assert precision_at_k([], ["A"]) == 0.0


def test_missed_and_over_refusals_are_reported_separately():
    results = [
        CaseResult("a", should_refuse=True, refused=False),    # missed — expensive
        CaseResult("b", should_refuse=True, refused=True),
        CaseResult("c", should_refuse=False, refused=True),    # over-refusal
        CaseResult("d", should_refuse=False, refused=False),
    ]
    s = summarize(results)
    assert s["missed_refusals"] == ["a"]
    assert s["over_refusals"] == ["c"]
    assert s["missed_refusal_rate"] == 0.5
    assert s["over_refusal_rate"] == 0.5
    assert s["refusal_accuracy"] == 0.5


def test_fabricated_citations_lower_the_grounding_score():
    ok = CaseResult("a", should_refuse=False, refused=False, citations_all_real=True)
    bad = CaseResult("b", should_refuse=False, refused=False, citations_all_real=False)
    assert summarize([ok, bad])["grounding_ok"] == 0.5


def test_must_cite_any_treats_alternatives_as_satisfied_by_one():
    """Some topics are genuinely split across documents. Listing the
    alternatives must not be scored as a partial miss."""
    from run_eval import recall_for
    case = {"must_cite_any": ["A", "B", "C"]}
    assert recall_for(case, ["B#2"]) == 1.0
    assert recall_for(case, ["Z#0"]) == 0.0


def test_must_cite_still_requires_every_listed_document():
    from run_eval import recall_for
    case = {"must_cite": ["A", "B"]}
    assert recall_for(case, ["A#0"]) == 0.5
    assert recall_for(case, ["A#0", "B#1"]) == 1.0


def test_refusal_cases_have_no_retrieval_expectation():
    from run_eval import recall_for
    assert recall_for({"must_cite": []}, []) == 1.0
