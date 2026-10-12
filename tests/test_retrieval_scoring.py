"""The offline cosine search must agree with what pgvector computes, or the
numbers from retrieval_eval.py describe a system that does not exist."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from retrieval_eval import cosine_search  # noqa: E402


def row(sid: str, vec: list[float]) -> dict:
    return {"source_id": sid, "embedding": vec}


ROWS = [
    row("A#0", [1.0, 0.0, 0.0]),
    row("B#0", [0.0, 1.0, 0.0]),
    row("C#0", [0.7071, 0.7071, 0.0]),
]


def test_ranks_by_cosine_similarity():
    hits = cosine_search([1.0, 0.0, 0.0], ROWS, k=3, floor=-1.0)
    assert [sid for sid, _ in hits] == ["A#0", "C#0", "B#0"]


def test_similarity_floor_excludes_weak_matches():
    hits = cosine_search([1.0, 0.0, 0.0], ROWS, k=3, floor=0.5)
    assert [sid for sid, _ in hits] == ["A#0", "C#0"]   # B scores 0.0


def test_k_caps_results_before_the_floor_is_applied():
    hits = cosine_search([1.0, 0.0, 0.0], ROWS, k=1, floor=-1.0)
    assert len(hits) == 1 and hits[0][0] == "A#0"


def test_no_match_above_floor_returns_empty_not_best_effort():
    """An empty result is what drives the graph to refuse. It must not
    silently degrade to 'closest anyway'."""
    assert cosine_search([0.0, 0.0, 1.0], ROWS, k=3, floor=0.5) == []


def test_dot_product_equals_cosine_for_normalised_vectors():
    """pgvector computes 1 - (a <=> b). For unit vectors that is the dot
    product, which is what this implements."""
    import math
    a = [0.6, 0.8, 0.0]
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-6)
    hits = cosine_search(a, [row("X#0", a)], k=1, floor=-1.0)
    assert math.isclose(hits[0][1], 1.0, rel_tol=1e-6)
