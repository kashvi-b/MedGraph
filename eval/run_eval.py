#!/usr/bin/env python
"""Evaluation harness.

Four metrics, reported per run so regressions are visible:

  retrieval   precision@k / recall@k against the sources a case must cite
  grounding   share of answered cases whose citations are all real
  refusal     did the graph decline exactly when it should have
  redaction   were identifiers removed before any provider call

Refusal accuracy is the headline. A missed refusal — answering when the corpus
cannot support it — is the expensive error, so it is reported separately from
over-refusal rather than averaged into one number.

    python eval/run_eval.py --fake        # exercise the harness itself, no keys
    python eval/run_eval.py               # real run; needs GROQ + GOOGLE keys and a DB
    python eval/run_eval.py --limit 5     # small real run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import medgraph.config  # noqa: E402,F401  — loads .env before any os.environ read

GOLDEN = Path(__file__).parent / "golden.jsonl"
RESULTS = Path(__file__).parent / "results"


def load_cases() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------
# Scoring — pure functions, unit-tested offline
# --------------------------------------------------------------------------

def _matches(retrieved_id: str, expected: str) -> bool:
    return retrieved_id.split("#")[0].upper() == expected.upper()


def precision_at_k(retrieved: list[str], expected: list[str]) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if any(_matches(r, e) for e in expected))
    return hits / len(retrieved)


def recall_at_k(retrieved: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    hits = sum(1 for e in expected if any(_matches(r, e) for r in retrieved))
    return hits / len(expected)


def expectations(case: dict) -> tuple[list[str], bool]:
    """Return (expected ids, any_of).

    `must_cite` means every listed document must be retrieved. `must_cite_any`
    means the answer is supported if ANY of them is — used where a topic is
    genuinely split across several documents, so listing alternatives must not
    be punished as a miss.
    """
    if case.get("must_cite_any"):
        return list(case["must_cite_any"]), True
    return list(case.get("must_cite", [])), False


def recall_for(case: dict, retrieved: list[str]) -> float:
    expected, any_of = expectations(case)
    if not expected:
        return 1.0
    if any_of:
        return 1.0 if any(
            any(_matches(r, e) for r in retrieved) for e in expected) else 0.0
    return recall_at_k(retrieved, expected)


def precision_for(case: dict, retrieved: list[str]) -> float:
    expected, _ = expectations(case)
    return precision_at_k(retrieved, expected)


@dataclass
class CaseResult:
    case_id: str
    should_refuse: bool
    refused: bool
    precision: float = 0.0
    recall: float = 0.0
    citations_all_real: bool = True
    hallucinated: list[str] = field(default_factory=list)
    redaction_ok: bool = True
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def refusal_correct(self) -> bool:
        return self.refused == self.should_refuse


def score(case: dict, state: dict) -> CaseResult:
    answer = state.get("answer")
    refused = bool(answer.refused) if answer else True
    retrieved = [c.source_id for c in (state.get("citations") or [])]
    trace = state.get("trace")

    redaction_ok = True
    if case.get("must_redact"):
        redacted = set(state.get("redacted") or [])
        redaction_ok = set(case["must_redact"]) <= redacted

    return CaseResult(
        case_id=case["id"],
        should_refuse=bool(case.get("should_refuse")),
        refused=refused,
        precision=precision_for(case, retrieved),
        recall=recall_for(case, retrieved),
        citations_all_real=not state.get("hallucinated_source_ids"),
        hallucinated=list(state.get("hallucinated_source_ids") or []),
        redaction_ok=redaction_ok,
        latency_ms=trace.latency_ms if trace else 0,
        input_tokens=trace.input_tokens if trace else 0,
        output_tokens=trace.output_tokens if trace else 0,
    )


def summarize(results: list[CaseResult]) -> dict:
    answered = [r for r in results if not r.should_refuse]
    to_refuse = [r for r in results if r.should_refuse]
    redaction_cases = [r for r in results if not r.redaction_ok] 

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    missed = [r.case_id for r in to_refuse if not r.refused]
    over = [r.case_id for r in answered if r.refused]

    return {
        "cases": len(results),
        "retrieval_precision": round(mean([r.precision for r in answered]), 3),
        "retrieval_recall": round(mean([r.recall for r in answered]), 3),
        "grounding_ok": round(
            mean([1.0 if r.citations_all_real else 0.0 for r in answered]), 3),
        "refusal_accuracy": round(
            mean([1.0 if r.refusal_correct else 0.0 for r in results]), 3),
        "missed_refusals": missed,
        "missed_refusal_rate": round(len(missed) / len(to_refuse), 3) if to_refuse else 0.0,
        "over_refusals": over,
        "over_refusal_rate": round(len(over) / len(answered), 3) if answered else 0.0,
        "redaction_failures": [r.case_id for r in redaction_cases],
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
        "errors": [r.case_id for r in results if r.error],
    }


def render(summary: dict) -> str:
    lines = [
        "",
        "  MedGraph evaluation",
        "  " + "-" * 46,
        f"  cases                  {summary['cases']}",
        f"  retrieval precision@k  {summary['retrieval_precision']}",
        f"  retrieval recall@k     {summary['retrieval_recall']}",
        f"  grounding (no fabs)    {summary['grounding_ok']}",
        f"  refusal accuracy       {summary['refusal_accuracy']}",
        "",
        f"  MISSED refusals        {summary['missed_refusal_rate']}  {summary['missed_refusals'] or ''}",
        f"  over-refusals          {summary['over_refusal_rate']}  {summary['over_refusals'] or ''}",
        f"  redaction failures     {summary['redaction_failures'] or 'none'}",
        "",
        f"  tokens                 {summary['input_tokens']:,} in / {summary['output_tokens']:,} out",
    ]
    if summary["errors"]:
        lines.append(f"  errors                 {summary['errors']}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def build(fake: bool):
    if fake:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tests.fakes import FakeEmbedder, FakeRetriever, make_invoke

        from medgraph.graph import build_graph
        return build_graph(retriever=FakeRetriever(), embedder=FakeEmbedder(),
                           invoke=make_invoke())

    from medgraph.graph import build_graph
    return build_graph()


def run_once(graph, cases) -> list[CaseResult]:
    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['id']}", end="\r", flush=True)
        started = time.monotonic()
        try:
            state = graph.invoke(
                {"query": case["query"], "run_id": f"eval-{case['id']}"},
                config={"configurable": {"thread_id": f"eval-{case['id']}"}},
            )
            result = score(case, state)
            result.latency_ms = int((time.monotonic() - started) * 1000)
        except Exception as exc:                                  # noqa: BLE001
            result = CaseResult(case_id=case["id"],
                                should_refuse=bool(case.get("should_refuse")),
                                refused=False, error=f"{type(exc).__name__}: {exc}")
        results.append(result)
    print(" " * 60, end="\r")
    return results


def repeated(graph, cases, args) -> int:
    """Run the set N times and report the spread.

    An LLM-judged metric is a sample, not a measurement. Reporting one run as
    though it were a fixed property of the system is the most common way eval
    numbers mislead — and this system demonstrably flips verdicts between runs on
    identical input.
    """
    METRICS = ("retrieval_precision", "retrieval_recall", "grounding_ok",
               "refusal_accuracy", "missed_refusal_rate", "over_refusal_rate")
    runs, flapped = [], {}

    for r in range(1, args.repeat + 1):
        print(f"  run {r}/{args.repeat}")
        results = run_once(graph, cases)
        runs.append(summarize(results))
        for c in results:
            flapped.setdefault(c.case_id, set()).add(c.refused)

    print()
    print("  MedGraph evaluation — {} runs".format(args.repeat))
    print("  " + "-" * 58)
    print(f"  {'metric':<24}{'min':>8}{'mean':>8}{'max':>8}")
    for m in METRICS:
        vals = [run[m] for run in runs]
        mean = sum(vals) / len(vals)
        print(f"  {m:<24}{min(vals):>8.3f}{mean:>8.3f}{max(vals):>8.3f}")

    unstable = sorted(cid for cid, outcomes in flapped.items() if len(outcomes) > 1)
    print()
    print(f"  cases that flipped verdict across runs: {unstable or 'none'}")
    if unstable:
        print("  (identical input, different refuse/answer outcome — LLM-judged,")
        print("   not deterministic; treat single-run refusal numbers with caution)")
    print()

    out = Path(args.out) if args.out else RESULTS / "repeated.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": runs, "unstable_cases": unstable}, indent=2))
    print(f"  written to {out}\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true",
                    help="run against canned model output to exercise the harness")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    ap.add_argument("--repeat", type=int, default=1, metavar="N",
                    help="run the set N times and report per-metric spread. The "
                         "verify node is LLM-judged and not deterministic even at "
                         "temperature 0, so a single run is a sample, not a measurement.")
    args = ap.parse_args()

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    if not args.fake:
        missing = [k for k in ("GROQ_API_KEY", "GOOGLE_API_KEY", "DATABASE_URL")
                   if not os.environ.get(k)]
        if missing:
            print(f"missing environment: {', '.join(missing)}", file=sys.stderr)
            print("(use --fake to exercise the harness without them)", file=sys.stderr)
            return 1

    if args.fake:
        print("\n  NOTE: --fake returns canned model output. It exercises the harness\n"
              "  plumbing and scoring; the scores themselves are not meaningful.\n")

    graph = build(args.fake)

    if args.repeat > 1:
        return repeated(graph, cases, args)

    results: list[CaseResult] = []

    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['id']}", end="\r", flush=True)
        started = time.monotonic()
        try:
            state = graph.invoke(
                {"query": case["query"], "run_id": f"eval-{case['id']}"},
                config={"configurable": {"thread_id": f"eval-{case['id']}"}},
            )
            result = score(case, state)
            result.latency_ms = int((time.monotonic() - started) * 1000)
        except Exception as exc:                       # noqa: BLE001
            result = CaseResult(case_id=case["id"],
                                should_refuse=bool(case.get("should_refuse")),
                                refused=False, error=f"{type(exc).__name__}: {exc}")
        results.append(result)

    print(" " * 60, end="\r")
    summary = summarize(results)
    print(render(summary))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS / "latest.json"
    out.write_text(json.dumps(
        {"summary": summary, "cases": [r.__dict__ for r in results]}, indent=2))
    print(f"  written to {out}\n")
    return 1 if summary["errors"] or summary["missed_refusals"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
