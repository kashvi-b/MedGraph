#!/usr/bin/env python
"""Rewrite the README's results blocks from the committed eval artifacts — or fail if stale.

    python scripts/update_readme.py --write    # regenerate the blocks
    python scripts/update_readme.py --check    # exit 1 with a diff if they are stale
    python scripts/update_readme.py --gate     # exit 1 if a published result breaks a promise

This is the mechanism that makes a hand-typed number impossible to merge. It exists
because the failure it prevents already happened here: the README simultaneously
published a measured results table and, nine lines above it, claimed those numbers did
not exist yet. Both were hand-edited, and nothing checked them against each other.

Three properties make the check meaningful rather than flaky:

*   **It renders from committed artifacts, not from a live run.** `retrieval.json` and
    `repeated.json` are in the repository. Rendering them is a pure function, so the
    check is byte-exact even though *producing* them is not reproducible — the verify
    node is LLM-judged and returns different verdicts on identical input.
*   **Nothing machine-dependent goes in the blocks.** No latency, no timestamps, no run
    ids. Those live in the JSON. A wall-clock number in a committed table fails the build
    on a slower runner, and the usual fix for that is to make the check toothless.
*   **The gate is separate from the diff.** `--check` asks "does the README match the
    artifact"; `--gate` asks "is the artifact acceptable". Conflating them means a
    regression can be laundered into the README by regenerating it.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RETRIEVAL = ROOT / "eval" / "results" / "retrieval.json"
REPEATED = ROOT / "eval" / "results" / "repeated.json"

EVAL_BEGIN, EVAL_END = "<!-- EVAL:BEGIN -->", "<!-- EVAL:END -->"
LIVE_BEGIN, LIVE_END = "<!-- LIVE:BEGIN -->", "<!-- LIVE:END -->"

GENERATED = ("<!-- Written by scripts/update_readme.py from the files in eval/results/.\n"
             "     Do not edit by hand: CI runs `--check` and fails when this block is\n"
             "     stale or hand-edited. -->")

# The promise the project is built around. If a published run breaks it, the build
# fails rather than the number quietly appearing in the README.
MAX_MISSED_REFUSAL_RATE = 0.0


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def build_eval_block() -> str:
    """Retrieval and corpus facts. Deterministic: no provider is involved."""
    d = json.loads(RETRIEVAL.read_text())
    s = d["summary"]
    probes = s.get("floor_probes") or {}

    lines = [
        "",
        GENERATED,
        "",
        "| | |",
        "|---|---|",
        f"| corpus | {s.get('documents', '?')} documents → **{s['chunks']} chunks** |",
    ]
    if s.get("max_chunk_tokens"):
        lines.append(f"| largest chunk | {s['max_chunk_tokens']} tokens (model window 512) |")
    lines += [
        f"| queries | {s['cases']} |",
        f"| k / similarity floor | {s['k']} / {s['similarity_floor']} |",
        f"| **recall@k** | **{_fmt(s['recall_at_k'])}** |",
        f"| precision@k | {_fmt(s['precision_at_k'])} |",
        "",
    ]

    if probes:
        lines += [
            "**How much does the similarity floor actually exclude?** Published because it is "
            "unflattering: the floor is close to inert for anything health-adjacent, so the "
            "refusals on out-of-corpus questions are coming from `classify` and `verify`, not "
            "from the floor.",
            "",
            "| probe | top score | chunks above the floor |",
            "|---|---|---|",
        ]
        for label, p in probes.items():
            lines.append(f"| {label} | {p['top']:.3f} | {p['above_floor']} of {s['chunks']} |")
        lines.append("")

    return "\n".join(lines)


def build_live_block() -> str:
    """Live provider results. Rendered from the committed multi-run artifact."""
    if not REPEATED.exists():
        return "\n" + GENERATED + "\n\n_No live evaluation has been published yet._\n"

    d = json.loads(REPEATED.read_text())
    runs = d["runs"]
    unstable = d.get("unstable_cases") or []

    def span(metric: str) -> tuple[float, float, float]:
        v = [r[metric] for r in runs]
        return min(v), sum(v) / len(v), max(v)

    rows = [
        ("retrieval precision@k", "retrieval_precision", False),
        ("retrieval recall@k", "retrieval_recall", False),
        ("grounding — no fabricated citations", "grounding_ok", False),
        ("refusal accuracy", "refusal_accuracy", False),
        ("**missed refusals**", "missed_refusal_rate", True),
        ("over-refusals", "over_refusal_rate", False),
    ]

    lines = [
        "",
        GENERATED,
        "",
        f"Measured against live providers over the {runs[0]['cases']}-case golden set, "
        f"**{len(runs)} runs**:",
        "",
        "| Metric | min | mean | max |",
        "|---|---|---|---|",
    ]
    for label, key, bold in rows:
        lo, mean, hi = span(key)
        cells = [_fmt(lo), _fmt(mean), _fmt(hi)]
        if bold:
            cells = [f"**{c}**" for c in cells]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")

    lines += [
        "",
        "Reported as a range because `verify` is LLM-judged and not deterministic even at "
        "temperature 0. "
        + (f"Across these runs {len(unstable)} case(s) returned a different refuse/answer "
           f"verdict on identical input: `{'`, `'.join(unstable)}`. "
           if unstable else "No case flipped verdict across these runs. ")
        + "`python eval/run_eval.py --repeat N` reports the spread and names every case that "
          "flipped; a single run of an LLM-judged metric is a sample, not a measurement.",
        "",
    ]
    return "\n".join(lines)


def replace(text: str, begin: str, end: str, block: str) -> str:
    i, j = text.find(begin), text.find(end)
    if i == -1 or j == -1:
        raise SystemExit(f"markers {begin} / {end} not found in README.md")
    return text[: i + len(begin)] + block + text[j:]


def render() -> str:
    text = README.read_text()
    text = replace(text, EVAL_BEGIN, EVAL_END, build_eval_block())
    text = replace(text, LIVE_BEGIN, LIVE_END, build_live_block())
    return text


def gate() -> int:
    """Is the published result acceptable? Independent of whether the README matches it."""
    if not REPEATED.exists():
        print("no live results to gate")
        return 0
    runs = json.loads(REPEATED.read_text())["runs"]
    worst = max(r["missed_refusal_rate"] for r in runs)
    if worst > MAX_MISSED_REFUSAL_RATE:
        print(f"GATE FAILED: missed_refusal_rate reached {worst:.3f} across "
              f"{len(runs)} runs (limit {MAX_MISSED_REFUSAL_RATE:.3f}).", file=sys.stderr)
        print("The system answered a question its corpus could not support. That is the "
              "one failure this project exists to prevent.", file=sys.stderr)
        return 1
    print(f"gate passed: missed_refusal_rate {worst:.3f} across {len(runs)} runs")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="regenerate the blocks in README.md")
    g.add_argument("--check", action="store_true", help="fail if the blocks are stale")
    g.add_argument("--gate", action="store_true", help="fail if a published result breaks a promise")
    args = ap.parse_args()

    if args.gate:
        return gate()

    current = README.read_text()
    updated = render()

    if args.write:
        if current == updated:
            print("README.md already up to date")
            return 0
        README.write_text(updated)
        print("README.md updated from eval/results/")
        return 0

    if current != updated:
        print("README.md is stale — regenerate with: python scripts/update_readme.py --write\n",
              file=sys.stderr)
        sys.stderr.writelines(difflib.unified_diff(
            current.splitlines(keepends=True), updated.splitlines(keepends=True),
            fromfile="README.md (committed)", tofile="README.md (generated)", n=2))
        return 1

    print("README.md matches eval/results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
