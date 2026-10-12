#!/usr/bin/env python
"""Retrieval-only evaluation.

Retrieval is the one part of the pipeline that needs no provider and no API key:
embeddings run locally and the search is arithmetic. So retrieval quality can be
measured — and regressed against — before a single token is ever spent.

Worth isolating for a second reason. When an end-to-end answer is wrong, the
cause is either retrieval or generation, and the combined harness cannot tell you
which. Measuring retrieval alone turns "the answer was bad" into a specific
number that moves when chunking or the similarity floor changes.

    python eval/retrieval_eval.py --vectors data/vectors.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import medgraph.config  # noqa: E402,F401
from run_eval import expectations, load_cases, precision_for, recall_for  # noqa: E402


def cosine_search(query_vec: list[float], rows: list[dict], k: int,
                  floor: float) -> list[tuple[str, float]]:
    """Exact cosine over normalised vectors — the dot product IS the cosine.

    Mirrors what pgvector computes with `1 - (embedding <=> query)`, so a number
    measured here is the number the database will produce.
    """
    scored = []
    for row in rows:
        v = row["embedding"]
        scored.append((row["source_id"], sum(a * b for a, b in zip(query_vec, v))))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [(sid, s) for sid, s in scored[:k] if s > floor]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="data/vectors.jsonl")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--out", default="eval/results/retrieval.json")
    args = ap.parse_args()

    from medgraph.config import RETRIEVAL_K, SIMILARITY_FLOOR
    from medgraph.embeddings import LocalEmbeddings

    k = args.k or RETRIEVAL_K
    floor = args.floor if args.floor is not None else SIMILARITY_FLOOR

    path = Path(args.vectors)
    if not path.exists():
        print(f"{path} not found — run: python scripts/ingest.py --out {path}",
              file=sys.stderr)
        return 1

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print(f"  {len(rows)} chunks loaded from {path}")
    documents = len({r["source_id"].split("#")[0] for r in rows})

    max_chunk_tokens = None
    try:                                        # optional: only for the report
        from transformers import AutoTokenizer

        from medgraph.config import EMBEDDING_MODEL
        _tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        max_chunk_tokens = max(len(_tok.encode(r["chunk"])) for r in rows)
    except Exception:                           # noqa: BLE001
        pass

    # Only cases that should produce an answer have retrieval expectations.
    cases = [c for c in load_cases() if not c.get("should_refuse")]
    embedder = LocalEmbeddings()

    # Warm up before timing. The first embed_query pays the one-time model load
    # (tens of seconds on a 2-core CPU); folding that into the mean reported a
    # per-query latency ~70x the steady-state figure.
    print("  warming up the embedding model...", end="", flush=True)
    warm_started = time.monotonic()
    embedder.embed_query("warmup")
    print(f" {time.monotonic() - warm_started:.1f}s")

    # Timed separately on purpose. The Python cosine loop here is a test
    # harness, not the system: in production pgvector does this scan in C.
    # Reporting one combined number would overstate real query latency.
    per_case, embed_ms, search_ms = [], [], []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['id']}", end="\r", flush=True)
        t0 = time.monotonic()
        qv = embedder.embed_query(case["query"])
        t1 = time.monotonic()
        hits = cosine_search(qv, rows, k, floor)
        t2 = time.monotonic()
        embed_ms.append((t1 - t0) * 1000)
        search_ms.append((t2 - t1) * 1000)

        ids = [sid for sid, _ in hits]
        expected, any_of = expectations(case)
        per_case.append({
            "id": case["id"],
            "precision": round(precision_for(case, ids), 3),
            "recall": round(recall_for(case, ids), 3),
            "any_of": any_of,
            "top_score": round(hits[0][1], 3) if hits else 0.0,
            "n_hits": len(ids),
            "expected": expected,
            "retrieved": ids[:3],
        })

    print(" " * 60, end="\r")

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    # How much does the similarity floor actually exclude? A floor that nothing
    # falls below is not filtering anything, and the refusal it appears to cause
    # is really coming from somewhere else in the graph. Published because it is
    # unflattering: see the note this renders into the README.
    floor_probes = {}
    for label, probe_q in (("in-corpus", cases[0]["query"]),
                           ("adjacent", "Does Medicare cover continuous glucose monitors?"),
                           ("off-topic", "Write me a Python function to sort a list."),
                           ("unrelated", "What is the capital of France?")):
        pv = embedder.embed_query(probe_q)
        scores = sorted((sum(a * b for a, b in zip(pv, r["embedding"])) for r in rows),
                        reverse=True)
        floor_probes[label] = {
            "top": round(scores[0], 3),
            "above_floor": sum(1 for x in scores if x > floor),
        }

    summary = {
        "chunks": len(rows),
        "documents": documents,
        "max_chunk_tokens": max_chunk_tokens,
        "floor_probes": floor_probes,
        "cases": len(per_case),
        "k": k,
        "similarity_floor": floor,
        "precision_at_k": round(mean([c["precision"] for c in per_case]), 3),
        "recall_at_k": round(mean([c["recall"] for c in per_case]), 3),
        "cases_with_zero_hits": [c["id"] for c in per_case if c["n_hits"] == 0],
        "cases_with_zero_recall": [c["id"] for c in per_case if c["recall"] == 0.0],
        "mean_embed_ms": round(mean(embed_ms), 1),
        "median_embed_ms": round(sorted(embed_ms)[len(embed_ms) // 2], 1),
        "mean_python_search_ms": round(mean(search_ms), 1),
    }

    print(f"""
  MedGraph retrieval evaluation
  {'-' * 46}
  corpus                 {summary['chunks']} chunks
  cases                  {summary['cases']}
  k / similarity floor   {k} / {floor}

  precision@k            {summary['precision_at_k']}
  recall@k               {summary['recall_at_k']}

  zero-hit cases         {summary['cases_with_zero_hits'] or 'none'}
  zero-recall cases      {summary['cases_with_zero_recall'] or 'none'}
  query embed (mean)     {summary['mean_embed_ms']} ms
  query embed (median)   {summary['median_embed_ms']} ms
  python cosine scan     {summary['mean_python_search_ms']} ms  (pgvector does this in C)
""")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "cases": per_case}, indent=2))
    print(f"  written to {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
