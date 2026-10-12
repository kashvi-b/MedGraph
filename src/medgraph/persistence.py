"""Write run traces to the `runs` table so budgets are enforceable and evals
can be compared across days."""
from __future__ import annotations

from .schemas import RunTrace

INSERT = """
    INSERT INTO runs (run_id, query, nodes_visited, model, refused,
                      input_tokens, output_tokens, cost_usd, latency_ms)
    VALUES (%(run_id)s, %(query)s, %(nodes)s, %(model)s, %(refused)s,
            %(input_tokens)s, %(output_tokens)s, %(cost_usd)s, %(latency_ms)s)
    ON CONFLICT (run_id) DO NOTHING
"""


def record(dsn: str, trace: RunTrace, query: str, refused: bool) -> None:
    """One row per LLM call, sharing the run's id via a suffix.

    Per-call rows rather than per-run: the daily budget is enforced per MODEL,
    and a single run spans two providers and up to three distinct models.
    """
    import psycopg

    if not trace.calls:
        return

    rows = [
        {
            "run_id": f"{trace.run_id}:{i}",
            "query": query[:2000],
            "nodes": trace.nodes_visited,
            "model": call.model,
            "refused": refused,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "cost_usd": 0.0,
            "latency_ms": call.latency_ms,
        }
        for i, call in enumerate(trace.calls)
    ]

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.executemany(INSERT, rows)
        conn.commit()
