"""Free-tier budget enforcement.

Groq allows 200,000 tokens/day per model. Exceeding it does not degrade — the
requests simply fail — so an evaluation run that starts without checking can die
half-finished and waste the day's quota. This refuses to start instead.
"""
from __future__ import annotations

from dataclasses import dataclass

# Per-model daily token allowance, with headroom below the real 200k ceiling.
DAILY_TOKEN_BUDGET = 160_000


@dataclass(frozen=True)
class BudgetStatus:
    model: str
    used: int
    budget: int

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.budget

    def can_afford(self, estimated_tokens: int) -> bool:
        return self.used + estimated_tokens <= self.budget

    def __str__(self) -> str:
        pct = (self.used / self.budget * 100) if self.budget else 0
        return f"{self.model}: {self.used:,}/{self.budget:,} tokens ({pct:.0f}%)"


def tokens_used_today(dsn: str, model: str) -> int:
    """Sum today's tokens for one model from the `runs` table."""
    import psycopg

    # A range predicate, not `created_at::date = CURRENT_DATE`. Two reasons:
    # the cast is not sargable so it cannot use the index, and casting
    # timestamptz to date resolves in the session's timezone. Provider quotas
    # reset on UTC midnight, so "today" is computed in UTC explicitly — counting
    # a day boundary in local time would reset the budget at the wrong moment.
    sql = """
        SELECT COALESCE(SUM(input_tokens + output_tokens), 0)
        FROM runs
        WHERE model = %(model)s
          AND created_at >= date_trunc('day', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
    """
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, {"model": model})
        return int(cur.fetchone()[0])


def status(dsn: str, model: str, budget: int = DAILY_TOKEN_BUDGET) -> BudgetStatus:
    return BudgetStatus(model=model, used=tokens_used_today(dsn, model), budget=budget)
