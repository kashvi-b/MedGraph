"""Free-tier budget arithmetic."""
from medgraph.budget import DAILY_TOKEN_BUDGET, BudgetStatus


def test_remaining_and_exhaustion():
    s = BudgetStatus("m", used=150_000, budget=160_000)
    assert s.remaining == 10_000 and not s.exhausted
    assert BudgetStatus("m", used=160_000, budget=160_000).exhausted


def test_remaining_never_goes_negative():
    assert BudgetStatus("m", used=999_999, budget=160_000).remaining == 0


def test_can_afford_blocks_a_run_that_would_overshoot():
    s = BudgetStatus("m", used=155_000, budget=160_000)
    assert s.can_afford(5_000)
    assert not s.can_afford(5_001)


def test_budget_sits_below_the_real_free_tier_ceiling():
    """Groq's hard limit is 200k/day; the budget must leave headroom."""
    assert DAILY_TOKEN_BUDGET < 200_000
