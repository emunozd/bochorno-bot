"""
signals/stats.py — Bootstrap confidence interval for win rate.
Unchanged from bochorno-bot — works over any trade history.
"""

import random
import math
from typing import Optional, Tuple

BOOTSTRAP_MIN_TRADES = 30
BOOTSTRAP_ITERATIONS = 10_000


def bootstrap_win_rate_ci(
    trades: list,
    confidence: float = 0.95,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> Optional[Tuple[float, float]]:
    if len(trades) < BOOTSTRAP_MIN_TRADES:
        return None

    wins = [1 if t.get("pnl", 0) > 0 else 0 for t in trades]
    n    = len(wins)

    sample_rates = []
    for _ in range(iterations):
        sample = [wins[int(random.random() * n)] for _ in range(n)]
        sample_rates.append(sum(sample) / n)

    sample_rates.sort()
    alpha  = 1 - confidence
    lo_idx = int(alpha / 2 * iterations)
    hi_idx = int((1 - alpha / 2) * iterations) - 1

    return (
        round(sample_rates[lo_idx], 4),
        round(sample_rates[hi_idx], 4),
    )


def edge_verdict(ci: Optional[Tuple[float, float]]) -> Tuple[str, str]:
    if ci is None:
        return "Not enough data", "dim"
    lo, hi = ci
    if lo > 0.60: return f"Edge confirmed [{lo:.0%}–{hi:.0%}]", "bold green"
    if lo > 0.55: return f"Edge likely   [{lo:.0%}–{hi:.0%}]", "green"
    if hi < 0.50: return f"Losing system [{lo:.0%}–{hi:.0%}]", "bold red"
    if lo >= 0.50: return f"Marginal edge [{lo:.0%}–{hi:.0%}]", "yellow"
    return f"Inconclusive  [{lo:.0%}–{hi:.0%}]", "orange3"
