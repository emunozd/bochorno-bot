"""
trading/sizing.py — Kelly criterion position sizing. Identical to bochorno-bot.
"""

from src.config import (
    KELLY_FRACTION, MAX_POS_PCT, MIN_POS_USDC,
    KELLY_MIN_TRADES, KELLY_FALLBACK
)


def kelly_win_rate(recent_trades: list) -> float:
    if len(recent_trades) < KELLY_MIN_TRADES:
        return 0.0
    winners = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
    return winners / len(recent_trades)


def kelly_fraction_dynamic(pip: float, mkt_price: float,
                            recent_trades: list) -> float:
    if mkt_price <= 0 or mkt_price >= 1:
        return 0.0

    p = pip
    n = len(recent_trades)
    if n >= KELLY_MIN_TRADES:
        wr = kelly_win_rate(recent_trades)
        w  = min(n / 30, 0.5)
        p  = pip * (1 - w) + wr * w

    q = 1 - p
    b = (1 - mkt_price) / mkt_price

    f_full = (p * b - q) / b
    if f_full <= 0:
        return 0.0

    return min(f_full * KELLY_FRACTION, MAX_POS_PCT)


def position_size_usdc(capital: float, pip: float, mkt_price: float,
                        recent_trades: list) -> float:
    if capital <= 0:
        return 0.0
    frac = kelly_fraction_dynamic(pip, mkt_price, recent_trades)
    if frac <= 0:
        return 0.0
    usdc = round(capital * frac, 2)
    return usdc if usdc >= MIN_POS_USDC else 0.0


def calc_stop_take(entry_price: float) -> tuple:
    from src.config import STOP_LOSS_PCT, TAKE_PROFIT_PCT
    stop = round(entry_price * (1 - STOP_LOSS_PCT), 3)
    take = round(entry_price * (1 + TAKE_PROFIT_PCT), 3)
    take = min(take, 0.97)
    return stop, take
