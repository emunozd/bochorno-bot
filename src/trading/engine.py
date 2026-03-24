"""
trading/engine.py — Position lifecycle for bochorno-bot.

Adapted from bochorno-bot: same open/monitor/close pattern,
city_key instead of asset, outcome_val instead of YES/NO direction.
"""

import math
import logging
from datetime import datetime
from typing import Optional

from src.config import WATCH_CITIES, EDGE_MIN, ET
from src.models import PolyPosition, ClosedTrade
from src.signals.weather_indicators import expected_value
from src.trading.sizing import position_size_usdc, calc_stop_take
from src.trading import execution as CLOB
from src.data import database as DB

log = logging.getLogger("bochorno-bot.engine")


def _log_return(entry: float, exit_: float) -> float:
    if entry <= 0 or exit_ <= 0:
        return 0.0
    return round(math.log(exit_ / entry), 6)


def open_position(
    city_key: str,
    opp: dict,
    capital: float,
    recent_trades: list,
    poly_client,
    state: dict,
) -> Optional[PolyPosition]:
    """
    Open a YES position on the best temperature outcome.
    """
    if state.get("engine_paused"):
        return None

    pip       = opp["pip"]
    mkt_price = opp["mkt_price"]
    token_id  = opp.get("token_id", "")
    outcome   = opp["outcome_val"]

    usdc = position_size_usdc(capital, pip, mkt_price, recent_trades)
    if usdc <= 0:
        log.debug(f"{city_key}/{outcome}: position size 0 — skipping")
        return None

    order_id = CLOB.buy(poly_client, token_id, usdc, mkt_price)
    if order_id is None:
        log.warning(f"{city_key}/{outcome}: CLOB buy failed")
        return None

    shares     = round(usdc / mkt_price, 2)
    stop, take = calc_stop_take(mkt_price)

    pos = PolyPosition(
        city_key    = city_key,
        outcome_val = outcome,
        side        = "YES",
        token_id    = token_id,
        shares      = shares,
        entry_price = mkt_price,
        usdc_spent  = usdc,
        entry_time  = datetime.now(ET),
        entry_wcs   = opp.get("wcs", 0.0),
        entry_pip   = pip,
        stop_loss   = stop,
        take_profit = take,
        order_id    = order_id,
        status      = "OPEN",
    )

    DB.save_position(city_key, pos)

    import src.telegram.bot as _tg
    _tg.alert_position_opened(
        city_key, outcome, opp.get("unit", "C"),
        shares, mkt_price, usdc, stop, take,
        opp.get("wcs", 0), pip,
        opp.get("T_predicted"), opp.get("T_std"),
    )

    with state.get("_lock_ref", __import__("contextlib").nullcontext()):
        state["positions"][city_key]     = pos
        state["capital_usdc"]           -= usdc
        state["peak_capital"]            = max(
            state["peak_capital"], state["capital_usdc"]
        )

    DB.save_portfolio(state["capital_usdc"], state["peak_capital"])
    log.info(f"Opened {city_key}/{outcome}°: {shares:.2f} shares @ {mkt_price:.3f}")
    return pos


def close_position(
    city_key: str,
    pos: PolyPosition,
    reason: str,
    exit_price: float,
    poly_client,
    state: dict,
) -> Optional[ClosedTrade]:
    """Close an open position and record the trade."""
    sold = CLOB.sell(poly_client, pos.token_id, pos.shares, exit_price)
    if not sold:
        log.warning(f"{city_key}: CLOB sell failed — forcing close anyway")

    pnl     = round((exit_price - pos.entry_price) * pos.shares, 4)
    pnl_pct = round(pnl / pos.usdc_spent * 100, 2) if pos.usdc_spent else 0.0
    lr      = _log_return(pos.entry_price, exit_price)
    ev      = expected_value(pos.entry_pip, pos.entry_price)

    # Duration
    elapsed = datetime.now(ET) - pos.entry_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, _    = divmod(rem, 60)
    duration = f"{h}h{m}m"

    trade = ClosedTrade(
        city_key    = city_key,
        outcome_val = pos.outcome_val,
        side        = "YES",
        entry_price = pos.entry_price,
        exit_price  = exit_price,
        shares      = pos.shares,
        pnl         = pnl,
        pnl_pct     = pnl_pct,
        log_return  = lr,
        ev_at_entry = ev,
        pip_at_entry= pos.entry_pip,
        duration    = duration,
        reason      = reason,
        time        = datetime.now(ET).isoformat(),
    )

    DB.save_trade(trade)
    DB.delete_position(city_key)

    recovered = pos.usdc_spent + pnl
    with state.get("_lock_ref", __import__("contextlib").nullcontext()):
        state["capital_usdc"] = round(state["capital_usdc"] + recovered, 4)
        state["peak_capital"] = max(state["peak_capital"], state["capital_usdc"])
        state["positions"].pop(city_key, None)
        state["trades"].append(trade)

    DB.save_portfolio(state["capital_usdc"], state["peak_capital"])

    import src.telegram.bot as _tg
    _tg.alert_position_closed(
        city_key, pos.outcome_val, pos.city_key,
        reason, pnl, pnl_pct, pos.entry_price, exit_price
    )

    log.info(f"Closed {city_key}/{pos.outcome_val}°: PnL={pnl:+.4f} ({reason})")
    return trade


def monitor_position(
    city_key: str,
    pos: PolyPosition,
    cur_price: Optional[float],
    wcs_score: float,
    poly_client,
    state: dict,
) -> Optional[ClosedTrade]:
    """
    Monitor an open position for stop-loss, take-profit, or signal reversal.
    cur_price: current YES price for the position's outcome token.
    """
    if cur_price is None:
        return None

    reason = None

    if cur_price <= pos.stop_loss:
        reason = "STOP_LOSS"
        import src.telegram.bot as _tg
        _tg.alert_stop_loss(city_key, pos.outcome_val, cur_price,
                            (cur_price - pos.entry_price) * pos.shares)

    elif cur_price >= pos.take_profit:
        reason = "TAKE_PROFIT"
        import src.telegram.bot as _tg
        _tg.alert_take_profit(city_key, pos.outcome_val, cur_price,
                              (cur_price - pos.entry_price) * pos.shares)

    # WCS signal reversal: if confidence collapses, exit early
    elif wcs_score < 30:
        reason = "SIGNAL_REVERSED"

    if reason:
        return close_position(city_key, pos, reason, cur_price, poly_client, state)

    return None
