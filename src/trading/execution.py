"""
trading/execution.py — Polymarket CLOB order execution. Identical to bochorno-bot.
"""

import time
from typing import Optional
from src.config import CLOB_MAX_RETRIES, CLOB_RETRY_DELAY, CLOB_LIMIT_SLIP

try:
    from py_clob_client.clob_types import MarketOrderArgs, OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL
    HAS_CLOB = True
except ImportError:
    HAS_CLOB = False


def buy(client, token_id: str, usdc_amount: float,
        ref_price: float) -> Optional[str]:
    if client is None:
        return "paper"
    if not HAS_CLOB:
        return None

    for attempt in range(1, CLOB_MAX_RETRIES + 1):
        try:
            if attempt <= 2:
                mo     = MarketOrderArgs(token_id=token_id, amount=usdc_amount,
                                         side=BUY, order_type=OrderType.FOK)
                signed = client.create_market_order(mo)
                resp   = client.post_order(signed, OrderType.FOK)
            else:
                price  = min(ref_price + CLOB_LIMIT_SLIP, 0.98)
                shares = round(usdc_amount / price, 2)
                lo     = OrderArgs(token_id=token_id, price=price,
                                   size=shares, side=BUY)
                signed = client.create_order(lo)
                resp   = client.post_order(signed, OrderType.GTC)

            order_id = resp.get("orderID", "") if resp else ""
            if order_id:
                return order_id
        except Exception:
            pass

        if attempt < CLOB_MAX_RETRIES:
            time.sleep(CLOB_RETRY_DELAY)

    return None


def sell(client, token_id: str, shares: float,
         ref_price: float) -> bool:
    if client is None:
        return True
    if not HAS_CLOB:
        return False

    for attempt in range(1, CLOB_MAX_RETRIES + 1):
        try:
            if attempt <= 2:
                mo     = MarketOrderArgs(token_id=token_id,
                                         amount=shares * ref_price,
                                         side=SELL, order_type=OrderType.FOK)
                signed = client.create_market_order(mo)
                resp   = client.post_order(signed, OrderType.FOK)
            else:
                price  = max(ref_price - CLOB_LIMIT_SLIP, 0.01)
                lo     = OrderArgs(token_id=token_id, price=price,
                                   size=shares, side=SELL)
                signed = client.create_order(lo)
                resp   = client.post_order(signed, OrderType.GTC)

            if resp and resp.get("orderID"):
                return True
        except Exception:
            pass

        if attempt < CLOB_MAX_RETRIES:
            time.sleep(CLOB_RETRY_DELAY)

    return False
