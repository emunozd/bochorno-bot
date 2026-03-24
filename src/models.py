"""
models.py — Pure data models (dataclasses only, zero logic).

Single Responsibility: define the shape of data that flows through the bot.
No imports from other bot modules — this is the base layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict


@dataclass
class WeatherForecast:
    """
    Ensemble forecast for a single city on a single target date.
    T_predicted and T_std are in the market's native unit (C or F).
    """
    city_key:      str
    target_date:   str            # "YYYY-MM-DD"
    unit:          str            # "C" | "F"
    T_predicted:   float          # ensemble weighted mean
    T_std:         float          # ensemble spread (1-sigma)
    model_temps:   Dict[str, float] = field(default_factory=dict)  # {model: T}
    model_weights: Dict[str, float] = field(default_factory=dict)  # {model: weight}
    fetched_at:    str = ""


@dataclass
class CitySignal:
    """
    The full signal output for a city at a point in time.
    Analogous to the Signal dataclass in bochorno-bot.
    """
    city_key:      str
    wcs:           float               # Weather Confidence Score 0–100
    wcs_zone:      str                 # HIGH_CONF | MEDIUM_CONF | LOW_CONF | VERY_LOW
    wcs_blocked:   bool                # True if sigma > SIGMA_MAX
    best_outcome:  Optional[int]       # e.g. 26 (in market unit)
    best_prob:     Optional[float]     # our probability for that outcome
    mkt_price:     Optional[float]     # Polymarket price for that outcome
    edge:          Optional[float]     # best_prob - mkt_price
    T_predicted:   Optional[float]     # central forecast
    T_std:         Optional[float]     # ensemble spread
    unit:          str                 # "C" | "F"
    all_probs:     Dict[int, float] = field(default_factory=dict)
    opportunity:   Optional[dict] = None
    wcs_breakdown: dict = field(default_factory=dict)


@dataclass
class PolyPosition:
    """An open position on Polymarket — identical to bochorno-bot except asset→city_key."""
    city_key:     str            # e.g. "BUENOS_AIRES"
    outcome_val:  int            # e.g. 26 (the temperature outcome we bought)
    side:         str            # always "YES" for bochorno-bot
    token_id:     str            # CLOB token ID
    shares:       float
    entry_price:  float          # price paid per share (0.00–1.00)
    usdc_spent:   float
    entry_time:   datetime
    entry_wcs:    float
    entry_pip:    float
    stop_loss:    float = 0.0
    take_profit:  float = 0.0
    order_id:     str   = ""
    status:       str   = "OPEN"
    pnl:          float = 0.0


@dataclass
class ClosedTrade:
    """Record of a completed trade."""
    city_key:     str
    outcome_val:  int
    side:         str
    entry_price:  float
    exit_price:   float
    shares:       float
    pnl:          float
    pnl_pct:      float
    log_return:   float
    ev_at_entry:  float
    pip_at_entry: float
    duration:     str
    reason:       str
    time:         str
