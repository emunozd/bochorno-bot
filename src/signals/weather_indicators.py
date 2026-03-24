"""
signals/weather_indicators.py — Core weather prediction math.

Responsibilities:
  1. Compute ensemble weighted mean and spread
  2. Apply climatological bias correction (quantile mapping)
  3. Compute probability of the closest-bin outcome (PIP)
  4. Unit conversion utilities

Pure functions — no I/O, no state.
"""

import math
import statistics
import logging
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("bochorno-bot.indicators")

# scipy is optional — we have a pure-Python fallback for norm.cdf
try:
    from scipy.stats import norm as _scipy_norm
    def _norm_cdf(x: float, mu: float, sigma: float) -> float:
        return float(_scipy_norm.cdf(x, loc=mu, scale=sigma))
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    def _norm_cdf(x: float, mu: float, sigma: float) -> float:
        """Pure-Python approximation of the normal CDF (Abramowitz & Stegun)."""
        if sigma <= 0:
            return 1.0 if x >= mu else 0.0
        z = (x - mu) / sigma
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        poly = t * (0.319381530
                    + t * (-0.356563782
                    + t * (1.781477937
                    + t * (-1.821255978
                    + t * 1.330274429))))
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        p   = 1.0 - pdf * poly
        return p if z >= 0 else 1.0 - p


# ── Unit conversion ─────────────────────────────────────────────────────────

def celsius_to_fahrenheit(t: float) -> float:
    return round(t * 9.0 / 5.0 + 32.0, 2)

def fahrenheit_to_celsius(t: float) -> float:
    return round((t - 32.0) * 5.0 / 9.0, 2)

def convert_temp(t: float, from_unit: str, to_unit: str) -> float:
    if from_unit == to_unit:
        return t
    if from_unit == "C" and to_unit == "F":
        return celsius_to_fahrenheit(t)
    if from_unit == "F" and to_unit == "C":
        return fahrenheit_to_celsius(t)
    raise ValueError(f"Unknown units: {from_unit} → {to_unit}")


# ── Ensemble statistics ─────────────────────────────────────────────────────

def compute_ensemble_stats(
    model_temps: Dict[str, float],
    model_weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Compute weighted ensemble mean and spread from NWP model forecasts.

    model_temps:   {model_name: T_celsius}
    model_weights: {model_name: weight} (must sum to 1.0 approximately)
                   If None, equal weights are used.

    Returns dict with:
      T_mean:        weighted mean temperature (°C)
      T_std:         ensemble spread (1-sigma, °C)
      agreement:     0.0–1.0 (1 = all models agree perfectly)
      model_weights: the weights actually used
      n_models:      number of models with data
    """
    if not model_temps:
        return {"T_mean": None, "T_std": None, "agreement": 0.0,
                "model_weights": {}, "n_models": 0}

    models = list(model_temps.keys())
    temps  = [model_temps[m] for m in models]
    n      = len(temps)

    # Build or normalize weights
    if model_weights:
        weights = [model_weights.get(m, 1.0 / n) for m in models]
    else:
        weights = [1.0 / n] * n

    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    # Weighted mean
    T_mean = sum(w * t for w, t in zip(weights, temps))

    # Weighted standard deviation
    variance = sum(w * (t - T_mean) ** 2 for w, t in zip(weights, temps))
    T_std    = math.sqrt(variance) if variance > 0 else 0.1

    # Agreement score: 1 - (T_std / typical_variability)
    # Typical model disagreement for T_max ~ 2°C. Below 1°C = excellent agreement.
    agreement = max(0.0, min(1.0, 1.0 - T_std / 2.0))

    return {
        "T_mean":        round(T_mean, 2),
        "T_std":         round(T_std, 3),
        "agreement":     round(agreement, 3),
        "model_weights": dict(zip(models, weights)),
        "n_models":      n,
    }


# ── Model weights from MAE ──────────────────────────────────────────────────

def weights_from_mae(mae_dict: Dict[str, float],
                     fallback_weights: Dict[str, float]) -> Dict[str, float]:
    """
    Convert MAE values to model weights: w_i = 1/MAE_i (normalized).

    mae_dict: {model: mae} — lower MAE = higher weight
    Returns {model: weight} summing to 1.0.
    """
    if not mae_dict:
        return fallback_weights

    raw = {}
    for model, mae in mae_dict.items():
        raw[model] = 1.0 / max(mae, 0.1)  # avoid div-by-zero

    total = sum(raw.values())
    return {m: round(w / total, 4) for m, w in raw.items()}


# ── Climatological bias correction ──────────────────────────────────────────

def compute_climate_percentiles(
    history: List[dict],
    day_of_year: int,
    window_days: int = 14,
) -> Optional[Dict]:
    """
    Compute climatological percentile distribution for a given day-of-year.

    history: list of {date: 'YYYY-MM-DD', T_max: float} in °C
    day_of_year: 1–365
    window_days: ±days around the target day (default ±14 → 29-day window)

    Returns dict with 'values' (sorted list of historical T_max),
    'mean', 'std', 'p10', 'p25', 'p50', 'p75', 'p90'.
    Returns None if fewer than 20 samples.
    """
    import datetime as _dt

    target_values = []
    for rec in history:
        try:
            d   = _dt.date.fromisoformat(rec["date"])
            doy = d.timetuple().tm_yday
            # Handle year-wrap (e.g. day 360 and day 5)
            diff = min(abs(doy - day_of_year),
                       365 - abs(doy - day_of_year))
            if diff <= window_days:
                target_values.append(float(rec["T_max"]))
        except Exception:
            continue

    if len(target_values) < 20:
        return None

    vals = sorted(target_values)
    n    = len(vals)

    def percentile(p: float) -> float:
        idx = (p / 100) * (n - 1)
        lo  = int(idx)
        hi  = min(lo + 1, n - 1)
        return vals[lo] + (idx - lo) * (vals[hi] - vals[lo])

    return {
        "values": vals,
        "mean":   round(statistics.mean(vals), 2),
        "std":    round(statistics.stdev(vals) if n > 1 else 0.0, 2),
        "p10":    round(percentile(10), 2),
        "p25":    round(percentile(25), 2),
        "p50":    round(percentile(50), 2),
        "p75":    round(percentile(75), 2),
        "p90":    round(percentile(90), 2),
        "n":      n,
    }


def apply_bias_correction(
    T_raw: float,
    T_std_raw: float,
    climate_stats: Optional[Dict],
) -> Tuple[float, float]:
    """
    Apply simple climatological bias correction.

    Method: if the ensemble mean deviates more than 1 sigma from the
    climatological mean, nudge it back by 30%. This prevents overconfident
    forecasts on unusual weather days.

    Returns (T_corrected, T_std_corrected).
    """
    if climate_stats is None:
        return T_raw, T_std_raw

    clim_mean = climate_stats["mean"]
    clim_std  = climate_stats["std"]

    if clim_std > 0:
        z_score = (T_raw - clim_mean) / clim_std
        # If ensemble says something very unusual (|z| > 1.5),
        # increase uncertainty
        if abs(z_score) > 1.5:
            T_std_raw = round(T_std_raw * 1.3, 3)

        # Nudge toward climatology if ensemble is very far off
        if abs(z_score) > 2.0:
            T_raw = round(T_raw * 0.7 + clim_mean * 0.3, 2)

    return T_raw, T_std_raw


# ── Outcome probability ─────────────────────────────────────────────────────

def select_best_outcome(
    T_predicted: float,
    outcomes: List[int],
) -> int:
    """
    Select the outcome bin closest to T_predicted.

    For the bin selection we simply round T_predicted to the nearest
    integer and find the closest outcome in the list.
    The Polymarket outcomes are typically integers like [21,22,23,...,30].
    """
    if not outcomes:
        raise ValueError("outcomes list is empty")
    return min(outcomes, key=lambda k: abs(k - T_predicted))


def _bin_half_width(outcome_val: int, outcomes: List[int]) -> float:
    """
    Compute the half-width of the bin for a given outcome.
    For evenly-spaced outcomes: half_width = step / 2.
    For a single outcome: half_width = 0.5.
    """
    outcomes_sorted = sorted(outcomes)
    if len(outcomes_sorted) < 2:
        return 0.5
    # Use minimum gap between consecutive outcomes as the step
    step = min(
        outcomes_sorted[i+1] - outcomes_sorted[i]
        for i in range(len(outcomes_sorted) - 1)
    )
    return step / 2.0


def outcome_probability(
    T_predicted: float,
    T_std: float,
    outcome_val: int,
    outcomes: List[int],
    unit: str = "C",
) -> float:
    """
    Compute P(temperature falls in the bin for outcome_val).

    Bin width is determined dynamically from the outcome spacing:
      - Outcomes spaced 1°C apart → each bin covers [k-0.5, k+0.5)
      - Outcomes spaced 2°F apart → each bin covers [k-1.0, k+1.0)
      - Lowest outcome: (-∞, k + half_width]
      - Highest outcome: [k - half_width, +∞)

    T_predicted and T_std must be in the same unit as outcomes.
    """
    if T_std <= 0:
        T_std = 0.5  # minimum spread

    sigma      = T_std
    hw         = _bin_half_width(outcome_val, outcomes)
    outcomes_s = sorted(outcomes)
    is_lowest  = outcome_val == outcomes_s[0]
    is_highest = outcome_val == outcomes_s[-1]

    if is_lowest:
        p = _norm_cdf(outcome_val + hw, T_predicted, sigma)
    elif is_highest:
        p = 1.0 - _norm_cdf(outcome_val - hw, T_predicted, sigma)
    else:
        p = (_norm_cdf(outcome_val + hw, T_predicted, sigma) -
             _norm_cdf(outcome_val - hw, T_predicted, sigma))

    return round(max(0.001, min(0.999, p)), 4)


def all_outcome_probabilities(
    T_predicted: float,
    T_std: float,
    outcomes: List[int],
    unit: str = "C",
) -> Dict[int, float]:
    """
    Compute probabilities for all outcomes.
    Returns {outcome_val: probability}.
    """
    return {
        k: outcome_probability(T_predicted, T_std, k, outcomes, unit)
        for k in outcomes
    }


def expected_value(pip: float, mkt_price: float) -> float:
    """EV per dollar invested: pip * (1/price) - 1"""
    if mkt_price <= 0 or mkt_price >= 1:
        return 0.0
    return round(pip * (1.0 / mkt_price) - 1.0, 4)


def expected_log_return(pip: float, mkt_price: float) -> float:
    """
    Expected log return for a binary market YES position.
    Win: payout = 1/price per dollar invested → log return = ln(1/price)
    Lose: payout = 0 → clipped to small epsilon to avoid -inf
    ELR = pip * ln(1/price) + (1-pip) * ln(epsilon)
    Positive when pip > mkt_price (positive edge).
    """
    epsilon = 1e-6
    if mkt_price <= 0 or mkt_price >= 1:
        return 0.0
    return round(
        pip * math.log(1.0 / mkt_price) +
        (1.0 - pip) * math.log(epsilon),
        6
    )
