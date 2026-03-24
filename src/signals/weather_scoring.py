"""
signals/weather_scoring.py — Weather Confidence Score (WCS) and Temperature Prediction Score (TPS).

WCS (0-100): how reliable is the ensemble forecast?  → analogous to MHS
TPS: produces best outcome, PIP, and edge.            → analogous to DBS + PIP

No I/O. Pure functions over forecast data.
"""

import logging
from typing import Dict, List, Optional

from src.config import WATCH_CITIES, NWP_FALLBACK_WEIGHTS, SIGMA_MAX, EDGE_MIN
from src.signals.weather_indicators import (
    compute_ensemble_stats, weights_from_mae, apply_bias_correction,
    compute_climate_percentiles, select_best_outcome, outcome_probability,
    all_outcome_probabilities, convert_temp, expected_value
)

log = logging.getLogger("bochorno-bot.scoring")


# ── Weather Confidence Score ────────────────────────────────────────────────

def compute_wcs(
    model_temps: Dict[str, float],
    model_weights: Dict[str, float],
    climate_stats: Optional[Dict],
    horizon_hours: int = 12,
) -> Dict:
    """
    Compute Weather Confidence Score (0–100).

    Components:
      Ensemble agreement  40% — how closely do models agree?
      Historical skill    25% — are we using calibrated weights?
      Forecast horizon    20% — closer = more confident
      Climate variability 15% — is this a stable date historically?

    Returns dict: {score, zone, blocked, breakdown}
    """
    ensemble = compute_ensemble_stats(model_temps, model_weights)
    T_std    = ensemble.get("T_std") or 9.9
    agree    = ensemble.get("agreement", 0.0)
    n_models = ensemble.get("n_models", 0)

    # ── Component 1: ensemble agreement (40%) ──────────────────────────────
    # sigma < 0.5°C → 100, sigma < 1°C → 80, sigma < 2°C → 50, sigma >= 3°C → 0
    if   T_std < 0.5: agree_score = 100.0
    elif T_std < 1.0: agree_score = 80.0
    elif T_std < 1.5: agree_score = 60.0
    elif T_std < 2.0: agree_score = 40.0
    elif T_std < 2.5: agree_score = 20.0
    else:             agree_score = 0.0

    # More models = more confidence
    if n_models >= 4: agree_score = min(100.0, agree_score + 10)
    elif n_models == 1: agree_score = max(0.0, agree_score - 20)

    # ── Component 2: historical skill (25%) ────────────────────────────────
    # If we have calibrated weights (from MAE), score is higher
    has_calibration = bool(model_weights and model_weights != NWP_FALLBACK_WEIGHTS)
    skill_score     = 80.0 if has_calibration else 50.0

    # ── Component 3: forecast horizon (20%) ────────────────────────────────
    if   horizon_hours <= 6:  horizon_score = 100.0
    elif horizon_hours <= 12: horizon_score = 85.0
    elif horizon_hours <= 24: horizon_score = 65.0
    elif horizon_hours <= 36: horizon_score = 40.0
    else:                     horizon_score = 20.0

    # ── Component 4: climate variability (15%) ─────────────────────────────
    if climate_stats and climate_stats.get("std", 0) > 0:
        clim_std = climate_stats["std"]
        if   clim_std < 2.0: clim_score = 90.0  # very stable date
        elif clim_std < 3.5: clim_score = 70.0
        elif clim_std < 5.0: clim_score = 50.0
        else:                clim_score = 30.0
    else:
        clim_score = 50.0  # no history = neutral

    wcs = round(
        agree_score   * 0.40 +
        skill_score   * 0.25 +
        horizon_score * 0.20 +
        clim_score    * 0.15,
        1
    )

    blocked = T_std > SIGMA_MAX

    if   blocked: zone = "VERY_LOW"
    elif wcs >= 80: zone = "HIGH_CONF"
    elif wcs >= 65: zone = "MEDIUM_CONF"
    elif wcs >= 45: zone = "LOW_CONF"
    else:           zone = "VERY_LOW"

    return {
        "score":   wcs,
        "zone":    zone,
        "blocked": blocked,
        "T_std":   T_std,
        "breakdown": {
            "agreement": round(agree_score, 1),
            "skill":     round(skill_score, 1),
            "horizon":   round(horizon_score, 1),
            "climate":   round(clim_score, 1),
        },
    }


# ── Temperature Prediction Score ───────────────────────────────────────────

def compute_tps(
    model_temps: Dict[str, float],
    model_weights: Dict[str, float],
    climate_stats: Optional[Dict],
    outcomes: List[int],
    mkt_prices: Dict[int, float],
    unit: str = "C",
) -> Dict:
    """
    Compute the Temperature Prediction Score.

    Steps:
      1. Ensemble weighted mean + spread (in °C from Open-Meteo)
      2. Bias correction using climate history
      3. Convert to market unit (°C or °F)
      4. Select closest bin outcome
      5. Compute probability of that bin (PIP)
      6. Compute edge vs market price

    Returns dict with best_outcome, best_prob, mkt_price, edge,
    T_predicted (in market unit), T_std, all_probs, direction.
    """
    if not outcomes:
        return {"best_outcome": None, "edge": None, "best_prob": None,
                "mkt_price": None, "T_predicted": None, "T_std": None,
                "all_probs": {}, "direction": "NEUTRAL"}

    # Step 1: ensemble stats (always in °C from Open-Meteo)
    ensemble = compute_ensemble_stats(model_temps, model_weights)
    T_mean_c = ensemble.get("T_mean")
    T_std_c  = ensemble.get("T_std") or 1.5

    if T_mean_c is None:
        return {"best_outcome": None, "edge": None, "best_prob": None,
                "mkt_price": None, "T_predicted": None, "T_std": None,
                "all_probs": {}, "direction": "NEUTRAL"}

    # Step 2: bias correction (in °C)
    T_corr_c, T_std_c = apply_bias_correction(T_mean_c, T_std_c, climate_stats)

    # Step 3: convert to market unit
    T_predicted = convert_temp(T_corr_c, "C", unit)
    # Scale T_std proportionally (°F spread = °C spread × 9/5)
    T_std = round(T_std_c * (9.0 / 5.0) if unit == "F" else T_std_c, 3)

    # Step 4: select best outcome
    best_outcome = select_best_outcome(T_predicted, outcomes)

    # Step 5: compute probabilities
    all_probs = all_outcome_probabilities(T_predicted, T_std, outcomes, unit)
    best_prob = all_probs.get(best_outcome, 0.0)

    # Step 6: edge
    mkt_price = mkt_prices.get(best_outcome)
    edge      = round(best_prob - mkt_price, 4) if mkt_price is not None else None
    ev        = expected_value(best_prob, mkt_price) if mkt_price else None

    return {
        "best_outcome": best_outcome,
        "best_prob":    best_prob,
        "mkt_price":    mkt_price,
        "edge":         edge,
        "ev":           ev,
        "T_predicted":  round(T_predicted, 2),
        "T_std":        T_std,
        "unit":         unit,
        "all_probs":    all_probs,
        "direction":    "LONG",  # bochorno-bot always buys YES on best outcome
    }


# ── Opportunity detection ───────────────────────────────────────────────────

def detect_opportunity(
    city_key: str,
    wcs_data: Dict,
    tps_data: Dict,
    pip_final: float,
) -> Optional[Dict]:
    """
    Return a trade signal dict if entry conditions are met.

    Conditions:
      - WCS score >= WCS_MIN threshold
      - Not blocked (sigma too high)
      - Edge >= EDGE_MIN after LLM validation
      - Market price available
    """
    from src.config import WCS_MIN

    if wcs_data.get("blocked"):
        return None
    if wcs_data.get("score", 0) < WCS_MIN:
        return None

    best_outcome = tps_data.get("best_outcome")
    mkt_price    = tps_data.get("mkt_price")

    if best_outcome is None or mkt_price is None:
        return None

    edge = round(pip_final - mkt_price, 4)
    if edge < EDGE_MIN:
        return None

    ev = expected_value(pip_final, mkt_price)

    cfg  = WATCH_CITIES.get(city_key, {})
    unit = tps_data.get("unit", cfg.get("temp_unit", "C"))

    return {
        "city_key":    city_key,
        "outcome_val": best_outcome,
        "unit":        unit,
        "side":        "YES",
        "pip":         pip_final,
        "mkt_price":   mkt_price,
        "edge":        edge,
        "ev":          ev,
        "T_predicted": tps_data.get("T_predicted"),
        "T_std":       tps_data.get("T_std"),
        "wcs":         wcs_data.get("score"),
        "token_id":    cfg.get("token_ids", {}).get(best_outcome, ""),
    }
