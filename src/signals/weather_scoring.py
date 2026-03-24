"""
signals/weather_scoring.py — Weather Confidence Score (WCS) and Temperature Prediction Score (TPS).

WCS (0-100): how reliable is the ensemble forecast?
TPS: produces best outcome, PIP, and edge.

Improvements over v1:
  1. Horizon gate    — entry blocked before MIN_HORIZON_HOURS (default 4h)
  2. Obs anchor      — current observed temp narrows uncertainty near resolution
  3. Forecast drift  — penalises WCS when model changed significantly since last run
  4. Climatological  — penalises pip when predicted temp is climatological outlier
  5. Model consensus — checks directional agreement, not just magnitude spread
"""

import logging
import math
from typing import Dict, List, Optional

from src.config import WATCH_CITIES, NWP_FALLBACK_WEIGHTS, SIGMA_MAX, EDGE_MIN
from src.signals.weather_indicators import (
    compute_ensemble_stats, weights_from_mae, apply_bias_correction,
    compute_climate_percentiles, select_best_outcome, outcome_probability,
    all_outcome_probabilities, convert_temp, expected_value, _norm_cdf
)

log = logging.getLogger("bochorno-bot.scoring")

# ── Tuneable constants ──────────────────────────────────────────────────────
MIN_HORIZON_HOURS  = 4    # never enter with less than this many hours to resolve
MAX_HORIZON_HOURS  = 20   # WCS heavily penalised beyond this (too early)
OBS_ANCHOR_WEIGHT  = 0.35 # how much the current observation pulls the ensemble mean
DRIFT_PENALTY_DEG  = 1.5  # °C change between consecutive forecasts that triggers penalty
CLIM_OUTLIER_SIGMA = 1.5  # z-score beyond which PIP is clipped down


# ── Weather Confidence Score ────────────────────────────────────────────────

def compute_wcs(
    model_temps: Dict[str, float],
    model_weights: Dict[str, float],
    climate_stats: Optional[Dict],
    horizon_hours: float = 12.0,
    prev_ensemble_mean: Optional[float] = None,   # change 3: drift detection
    obs_temp_c: Optional[float] = None,           # change 2: current observation
) -> Dict:
    """
    Compute Weather Confidence Score (0–100).

    Components:
      Ensemble agreement   35% — spread + directional consensus
      Forecast horizon     25% — closer = more confident, gated at MIN_HORIZON_HOURS
      Historical skill     20% — calibrated weights vs fallback
      Climate variability  15% — how stable is this date historically
      Obs consistency       5% — does observation agree with ensemble (when available)
    """
    ensemble = compute_ensemble_stats(model_temps, model_weights)
    T_std    = ensemble.get("T_std") or 9.9
    T_mean   = ensemble.get("T_mean")
    n_models = ensemble.get("n_models", 0)

    # ── Hard gate: horizon too short or too early ───────────────────────────
    if horizon_hours < MIN_HORIZON_HOURS:
        return {
            "score": 0.0, "zone": "VERY_LOW", "blocked": True,
            "T_std": T_std,
            "blocked_reason": f"horizon {horizon_hours:.0f}h < minimum {MIN_HORIZON_HOURS}h",
            "breakdown": {"agreement": 0, "horizon": 0, "skill": 0, "climate": 0, "obs": 0},
        }

    # ── Component 1: ensemble agreement + directional consensus (35%) ───────
    if   T_std < 0.5: agree_score = 100.0
    elif T_std < 1.0: agree_score = 80.0
    elif T_std < 1.5: agree_score = 60.0
    elif T_std < 2.0: agree_score = 40.0
    elif T_std < 2.5: agree_score = 20.0
    else:             agree_score = 0.0

    if n_models >= 4: agree_score = min(100.0, agree_score + 10)
    elif n_models == 1: agree_score = max(0.0, agree_score - 20)

    # Change 5: directional consensus — are models all pointing same side of ensemble mean?
    if T_mean and len(model_temps) >= 2:
        above = sum(1 for t in model_temps.values() if t >= T_mean)
        below = sum(1 for t in model_temps.values() if t < T_mean)
        n     = len(model_temps)
        # Perfect consensus: all above or all below → bonus
        # Split: half and half → penalty
        consensus_ratio = max(above, below) / n  # 1.0 = full consensus, 0.5 = split
        agree_score = round(agree_score * (0.7 + 0.3 * consensus_ratio), 1)

    # ── Component 2: historical skill (20%) ────────────────────────────────
    has_calibration = bool(model_weights and model_weights != NWP_FALLBACK_WEIGHTS)
    skill_score     = 80.0 if has_calibration else 50.0

    # ── Component 3: forecast horizon (25%) ────────────────────────────────
    # Sharper curve: < 4h gated above, 4-8h excellent, 8-16h good, 16-24h fair, >24h poor
    if   horizon_hours <= 4:  horizon_score = 100.0
    elif horizon_hours <= 8:  horizon_score = 90.0
    elif horizon_hours <= 12: horizon_score = 75.0
    elif horizon_hours <= 16: horizon_score = 55.0
    elif horizon_hours <= 20: horizon_score = 35.0
    else:                     horizon_score = 15.0

    # ── Component 4: climate variability (15%) ─────────────────────────────
    if climate_stats and climate_stats.get("std", 0) > 0:
        clim_std = climate_stats["std"]
        if   clim_std < 2.0: clim_score = 90.0
        elif clim_std < 3.5: clim_score = 70.0
        elif clim_std < 5.0: clim_score = 50.0
        else:                clim_score = 30.0
    else:
        clim_score = 50.0

    # ── Component 5: observation consistency (5%) ───────────────────────────
    # Change 2 partial: penalise WCS when current obs is far from ensemble
    obs_score = 70.0  # neutral when no obs
    if obs_temp_c is not None and T_mean is not None:
        obs_delta = abs(obs_temp_c - T_mean)
        if   obs_delta < 1.0: obs_score = 100.0
        elif obs_delta < 2.0: obs_score = 80.0
        elif obs_delta < 3.0: obs_score = 50.0
        else:                 obs_score = 20.0

    wcs = round(
        agree_score   * 0.35 +
        skill_score   * 0.20 +
        horizon_score * 0.25 +
        clim_score    * 0.15 +
        obs_score     * 0.05,
        1
    )

    # Change 3: forecast drift penalty — if ensemble shifted >DRIFT_PENALTY_DEG since last run
    drift_penalty = 0.0
    if prev_ensemble_mean is not None and T_mean is not None:
        drift = abs(T_mean - prev_ensemble_mean)
        if drift > DRIFT_PENALTY_DEG * 2:
            drift_penalty = 15.0
            log.info(f"Large forecast drift {drift:.1f}°C — WCS penalised -{drift_penalty}")
        elif drift > DRIFT_PENALTY_DEG:
            drift_penalty = 7.0
            log.info(f"Forecast drift {drift:.1f}°C — WCS penalised -{drift_penalty}")

    wcs = max(0.0, round(wcs - drift_penalty, 1))

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
        "drift_penalty": drift_penalty,
        "breakdown": {
            "agreement": round(agree_score, 1),
            "skill":     round(skill_score, 1),
            "horizon":   round(horizon_score, 1),
            "climate":   round(clim_score, 1),
            "obs":       round(obs_score, 1),
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
    obs_temp_c: Optional[float] = None,    # change 2: observed temp anchor
    horizon_hours: float = 12.0,           # change 2: used for obs blending
) -> Dict:
    """
    Compute the Temperature Prediction Score.

    Steps:
      1. Ensemble weighted mean + spread (in °C from Open-Meteo)
      2. Obs anchor blend — pull ensemble toward current observation near resolution
      3. Bias correction using climate history
      4. Climatological outlier clip (change 4)
      5. Convert to market unit (°C or °F)
      6. Select closest bin outcome
      7. Compute probability of that bin (PIP)
      8. Compute edge vs market price
    """
    if not outcomes:
        return {"best_outcome": None, "edge": None, "best_prob": None,
                "mkt_price": None, "T_predicted": None, "T_std": None,
                "all_probs": {}, "direction": "NEUTRAL"}

    # Step 1: ensemble stats (always in °C)
    ensemble = compute_ensemble_stats(model_temps, model_weights)
    T_mean_c = ensemble.get("T_mean")
    T_std_c  = ensemble.get("T_std") or 1.5

    if T_mean_c is None:
        return {"best_outcome": None, "edge": None, "best_prob": None,
                "mkt_price": None, "T_predicted": None, "T_std": None,
                "all_probs": {}, "direction": "NEUTRAL"}

    # Step 2: observation anchor — blend ensemble with current obs as horizon shrinks
    # Change 2: the closer to resolution, the more we trust what the thermometer says now.
    # At T-4h: 35% obs weight. At T-12h: ~10%. At T-20h+: 0%.
    obs_blend = 0.0
    if obs_temp_c is not None and horizon_hours <= 20:
        # Weight grows linearly as we approach resolution, capped at OBS_ANCHOR_WEIGHT
        obs_blend = max(0.0, min(OBS_ANCHOR_WEIGHT,
                                 OBS_ANCHOR_WEIGHT * (1.0 - horizon_hours / 20.0)))
        T_anchored = T_mean_c * (1 - obs_blend) + obs_temp_c * obs_blend
        # Observation also reduces uncertainty
        T_std_c    = round(T_std_c * (1 - obs_blend * 0.5), 3)
        log.debug(f"Obs anchor: blend={obs_blend:.2f} "
                  f"T {T_mean_c:.1f}→{T_anchored:.1f}°C")
        T_mean_c = T_anchored

    # Step 3: bias correction
    T_corr_c, T_std_c = apply_bias_correction(T_mean_c, T_std_c, climate_stats)

    # Step 4: climatological outlier clip (change 4)
    # If our prediction is a z-score outlier vs history, PIP gets a credibility penalty
    clim_outlier_factor = 1.0  # multiplier applied to pip later
    if climate_stats and climate_stats.get("std", 0) > 0 and climate_stats.get("mean"):
        z = (T_corr_c - climate_stats["mean"]) / climate_stats["std"]
        if abs(z) > CLIM_OUTLIER_SIGMA:
            # Compress PIP toward 50% the more extreme the outlier
            excess    = abs(z) - CLIM_OUTLIER_SIGMA
            clim_outlier_factor = max(0.5, 1.0 - excess * 0.15)
            log.info(f"Climatological outlier z={z:.1f} — PIP factor {clim_outlier_factor:.2f}")

    # Step 5: convert to market unit
    T_predicted = convert_temp(T_corr_c, "C", unit)
    T_std       = round(T_std_c * (9.0 / 5.0) if unit == "F" else T_std_c, 3)

    # Step 6: select best outcome
    best_outcome = select_best_outcome(T_predicted, outcomes)

    # Step 7: compute probabilities + apply outlier factor
    all_probs = all_outcome_probabilities(T_predicted, T_std, outcomes, unit)
    best_prob = all_probs.get(best_outcome, 0.0)

    if clim_outlier_factor < 1.0:
        # Pull pip toward 0.5 (base rate) by the outlier factor
        best_prob = round(best_prob * clim_outlier_factor +
                          0.5 * (1 - clim_outlier_factor), 4)
        best_prob = max(0.001, min(0.999, best_prob))

    # Step 8: edge
    mkt_price = mkt_prices.get(best_outcome)
    edge      = round(best_prob - mkt_price, 4) if mkt_price is not None else None
    ev        = expected_value(best_prob, mkt_price) if mkt_price else None

    return {
        "best_outcome":        best_outcome,
        "best_prob":           best_prob,
        "mkt_price":           mkt_price,
        "edge":                edge,
        "ev":                  ev,
        "T_predicted":         round(T_predicted, 2),
        "T_std":               T_std,
        "unit":                unit,
        "all_probs":           all_probs,
        "direction":           "LONG",
        "obs_blend":           round(obs_blend, 3),
        "clim_outlier_factor": round(clim_outlier_factor, 3),
    }


# ── Opportunity detection ───────────────────────────────────────────────────

def detect_opportunity(
    city_key: str,
    wcs_data: Dict,
    tps_data: Dict,
    pip_final: float,
) -> Optional[Dict]:
    """
    Return a trade signal dict if all entry conditions are met.

    Filters (in order):
      1. WCS not blocked (sigma too high, horizon too short)
      2. WCS >= WCS_MIN
      3. Market is open (not within 30min of resolution)
      4. Outcome price >= MKT_PRICE_MIN (liquidity check)
      5. Edge >= EDGE_MIN
    """
    from src.config import WCS_MIN, MKT_PRICE_MIN
    from src.data.weather_data import market_is_open

    if wcs_data.get("blocked"):
        reason = wcs_data.get("blocked_reason", "ensemble blocked")
        log.debug(f"{city_key}: blocked — {reason}")
        return None
    if wcs_data.get("score", 0) < WCS_MIN:
        return None
    if not market_is_open(city_key):
        return None

    best_outcome = tps_data.get("best_outcome")
    mkt_price    = tps_data.get("mkt_price")

    if best_outcome is None or mkt_price is None:
        return None

    if mkt_price < MKT_PRICE_MIN:
        log.debug(f"{city_key}: mkt_price {mkt_price:.4f} illiquid — skip")
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
        "end_date":    cfg.get("end_date", ""),
        "target_date": cfg.get("target_date", ""),
    }