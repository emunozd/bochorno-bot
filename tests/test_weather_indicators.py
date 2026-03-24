"""
tests/test_weather_indicators.py

Unit tests for the core weather math:
  - Unit conversion
  - Ensemble stats
  - Outcome probability
  - Best outcome selection
  - Bias correction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.signals.weather_indicators import (
    celsius_to_fahrenheit, fahrenheit_to_celsius, convert_temp,
    compute_ensemble_stats, weights_from_mae,
    select_best_outcome, outcome_probability, all_outcome_probabilities,
    apply_bias_correction, expected_value, expected_log_return,
)


# ── Unit conversion ─────────────────────────────────────────────────────────

def test_c_to_f_freezing():
    assert celsius_to_fahrenheit(0) == 32.0

def test_c_to_f_boiling():
    assert celsius_to_fahrenheit(100) == 212.0

def test_f_to_c_body_temp():
    assert abs(fahrenheit_to_celsius(98.6) - 37.0) < 0.1

def test_roundtrip():
    for t in [-20, 0, 15, 26, 40]:
        assert abs(fahrenheit_to_celsius(celsius_to_fahrenheit(t)) - t) < 0.01

def test_convert_noop():
    assert convert_temp(25.0, "C", "C") == 25.0
    assert convert_temp(77.0, "F", "F") == 77.0

def test_convert_c_to_f():
    assert convert_temp(26.0, "C", "F") == pytest.approx(78.8, abs=0.1)

def test_convert_f_to_c():
    assert convert_temp(72.0, "F", "C") == pytest.approx(22.22, abs=0.1)


# ── Ensemble stats ──────────────────────────────────────────────────────────

def test_ensemble_empty():
    result = compute_ensemble_stats({})
    assert result["T_mean"] is None
    assert result["n_models"] == 0

def test_ensemble_single_model():
    result = compute_ensemble_stats({"ecmwf": 26.0})
    assert result["T_mean"] == 26.0
    assert result["T_std"] < 0.5   # near-zero spread for single model

def test_ensemble_equal_weights():
    temps = {"ecmwf": 25.0, "gfs": 27.0}
    result = compute_ensemble_stats(temps)
    assert result["T_mean"] == pytest.approx(26.0, abs=0.01)
    assert result["T_std"] > 0

def test_ensemble_custom_weights():
    temps   = {"ecmwf": 24.0, "gfs": 28.0}
    weights = {"ecmwf": 0.75, "gfs": 0.25}
    result  = compute_ensemble_stats(temps, weights)
    # 24*0.75 + 28*0.25 = 25.0
    assert result["T_mean"] == pytest.approx(25.0, abs=0.01)

def test_ensemble_agreement_low_spread():
    # All models agree → high agreement score
    temps  = {"m1": 26.0, "m2": 26.1, "m3": 25.9, "m4": 26.0}
    result = compute_ensemble_stats(temps)
    assert result["agreement"] > 0.9

def test_ensemble_agreement_high_spread():
    # Models disagree widely → low agreement
    temps  = {"m1": 20.0, "m2": 30.0}
    result = compute_ensemble_stats(temps)
    assert result["agreement"] < 0.5


# ── Weights from MAE ────────────────────────────────────────────────────────

def test_weights_from_mae_lower_is_better():
    mae     = {"ecmwf": 1.0, "gfs": 2.0}
    weights = weights_from_mae(mae, {})
    # ecmwf should have higher weight (lower MAE)
    assert weights["ecmwf"] > weights["gfs"]
    assert abs(sum(weights.values()) - 1.0) < 0.001

def test_weights_fallback_on_empty():
    fallback = {"ecmwf": 0.5, "gfs": 0.5}
    result   = weights_from_mae({}, fallback)
    assert result == fallback


# ── Outcome selection ───────────────────────────────────────────────────────

def test_select_exact_match():
    outcomes = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    assert select_best_outcome(26.0, outcomes) == 26

def test_select_rounds_up():
    outcomes = [21, 22, 23, 24, 25, 26, 27]
    assert select_best_outcome(25.6, outcomes) == 26

def test_select_rounds_down():
    outcomes = [21, 22, 23, 24, 25, 26, 27]
    assert select_best_outcome(25.3, outcomes) == 25

def test_select_below_range():
    outcomes = [21, 22, 23, 24, 25]
    # 15°C → closest is 21
    assert select_best_outcome(15.0, outcomes) == 21

def test_select_above_range():
    outcomes = [21, 22, 23, 24, 25]
    # 35°C → closest is 25
    assert select_best_outcome(35.0, outcomes) == 25


# ── Outcome probability ─────────────────────────────────────────────────────

def test_prob_sums_to_1():
    outcomes = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    probs    = all_outcome_probabilities(25.0, 1.5, outcomes)
    total    = sum(probs.values())
    assert abs(total - 1.0) < 0.02   # small float tolerance

def test_prob_peak_at_predicted():
    outcomes = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    probs    = all_outcome_probabilities(26.0, 1.0, outcomes)
    # The bin containing 26 should have highest prob
    best = max(probs, key=probs.get)
    assert best == 26

def test_prob_lowest_bin_catches_tail():
    # If T_predicted is much higher than outcomes, lowest bin should have tiny prob
    outcomes = [21, 22, 23, 24, 25]
    probs    = all_outcome_probabilities(28.0, 1.0, outcomes)
    assert probs[21] < 0.01

def test_prob_highest_bin_catches_tail():
    # If T_predicted is much higher than outcomes, highest bin should have most prob
    outcomes = [21, 22, 23, 24, 25]
    probs    = all_outcome_probabilities(30.0, 1.0, outcomes)
    assert probs[25] > 0.95

def test_prob_with_small_sigma():
    # Very tight spread → almost all probability on nearest bin
    outcomes = [24, 25, 26, 27, 28]
    probs    = all_outcome_probabilities(26.0, 0.2, outcomes)
    assert probs[26] > 0.95

def test_prob_fahrenheit_consistent():
    # Outcomes in °F — should still sum to ~1
    outcomes_f = [68, 70, 72, 74, 76, 78, 80]
    probs      = all_outcome_probabilities(73.0, 3.0, outcomes_f, unit="F")
    total      = sum(probs.values())
    assert abs(total - 1.0) < 0.02


# ── Bias correction ─────────────────────────────────────────────────────────

def test_bias_correction_noop_when_no_history():
    T, s = apply_bias_correction(26.0, 1.5, None)
    assert T == 26.0
    assert s == 1.5

def test_bias_correction_increases_sigma_on_outlier():
    # If ensemble says 35°C but climate mean is 26°C with std=2°C → z=4.5
    climate = {"mean": 26.0, "std": 2.0, "values": [], "n": 30}
    _, s_out = apply_bias_correction(35.0, 1.5, climate)
    assert s_out > 1.5  # sigma should increase

def test_bias_correction_nudges_extreme():
    # Extreme forecast should be nudged toward climatology
    climate = {"mean": 26.0, "std": 2.0, "values": [], "n": 30}
    T_out, _ = apply_bias_correction(40.0, 1.5, climate)
    # Should be between 26 and 40 (pulled toward clim mean)
    assert 26.0 < T_out < 40.0

def test_bias_correction_no_nudge_near_mean():
    climate = {"mean": 26.0, "std": 2.0, "values": [], "n": 30}
    T_out, s_out = apply_bias_correction(26.5, 1.5, climate)
    assert T_out == pytest.approx(26.5, abs=0.01)
    assert s_out == pytest.approx(1.5, abs=0.01)


# ── Expected value and log return ───────────────────────────────────────────

def test_ev_positive_when_pip_above_price():
    # We think p=0.35, market says 0.27 → positive EV
    ev = expected_value(0.35, 0.27)
    assert ev > 0

def test_ev_negative_when_pip_below_price():
    ev = expected_value(0.20, 0.27)
    assert ev < 0

def test_ev_zero_at_boundary():
    ev = expected_value(0.5, 0.0)
    assert ev == 0.0
    ev = expected_value(0.5, 1.0)
    assert ev == 0.0

def test_log_return_positive_edge():
    # ELR is always negative (total loss case dominates) but higher pip → less negative
    elr_high = expected_log_return(0.60, 0.27)
    elr_low  = expected_log_return(0.20, 0.27)
    # Higher pip should give a less negative (better) ELR
    assert elr_high > elr_low

def test_log_return_negative_edge():
    elr = expected_log_return(0.20, 0.27)
    assert elr < 0


# ── Integration: full Buenos Aires scenario ─────────────────────────────────

def test_buenos_aires_scenario():
    """
    Simulate a complete signal calculation for Buenos Aires.
    Model predicts 26.2°C with ±1.1°C spread.
    Market has outcomes [21..30] in °C.
    """
    model_temps = {
        "ecmwf_ifs04":       26.3,
        "gfs_seamless":      25.8,
        "icon_global":       26.1,
        "cma_grapes_global": 26.5,
    }
    outcomes  = list(range(21, 31))    # 21..30 in °C
    mkt_price_26 = 0.27                 # market underprices 26°C

    ensemble = compute_ensemble_stats(model_temps)
    assert ensemble["T_mean"] == pytest.approx(26.175, abs=0.1)
    assert ensemble["T_std"] < 0.5     # tight agreement

    best = select_best_outcome(ensemble["T_mean"], outcomes)
    assert best == 26

    pip = outcome_probability(ensemble["T_mean"], ensemble["T_std"], 26, outcomes)
    edge = pip - mkt_price_26
    assert edge > 0.05                 # expect meaningful positive edge


# ── Integration: Atlanta °F scenario ────────────────────────────────────────

def test_atlanta_fahrenheit_scenario():
    """
    Atlanta market uses °F. Model temps come in °C from Open-Meteo.
    Verify the conversion chain works correctly.
    """
    model_temps_c = {
        "ecmwf_ifs04":  22.0,   # ≈ 71.6°F
        "gfs_seamless": 22.5,   # ≈ 72.5°F
        "icon_global":  21.8,   # ≈ 71.2°F
    }
    outcomes_f = [66, 68, 70, 72, 74, 76, 78, 80]
    unit       = "F"

    ensemble   = compute_ensemble_stats(model_temps_c)
    T_mean_c   = ensemble["T_mean"]
    T_mean_f   = convert_temp(T_mean_c, "C", "F")
    T_std_f    = ensemble["T_std"] * (9.0 / 5.0)

    assert 71.0 < T_mean_f < 74.0     # should be around 72°F

    best_f = select_best_outcome(T_mean_f, outcomes_f)
    assert best_f == 72               # closest bin

    pip = outcome_probability(T_mean_f, T_std_f, 72, outcomes_f, unit)
    assert pip > 0.10                 # should have reasonable probability
