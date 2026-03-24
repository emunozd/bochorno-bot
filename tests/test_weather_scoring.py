"""
tests/test_weather_scoring.py

Tests for WCS and TPS scoring functions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.signals.weather_scoring import compute_wcs, compute_tps, detect_opportunity
from src.config import NWP_FALLBACK_WEIGHTS


OUTCOMES_C = list(range(21, 31))   # 21..30 °C
OUTCOMES_F = list(range(66, 83, 2)) # 66,68,70,...82 °F

MODEL_TEMPS_GOOD = {
    "ecmwf_ifs04":       26.1,
    "gfs_seamless":      26.0,
    "icon_global":       26.2,
    "cma_grapes_global": 26.1,
}

MODEL_TEMPS_SPREAD = {
    "ecmwf_ifs04":       23.0,
    "gfs_seamless":      28.0,
    "icon_global":       25.0,
    "cma_grapes_global": 30.0,
}


# ── WCS ─────────────────────────────────────────────────────────────────────

def test_wcs_high_conf_tight_ensemble():
    wcs = compute_wcs(MODEL_TEMPS_GOOD, NWP_FALLBACK_WEIGHTS, None, horizon_hours=12)
    assert wcs["score"] >= 65
    assert wcs["zone"] in ("HIGH_CONF", "MEDIUM_CONF")
    assert not wcs["blocked"]

def test_wcs_low_conf_wide_spread():
    wcs = compute_wcs(MODEL_TEMPS_SPREAD, NWP_FALLBACK_WEIGHTS, None, horizon_hours=12)
    # Wide spread should lower WCS significantly
    assert wcs["score"] < 65

def test_wcs_blocked_when_sigma_exceeds_max(monkeypatch):
    monkeypatch.setattr("src.signals.weather_scoring.SIGMA_MAX", 2.0)
    # MODEL_TEMPS_SPREAD has sigma ~3°C → should be blocked
    wcs = compute_wcs(MODEL_TEMPS_SPREAD, NWP_FALLBACK_WEIGHTS, None)
    assert wcs["blocked"]
    assert wcs["zone"] == "VERY_LOW"

def test_wcs_breakdown_keys():
    wcs = compute_wcs(MODEL_TEMPS_GOOD, NWP_FALLBACK_WEIGHTS, None)
    brk = wcs["breakdown"]
    assert all(k in brk for k in ("agreement", "skill", "horizon", "climate"))

def test_wcs_better_horizon():
    # 6h horizon should score higher than 36h horizon
    wcs_6h  = compute_wcs(MODEL_TEMPS_GOOD, NWP_FALLBACK_WEIGHTS, None, horizon_hours=6)
    wcs_36h = compute_wcs(MODEL_TEMPS_GOOD, NWP_FALLBACK_WEIGHTS, None, horizon_hours=36)
    assert wcs_6h["score"] > wcs_36h["score"]

def test_wcs_with_stable_climate():
    stable_climate = {"mean": 26.0, "std": 1.5, "values": [], "n": 60}
    wcs_stable = compute_wcs(MODEL_TEMPS_GOOD, NWP_FALLBACK_WEIGHTS, stable_climate)

    variable_climate = {"mean": 26.0, "std": 6.0, "values": [], "n": 60}
    wcs_variable = compute_wcs(MODEL_TEMPS_GOOD, NWP_FALLBACK_WEIGHTS, variable_climate)

    assert wcs_stable["score"] > wcs_variable["score"]


# ── TPS ─────────────────────────────────────────────────────────────────────

def test_tps_selects_correct_outcome():
    mkt_prices = {k: 0.10 for k in OUTCOMES_C}
    mkt_prices[26] = 0.18   # give 26°C a real price
    tps = compute_tps(MODEL_TEMPS_GOOD, NWP_FALLBACK_WEIGHTS, None,
                      OUTCOMES_C, mkt_prices, "C")
    assert tps["best_outcome"] == 26
    assert tps["T_predicted"] is not None
    assert tps["T_std"] is not None

def test_tps_edge_positive_when_underpriced():
    mkt_prices = {k: 0.05 for k in OUTCOMES_C}
    mkt_prices[26] = 0.10  # underpriced (our prob will be ~0.25-0.35)
    tps = compute_tps(MODEL_TEMPS_GOOD, NWP_FALLBACK_WEIGHTS, None,
                      OUTCOMES_C, mkt_prices, "C")
    assert tps["edge"] is not None
    assert tps["edge"] > 0

def test_tps_edge_negative_when_overpriced():
    # Market prices bin 26 at 0.80, but with high spread our pip is only ~0.20
    model_temps_wide = {
        "ecmwf_ifs04": 23.0, "gfs_seamless": 29.0,  # very wide spread
    }
    mkt_prices = {k: 0.05 for k in OUTCOMES_C}
    mkt_prices[26] = 0.80   # overpriced vs our ~0.15 probability
    tps = compute_tps(model_temps_wide, NWP_FALLBACK_WEIGHTS, None,
                      OUTCOMES_C, mkt_prices, "C")
    # Best outcome will be ~26 (midpoint), pip should be low given spread
    if tps["best_outcome"] == 26:
        assert tps["edge"] < 0

def test_tps_unit_f():
    # Atlanta: model temps in °C, outcomes in °F
    model_c = {"ecmwf_ifs04": 22.0, "gfs_seamless": 22.5, "icon_global": 21.8}
    mkt_f   = {k: 0.10 for k in OUTCOMES_F}
    mkt_f[72] = 0.18
    tps = compute_tps(model_c, NWP_FALLBACK_WEIGHTS, None, OUTCOMES_F, mkt_f, "F")
    assert tps["unit"] == "F"
    # T_predicted should be in °F (~71-73)
    assert 68 < tps["T_predicted"] < 76

def test_tps_all_probs_sum_to_1():
    mkt_prices = {k: 0.10 for k in OUTCOMES_C}
    tps   = compute_tps(MODEL_TEMPS_GOOD, NWP_FALLBACK_WEIGHTS, None,
                        OUTCOMES_C, mkt_prices, "C")
    total = sum(tps["all_probs"].values())
    assert abs(total - 1.0) < 0.02

def test_tps_no_outcomes():
    tps = compute_tps(MODEL_TEMPS_GOOD, NWP_FALLBACK_WEIGHTS, None, [], {}, "C")
    assert tps["best_outcome"] is None
    assert tps["edge"] is None

def test_tps_direction_always_long():
    mkt_prices = {k: 0.10 for k in OUTCOMES_C}
    tps = compute_tps(MODEL_TEMPS_GOOD, NWP_FALLBACK_WEIGHTS, None,
                      OUTCOMES_C, mkt_prices, "C")
    assert tps["direction"] == "LONG"


# ── detect_opportunity ──────────────────────────────────────────────────────

def test_opportunity_detected_with_good_signal(monkeypatch):
    monkeypatch.setattr("src.config.WCS_MIN",  65.0)
    monkeypatch.setattr("src.config.EDGE_MIN", 0.05)

    wcs_data = {"score": 75.0, "zone": "MEDIUM_CONF", "blocked": False}
    tps_data = {
        "best_outcome": 26, "best_prob": 0.35,
        "mkt_price": 0.20, "edge": 0.15,
        "T_predicted": 26.2, "T_std": 1.1, "unit": "C",
    }

    import src.config as cfg
    cfg.WATCH_CITIES["TEST_CITY"] = {
        "name": "Test", "temp_unit": "C",
        "token_ids": {26: "abc123"}, "mkt_prices": {26: 0.20}
    }
    opp = detect_opportunity("TEST_CITY", wcs_data, tps_data, pip_final=0.35)
    del cfg.WATCH_CITIES["TEST_CITY"]

    assert opp is not None
    assert opp["outcome_val"] == 26
    assert opp["edge"] > 0.05

def test_no_opportunity_when_blocked():
    wcs_data = {"score": 80.0, "zone": "HIGH_CONF", "blocked": True}
    tps_data = {"best_outcome": 26, "best_prob": 0.35, "mkt_price": 0.20, "edge": 0.15}
    opp = detect_opportunity("BUENOS_AIRES", wcs_data, tps_data, pip_final=0.35)
    assert opp is None

def test_no_opportunity_when_wcs_below_min(monkeypatch):
    monkeypatch.setattr("src.config.WCS_MIN", 65.0)
    wcs_data = {"score": 50.0, "zone": "LOW_CONF", "blocked": False}
    tps_data = {"best_outcome": 26, "best_prob": 0.35, "mkt_price": 0.20, "edge": 0.15}
    opp = detect_opportunity("BUENOS_AIRES", wcs_data, tps_data, pip_final=0.35)
    assert opp is None

def test_no_opportunity_when_edge_too_small(monkeypatch):
    monkeypatch.setattr("src.config.WCS_MIN",  65.0)
    monkeypatch.setattr("src.config.EDGE_MIN", 0.08)
    wcs_data = {"score": 75.0, "zone": "MEDIUM_CONF", "blocked": False}
    tps_data = {
        "best_outcome": 26, "best_prob": 0.25,
        "mkt_price": 0.22, "edge": 0.03,
        "T_predicted": 26.2, "T_std": 1.1, "unit": "C",
    }
    import src.config as cfg
    cfg.WATCH_CITIES["TEST_CITY2"] = {
        "name": "Test2", "temp_unit": "C",
        "token_ids": {26: "abc"}, "mkt_prices": {26: 0.22}
    }
    opp = detect_opportunity("TEST_CITY2", wcs_data, tps_data, pip_final=0.25)
    del cfg.WATCH_CITIES["TEST_CITY2"]
    assert opp is None
