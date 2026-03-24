"""
tests/test_market_discovery.py

Tests for weather market discovery helpers:
  - Slug generation
  - Outcome value parsing
  - Unit detection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
from zoneinfo import ZoneInfo
import pytest

from src.data.weather_data import (
    _make_weather_slug,
    _parse_outcome_value,
    _detect_unit_from_market,
)


# ── Slug generation ─────────────────────────────────────────────────────────

def test_slug_buenos_aires():
    dt   = datetime(2026, 3, 25, tzinfo=ZoneInfo("UTC"))
    slug = _make_weather_slug("highest-temperature-in-buenos-aires-on", dt)
    assert slug == "highest-temperature-in-buenos-aires-on-march-25-2026"

def test_slug_atlanta():
    dt   = datetime(2026, 3, 25, tzinfo=ZoneInfo("UTC"))
    slug = _make_weather_slug("highest-temperature-in-atlanta-on", dt)
    assert slug == "highest-temperature-in-atlanta-on-march-25-2026"

def test_slug_no_leading_zero_on_day():
    dt   = datetime(2026, 3, 5, tzinfo=ZoneInfo("UTC"))
    slug = _make_weather_slug("highest-temperature-in-seoul-on", dt)
    assert "march-5-2026" in slug
    assert "march-05-2026" not in slug

def test_slug_december():
    dt   = datetime(2026, 12, 1, tzinfo=ZoneInfo("UTC"))
    slug = _make_weather_slug("highest-temperature-in-shanghai-on", dt)
    assert "december-1-2026" in slug


# ── Outcome value parsing ───────────────────────────────────────────────────

def test_parse_simple_celsius():
    assert _parse_outcome_value("26°C", "C") == 26

def test_parse_simple_fahrenheit():
    assert _parse_outcome_value("72°F", "F") == 72

def test_parse_or_higher():
    assert _parse_outcome_value("30°C or higher", "C") == 30

def test_parse_or_lower():
    assert _parse_outcome_value("21°C or lower", "C") == 21

def test_parse_or_below():
    assert _parse_outcome_value("5°C or below", "C") == 5

def test_parse_no_unit():
    assert _parse_outcome_value("26", "C") == 26

def test_parse_negative_temp():
    assert _parse_outcome_value("-5°C", "C") == -5

def test_parse_fahrenheit_high():
    assert _parse_outcome_value("100°F or higher", "F") == 100

def test_parse_none_on_garbage():
    result = _parse_outcome_value("No data", "C")
    assert result is None


# ── Unit detection ──────────────────────────────────────────────────────────

def test_detect_celsius():
    market = {"question": "Highest temperature in Seoul on March 25? 11°C"}
    assert _detect_unit_from_market(market) == "C"

def test_detect_fahrenheit():
    market = {"question": "Highest temperature in Atlanta on March 25? 72°F"}
    assert _detect_unit_from_market(market) == "F"

def test_detect_fahrenheit_word():
    market = {"question": "Will temp exceed 70 Fahrenheit?"}
    assert _detect_unit_from_market(market) == "F"

def test_detect_celsius_word():
    market = {"question": "Will temp be above 20 Celsius?"}
    assert _detect_unit_from_market(market) == "C"

def test_detect_none_on_ambiguous():
    market = {"question": "What will the temperature be?"}
    assert _detect_unit_from_market(market) is None

def test_detect_from_description():
    market = {
        "question": "Highest temperature tomorrow?",
        "description": "Resolution in degrees Fahrenheit."
    }
    assert _detect_unit_from_market(market) == "F"


# ── Full outcome extraction simulation ─────────────────────────────────────

def test_outcome_extraction_complete():
    """
    Simulate what _extract_outcome_tokens would return for a
    realistic Polymarket event structure.
    """
    import json
    from src.data.weather_data import _extract_outcome_tokens

    fake_event = {
        "question": "Highest temperature in Buenos Aires on March 25?",
        "markets": [
            {
                "question":      "21°C",
                "clobTokenIds":  json.dumps(["TOKEN_21_YES", "TOKEN_21_NO"]),
                "outcomePrices": json.dumps(["0.006", "0.994"]),
            },
            {
                "question":      "26°C",
                "clobTokenIds":  json.dumps(["TOKEN_26_YES", "TOKEN_26_NO"]),
                "outcomePrices": json.dumps(["0.32", "0.68"]),
            },
            {
                "question":      "30°C or higher",
                "clobTokenIds":  json.dumps(["TOKEN_30_YES", "TOKEN_30_NO"]),
                "outcomePrices": json.dumps(["0.02", "0.98"]),
            },
        ]
    }
    city_cfg = {"temp_unit": "C"}
    token_ids, mkt_prices, unit = _extract_outcome_tokens(fake_event, city_cfg)

    assert token_ids[21]  == "TOKEN_21_YES"
    assert token_ids[26]  == "TOKEN_26_YES"
    assert token_ids[30]  == "TOKEN_30_YES"
    assert mkt_prices[21] == pytest.approx(0.006, abs=0.001)
    assert mkt_prices[26] == pytest.approx(0.32,  abs=0.001)
    assert unit == "C"
