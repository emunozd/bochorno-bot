"""
data/weather_data.py — Weather data fetching and Polymarket market discovery.

Responsibilities:
  1. Fetch NWP ensemble forecasts from Open-Meteo (free, no key required)
  2. Fetch secondary validation from OpenWeatherMap (optional)
  3. Download ERA5 climate history from Open-Meteo Historical API
  4. Discover temperature markets on Polymarket and extract all outcome token IDs
"""

import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

from src.config import (
    WATCH_CITIES, NWP_MODELS, GAMMA_API, OWM_API_KEY, ET, UTC
)
from src.data import database as DB

log = logging.getLogger("bochorno-bot.weather_data")

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORY  = "https://archive-api.open-meteo.com/v1/archive"
OWM_CURRENT         = "https://api.openweathermap.org/data/2.5/weather"


# ── Forecast fetching ───────────────────────────────────────────────────────

def fetch_ensemble(city_key: str) -> Optional[dict]:
    """
    Fetch T_max forecast from all NWP models for the target date.
    Returns dict: {model_name: T_max_celsius, target_date: str} or None.
    Open-Meteo always returns °C — unit conversion happens in weather_indicators.py.
    """
    cfg = WATCH_CITIES.get(city_key)
    if not cfg:
        return None

    local_tz  = ZoneInfo(cfg["timezone"])
    local_now = datetime.now(local_tz)
    target_dt = local_now.date()

    params = {
        "latitude":      cfg["lat"],
        "longitude":     cfg["lon"],
        "daily":         "temperature_2m_max",
        "models":        ",".join(NWP_MODELS),
        "forecast_days": 2,
        "timezone":      cfg["timezone"],
    }

    try:
        r = requests.get(OPEN_METEO_FORECAST, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"Open-Meteo forecast failed for {city_key}: {e}")
        return None

    daily  = data.get("daily", {})
    dates  = daily.get("time", [])

    target_str = str(target_dt)
    try:
        idx = dates.index(target_str)
    except ValueError:
        tomorrow_str = str(target_dt + timedelta(days=1))
        try:
            idx = dates.index(tomorrow_str)
            target_str = tomorrow_str
        except ValueError:
            log.warning(f"Target date {target_str} not in Open-Meteo response for {city_key}")
            return None

    model_temps = {}
    for model in NWP_MODELS:
        key = "temperature_2m_max"
        val = daily.get(key, [None] * (idx + 1))
        if isinstance(val, list) and idx < len(val) and val[idx] is not None:
            model_temps[model] = float(val[idx])
        else:
            model_key = f"{key}_{model}"
            val2 = daily.get(model_key, [None] * (idx + 1))
            if isinstance(val2, list) and idx < len(val2) and val2[idx] is not None:
                model_temps[model] = float(val2[idx])

    if not model_temps:
        t_max = daily.get("temperature_2m_max", [])
        if isinstance(t_max, list) and idx < len(t_max) and t_max[idx] is not None:
            for model in NWP_MODELS:
                model_temps[model] = float(t_max[idx])

    if not model_temps:
        log.warning(f"No model temperature data extracted for {city_key}")
        return None

    log.info(f"{city_key} ensemble ({target_str}): {model_temps}")
    return {"model_temps": model_temps, "target_date": target_str}


def fetch_current_obs(city_key: str) -> Optional[float]:
    """Fetch current observed temperature (°C) for position monitoring."""
    cfg = WATCH_CITIES.get(city_key)
    if not cfg:
        return None

    if OWM_API_KEY:
        try:
            r = requests.get(OWM_CURRENT, params={
                "lat": cfg["lat"], "lon": cfg["lon"],
                "appid": OWM_API_KEY, "units": "metric"
            }, timeout=6)
            return float(r.json()["main"]["temp"])
        except Exception:
            pass

    try:
        r = requests.get(OPEN_METEO_FORECAST, params={
            "latitude":  cfg["lat"],
            "longitude": cfg["lon"],
            "current":   "temperature_2m",
            "timezone":  cfg["timezone"],
        }, timeout=6)
        return float(r.json()["current"]["temperature_2m"])
    except Exception as e:
        log.warning(f"Could not fetch current obs for {city_key}: {e}")
        return None


# ── Climate history ─────────────────────────────────────────────────────────

def fetch_climate_history(city_key: str, years: int = 2) -> bool:
    """Download ERA5 T_max history. Stored in °C. Returns True on success."""
    cfg = WATCH_CITIES.get(city_key)
    if not cfg:
        return False

    if DB.is_hist_loaded(city_key) and DB.climate_history_count(city_key) > 300:
        log.info(f"{city_key} climate history already loaded ({DB.climate_history_count(city_key)} records)")
        return True

    end_date   = (datetime.now(UTC) - timedelta(days=5)).strftime("%Y-%m-%d")
    start_date = (datetime.now(UTC) - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    try:
        r = requests.get(OPEN_METEO_HISTORY, params={
            "latitude":   cfg["lat"],
            "longitude":  cfg["lon"],
            "start_date": start_date,
            "end_date":   end_date,
            "daily":      "temperature_2m_max",
            "timezone":   cfg["timezone"],
        }, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.error(f"Climate history fetch failed for {city_key}: {e}")
        return False

    dates  = data.get("daily", {}).get("time", [])
    t_maxs = data.get("daily", {}).get("temperature_2m_max", [])

    records = [
        {"date": d, "T_max": float(t)}
        for d, t in zip(dates, t_maxs)
        if t is not None
    ]

    if records:
        DB.save_climate_history(city_key, records)
        DB.set_hist_loaded(city_key)
        log.info(f"{city_key} climate history loaded: {len(records)} records")
        return True

    return False


# ── Polymarket market discovery ─────────────────────────────────────────────

def _make_weather_slug(prefix: str, dt: datetime) -> str:
    """Build slug: highest-temperature-in-buenos-aires-on-march-25-2026"""
    month = dt.strftime("%B").lower()
    day   = str(dt.day)
    year  = str(dt.year)
    return f"{prefix}-{month}-{day}-{year}"


def _parse_outcome_value(question: str) -> Optional[int]:
    """
    Extract integer temperature value from a Polymarket outcome question.
    Works for any unit — the unit is detected separately at the event level.

    Examples:
      "26°C"           → 26
      "72°F"           → 72
      "30°C or higher" → 30
      "21°C or lower"  → 21
      "-5°C"           → -5
    """
    cleaned = question.replace("°C", "").replace("°F", "").replace("°", "").strip()
    match = re.search(r"-?\d+", cleaned)
    if match:
        return int(match.group())
    return None


def _detect_unit_from_event(event: dict, city_cfg: dict) -> str:
    """
    Detect temperature unit for a market event.

    Priority:
      1. Outcome questions inside markets — most reliable ('26°C', '72°F')
      2. Event-level question / description text
      3. City config fallback (never trust silently — log a warning if used)

    Returns 'C' or 'F'.
    """
    # 1. Scan individual outcome questions — they carry the actual symbol
    for mkt in event.get("markets", []):
        q = mkt.get("question", "") or mkt.get("outcomeName", "")
        if "°F" in q:
            return "F"
        if "°C" in q:
            return "C"

    # 2. Event-level text
    text = event.get("question", "") + " " + str(event.get("description", ""))
    if "°F" in text or "Fahrenheit" in text or "fahrenheit" in text:
        return "F"
    if "°C" in text or "Celsius" in text or "celsius" in text:
        return "C"

    # 3. Fall back to config — log so it's visible
    fallback = city_cfg.get("temp_unit", "C")
    log.warning(
        f"Could not detect unit from market data — using config fallback '{fallback}'"
    )
    return fallback


def _parse_end_date(event: dict) -> Optional[str]:
    """
    Extract resolution date/time from event.
    Returns ISO string or None.
    Checks endDate, end_date, and nested market endDate.
    """
    for key in ("endDate", "end_date", "endDateIso"):
        val = event.get(key)
        if val:
            return str(val)

    # Try first nested market
    markets = event.get("markets", [])
    if markets:
        for key in ("endDate", "end_date"):
            val = markets[0].get(key)
            if val:
                return str(val)

    return None


def _fetch_event_by_slug(slug: str) -> Optional[dict]:
    """Fetch a Polymarket event by its slug from Gamma API."""
    try:
        r = requests.get(f"{GAMMA_API}/events",
                         params={"slug": slug}, timeout=8)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict) and data.get("slug") == slug:
            return data
    except Exception as e:
        log.debug(f"Event fetch failed for slug={slug}: {e}")
    return None


def _extract_outcome_tokens(
    event: dict,
    city_cfg: dict,
) -> Tuple[Dict[int, str], Dict[int, float], str]:
    """
    Extract {outcome_val: token_yes_id}, {outcome_val: price}, and unit
    from a Polymarket event dict.
    """
    token_ids  = {}
    mkt_prices = {}

    unit = _detect_unit_from_event(event, city_cfg)

    markets = event.get("markets", [])
    if not markets:
        markets = event.get("market", [])
        if not isinstance(markets, list):
            markets = [markets] if markets else []

    for mkt in markets:
        question    = mkt.get("question", "") or mkt.get("outcomeName", "")
        outcome_val = _parse_outcome_value(question)
        if outcome_val is None:
            continue

        # Token IDs
        raw = mkt.get("clobTokenIds", [])
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = []
        if isinstance(raw, list) and len(raw) >= 1:
            token_ids[outcome_val] = str(raw[0])  # index 0 = YES token

        # Prices
        prices_raw = mkt.get("outcomePrices", [])
        if isinstance(prices_raw, str):
            try:
                prices_raw = json.loads(prices_raw)
            except Exception:
                prices_raw = []
        if isinstance(prices_raw, list) and len(prices_raw) >= 1:
            try:
                mkt_prices[outcome_val] = float(prices_raw[0])
            except (ValueError, TypeError):
                pass

        if outcome_val not in mkt_prices:
            ask = mkt.get("bestAsk")
            bid = mkt.get("bestBid")
            if ask and bid:
                try:
                    mkt_prices[outcome_val] = round((float(ask) + float(bid)) / 2, 3)
                except (ValueError, TypeError):
                    pass

    return token_ids, mkt_prices, unit


def discover_weather_markets(watch_cities: dict) -> Dict[str, dict]:
    """
    Discover today's active temperature markets for each watched city.

    For each market found, stores in watch_cities:
      - token_ids:   {outcome_val: token_yes_id}
      - mkt_prices:  {outcome_val: price}
      - temp_unit:   detected unit ('C' or 'F')
      - poly_slug:   slug used
      - end_date:    ISO resolution datetime string
      - target_date: YYYY-MM-DD of the market being traded

    Returns dict of {city_key: info} for found markets.
    """
    found: Dict[str, dict] = {}
    now      = datetime.now(ET)
    tomorrow = now + timedelta(days=1)

    for city_key, cfg in watch_cities.items():
        prefix = cfg.get("poly_slug_prefix", "")
        if not prefix:
            continue

        event     = None
        slug_used = ""

        for dt in [now, tomorrow]:
            slug  = _make_weather_slug(prefix, dt)
            event = _fetch_event_by_slug(slug)
            if event:
                slug_used = slug
                break

        if not event:
            city_name_lower = cfg["name"].lower().replace(" ", "-")
            try:
                r = requests.get(f"{GAMMA_API}/markets", params={
                    "active": True, "closed": False,
                    "tag_slug": "weather", "limit": 50
                }, timeout=8)
                for mkt in r.json():
                    q = mkt.get("question", "").lower()
                    if city_name_lower in q and "temperature" in q:
                        event     = {"markets": [mkt], "question": mkt.get("question", "")}
                        slug_used = mkt.get("slug", "")
                        break
            except Exception:
                pass

        if not event:
            log.info(f"No market found for {city_key}")
            continue

        token_ids, mkt_prices, unit = _extract_outcome_tokens(event, cfg)

        if not token_ids:
            log.warning(f"{city_key}: event found but no outcome tokens extracted")
            continue

        end_date    = _parse_end_date(event)
        target_date = slug_used.split("-on-")[-1] if "-on-" in slug_used else ""

        # Log clearly — unit mismatch is now informational only, detection wins
        config_unit = cfg.get("temp_unit", "C")
        if unit != config_unit:
            log.info(
                f"{city_key}: market unit is '{unit}' "
                f"(config had '{config_unit}') — using '{unit}'"
            )

        info = {
            "token_ids":   token_ids,
            "mkt_prices":  mkt_prices,
            "unit":        unit,
            "slug":        slug_used,
            "question":    event.get("question", ""),
            "outcomes":    sorted(token_ids.keys()),
            "end_date":    end_date,
            "target_date": target_date,
        }
        found[city_key] = info

        watch_cities[city_key]["token_ids"]   = token_ids
        watch_cities[city_key]["mkt_prices"]  = mkt_prices
        watch_cities[city_key]["temp_unit"]   = unit
        watch_cities[city_key]["poly_slug"]   = slug_used
        watch_cities[city_key]["end_date"]    = end_date
        watch_cities[city_key]["target_date"] = target_date

        log.info(
            f"{city_key}: {len(token_ids)} outcomes {sorted(token_ids.keys())} "
            f"unit={unit} end_date={end_date} slug={slug_used}"
        )

    return found


def market_is_open(city_key: str, block_minutes_before: int = 30) -> bool:
    """
    Returns False if the market resolves within `block_minutes_before` minutes
    or has already resolved. Prevents entering positions too close to cutoff.
    """
    cfg      = WATCH_CITIES.get(city_key, {})
    end_date = cfg.get("end_date")
    if not end_date:
        return True  # unknown — allow, rely on WCS to filter

    try:
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        mins_left = (end_dt - datetime.now(end_dt.tzinfo)).total_seconds() / 60
        if mins_left < block_minutes_before:
            log.info(f"{city_key}: market closes in {mins_left:.0f}min — blocking entry")
            return False
        return True
    except Exception:
        return True  # parse error — allow


def fetch_poly_prices(
    city_key: str,
    token_ids: Dict[int, str],
    poly_client=None,
) -> Dict[int, float]:
    """
    Fetch current YES prices for all outcomes of a city.
    Returns {outcome_val: price}.
    """
    prices = {}

    for outcome_val, token_id in token_ids.items():
        if not token_id:
            continue

        price = None

        if poly_client:
            try:
                buy_r  = requests.get(
                    "https://clob.polymarket.com/book",
                    params={"token_id": token_id, "side": "buy"},
                    timeout=5
                )
                sell_r = requests.get(
                    "https://clob.polymarket.com/book",
                    params={"token_id": token_id, "side": "sell"},
                    timeout=5
                )
                buy_p  = float(buy_r.json().get("price", 0))
                sell_p = float(sell_r.json().get("price", 0))
                if buy_p > 0 and sell_p > 0:
                    price = round((buy_p + sell_p) / 2, 3)
                elif buy_p > 0:
                    price = buy_p
            except Exception:
                pass

        if price is None:
            try:
                r = requests.get(
                    f"{GAMMA_API}/markets",
                    params={"clobTokenIds": token_id},
                    timeout=5
                )
                data = r.json()
                if data:
                    mkt = data[0] if isinstance(data, list) else data
                    raw = mkt.get("outcomePrices", [])
                    if isinstance(raw, str):
                        raw = json.loads(raw)
                    if raw:
                        price = float(raw[0])
            except Exception:
                pass

        if price is not None:
            prices[outcome_val] = round(price, 3)

    return prices