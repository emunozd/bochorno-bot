"""
config.py — Single source of truth for all bochorno-bot configuration.

All values can be overridden via environment variables.
No logic here — only constants and typed settings.
"""

import os
from zoneinfo import ZoneInfo
from typing import Dict, Tuple

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ── Watched cities ─────────────────────────────────────────────────────────
# Each city maps to its Polymarket daily temperature market.
# token_ids is a dict {outcome_value: token_yes_id} discovered at runtime.
# temp_unit must match what Polymarket uses for that city's market.
WATCH_CITIES: Dict[str, dict] = {
    "BUENOS_AIRES": {
        "name":             "Buenos Aires",
        "country":          "AR",
        "lat":              -34.6037,
        "lon":              -58.3816,
        "temp_unit":        "C",
        "timezone":         "America/Argentina/Buenos_Aires",
        "poly_slug_prefix": "highest-temperature-in-buenos-aires-on",
        "poly_slug":        "",          # built dynamically at runtime
        "token_ids":        {},          # {outcome_val: token_yes_id}
        "mkt_prices":       {},          # {outcome_val: float}
        "resolves_hour":    12,
        "resolves_minute":  0,
    },
    "ATLANTA": {
        "name":             "Atlanta",
        "country":          "US",
        "lat":              33.7490,
        "lon":              -84.3880,
        "temp_unit":        "F",
        "timezone":         "America/New_York",
        "poly_slug_prefix": "highest-temperature-in-atlanta-on",
        "poly_slug":        "",
        "token_ids":        {},
        "mkt_prices":       {},
        "resolves_hour":    12,
        "resolves_minute":  0,
    },
    "SEOUL": {
        "name":             "Seoul",
        "country":          "KR",
        "lat":              37.5665,
        "lon":              126.9780,
        "temp_unit":        "C",
        "timezone":         "Asia/Seoul",
        "poly_slug_prefix": "highest-temperature-in-seoul-on",
        "poly_slug":        "",
        "token_ids":        {},
        "mkt_prices":       {},
        "resolves_hour":    12,
        "resolves_minute":  0,
    },
    "SHANGHAI": {
        "name":             "Shanghai",
        "country":          "CN",
        "lat":              31.2304,
        "lon":              121.4737,
        "temp_unit":        "C",
        "timezone":         "Asia/Shanghai",
        "poly_slug_prefix": "highest-temperature-in-shanghai-on",
        "poly_slug":        "",
        "token_ids":        {},
        "mkt_prices":       {},
        "resolves_hour":    12,
        "resolves_minute":  0,
    },
}

# ── NWP models available via Open-Meteo ────────────────────────────────────
# These are fetched in parallel and used to build the ensemble.
# Weights are recalibrated per-city at startup from historical MAE.
# Fallback weights used until calibration data is available.
NWP_MODELS = [
    "ecmwf_ifs04",        # ECMWF IFS — global reference
    "gfs_seamless",       # NOAA GFS  — strong for Americas
    "icon_global",        # DWD ICON  — strong for Europe/Korea
    "cma_grapes_global",  # CMA       — strong for Asia-Pacific
]

NWP_FALLBACK_WEIGHTS = {
    "ecmwf_ifs04":       0.35,
    "gfs_seamless":      0.30,
    "icon_global":       0.20,
    "cma_grapes_global": 0.15,
}

# ── Signal thresholds ──────────────────────────────────────────────────────
WCS_MIN        = float(os.environ.get("WCS_MIN",    "65"))   # Weather Confidence Score
EDGE_MIN       = float(os.environ.get("EDGE_MIN",   "0.08")) # min PIP - market_price
MKT_PRICE_MIN      = float(os.environ.get("MKT_PRICE_MIN",  "0.02"))  # skip illiquid outcomes below this price
COLLAPSE_PRICE     = float(os.environ.get("COLLAPSE_PRICE", "0.03"))  # market collapsed — block city for the day
ONE_ENTRY_PER_DAY  = os.environ.get("ONE_ENTRY_PER_DAY", "true").lower() == "true"  # no re-entry same city same day
SIGMA_MAX      = float(os.environ.get("SIGMA_MAX",  "3.0"))  # block if ensemble spread > this (°C/°F)

# ── Risk management ────────────────────────────────────────────────────────
STOP_LOSS_PCT   = float(os.environ.get("STOP_LOSS_PCT",   "0.35"))
TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", "0.50"))
KELLY_FRACTION  = float(os.environ.get("KELLY_FRACTION",  "0.25"))
MAX_POS_PCT     = float(os.environ.get("MAX_POS_PCT",     "0.15"))
MIN_POS_USDC    = float(os.environ.get("MIN_POS_USDC",    "5.0"))
KELLY_MIN_TRADES= int(os.environ.get("KELLY_MIN_TRADES",  "10"))
KELLY_FALLBACK  = float(os.environ.get("KELLY_FALLBACK",  "0.05"))
CAPITAL_INITIAL = float(os.environ.get("CAPITAL_INITIAL", "500.0"))

# ── CLOB execution ─────────────────────────────────────────────────────────
CLOB_MAX_RETRIES = int(os.environ.get("CLOB_MAX_RETRIES", "3"))
CLOB_RETRY_DELAY = float(os.environ.get("CLOB_RETRY_DELAY", "2.0"))
CLOB_LIMIT_SLIP  = float(os.environ.get("CLOB_LIMIT_SLIP",  "0.02"))

# ── API credentials ────────────────────────────────────────────────────────
# Open-Meteo is free with no key required.
# OWM_API_KEY is optional — used as secondary source for validation.
OWM_API_KEY  = os.environ.get("OWM_API_KEY",        "")
POLY_PK      = os.environ.get("POLY_PRIVATE_KEY",   "")
POLY_FUNDER  = os.environ.get("POLY_FUNDER_ADDRESS","")
POLY_HOST    = "https://clob.polymarket.com"
POLY_CHAIN   = 137  # Polygon mainnet
GAMMA_API    = "https://gamma-api.polymarket.com"

# ── LLM backend ────────────────────────────────────────────────────────────
LLM_BACKEND  = os.environ.get("LLM_BACKEND",  "openai").lower()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL    = os.environ.get("LLM_MODEL",    "llama3.2")
LLM_API_KEY  = os.environ.get("LLM_API_KEY",  "none")

MODEL_CONF_CAPS: Dict[str, float] = {
    "claude":   1.00, "gpt-4": 1.00, "gpt-4o": 1.00,
    "llama3.3": 0.80, "llama-3.3": 0.80, "deepseek": 0.75,
    "llama3.2": 0.45, "llama3.1": 0.50, "mistral": 0.45,
    "llama3":   0.50, "phi": 0.35, "gemma": 0.40,
}

# ── Database ───────────────────────────────────────────────────────────────
DB_FILE           = os.environ.get("BOT_DB", "bot.db")
CLIMATE_HIST_DAYS = int(os.environ.get("CLIMATE_HIST_DAYS", "730"))  # 2 years for calibration