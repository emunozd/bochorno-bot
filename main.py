#!/usr/bin/env python3
"""
main.py — Entry point for bochorno-bot — Polymarket temperature prediction bot.

Architecture mirrors bochorno-bot exactly:
  1. Initialise shared state and dependencies
  2. Run startup sequence (DB, climate history, market discovery)
  3. Drive the main clock loop

Climate model: Open-Meteo NWP ensemble → WCS + TPS → PIP → Kelly → CLOB
"""

import sys
import time
import logging
import threading
import datetime as _dt
from datetime import datetime
from dataclasses import fields as dc_fields

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("bochorno-bot.telegram").setLevel(logging.DEBUG)
logging.getLogger("bochorno-bot.llm").setLevel(logging.WARNING)

from src.config import (
    WATCH_CITIES, POLY_PK, POLY_FUNDER, CAPITAL_INITIAL,
    POLY_HOST, POLY_CHAIN, ET, LLM_BACKEND, LLM_MODEL,
    NWP_FALLBACK_WEIGHTS
)
from src.data import database as DB
from src.data.weather_data import (
    fetch_ensemble, fetch_current_obs, fetch_climate_history,
    discover_weather_markets, fetch_poly_prices, market_is_open
)
from src.signals.weather_indicators import (
    compute_ensemble_stats, weights_from_mae, compute_climate_percentiles,
    select_best_outcome, outcome_probability, all_outcome_probabilities,
    convert_temp, expected_value, expected_log_return
)
from src.signals.weather_scoring import compute_wcs, compute_tps, detect_opportunity
from src.signals.stats import bootstrap_win_rate_ci
from src.llm.weather_analysis import validate_weather_pip
from src.llm.client import is_available as llm_available
from src.trading.engine import open_position, close_position, monitor_position
from src.models import PolyPosition
from src.ui.display import build_layout, console
from src.telegram import bot as tg_bot

try:
    from py_clob_client.client import ClobClient
    HAS_CLOB = True
except ImportError:
    HAS_CLOB = False

from rich.live import Live
from rich.panel import Panel

# ── Live trading client ────────────────────────────────────────────────────
LIVE_MODE  = bool(POLY_PK and POLY_FUNDER and HAS_CLOB)
poly_client = None

if LIVE_MODE:
    try:
        poly_client = ClobClient(
            POLY_HOST, key=POLY_PK, chain_id=POLY_CHAIN,
            signature_type=1, funder=POLY_FUNDER,
        )
        poly_client.set_api_creds(poly_client.create_or_derive_api_creds())
        console.print("[green]✓ Connected to Polymarket CLOB (LIVE mode)[/]")
    except Exception as e:
        console.print(f"[red]⚠ Polymarket CLOB connection failed: {e}[/]")
        LIVE_MODE = False

# ── Shared state ───────────────────────────────────────────────────────────
lock = threading.Lock()

state = {
    # Per-city signal output
    "signals":       {c: {} for c in WATCH_CITIES},
    "pip":           {c: None for c in WATCH_CITIES},
    "pip_validated": {c: {} for c in WATCH_CITIES},

    # Per-city market prices {outcome_val: price}
    "poly_prices":   {c: {} for c in WATCH_CITIES},

    # Per-city forecast cache
    "forecasts":     {c: {} for c in WATCH_CITIES},
    "obs_temps":     {c: None for c in WATCH_CITIES},   # current observed T in °C
    "prev_ensemble": {c: None for c in WATCH_CITIES},   # previous ensemble mean °C

    # Positions and trades
    "positions":     {},
    "trades":        [],
    "capital_usdc":  CAPITAL_INITIAL,
    "peak_capital":  CAPITAL_INITIAL,
    "bootstrap_ci":  None,

    # Engine meta
    "last_update":   "—",
    "last_signal":   "—",
    "status":        "Starting...",
    "fetching":      set(),
    "countdown":     60,
    "engine_paused": False,
    "poly_client":   poly_client,
}


# ── Background task runner ─────────────────────────────────────────────────

def _run(fn):
    def wrapper():
        try:
            fn()
        except Exception as e:
            with lock:
                state["status"] = f"⚠ {fn.__name__}: {str(e)[:55]}"
    return wrapper

def bg(fn):
    threading.Thread(target=_run(fn), daemon=True).start()


# ── Fetch functions ────────────────────────────────────────────────────────

def do_fetch_forecasts():
    with lock: state["fetching"].add("forecasts")
    try:
        for city_key in WATCH_CITIES:
            result = fetch_ensemble(city_key)
            if result:
                with lock:
                    state["forecasts"][city_key] = result
                    state["last_update"] = datetime.now(ET).strftime("%H:%M:%S ET")
    finally:
        with lock: state["fetching"].discard("forecasts")


def do_fetch_obs():
    """Fetch current observed temperature for each city."""
    with lock: state["fetching"].add("obs")
    try:
        for city_key in WATCH_CITIES:
            t = fetch_current_obs(city_key)
            if t is not None:
                with lock:
                    state["obs_temps"][city_key] = t
    finally:
        with lock: state["fetching"].discard("obs")


def do_fetch_prices():
    with lock: state["fetching"].add("prices")
    try:
        for city_key, cfg in WATCH_CITIES.items():
            token_ids = cfg.get("token_ids", {})
            if not token_ids:
                continue
            prices = fetch_poly_prices(city_key, token_ids, poly_client)
            if prices:
                with lock:
                    state["poly_prices"][city_key] = prices
                    # Also update mkt_prices in WATCH_CITIES for scoring
                    WATCH_CITIES[city_key]["mkt_prices"] = prices
    finally:
        with lock: state["fetching"].discard("prices")


def do_run_signals():
    with lock: state["fetching"].add("signal")
    try:
        recent_trades = DB.load_recent_trades(limit=200)

        for city_key, cfg in WATCH_CITIES.items():
            forecast = state["forecasts"].get(city_key, {})
            model_temps = forecast.get("model_temps", {})
            target_date = forecast.get("target_date", "")

            if not model_temps:
                continue

            unit     = cfg.get("temp_unit", "C")
            outcomes = sorted(cfg.get("token_ids", {}).keys())
            mkt_prices = state["poly_prices"].get(city_key, {})

            if not outcomes:
                continue

            # Load calibrated weights from DB
            now_month = datetime.now(ET).month
            mae_dict  = DB.load_model_mae(city_key, now_month)
            weights   = weights_from_mae(mae_dict, NWP_FALLBACK_WEIGHTS) \
                        if mae_dict else NWP_FALLBACK_WEIGHTS

            # Load climate stats for today's day-of-year
            doy      = datetime.now(ET).timetuple().tm_yday
            history  = DB.load_climate_history(city_key, month=now_month)
            clim_stats = compute_climate_percentiles(history, doy)

            # Convert model temps to market unit for display
            model_temps_display = {
                m: round(convert_temp(t, "C", unit), 1)
                for m, t in model_temps.items()
            }

            # Compute ensemble stats (in °C for internal math)
            ensemble = compute_ensemble_stats(model_temps, weights)

            # Compute real horizon: minutes until end_date
            horizon_hours = 12.0  # fallback
            end_date = cfg.get("end_date", "")
            if end_date:
                try:
                    from datetime import timezone
                    end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    mins_left = (end_dt - datetime.now(end_dt.tzinfo)).total_seconds() / 60
                    horizon_hours = max(0.0, mins_left / 60.0)
                except Exception:
                    pass

            obs_temp_c      = state["obs_temps"].get(city_key)
            prev_ens_mean   = state["prev_ensemble"].get(city_key)

            # WCS — now with real horizon, obs consistency, drift detection
            wcs_data = compute_wcs(
                model_temps, weights, clim_stats,
                horizon_hours=horizon_hours,
                prev_ensemble_mean=prev_ens_mean,
                obs_temp_c=obs_temp_c,
            )

            # TPS — now with obs anchor and climatological outlier clip
            tps_data = compute_tps(
                model_temps, weights, clim_stats,
                outcomes, mkt_prices, unit,
                obs_temp_c=obs_temp_c,
                horizon_hours=horizon_hours,
            )

            # Save ensemble mean for drift detection next cycle
            new_ens_mean = ensemble.get("T_mean")
            if new_ens_mean is not None:
                with lock:
                    state["prev_ensemble"][city_key] = new_ens_mean

            # PIP raw
            pip_raw = tps_data.get("best_prob") or 0.5

            # Bayesian LLM validation
            pip_validated = {}
            pip_final     = pip_raw
            if (wcs_data["score"] >= 65 and not wcs_data["blocked"]
                    and tps_data.get("best_outcome") is not None):
                pip_validated = validate_weather_pip(
                    city_key    = city_key,
                    city_name   = cfg["name"],
                    country     = cfg.get("country", ""),
                    target_date = target_date,
                    T_predicted = tps_data.get("T_predicted", 0),
                    T_std       = tps_data.get("T_std", 1.5),
                    unit        = unit,
                    best_outcome= tps_data["best_outcome"],
                    pip         = pip_raw,
                    mkt_price   = tps_data.get("mkt_price", 0.5),
                    wcs         = wcs_data["score"],
                    model_temps = model_temps_display,
                    n_models    = ensemble.get("n_models", 0),
                )
                if pip_validated.get("valid", True):
                    pip_final = pip_validated.get("adjusted_pip", pip_raw)

            # Detect opportunity (recalculate edge with validated PIP)
            opp = detect_opportunity(city_key, wcs_data, tps_data, pip_final)

            # Build signal dict for display and Telegram
            signal = {
                "wcs":          wcs_data["score"],
                "wcs_zone":     wcs_data["zone"],
                "wcs_blocked":  wcs_data["blocked"],
                "wcs_breakdown":wcs_data["breakdown"],
                "best_outcome": tps_data.get("best_outcome"),
                "best_prob":    pip_final,
                "mkt_price":    tps_data.get("mkt_price"),
                "edge":         round(pip_final - tps_data.get("mkt_price", pip_final), 4)
                                if tps_data.get("mkt_price") else None,
                "T_predicted":  tps_data.get("T_predicted"),
                "T_std":        tps_data.get("T_std"),
                "unit":         unit,
                "all_probs":    tps_data.get("all_probs", {}),
                "opportunity":  opp,
                "model_temps":  model_temps_display,
                "obs_temp_c":   obs_temp_c,
                "horizon_h":    round(horizon_hours, 1),
            }

            with lock:
                state["signals"][city_key]       = signal
                state["pip"][city_key]           = pip_raw
                state["pip_validated"][city_key] = pip_validated
                state["last_signal"]             = datetime.now(ET).strftime("%H:%M:%S ET")

            # Monitor open position
            pos = state["positions"].get(city_key)
            if pos and pos.status == "OPEN":
                cur_price = state["poly_prices"].get(city_key, {}).get(pos.outcome_val)
                monitor_position(city_key, pos, cur_price,
                                 wcs_data["score"], poly_client, state)
                continue

            # Open new position
            if opp and city_key not in state["positions"]:
                open_position(
                    city_key, opp, state["capital_usdc"],
                    recent_trades, poly_client, state
                )

        # Bootstrap CI
        current_count = len(recent_trades)
        if current_count != state.get("_last_bootstrap_n", -1):
            ci = bootstrap_win_rate_ci(recent_trades)
            with lock:
                state["bootstrap_ci"]      = ci
                state["_last_bootstrap_n"] = current_count

    finally:
        with lock: state["fetching"].discard("signal")


# ── Startup ────────────────────────────────────────────────────────────────

def startup():
    console.print(Panel.fit(
        "[bold cyan]bochorno-bot[/]\n"
        "[dim]Open-Meteo NWP ensemble · ERA5 history · LLM-agnostic · SQLite[/]\n"
        "[dim]WCS · TPS · PIP · Bayesian validation · Kelly · CLOB retry[/]\n"
        f"[dim]Mode: {'LIVE (Polymarket CLOB)' if LIVE_MODE else 'PAPER (no real orders)'}[/]\n"
        f"[dim]LLM:  {LLM_BACKEND}/{LLM_MODEL.split('/')[-1]}  "
        f"({'available' if llm_available() else 'unavailable'})[/]",
        border_style="cyan",
    ))

    # Database
    DB.init_db()
    for city_key, cfg in WATCH_CITIES.items():
        DB.upsert_city(
            city_key, cfg["name"], cfg.get("country", ""),
            cfg["lat"], cfg["lon"], cfg["temp_unit"], cfg.get("poly_slug", "")
        )

    # Recover portfolio
    portfolio = DB.load_portfolio()
    if portfolio:
        state["capital_usdc"] = portfolio["capital_usdc"]
        state["peak_capital"] = portfolio["peak_capital"]
        console.print(f"[dim]Capital restored: ${portfolio['capital_usdc']:.2f} USDC[/]")
    else:
        DB.save_portfolio(state["capital_usdc"], state["peak_capital"])

    # Recover open positions
    open_pos = DB.load_open_positions()
    if open_pos:
        console.print(f"[yellow]Recovering {len(open_pos)} open position(s)...[/]")
        valid_fields = {f.name for f in dc_fields(PolyPosition)}
        for city_key, p in open_pos.items():
            try:
                p["entry_time"] = datetime.fromisoformat(p["entry_time"])
                state["positions"][city_key] = PolyPosition(
                    **{k: v for k, v in p.items() if k in valid_fields}
                )
                state["capital_usdc"] -= p["usdc_spent"]
                console.print(f"[dim]  {city_key}: {p['outcome_val']}° ${p['usdc_spent']:.2f}[/]")
            except Exception as e:
                console.print(f"[red]  Could not restore {city_key}: {e}[/]")

    # Load climate history (ERA5)
    console.print("[dim]Loading climate history (ERA5)...[/]")
    for city_key in WATCH_CITIES:
        fetch_climate_history(city_key, years=2)

    # Discover today's markets
    console.print("[dim]Discovering today's temperature markets...[/]")
    found = discover_weather_markets(WATCH_CITIES)
    if found:
        for city_key, info in found.items():
            console.print(
                f"[green]✓ {city_key}: {len(info['outcomes'])} outcomes "
                f"{info['outcomes']} unit={info['unit']}[/]"
            )
    else:
        console.print("[yellow]⚠ No markets found. Check slugs in config.py[/]")

    # Initial data fetch
    console.print("[dim]Fetching initial forecasts...[/]")
    threads = [
        threading.Thread(target=_run(do_fetch_forecasts), daemon=True),
        threading.Thread(target=_run(do_fetch_prices), daemon=True),
        threading.Thread(target=_run(do_fetch_obs), daemon=True),
    ]
    for t in threads: t.start()
    for t in threads: t.join(timeout=45)

    do_run_signals()
    with lock:
        state["status"] = "Running"


# ── Main loop ──────────────────────────────────────────────────────────────

def main():
    startup()
    tg_bot.init(state, lock)

    _last_forecast_min  = -1
    _last_price_ts      = 0.0
    _last_obs_ts        = 0.0
    _last_sig_min       = -1
    _last_market_h      = -1

    with Live(console=console, refresh_per_second=2, screen=True) as live:
        while True:
            now = datetime.now(ET)
            with lock:
                state["countdown"] = 60 - now.second

            live.update(build_layout(state, LIVE_MODE))
            time.sleep(0.5)

            # Polymarket prices: every 30 seconds
            if time.time() - _last_price_ts >= 30:
                _last_price_ts = time.time()
                bg(do_fetch_prices)

            # Current observed temperature: every 15 minutes
            if time.time() - _last_obs_ts >= 900:
                _last_obs_ts = time.time()
                bg(do_fetch_obs)

            # NWP forecasts: every 30 minutes (models update 2-4x/day)
            if now.minute in (0, 30) and now.minute != _last_forecast_min:
                _last_forecast_min = now.minute
                bg(do_fetch_forecasts)

            # Signal engine: every minute
            if now.minute != _last_sig_min:
                _last_sig_min = now.minute
                bg(do_run_signals)

            # Market re-discovery: every hour (for rollover to next day)
            if now.hour != _last_market_h:
                _last_market_h = now.hour
                def _do_discovery():
                    found = discover_weather_markets(WATCH_CITIES)
                    for city_key, info in found.items():
                        # Close orphaned positions if market rolled over
                        with lock:
                            pos = state["positions"].get(city_key)
                        if pos and pos.outcome_val not in info["token_ids"]:
                            last_pp = state["poly_prices"].get(city_key, {})
                            exit_p  = last_pp.get(pos.outcome_val, pos.entry_price)
                            with lock:
                                close_position(city_key, pos, "MARKET_EXPIRED",
                                               exit_p, poly_client, state)
                bg(_do_discovery)

            # Daily snapshot at 16:00 ET
            if now.hour == 16 and now.minute == 0:
                today    = now.strftime("%Y-%m-%d")
                recent   = DB.load_recent_trades(limit=100)
                today_ts = [t for t in recent if t["time"][:10] == today]
                DB.save_daily_snapshot(state["capital_usdc"], today_ts)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped. Goodbye.[/]")