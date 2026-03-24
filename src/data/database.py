"""
data/database.py — SQLite persistence layer.

Adapted from bochorno-bot: assets→cities, candles removed,
weather_forecasts and climate_history tables added.
Everything else (trades, positions, portfolio, bot_config) is identical.
"""

import sqlite3
import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import asdict

from src.config import DB_FILE, ET


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS cities (
            city_key        TEXT PRIMARY KEY,
            name            TEXT,
            country         TEXT,
            lat             REAL,
            lon             REAL,
            temp_unit       TEXT DEFAULT 'C',
            poly_slug       TEXT DEFAULT '',
            hist_loaded     INTEGER DEFAULT 0,
            updated_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS weather_forecasts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            city_key    TEXT NOT NULL,
            target_date TEXT NOT NULL,
            unit        TEXT NOT NULL,
            T_predicted REAL,
            T_std       REAL,
            model_temps TEXT DEFAULT '{}',
            model_weights TEXT DEFAULT '{}',
            fetched_at  TEXT,
            UNIQUE(city_key, target_date, fetched_at)
        );

        CREATE TABLE IF NOT EXISTS climate_history (
            city_key    TEXT NOT NULL,
            date        TEXT NOT NULL,
            T_max       REAL NOT NULL,
            PRIMARY KEY (city_key, date)
        );

        CREATE TABLE IF NOT EXISTS model_mae (
            city_key    TEXT NOT NULL,
            model       TEXT NOT NULL,
            month       INTEGER NOT NULL,
            mae         REAL NOT NULL,
            n_samples   INTEGER DEFAULT 0,
            updated_at  TEXT,
            PRIMARY KEY (city_key, model, month)
        );

        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            city_key    TEXT,
            outcome_val INTEGER,
            side        TEXT,
            entry_price REAL,
            exit_price  REAL,
            shares      REAL,
            pnl         REAL,
            pnl_pct     REAL,
            log_return  REAL DEFAULT 0,
            ev_at_entry REAL DEFAULT 0,
            pip_at_entry REAL DEFAULT 0.5,
            duration    TEXT,
            reason      TEXT,
            time        TEXT
        );

        CREATE TABLE IF NOT EXISTS positions (
            city_key    TEXT PRIMARY KEY,
            outcome_val INTEGER,
            side        TEXT,
            token_id    TEXT,
            shares      REAL,
            entry_price REAL,
            usdc_spent  REAL,
            stop_loss   REAL DEFAULT 0,
            take_profit REAL DEFAULT 1,
            entry_time  TEXT,
            entry_wcs   REAL DEFAULT 0,
            entry_pip   REAL DEFAULT 0.5,
            order_id    TEXT DEFAULT '',
            status      TEXT DEFAULT 'OPEN'
        );

        CREATE TABLE IF NOT EXISTS portfolio (
            id              INTEGER PRIMARY KEY DEFAULT 1,
            capital_usdc    REAL,
            peak_capital    REAL,
            updated_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS bot_config (
            key         TEXT PRIMARY KEY,
            value       TEXT
        );

        CREATE TABLE IF NOT EXISTS daily_snapshots (
            date            TEXT PRIMARY KEY,
            capital_usdc    REAL,
            n_trades        INTEGER,
            day_pnl         REAL,
            updated_at      TEXT
        );
        """)


# ── Cities ─────────────────────────────────────────────────────────────────

def upsert_city(city_key: str, name: str, country: str, lat: float,
                lon: float, temp_unit: str, poly_slug: str = "") -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT INTO cities (city_key, name, country, lat, lon,
                                temp_unit, poly_slug, updated_at)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(city_key) DO UPDATE SET
                name=excluded.name, temp_unit=excluded.temp_unit,
                poly_slug=excluded.poly_slug, updated_at=excluded.updated_at
        """, (city_key, name, country, lat, lon, temp_unit, poly_slug,
              datetime.now(ET).isoformat()))


def set_hist_loaded(city_key: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE cities SET hist_loaded=1, updated_at=? WHERE city_key=?",
            (datetime.now(ET).isoformat(), city_key)
        )


def is_hist_loaded(city_key: str) -> bool:
    with _conn() as conn:
        row = conn.execute(
            "SELECT hist_loaded FROM cities WHERE city_key=?", (city_key,)
        ).fetchone()
        return bool(row and row["hist_loaded"])


# ── Weather forecasts ───────────────────────────────────────────────────────

def save_forecast(city_key: str, target_date: str, unit: str,
                  T_predicted: float, T_std: float,
                  model_temps: dict, model_weights: dict) -> None:
    now = datetime.now(ET).isoformat()
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO weather_forecasts
                (city_key, target_date, unit, T_predicted, T_std,
                 model_temps, model_weights, fetched_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (city_key, target_date, unit, T_predicted, T_std,
              json.dumps(model_temps), json.dumps(model_weights), now))


def load_latest_forecast(city_key: str, target_date: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute("""
            SELECT * FROM weather_forecasts
            WHERE city_key=? AND target_date=?
            ORDER BY fetched_at DESC LIMIT 1
        """, (city_key, target_date)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["model_temps"]   = json.loads(d.get("model_temps", "{}"))
    d["model_weights"] = json.loads(d.get("model_weights", "{}"))
    return d


# ── Climate history ─────────────────────────────────────────────────────────

def save_climate_history(city_key: str, records: List[dict]) -> None:
    """records: list of {date: 'YYYY-MM-DD', T_max: float}"""
    with _conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO climate_history (city_key, date, T_max) VALUES (?,?,?)",
            [(city_key, r["date"], r["T_max"]) for r in records]
        )


def load_climate_history(city_key: str,
                         month: Optional[int] = None,
                         day_of_year_range: Optional[tuple] = None) -> List[dict]:
    """
    Load historical T_max records for a city.
    Optionally filter by calendar month (1-12).
    """
    with _conn() as conn:
        if month:
            rows = conn.execute("""
                SELECT date, T_max FROM climate_history
                WHERE city_key=? AND CAST(strftime('%m', date) AS INTEGER)=?
                ORDER BY date
            """, (city_key, month)).fetchall()
        else:
            rows = conn.execute(
                "SELECT date, T_max FROM climate_history WHERE city_key=? ORDER BY date",
                (city_key,)
            ).fetchall()
    return [dict(r) for r in rows]


def climate_history_count(city_key: str) -> int:
    with _conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as n FROM climate_history WHERE city_key=?", (city_key,)
        ).fetchone()
        return row["n"] if row else 0


# ── Model MAE (calibration) ─────────────────────────────────────────────────

def save_model_mae(city_key: str, model: str, month: int,
                   mae: float, n_samples: int) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO model_mae
                (city_key, model, month, mae, n_samples, updated_at)
            VALUES (?,?,?,?,?,?)
        """, (city_key, model, month, mae, n_samples, datetime.now(ET).isoformat()))


def load_model_mae(city_key: str, month: int) -> Dict[str, float]:
    """Returns {model: mae} for a city+month. Empty dict if not calibrated."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT model, mae FROM model_mae WHERE city_key=? AND month=?",
            (city_key, month)
        ).fetchall()
    return {r["model"]: r["mae"] for r in rows}


# ── Trades ──────────────────────────────────────────────────────────────────

def save_trade(trade) -> int:
    d = trade if isinstance(trade, dict) else asdict(trade)
    with _conn() as conn:
        cur = conn.execute("""
            INSERT INTO trades
                (city_key, outcome_val, side, entry_price, exit_price,
                 shares, pnl, pnl_pct, log_return, ev_at_entry,
                 pip_at_entry, duration, reason, time)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (d["city_key"], d.get("outcome_val", 0), d["side"],
              d["entry_price"], d["exit_price"], d["shares"],
              d["pnl"], d["pnl_pct"], d.get("log_return", 0),
              d.get("ev_at_entry", 0), d.get("pip_at_entry", 0.5),
              d["duration"], d["reason"], d["time"]))
        return cur.lastrowid


def load_recent_trades(limit: int = 200) -> List[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def trade_stats() -> dict:
    with _conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS winners,
                SUM(pnl)         AS total_pnl,
                SUM(log_return)  AS total_log_return,
                AVG(ev_at_entry) AS avg_ev
            FROM trades
        """).fetchone()
    return dict(row) if row else {}


# ── Positions ───────────────────────────────────────────────────────────────

def save_position(city_key: str, pos) -> None:
    d = pos if isinstance(pos, dict) else asdict(pos)
    entry_time = d["entry_time"]
    if not isinstance(entry_time, str):
        entry_time = entry_time.isoformat()
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO positions
                (city_key, outcome_val, side, token_id, shares, entry_price,
                 usdc_spent, stop_loss, take_profit, entry_time,
                 entry_wcs, entry_pip, order_id, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (city_key, d.get("outcome_val", 0), d["side"],
              d.get("token_id", ""), d["shares"], d["entry_price"],
              d["usdc_spent"], d.get("stop_loss", 0), d.get("take_profit", 1),
              entry_time, d.get("entry_wcs", 0), d.get("entry_pip", 0.5),
              d.get("order_id", ""), d.get("status", "OPEN")))


def delete_position(city_key: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM positions WHERE city_key=?", (city_key,))


def load_open_positions() -> Dict[str, dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE status='OPEN'"
        ).fetchall()
    return {r["city_key"]: dict(r) for r in rows}


# ── Portfolio ────────────────────────────────────────────────────────────────

def save_portfolio(capital: float, peak: float) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO portfolio (id, capital_usdc, peak_capital, updated_at)
            VALUES (1, ?, ?, ?)
        """, (capital, peak, datetime.now(ET).isoformat()))


def load_portfolio() -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM portfolio WHERE id=1").fetchone()
    return dict(row) if row else None


# ── Bot config (owner_chat_id etc.) ─────────────────────────────────────────

def set_bot_config(key: str, value: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO bot_config (key, value) VALUES (?,?)",
            (key, value)
        )


def get_bot_config(key: str) -> Optional[str]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT value FROM bot_config WHERE key=?", (key,)
        ).fetchone()
    return row["value"] if row else None


def del_bot_config(key: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM bot_config WHERE key=?", (key,))


# ── Daily snapshots ──────────────────────────────────────────────────────────

def save_daily_snapshot(capital: float, day_trades: list) -> None:
    today   = datetime.now(ET).strftime("%Y-%m-%d")
    day_pnl = sum(t.get("pnl", 0) for t in day_trades)
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO daily_snapshots
                (date, capital_usdc, n_trades, day_pnl, updated_at)
            VALUES (?,?,?,?,?)
        """, (today, capital, len(day_trades), day_pnl,
              datetime.now(ET).isoformat()))


def db_stats() -> dict:
    with _conn() as conn:
        trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        hist   = conn.execute("SELECT COUNT(*) FROM climate_history").fetchone()[0]
    size_kb = os.path.getsize(DB_FILE) / 1024 if os.path.exists(DB_FILE) else 0
    return {"trades": trades, "climate_hist": hist, "size_kb": size_kb}
