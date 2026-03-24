"""
src/telegram/bot.py — Telegram interface for bochorno-bot.

Same auth flow: seed phrase vinculation, push alerts, commands.
"""

import os
import threading
import logging
import functools
from datetime import datetime
from typing import Optional

from src.config import ET, WATCH_CITIES
from src.data import database as DB

log = logging.getLogger("bochorno-bot.telegram")

try:
    from telegram import Update, BotCommand, ReplyKeyboardRemove
    from telegram.ext import (
        ApplicationBuilder, CommandHandler, ContextTypes,
    )
    from telegram.constants import ParseMode
    HAS_TG = True
except ImportError:
    HAS_TG = False
    log.warning("python-telegram-bot not installed. Telegram disabled.")

_state:      Optional[dict]           = None
_lock:       Optional[threading.Lock] = None
_app:        Optional[object]         = None
_start_time: datetime                 = datetime.now(ET)
_paused      = threading.Event()


def init(state: dict, lock: threading.Lock) -> None:
    global _state, _lock
    if not HAS_TG:
        return
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return
    _state = state
    _lock  = lock
    state["_lock_ref"] = lock
    threading.Thread(target=_run_polling, args=(token,), daemon=True).start()
    log.info("Telegram thread started.")


def _run_polling(token: str) -> None:
    import asyncio

    async def _main():
        global _app
        _app = ApplicationBuilder().token(token).build()
        for name, fn in [
            ("start",       cmd_start),
            ("vincular",    cmd_vincular),
            ("desvincular", cmd_desvincular),
            ("help",        cmd_help),
            ("positions",   cmd_positions),
            ("portfolio",   cmd_portfolio),
            ("signals",     cmd_signals),
            ("trades",      cmd_trades),
            ("status",      cmd_status),
            ("close",       cmd_close),
            ("pause",       cmd_pause),
            ("resume",      cmd_resume),
        ]:
            _app.add_handler(CommandHandler(name, fn))

        await _app.bot.set_my_commands([
            BotCommand("positions",   "Posiciones abiertas"),
            BotCommand("signals",     "Señales climáticas activas"),
            BotCommand("portfolio",   "Capital y performance"),
            BotCommand("trades",      "Últimos trades"),
            BotCommand("status",      "Estado del bot"),
            BotCommand("close",       "Cerrar posición: /close CIUDAD"),
            BotCommand("pause",       "Pausar nuevas entradas"),
            BotCommand("resume",      "Reactivar entradas"),
            BotCommand("help",        "Ayuda"),
        ])

        await _app.run_polling(drop_pending_updates=True)

    asyncio.run(_main())


# ── Auth helpers ────────────────────────────────────────────────────────────

def _get_owner() -> Optional[str]:
    return DB.get_bot_config("owner_chat_id")


def _require_auth(fn):
    @functools.wraps(fn)
    async def wrapper(update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        sender_id = str(update.effective_chat.id)
        owner     = _get_owner()
        if not owner or owner != sender_id:
            return
        return await fn(update, ctx)
    return wrapper


# ── Text helpers ────────────────────────────────────────────────────────────

def _e(text: str) -> str:
    """Escape MarkdownV2 special chars."""
    for ch in r"\_*[]()~`>#+-=|{}.!":
        text = text.replace(ch, f"\\{ch}")
    return text


def _fmt_pnl(pnl: float) -> str:
    return f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"


def _pnl_bar(pct: float, width: int = 10) -> str:
    filled = max(0, min(width, round(abs(pct) / 10 * width)))
    return "█" * filled + "░" * (width - filled)


def _uptime() -> str:
    delta = datetime.now(ET) - _start_time
    h, rem = divmod(int(delta.total_seconds()), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


def _positions_snap() -> dict:
    if _state is None: return {}
    with _lock: return dict(_state.get("positions", {}))


def _poly_prices_snap() -> dict:
    if _state is None: return {}
    with _lock: return dict(_state.get("poly_prices", {}))


def _fmt_end_date(end_date: str) -> str:
    """Format end_date ISO string into a short human-readable string."""
    if not end_date:
        return "unknown"
    try:
        dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        return dt.strftime("%b %d %H:%M UTC")
    except Exception:
        return end_date[:16]


# ── Commands ────────────────────────────────────────────────────────────────

async def cmd_start(update: "Update", ctx: "ContextTypes.DEFAULT_TYPE") -> None:
    sender_id = str(update.effective_chat.id)
    owner     = _get_owner()
    if owner and owner == sender_id:
        await update.message.reply_text(
            "🌡 bochorno-bot activo.\n\nUsa el menú inferior o escribe un comando."
        )
        return
    if owner and owner != sender_id:
        return
    await update.message.reply_text(
        "🤖 Bienvenido a bochorno-bot\n\n"
        "Este bot no tiene dueño aún.\n\n"
        "Para reclamarlo envía:\n\n"
        "/vincular <frase_secreta>\n\n"
        "La frase secreta es el valor de TELEGRAM_LINK_SECRET en el .env del servidor."
    )


async def cmd_vincular(update: "Update", ctx: "ContextTypes.DEFAULT_TYPE") -> None:
    sender_id  = str(update.effective_chat.id)
    owner      = _get_owner()

    if owner and owner != sender_id:
        # Already claimed by someone else — silent
        return

    if owner and owner == sender_id:
        await update.message.reply_text("✅ Ya estás vinculado como dueño de este bot.")
        return

    secret_env = os.environ.get("TELEGRAM_LINK_SECRET", "").strip()
    if not secret_env:
        await update.message.reply_text(
            "⚠️ TELEGRAM_LINK_SECRET no está configurado en el servidor."
        )
        return

    provided = " ".join(ctx.args).strip() if ctx.args else ""
    if not provided:
        await update.message.reply_text(
            "Uso: /vincular <frase_secreta>"
        )
        return

    if provided != secret_env:
        log.warning(f"Wrong secret attempt — chat_id={sender_id}")
        await update.message.reply_text("❌ Frase incorrecta. Intentá de nuevo.")
        return

    DB.set_bot_config("owner_chat_id", sender_id)
    log.info(f"Bot claimed by chat_id={sender_id}")
    await update.message.reply_text(
        "🔐 Bot vinculado correctamente.\n\n"
        "Eres el único dueño de esta instancia.\n"
        "El menú de comandos ya está disponible abajo 👇"
    )


async def cmd_desvincular(update: "Update", ctx: "ContextTypes.DEFAULT_TYPE") -> None:
    sender_id = str(update.effective_chat.id)
    owner     = _get_owner()
    if not owner or owner != sender_id:
        return
    DB.del_bot_config("owner_chat_id")
    await update.message.reply_text(
        "🔓 Bot desvinculado.\n\nUsa /vincular <secreto> para reclamarlo de nuevo."
    )


@_require_auth
async def cmd_help(update: "Update", ctx: "ContextTypes.DEFAULT_TYPE") -> None:
    await update.message.reply_text(
        "🌡 bochorno-bot — Predicción climática en Polymarket\n\n"
        "Posiciones y portafolio\n"
        "  /positions — Posiciones abiertas con SL · TP · PnL\n"
        "  /portfolio — Capital, drawdown, win-rate\n"
        "  /trades    — Últimos 10 trades cerrados\n\n"
        "Señales\n"
        "  /signals   — WCS · TPS · PIP por ciudad\n\n"
        "Control\n"
        "  /close BUENOS_AIRES — Cierre forzado\n"
        "  /pause     — Detiene nuevas entradas\n"
        "  /resume    — Reactiva entradas\n\n"
        "Cuenta\n"
        "  /desvincular — Libera este bot\n\n"
        "Info\n"
        "  /status    — Estado del bot y uptime\n"
    )


@_require_auth
async def cmd_signals(update: "Update", ctx: "ContextTypes.DEFAULT_TYPE") -> None:
    if _state is None:
        await update.message.reply_text("⚠️ Estado no disponible.")
        return
    with _lock:
        signals = dict(_state.get("signals", {}))

    if not signals:
        await update.message.reply_text("📡 No hay señales aún. El motor está calculando.")
        return

    lines = ["📡 Señales Climáticas\n"]
    for city_key, sig in signals.items():
        cfg         = WATCH_CITIES.get(city_key, {})
        name        = cfg.get("name", city_key)
        unit        = sig.get("unit", "C")
        wcs         = sig.get("wcs", 0) or 0
        zone        = sig.get("wcs_zone", "—")
        T           = sig.get("T_predicted")
        std         = sig.get("T_std")
        bo          = sig.get("best_outcome")
        pip         = sig.get("best_prob")
        mkt         = sig.get("mkt_price")
        edge        = sig.get("edge")
        opp         = sig.get("opportunity")
        mdl         = sig.get("model_temps", {})
        end_date    = cfg.get("end_date", "")
        target_date = cfg.get("target_date", "")

        T_str    = f"{T:.1f}°{unit}" if T is not None else "—"
        std_str  = f"±{std:.1f}" if std is not None else ""
        pip_str  = f"{pip:.0%}" if pip is not None else "—"
        mkt_str  = f"{mkt:.0%}" if mkt is not None else "—"
        edge_str = f"{edge:+.1%}" if edge is not None else "—"
        bo_str   = f"{bo}°{unit}" if bo is not None else "—"
        end_str  = _fmt_end_date(end_date)

        model_line = "  ".join(
            f"{m.split('_')[0].upper()}: {t:.1f}°{unit}"
            for m, t in list(mdl.items())[:4]
        )

        signal_icon = "🚀" if opp else ("⚠️" if wcs < 45 else "📊")

        lines.append(
            f"🌡 {name} ({unit}) — {target_date}\n"
            f"  Resolves: {end_str}\n"
            f"  T pred: {T_str} {std_str}\n"
            f"  Outcome: {bo_str} | PIP: {pip_str} | Mkt: {mkt_str} | Edge: {edge_str}\n"
            f"  WCS: {wcs:.0f}/100 ({zone})\n"
            f"  {model_line}\n"
            f"  {signal_icon} {'SEÑAL ACTIVA' if opp else 'Sin señal'}\n"
        )

    await update.message.reply_text("\n".join(lines))


@_require_auth
async def cmd_positions(update: "Update", ctx: "ContextTypes.DEFAULT_TYPE") -> None:
    positions   = _positions_snap()
    poly_prices = _poly_prices_snap()

    if not positions:
        await update.message.reply_text("📭 No hay posiciones abiertas ahora mismo.")
        return

    lines = [f"📈 Posiciones Abiertas ({len(positions)} total)\n"]
    for city_key, pos in positions.items():
        cfg       = WATCH_CITIES.get(city_key, {})
        name      = cfg.get("name", city_key)
        unit      = cfg.get("temp_unit", "C")
        end_date  = cfg.get("end_date", "")
        pp        = poly_prices.get(city_key, {})
        cur_price = pp.get(pos.outcome_val, pos.entry_price)
        cur_val   = pos.shares * cur_price
        pnl_usd   = cur_val - pos.usdc_spent
        pnl_pct   = (pnl_usd / pos.usdc_spent * 100) if pos.usdc_spent else 0.0
        pnl_icon  = "🟢" if pnl_usd >= 0 else "🔴"
        arrow     = "▲" if pnl_usd >= 0 else "▼"
        entry_dt  = pos.entry_time.strftime("%b %d %H:%M") if hasattr(pos.entry_time, "strftime") else str(pos.entry_time)

        lines.append(
            f"{name} — {pos.outcome_val}°{unit} YES\n"
            f"  Resolves: {_fmt_end_date(end_date)}\n"
            f"  {pos.shares:.2f} shares @ ${pos.entry_price:.3f}\n"
            f"  ${pos.usdc_spent:.2f} → ${cur_val:.2f}\n"
            f"  {pnl_icon} PnL: {_fmt_pnl(pnl_usd)} USDC ({arrow}{abs(pnl_pct):.1f}%) [{_pnl_bar(pnl_pct)}]\n"
            f"  SL: ${pos.stop_loss:.3f}   TP: ${pos.take_profit:.3f}\n"
            f"  Entered: {entry_dt} ET  WCS:{pos.entry_wcs:.0f}  PIP:{pos.entry_pip:.3f}\n"
        )

    await update.message.reply_text("\n".join(lines))


@_require_auth
async def cmd_portfolio(update: "Update", ctx: "ContextTypes.DEFAULT_TYPE") -> None:
    if _state is None:
        await update.message.reply_text("⚠️ Estado no disponible.")
        return
    with _lock:
        capital      = _state.get("capital_usdc", 0.0)
        peak         = _state.get("peak_capital", capital)
        positions    = _state.get("positions", {})
        bootstrap_ci = _state.get("bootstrap_ci")

    recent   = DB.load_recent_trades(limit=200)
    invested = sum(p.usdc_spent for p in positions.values())
    total_val= capital + invested
    drawdown = ((peak - total_val) / peak * 100) if peak else 0.0
    wins     = [t for t in recent if (t.get("pnl") or 0) > 0]
    losses   = [t for t in recent if (t.get("pnl") or 0) <= 0]
    win_rate = (len(wins) / len(recent) * 100) if recent else 0.0
    total_pnl= sum(t.get("pnl", 0) for t in recent)
    dd_icon  = "🟢" if drawdown < 5 else ("🟡" if drawdown < 15 else "🔴")

    ci_txt = ""
    if bootstrap_ci:
        lo, hi = bootstrap_ci
        ci_txt = f"\n  Win CI 95%: {lo:.1%} – {hi:.1%}"

    await update.message.reply_text(
        f"💼 Resumen del Portafolio\n\n"
        f"  💵 Disponible: ${capital:.2f} USDC\n"
        f"  📊 Invertido:  ${invested:.2f} USDC\n"
        f"  🏦 Total:      ${total_val:.2f} USDC\n"
        f"  {dd_icon} Drawdown: {drawdown:.1f}% desde pico (${peak:.2f})\n"
        f"  📈 PnL total: {_fmt_pnl(total_pnl)} USDC\n\n"
        f"  Trades: {len(recent)}   Win rate: {win_rate:.0f}%"
        f"  ({len(wins)}W / {len(losses)}L)"
        f"{ci_txt}"
    )


@_require_auth
async def cmd_trades(update: "Update", ctx: "ContextTypes.DEFAULT_TYPE") -> None:
    recent = DB.load_recent_trades(limit=10)
    if not recent:
        await update.message.reply_text("📭 No hay trades registrados aún.")
        return

    lines = ["📋 Últimos trades\n"]
    for t in recent:
        cfg  = WATCH_CITIES.get(t.get("city_key", ""), {})
        unit = cfg.get("temp_unit", "C")
        col  = "🟢" if (t.get("pnl") or 0) > 0 else "🔴"
        lines.append(
            f"{col} {cfg.get('name', t.get('city_key','?'))} "
            f"{t.get('outcome_val','?')}°{unit} "
            f"— {_fmt_pnl(t.get('pnl', 0))} USDC "
            f"({t.get('reason','?')})"
        )

    await update.message.reply_text("\n".join(lines))


@_require_auth
async def cmd_status(update: "Update", ctx: "ContextTypes.DEFAULT_TYPE") -> None:
    if _state is None:
        await update.message.reply_text("⚠️ Estado no disponible.")
        return

    with _lock:
        capital  = _state.get("capital_usdc", 0)
        n_pos    = len(_state.get("positions", {}))
        fetching = list(_state.get("fetching", set()))
        status   = _state.get("status", "—")
        paused   = _state.get("engine_paused", False)
        lu       = _state.get("last_update", "—")
        ls       = _state.get("last_signal", "—")

    # Show market dates per city
    market_lines = []
    for city_key, cfg in WATCH_CITIES.items():
        end_date    = cfg.get("end_date", "")
        target_date = cfg.get("target_date", "")
        unit        = cfg.get("temp_unit", "C")
        n_outcomes  = len(cfg.get("token_ids", {}))
        market_lines.append(
            f"  {cfg['name']} ({unit}): {target_date} → resolves {_fmt_end_date(end_date)}"
            f"  [{n_outcomes} outcomes]"
        )

    db = DB.db_stats()

    await update.message.reply_text(
        f"🌡 bochorno-bot — Estado\n\n"
        f"  ⏱ Uptime: {_uptime()}\n"
        f"  {'⏸ PAUSADO' if paused else '▶️ Activo'}\n"
        f"  💵 Capital: ${capital:.2f} USDC\n"
        f"  📊 Posiciones abiertas: {n_pos}\n"
        f"  🔄 Fetching: {', '.join(fetching) or 'idle'}\n"
        f"  📡 Última señal: {ls}\n"
        f"  📥 Último fetch: {lu}\n\n"
        f"Mercados activos:\n" + "\n".join(market_lines) + "\n\n"
        f"  🗃 DB: {db.get('trades',0)} trades · "
        f"{db.get('climate_hist',0)} hist · {db.get('size_kb',0):.0f}KB\n"
        f"  ℹ️ {status}"
    )


@_require_auth
async def cmd_close(update: "Update", ctx: "ContextTypes.DEFAULT_TYPE") -> None:
    if not ctx.args:
        await update.message.reply_text("Uso: /close CIUDAD\n\nEjemplo: /close BUENOS_AIRES")
        return

    city_key = ctx.args[0].upper()
    if _state is None:
        return

    with _lock:
        pos = _state.get("positions", {}).get(city_key)

    if not pos:
        await update.message.reply_text(f"📭 No hay posición abierta para {city_key}.")
        return

    with _lock:
        pp = _state.get("poly_prices", {}).get(city_key, {})
    price = pp.get(pos.outcome_val) or pos.entry_price

    from src.trading.engine import close_position
    poly_client = _state.get("poly_client")
    close_position(city_key, pos, "MANUAL", price, poly_client, _state)
    await update.message.reply_text(f"✅ Posición {city_key} cerrada manualmente.")


@_require_auth
async def cmd_pause(update: "Update", ctx: "ContextTypes.DEFAULT_TYPE") -> None:
    if _paused.is_set():
        await update.message.reply_text("⏸ El motor ya está pausado.")
        return
    _paused.set()
    if _state is not None:
        with _lock: _state["engine_paused"] = True
    await update.message.reply_text(
        "⏸ Motor PAUSADO.\n\nNo se abrirán nuevas posiciones.\nUsa /resume para reactivar."
    )


@_require_auth
async def cmd_resume(update: "Update", ctx: "ContextTypes.DEFAULT_TYPE") -> None:
    if not _paused.is_set():
        await update.message.reply_text("▶️ El motor ya está corriendo.")
        return
    _paused.clear()
    if _state is not None:
        with _lock: _state["engine_paused"] = False
    await update.message.reply_text("▶️ Motor REACTIVADO.\nNuevas entradas habilitadas.")


def is_paused() -> bool:
    return _paused.is_set()


# ── Push alerts ─────────────────────────────────────────────────────────────

def _send_sync(text: str) -> None:
    if not HAS_TG or _app is None:
        return
    chat_id = _get_owner()
    if not chat_id:
        return
    def _push():
        import asyncio
        try:
            asyncio.run(_app.bot.send_message(chat_id=chat_id, text=text))
        except Exception as exc:
            log.warning(f"Telegram push failed: {exc}")
    threading.Thread(target=_push, daemon=True).start()


def alert_position_opened(city_key, outcome_val, unit,
                           shares, entry, usdc, sl, tp,
                           wcs, pip, T_pred, T_std):
    cfg      = WATCH_CITIES.get(city_key, {})
    name     = cfg.get("name", city_key)
    end_date = cfg.get("end_date", "")
    T_str    = f"{T_pred:.1f}°{unit}" if T_pred is not None else "—"
    std_str  = f"±{T_std:.1f}" if T_std is not None else ""
    _send_sync(
        f"🚀 Posición Abierta\n\n"
        f"  🌡 {name} — {outcome_val}°{unit} YES\n"
        f"  Resolves: {_fmt_end_date(end_date)}\n"
        f"  {shares:.2f} shares @ ${entry:.3f}\n"
        f"  Invertido: ${usdc:.2f} USDC\n\n"
        f"  🛑 SL: ${sl:.3f}   🎯 TP: ${tp:.3f}\n\n"
        f"  T pred: {T_str} {std_str}\n"
        f"  WCS:{wcs:.0f}  PIP:{pip:.3f}"
    )


def alert_position_closed(city_key, outcome_val, name_unused,
                           reason, pnl, pnl_pct, entry, exit_p):
    cfg  = WATCH_CITIES.get(city_key, {})
    name = cfg.get("name", city_key)
    unit = cfg.get("temp_unit", "C")
    icon = "🏆" if pnl >= 0 else "💸"
    _send_sync(
        f"{icon} Posición Cerrada — {reason}\n\n"
        f"  {name} {outcome_val}°{unit} YES\n"
        f"  ${entry:.3f} → ${exit_p:.3f}\n"
        f"  PnL: {_fmt_pnl(pnl)} USDC ({pnl_pct:+.1f}%)"
    )


def alert_stop_loss(city_key, outcome_val, trigger, pnl):
    cfg  = WATCH_CITIES.get(city_key, {})
    name = cfg.get("name", city_key)
    unit = cfg.get("temp_unit", "C")
    _send_sync(
        f"🛑 Stop-Loss Ejecutado\n\n"
        f"  {name} {outcome_val}°{unit}\n"
        f"  Precio: ${trigger:.3f}\n"
        f"  Pérdida: {_fmt_pnl(pnl)} USDC"
    )


def alert_take_profit(city_key, outcome_val, trigger, pnl):
    cfg  = WATCH_CITIES.get(city_key, {})
    name = cfg.get("name", city_key)
    unit = cfg.get("temp_unit", "C")
    _send_sync(
        f"🎯 Take-Profit Alcanzado!\n\n"
        f"  {name} {outcome_val}°{unit}\n"
        f"  Precio: ${trigger:.3f}\n"
        f"  Ganancia: {_fmt_pnl(pnl)} USDC"
    )