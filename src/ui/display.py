"""
ui/display.py — Rich terminal UI for bochorno-bot.
Adapted from bochorno-bot: city panels instead of asset panels.
"""

import math
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.rule import Rule
from rich.columns import Columns
from rich.console import Group

from src.config import ET, WATCH_CITIES
from src.signals.stats import edge_verdict
from src.data import database as DB

console = Console(width=200)


# ── Style helpers ───────────────────────────────────────────────────────────

def _wcs_style(s: Optional[float]) -> str:
    if s is None:    return "dim"
    if s >= 80:      return "bold green"
    if s >= 65:      return "green"
    if s >= 45:      return "yellow"
    return "red"

def _zone_style(z: str) -> str:
    return {
        "HIGH_CONF":   "bold green",
        "MEDIUM_CONF": "green",
        "LOW_CONF":    "yellow",
        "VERY_LOW":    "bold red",
    }.get(z, "dim")

def _edge_style(e: Optional[float]) -> str:
    if e is None:  return "dim"
    if e >= 0.10:  return "bold cyan"
    if e >= 0.05:  return "cyan"
    if e >= 0:     return "yellow"
    return "red"


# ── Header ──────────────────────────────────────────────────────────────────

def header_bar(state: dict, live_mode: bool) -> Panel:
    now  = datetime.now(ET)
    mode = Text(" [LIVE] ", style="bold green") if live_mode else \
           Text(" [PAPER] ", style="bold yellow")
    spin = Text("⟳ " + ", ".join(state["fetching"]) + "  ", style="yellow") \
           if state["fetching"] else Text("")

    t = Text()
    t.append("  BOCHORNO-BOT  ", style="bold white")
    t.append_text(mode)
    t.append(f"   {now.strftime('%a %d %b %Y  %H:%M:%S ET')}  ", style="dim")
    t.append_text(spin)
    t.append(f"   {state.get('status', '')}  ", style="dim")
    return Panel(t, style="on #080818", box=box.SIMPLE)


# ── City panel ──────────────────────────────────────────────────────────────

def city_panel(city_key: str, state: dict) -> Panel:
    cfg    = WATCH_CITIES[city_key]
    sig    = state["signals"].get(city_key, {})
    pos    = state["positions"].get(city_key)
    pp     = state["poly_prices"].get(city_key, {})

    wcs       = sig.get("wcs")
    wcs_zone  = sig.get("wcs_zone", "—")
    T_pred    = sig.get("T_predicted")
    T_std     = sig.get("T_std")
    best_out  = sig.get("best_outcome")
    best_prob = sig.get("best_prob")
    mkt_price = sig.get("mkt_price")
    edge      = sig.get("edge")
    opp       = sig.get("opportunity")
    all_probs = sig.get("all_probs", {})
    model_temps = sig.get("model_temps", {})
    unit      = cfg.get("temp_unit", "C")
    pip_val   = state["pip"].get(city_key)
    pip_valid = state["pip_validated"].get(city_key, {})
    blocked   = sig.get("wcs_blocked", False)

    border = "cyan"
    if pos:    border = "green"
    elif opp:  border = "cyan"
    elif blocked: border = "red"

    # Header
    hdr = Text()
    hdr.append(f"{city_key}  ", style=f"bold {border}")
    hdr.append(f"{cfg['name']}, {cfg['country']}  ", style="dim")
    if T_pred is not None:
        hdr.append(f"T̂ {T_pred:.1f}°{unit}", style="bold white")
    if T_std is not None:
        hdr.append(f"  ±{T_std:.1f}", style="dim")

    # Model temps row
    mt = Table.grid(padding=(0, 2))
    for _ in range(min(len(model_temps) * 2, 8)):
        mt.add_column()
    if model_temps:
        row_items = []
        for model, temp in model_temps.items():
            short = model.split("_")[0].upper()
            row_items.append(Text(short, style="dim"))
            row_items.append(Text(f"{temp:.1f}°{unit}", style="white"))
        mt.add_row(*row_items)

    # Scores
    sr = Table.grid(padding=(0, 3))
    sr.add_column(ratio=1); sr.add_column(ratio=1); sr.add_column(ratio=1)

    c1 = Table.grid()
    c1.add_row(Text("WCS  ", style="dim"),
               Text(f"{wcs:.1f}" if wcs else "—", style=_wcs_style(wcs)))
    c1.add_row(Text(wcs_zone, style=_zone_style(wcs_zone)))
    brk = sig.get("wcs_breakdown", {})
    if brk:
        c1.add_row(Text(
            f"A:{brk.get('agreement',0):.0f} S:{brk.get('skill',0):.0f} "
            f"H:{brk.get('horizon',0):.0f} C:{brk.get('climate',0):.0f}",
            style="dim"
        ))

    c2 = Table.grid()
    c2.add_row(Text("Outcome  ", style="dim"),
               Text(f"{best_out}°{unit}" if best_out else "—", style="cyan"))
    c2.add_row(Text("PIP      ", style="dim"),
               Text(f"{best_prob:.2%}" if best_prob else "—", style="cyan"))
    if pip_valid.get("reason"):
        c2.add_row(Text(f"LLM ({pip_valid.get('confidence','?')})", style="dim"),
                   Text(pip_valid["reason"][:35], style="dim"))

    c3 = Table.grid()
    c3.add_row(Text("Mkt   ", style="dim"),
               Text(f"{mkt_price:.3f}" if mkt_price else "—"))
    c3.add_row(Text("Edge  ", style="dim"),
               Text(f"{edge:+.2%}" if edge is not None else "—",
                    style=_edge_style(edge)))
    if all_probs and best_out:
        # Show nearby outcome probabilities for context
        outcomes_sorted = sorted(all_probs.keys())
        idx = outcomes_sorted.index(best_out) if best_out in outcomes_sorted else -1
        nearby = outcomes_sorted[max(0, idx-1):idx+3]
        prob_str = "  ".join(f"{k}°:{all_probs[k]:.0%}" for k in nearby)
        c3.add_row(Text(prob_str[:40], style="dim"))

    sr.add_row(c1, c2, c3)

    # Opportunity row
    opp_txt = Text("")
    if opp:
        opp_txt = Text()
        opp_txt.append("OPPORTUNITY  ", style="bold cyan")
        opp_txt.append(f"BUY YES {opp['outcome_val']}°{unit}  ", style="bold green")
        opp_txt.append(
            f"mkt={opp['mkt_price']:.3f}  pip={opp['pip']:.2%}  "
            f"edge={opp['edge']:.2%}",
            style="white"
        )
        if pip_valid.get("reason"):
            opp_txt.append(
                f"\n  LLM ({pip_valid.get('confidence','?')}): "
                f"{pip_valid['reason'][:60]}",
                style="dim"
            )

    # Open position row
    pos_txt = Text("")
    if pos:
        cur_price = pp.get(pos.outcome_val, pos.entry_price)
        pnl_now   = (cur_price - pos.entry_price) * pos.shares
        pc = "green" if pnl_now >= 0 else "red"
        pos_txt = Text()
        pos_txt.append(f"▶ YES {pos.outcome_val}°{unit}  ", style="bold green")
        pos_txt.append(
            f"entry={pos.entry_price:.3f}  shares={pos.shares:.1f}  "
            f"spent=${pos.usdc_spent:.2f}  ",
            style="dim"
        )
        pos_txt.append(f"PnL {'+' if pnl_now>=0 else ''}{pnl_now:.2f}  ",
                       style=f"bold {pc}")
        pos_txt.append(f"SL={pos.stop_loss:.3f}  TP={pos.take_profit:.3f}",
                       style="dim")

    return Panel(
        Group(hdr, Text(""), mt, Rule(style="dim"), sr, Text(""), opp_txt, pos_txt),
        border_style=border, padding=(1, 2),
        title=f"[bold {border}]{city_key}[/]  [dim]{unit}[/]",
        title_align="left",
    )


# ── Portfolio panel ─────────────────────────────────────────────────────────

def portfolio_panel(state: dict, live_mode: bool) -> Panel:
    import math as _math
    capital = state["capital_usdc"]
    peak    = state["peak_capital"]

    stats       = DB.trade_stats()
    total_pnl   = stats.get("total_pnl")   or 0
    winners     = stats.get("winners")     or 0
    total_count = stats.get("total")       or 0
    losers      = total_count - winners
    win_rate    = winners / total_count * 100 if total_count else 0.0
    total_lr    = stats.get("total_log_return") or 0
    total_lr_pct= (_math.exp(total_lr) - 1) * 100 if total_lr else 0.0
    avg_ev      = stats.get("avg_ev") or 0.0

    mtm = 0.0
    for ck, pos in state["positions"].items():
        pp     = state["poly_prices"].get(ck, {})
        cur    = pp.get(pos.outcome_val, pos.entry_price)
        mtm   += (cur - pos.entry_price) * pos.shares
    total_val = capital + mtm
    dd        = (peak - total_val) / peak * 100 if peak > 0 else 0

    g = Table.grid(padding=(0, 3))
    g.add_column(style="dim", width=16); g.add_column()
    g.add_column(style="dim", width=14); g.add_column()

    pc  = "green" if total_pnl >= 0 else "red"
    lrc = "green" if total_lr_pct >= 0 else "red"
    g.add_row("USDC capital", Text(f"${capital:.2f}", style="bold white"),
              "Total value",  Text(f"${total_val:.2f}", style="bold cyan"))
    g.add_row("Realized PnL", Text(f"${total_pnl:+.2f}", style=f"bold {pc}"),
              "Max drawdown", Text(f"-{dd:.1f}%", style="orange3" if dd > 10 else "dim"))
    g.add_row("Log return",   Text(f"{total_lr_pct:+.2f}% (ln={total_lr:+.4f})", style=f"bold {lrc}"),
              "Avg EV/USD",   Text(f"{avg_ev:+.4f}", style="cyan" if avg_ev > 0 else "red"))
    g.add_row("Trades",       Text(str(total_count), style="white"),
              "Win rate",     Text(f"{win_rate:.0f}%  ({winners}W / {losers}L)",
                                   style="green" if win_rate >= 55 else "yellow"))

    ci             = state.get("bootstrap_ci")
    verdict_label, verdict_style = edge_verdict(ci)
    g.add_row("Edge (95% CI)", Text(verdict_label, style=verdict_style), "", Text(""))

    tt = Table.grid(padding=(0, 1))
    for _ in range(6): tt.add_column()
    for trade in list(reversed(state.get("trades", [])))[:6]:
        cfg  = WATCH_CITIES.get(trade.city_key if hasattr(trade, "city_key") else
                                 trade.get("city_key", ""), {})
        unit = cfg.get("temp_unit", "C")
        col  = "green" if trade.pnl > 0 else "red"
        ov   = trade.outcome_val if hasattr(trade, "outcome_val") else trade.get("outcome_val", "?")
        tt.add_row(
            Text(trade.time if hasattr(trade, "time") else trade.get("time", ""), style="dim"),
            Text(trade.city_key if hasattr(trade, "city_key") else trade.get("city_key", ""), style="dim"),
            Text(f"{ov}°{unit}", style=col),
            Text(f"{trade.entry_price:.3f}→{trade.exit_price:.3f}" if hasattr(trade, "entry_price") else "", style="dim"),
            Text(f"${trade.pnl:+.2f}" if hasattr(trade, "pnl") else "", style=f"bold {col}"),
            Text(trade.reason if hasattr(trade, "reason") else trade.get("reason", ""), style="dim"),
        )

    mode = Text()
    mode.append("● LIVE — Polymarket CLOB", style="bold green") if live_mode else \
    mode.append("● PAPER — simulation only", style="bold yellow")

    return Panel(
        Group(mode, Text(""), g, Text(""), Text("── Recent trades", style="dim"), tt),
        title="[bold magenta]💼  Portfolio[/]",
        border_style="magenta", padding=(1, 2),
    )


# ── Status bar ──────────────────────────────────────────────────────────────

def status_bar(state: dict) -> Text:
    sig = state.get("last_signal", "—")
    upd = state.get("last_update", "—")
    cd  = state.get("countdown", 0)
    try:
        dbs     = DB.db_stats()
        db_info = f"  DB: {dbs['trades']} trades · {dbs['climate_hist']} hist · {dbs['size_kb']:.0f}KB"
    except Exception:
        db_info = ""
    return Text(
        f"  Data:{upd}  Signal:{sig}  •  next in {cd}s{db_info}  •  Ctrl+P Ctrl+Q to detach",
        style="dim",
    )


# ── Full layout ─────────────────────────────────────────────────────────────

def build_layout(state: dict, live_mode: bool):
    ag = Table.grid(expand=True, padding=(0, 1))
    for _ in WATCH_CITIES:
        ag.add_column(ratio=1)
    ag.add_row(*[city_panel(c, state) for c in WATCH_CITIES])

    root = Table.grid(expand=True)
    root.add_column()
    root.add_row(header_bar(state, live_mode))
    root.add_row(ag)
    root.add_row(portfolio_panel(state, live_mode))
    root.add_row(Rule(style="dim"))
    root.add_row(status_bar(state))
    return root