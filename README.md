# 🌡 bochorno-bot

Automated prediction and trading bot for daily **maximum temperature markets** on [Polymarket](https://polymarket.com).

Uses an ensemble of NWP (Numerical Weather Prediction) models — ECMWF, GFS, ICON, CMA — to forecast which temperature bin will resolve for a given city, compares that probability against the market price, and opens a position when edge is sufficient. Notifications and control via Telegram. Fully LLM-agnostic — defaults to any local OpenAI-compatible server (Ollama, LM Studio, vLLM).

---

## How it works

```
Open-Meteo (4 NWP models)
        │
        ▼
  Weighted ensemble mean + σ
  (weights calibrated by historical MAE vs ERA5)
        │
        ▼
  Select closest bin to forecast
  (e.g. 26°C from the list [21°C … 30°C])
        │
        ▼
  P(bin) via Gaussian distribution
  → PIP (our implied probability)
        │
        ▼
  Bayesian validation with local LLM
  (max ±0.10 adjustment)
        │
        ▼
  Edge = PIP − polymarket_price
  If edge ≥ 0.08 → Kelly sizing → open YES position
```

The **WCS** (Weather Confidence Score, 0–100) gates entries based on ensemble quality. If model spread is too high (σ > 3°C/°F) or WCS falls below the threshold, no position is opened.

---

## Supported markets

Polymarket lists daily maximum temperature markets for several cities. The bot discovers them automatically via the Gamma API at startup.

Pre-configured cities:

| City | Country | Unit |
|---|---|---|
| Buenos Aires | AR | °C |
| Atlanta | US | °F |
| Seoul | KR | °C |
| Shanghai | CN | °C |

Example URL: `polymarket.com/event/highest-temperature-in-buenos-aires-on-march-25-2026`

To add or remove cities, edit `WATCH_CITIES` in `src/config.py`.

---

## Stack

| Layer | Technology |
|---|---|
| Weather data | [Open-Meteo](https://open-meteo.com) (free, no API key required) |
| Climate history | ERA5 via Open-Meteo Historical API |
| NWP models | ECMWF IFS, NOAA GFS, DWD ICON, CMA GRAPES |
| LLM validation | Any OpenAI-compatible server (Ollama, LM Studio, Groq, etc.) |
| Markets | Polymarket Gamma API + CLOB |
| Persistence | SQLite (WAL mode) |
| Notifications | Telegram Bot |
| Terminal UI | Rich (live TUI) |

---

## Requirements

- Python 3.12+
- A local LLM server (Ollama recommended) **or** an API key for Groq / OpenAI / Anthropic
- Docker + Docker Compose (for deployment)
- Polygon wallet with USDC only for live trading — paper trading works without it

---

## Quick start

### 1. Clone and configure

```bash
git clone https://github.com/your-username/bochorno-bot.git
cd bochorno-bot
cp .env.example .env
```

Edit `.env`. The bare minimum:

```env
# Local LLM server (Ollama with llama3.2 by default)
LLM_BACKEND=openai
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=none
LLM_MODEL=llama3.2

# Telegram (optional but recommended)
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_LINK_SECRET=your-seed-phrase
```

### 2. Run with Docker Compose

```bash
docker-compose up --build
```

On first run the bot downloads ERA5 climate history automatically (~30 seconds per city), then enters the main loop.

### 3. Link Telegram

Once running, find your bot on Telegram and send:

```
/vincular your-seed-phrase
```

Where `your-seed-phrase` matches `TELEGRAM_LINK_SECRET` in your `.env`. This claims the bot for your chat as the sole owner.

---

## Running without Docker

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env
python main.py
```

---

## LLM configuration

The bot works with **any OpenAI-compatible server**. The LLM is only used for Bayesian PIP validation — if unavailable, the bot continues operating using the meteorological ensemble alone.

### Ollama (default)

```bash
ollama pull llama3.2
ollama serve
```

```env
LLM_BACKEND=openai
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=none
LLM_MODEL=llama3.2
```

### LM Studio

```env
LLM_BACKEND=openai
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=none
LLM_MODEL=loaded-model-name
```

### Groq (cloud, generous free tier)

```env
LLM_BACKEND=openai
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_your_key
LLM_MODEL=llama-3.3-70b-versatile
```

### Anthropic Claude

```env
LLM_BACKEND=anthropic
LLM_API_KEY=sk-ant-your_key
LLM_MODEL=claude-sonnet-4-20250514
```

> Smaller models (llama3.2, mistral) receive lower weight in the Bayesian blend than more capable ones (llama3.3-70b, GPT-4, Claude). This is configured in `MODEL_CONF_CAPS` in `src/config.py`.

---

## Telegram commands

| Command | Description |
|---|---|
| `/signals` | Active signals per city: WCS, T_pred, PIP, edge |
| `/positions` | Open positions with SL, TP and live PnL |
| `/portfolio` | Capital, drawdown, win rate, confidence interval |
| `/trades` | Last 10 closed trades |
| `/status` | Bot state, uptime, active fetches, DB stats |
| `/close CITY` | Force-close a position (e.g. `/close BUENOS_AIRES`) |
| `/pause` | Pause new entries (open positions keep being monitored) |
| `/resume` | Resume the entry engine |
| `/vincular <secret>` | Claim the bot for your chat |
| `/desvincular` | Release ownership |

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `openai` | `openai` or `anthropic` |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | LLM server URL |
| `LLM_MODEL` | `llama3.2` | Model to use |
| `LLM_API_KEY` | `none` | API key (not needed for local servers) |
| `WCS_MIN` | `65` | Minimum WCS to enter a position (0–100) |
| `EDGE_MIN` | `0.08` | Minimum edge over market price |
| `SIGMA_MAX` | `3.0` | Block entry if ensemble spread exceeds this (°C/°F) |
| `STOP_LOSS_PCT` | `0.35` | Stop-loss as fraction of entry price |
| `TAKE_PROFIT_PCT` | `0.50` | Take-profit as fraction of entry price |
| `KELLY_FRACTION` | `0.25` | Kelly multiplier (0.25 = quarter-Kelly) |
| `CAPITAL_INITIAL` | `500.0` | Starting USDC capital for paper trading |
| `BOT_DB` | `bot.db` | SQLite database file path |
| `CLIMATE_HIST_DAYS` | `730` | Days of ERA5 history to download |
| `OWM_API_KEY` | *(empty)* | OpenWeatherMap key (optional, secondary source) |
| `POLY_PRIVATE_KEY` | *(empty)* | Polygon wallet private key (live trading only) |
| `POLY_FUNDER_ADDRESS` | *(empty)* | Wallet address holding USDC (live trading only) |
| `TELEGRAM_BOT_TOKEN` | *(empty)* | Telegram bot token |
| `TELEGRAM_LINK_SECRET` | *(empty)* | Seed phrase for chat ownership claim |

---

## Project structure

```
bochorno-bot/
├── main.py                        # Entry point and main clock loop
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── src/
    ├── config.py                  # All configuration (cities, thresholds, LLM)
    ├── models.py                  # Dataclasses: WeatherForecast, PolyPosition, etc.
    ├── data/
    │   ├── database.py            # SQLite: forecasts, climate history, trades, positions
    │   └── weather_data.py        # Open-Meteo ensemble, ERA5, market discovery
    ├── signals/
    │   ├── weather_indicators.py  # Ensemble stats, Gaussian distribution, unit conversion
    │   ├── weather_scoring.py     # WCS + TPS scoring + detect_opportunity
    │   └── stats.py               # Bootstrap CI for win rate
    ├── llm/
    │   ├── client.py              # Generic OpenAI-compatible + Anthropic client
    │   └── weather_analysis.py    # Meteorological prompt + Bayesian blend
    ├── trading/
    │   ├── engine.py              # Position lifecycle (open, monitor, close)
    │   ├── execution.py           # CLOB buy/sell with retry
    │   └── sizing.py              # Kelly criterion sizing
    ├── telegram/
    │   └── bot.py                 # Commands, push alerts, seed-phrase ownership
    └── ui/
        └── display.py             # Live Rich TUI with per-city panels
```

---

## Tests

```bash
pip install pytest scipy numpy
python -m pytest tests/ -v
```

74 tests covering unit conversion, ensemble statistics, per-bin Gaussian probability, outcome selection, WCS/TPS scoring, opportunity detection, slug parsing, and market token extraction.

---

## Paper trading vs live trading

By default the bot runs in **paper trading** mode: all decisions are calculated and recorded but no real orders are sent to the Polymarket CLOB.

To enable live trading, add to `.env`:

```env
POLY_PRIVATE_KEY=0x_your_private_key
POLY_FUNDER_ADDRESS=0x_your_wallet_with_usdc
```

The bot shows `[LIVE]` in the TUI header when connected to the CLOB.

---

## Weather data notes

**Open-Meteo** is the primary source. It is completely free, requires no API key, and exposes the main global NWP models directly:

- **ECMWF IFS** — global reference model, best skill at 24–72h horizons
- **NOAA GFS** — strong for the Americas, updates 4 times per day
- **DWD ICON** — high resolution over Europe and East Asia (Seoul)
- **CMA GRAPES** — Chinese model, best skill for Shanghai and Asia-Pacific

Model weights are calibrated automatically per city and month using historical MAE against ERA5. Until enough prediction history is accumulated, fallback weights are used (ECMWF 35%, GFS 30%, ICON 20%, CMA 15%).

---

## License

MIT