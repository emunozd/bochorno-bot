# ── Stage 1: dependency builder ───────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="bochorno-bot"
LABEL description="bochorno-bot — Predicción de temperatura en Polymarket"

RUN groupadd -r botuser && useradd -r -g botuser botuser

WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/            ./src/
COPY main.py         .
COPY requirements.txt .

# Create /data and hand ownership to botuser at build time
RUN mkdir -p /data && chown botuser:botuser /data

ENV BOT_DB=/data/bot.db
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Drop privileges — no entrypoint script needed
USER botuser

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests, rich, scipy, openai; print('ok')" || exit 1

CMD ["python", "main.py"]