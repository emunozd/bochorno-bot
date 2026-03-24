"""
llm/weather_analysis.py — LLM-powered weather forecast validation.

Same Bayesian blend pattern as bochorno-bot's validate_pip,
adapted with a meteorological prompt.
"""

import json
from typing import Optional

from src.llm.client import llm_chat
from src.config import LLM_MODEL, MODEL_CONF_CAPS


_WEATHER_VALIDATION_PROMPT = """
You are a meteorological prediction market analyst with expertise in short-range weather forecasting.

City: {city_name} ({country})
Date: {target_date}
Ensemble T_max forecast: {T_predicted:.1f}°{unit}
Ensemble spread (1-sigma): ±{T_std:.1f}°{unit}
Number of NWP models: {n_models}
Weather Confidence Score (WCS): {wcs:.1f}/100
Outcome we are bidding YES on: {best_outcome}°{unit}
Our probability estimate (PIP): {pip:.2%}
Current Polymarket price for that outcome: {mkt_price:.2%}
Edge: {edge:+.2%}

Context from ensemble models: {model_summary}

Question: Is our {pip:.0%} probability estimate for T_max = {best_outcome}°{unit} reasonable?

Consider:
- Typical forecast skill for this city and season
- Whether {T_predicted:.1f}°{unit} with ±{T_std:.1f}°{unit} spread strongly supports outcome {best_outcome}
- Base rate: for a bin of ±0.5°{unit} around a forecast, the correct bin resolves ~25-35% of the time under normal spread
- Any known systematic biases for the region (e.g. marine influence, urban heat island)
- Adjust our PIP by at most ±0.10

Return ONLY valid JSON (no markdown, no extra text):
{{"valid": true, "adjusted_pip": {pip:.3f}, "confidence": "medium", "reason": "one sentence"}}

Rules:
- adjusted_pip must be between 0.25 and 0.75
- confidence: high | medium | low
- If you are uncertain, set confidence to low and keep adjusted_pip close to our estimate
"""


def _model_conf_cap() -> float:
    model = LLM_MODEL.lower()
    for key, cap in MODEL_CONF_CAPS.items():
        if key in model:
            return cap
    return 0.50


def validate_weather_pip(
    city_key: str,
    city_name: str,
    country: str,
    target_date: str,
    T_predicted: float,
    T_std: float,
    unit: str,
    best_outcome: int,
    pip: float,
    mkt_price: float,
    wcs: float,
    model_temps: dict,
    n_models: int,
) -> dict:
    """
    Bayesian PIP validation using LLM as meteorological second opinion.

    Process identical to bochorno-bot's validate_pip:
      pip_final = pip_raw * (1 - w) + llm_pip * w
      where w = confidence_base × model_cap
    """
    edge = round(pip - mkt_price, 4)

    # Build model summary string
    model_summary = ", ".join(
        f"{m.replace('_', '-')}: {t:.1f}°{unit}"
        for m, t in model_temps.items()
    ) or "unavailable"

    raw = llm_chat(_WEATHER_VALIDATION_PROMPT.format(
        city_name    = city_name,
        country      = country,
        target_date  = target_date,
        T_predicted  = T_predicted,
        T_std        = T_std,
        unit         = unit,
        n_models     = n_models,
        wcs          = wcs,
        best_outcome = best_outcome,
        pip          = pip,
        mkt_price    = mkt_price,
        edge         = edge,
        model_summary= model_summary,
    ), max_tokens=200)

    if raw is None:
        return {"valid": True, "adjusted_pip": pip, "confidence": "low",
                "reason": "LLM unavailable — using ensemble PIP"}

    try:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)

        llm_pip = float(result.get("adjusted_pip", pip))
        llm_pip = max(0.25, min(0.75, llm_pip))

        conf_base = {"high": 0.40, "medium": 0.25, "low": 0.10}.get(
            result.get("confidence", "low"), 0.15
        )
        weight    = conf_base * _model_conf_cap()
        final_pip = round(pip * (1 - weight) + llm_pip * weight, 3)
        final_pip = max(0.25, min(0.75, final_pip))

        return {
            "valid":        result.get("valid", True),
            "adjusted_pip": final_pip,
            "raw_llm_pip":  llm_pip,
            "confidence":   result.get("confidence", "low"),
            "reason":       result.get("reason", ""),
            "weight":       round(weight, 3),
        }
    except Exception:
        return {"valid": True, "adjusted_pip": pip, "confidence": "low",
                "reason": "LLM parse error — using ensemble PIP"}
