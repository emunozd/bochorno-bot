"""
llm/weather_analysis.py — LLM-powered weather forecast validation.

Same Bayesian blend pattern as bochorno-bot's validate_pip,
adapted with a meteorological prompt.
"""

import json
import time
from typing import Optional

from src.llm.client import llm_chat
from src.config import LLM_MODEL, MODEL_CONF_CAPS

# Throttle: one LLM call per city per N seconds — prevents hammering large local models
_LLM_THROTTLE_SECS = 600   # 10 minutes
_last_call: dict = {}       # {city_key: timestamp}


_WEATHER_VALIDATION_PROMPT = """You are a weather prediction market analyst.
{city_name} {target_date}: ensemble T_max={T_predicted:.1f}°{unit} ±{T_std:.1f}, models={model_summary}
Bidding YES on {best_outcome}°{unit}. Our PIP={pip:.0%}, market={mkt_price:.0%}, edge={edge:+.0%}.
Is PIP reasonable? Adjust ±0.10 max.
Reply ONLY JSON: {{"valid":true,"adjusted_pip":{pip:.3f},"confidence":"medium","reason":"one sentence"}}"""


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

    # Throttle — skip if called too recently for this city
    now = time.time()
    if now - _last_call.get(city_key, 0) < _LLM_THROTTLE_SECS:
        return {"valid": True, "adjusted_pip": pip, "confidence": "low",
                "reason": "throttled — using ensemble PIP"}
    _last_call[city_key] = now

    # Build model summary string (keep short)
    model_summary = " ".join(
        f"{m.split('-')[0][:4]}:{t:.0f}"
        for m, t in list(model_temps.items())[:4]
    ) or "n/a"

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