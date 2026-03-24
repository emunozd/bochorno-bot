"""
llm/client.py — Provider-agnostic LLM interface.

Default: local OpenAI-compatible server (Ollama, LM Studio, vLLM, etc.)
Also supports: Groq, Together, OpenAI, Anthropic — set via env vars.

  LLM_BACKEND  = openai     (default)
  LLM_BASE_URL = http://localhost:11434/v1   (default — Ollama)
  LLM_MODEL    = llama3.2   (default)
  LLM_API_KEY  = none       (default — local servers don't need a key)

To use a remote provider instead:
  LLM_BACKEND  = openai
  LLM_BASE_URL = https://api.groq.com/openai/v1
  LLM_API_KEY  = gsk_...
  LLM_MODEL    = llama-3.3-70b-versatile

  LLM_BACKEND  = anthropic
  LLM_API_KEY  = sk-ant-...
  LLM_MODEL    = claude-sonnet-4-20250514
"""

from typing import Optional
from src.config import LLM_BACKEND, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY

try:
    from openai import OpenAI as _OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic as _anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def _build_client():
    if LLM_BACKEND == "anthropic":
        if not HAS_ANTHROPIC:
            print("⚠  anthropic not installed: pip install anthropic")
            return None
        if not LLM_API_KEY or LLM_API_KEY == "none":
            print("⚠  LLM_API_KEY not set (required for anthropic backend)")
            return None
        return _anthropic.Anthropic(api_key=LLM_API_KEY)

    # Default: openai-compatible (local or remote)
    if not HAS_OPENAI:
        print("⚠  openai not installed: pip install openai")
        return None
    base = LLM_BASE_URL or "http://localhost:11434/v1"
    key  = LLM_API_KEY if LLM_API_KEY and LLM_API_KEY != "none" else "sk-local"
    return _OpenAI(base_url=base, api_key=key)


_client = _build_client()


def llm_chat(prompt: str, max_tokens: int = 500) -> Optional[str]:
    """
    Send a prompt to the configured LLM and return the response text.
    Returns None on any failure — callers must handle the None case gracefully.
    """
    if _client is None:
        return None
    try:
        if LLM_BACKEND == "anthropic":
            resp = _client.messages.create(
                model=LLM_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        else:
            resp = _client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
    except Exception:
        return None


def is_available() -> bool:
    return _client is not None
