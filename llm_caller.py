"""
Resilient LLM call wrapper with exponential backoff retry and provider fallback.

Retry policy:
  - Up to 3 attempts with backoff: 1 s, 2 s, 4 s
  - Retries on transient HTTP errors (429, 500, 502, 503, 504) and timeouts
  - Non-transient errors (400, 401, 403) are not retried

Fallback:
  - If all retries on the primary provider fail, one attempt is made on the
    fallback provider (no further retries on the fallback).

Logging:
  - Each attempt and final outcome is logged as a JSON line so CloudWatch
    Logs Insights can query by provider, error_code, and latency_ms.
"""

import json
import os
import time
import urllib.error

_TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_BACKOFF = 1  # seconds


def _log(event: str, **kwargs):
    print(json.dumps({"event": event, **kwargs}))


def _import_provider(name: str):
    if name == "gemini":
        from gemini_provider import ask_llm
    elif name == "openrouter":
        from openrouter_provider import ask_llm
    else:
        from groq_provider import ask_llm
    return ask_llm


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _TRANSIENT_HTTP_CODES
    # URLError (network issues, timeouts) are transient
    if isinstance(exc, urllib.error.URLError):
        return True
    # OSError covers socket timeouts
    if isinstance(exc, OSError):
        return True
    return False


def _call_with_retry(provider: str, user_message, conversation_history, user_context):
    ask_llm = _import_provider(provider)
    last_exc = None

    for attempt in range(1, _MAX_RETRIES + 1):
        t0 = time.time()
        try:
            result = ask_llm(user_message, conversation_history, user_context)
            latency_ms = int((time.time() - t0) * 1000)
            _log("llm_success", provider=provider, attempt=attempt, latency_ms=latency_ms)
            return result
        except Exception as exc:
            latency_ms = int((time.time() - t0) * 1000)
            error_code = exc.code if isinstance(exc, urllib.error.HTTPError) else None
            _log(
                "llm_failure",
                provider=provider,
                attempt=attempt,
                error=str(exc),
                error_code=error_code,
                latency_ms=latency_ms,
                transient=_is_transient(exc),
            )
            last_exc = exc

            if not _is_transient(exc):
                break

            if attempt < _MAX_RETRIES:
                time.sleep(_BASE_BACKOFF * (2 ** (attempt - 1)))

    raise last_exc


def _fallback_provider(primary: str) -> str:
    fallbacks = {
        "groq": "gemini",
        "gemini": "groq",
        "openrouter": "groq",
    }
    return fallbacks.get(primary, "gemini")


def call_llm(user_message, conversation_history=None, user_context=None):
    """
    Call the configured primary LLM with retry. On total failure, attempt the
    fallback provider once. Raises on complete failure.
    """
    if conversation_history is None:
        conversation_history = []

    primary = os.environ.get("LLM_PROVIDER", "groq").lower()

    try:
        return _call_with_retry(primary, user_message, conversation_history, user_context)
    except Exception as primary_exc:
        fallback = _fallback_provider(primary)
        _log("llm_fallback", primary=primary, fallback=fallback, reason=str(primary_exc))
        try:
            return _call_with_retry(fallback, user_message, conversation_history, user_context)
        except Exception as fallback_exc:
            _log("llm_total_failure", primary=primary, fallback=fallback, error=str(fallback_exc))
            raise
