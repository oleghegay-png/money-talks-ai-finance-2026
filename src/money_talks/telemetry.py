"""PRISM tracing (Block Convey) for LLM calls and guardrail decisions.

How it fits together:

* ``traced(call_llm)`` wraps the one function that talks to the LLM. It only
  remembers each call (prompts, reply or error, latency); it sends nothing.
* The pipeline then calls ``record(stage, ...)`` once the deterministic
  guardrail has decided, so one PRISM trace holds both what the model proposed
  and what the code decided.

Tracing is switched on by filling in ``PRISMTRACE_PROJECT_ID`` and
``PRISMTRACE_API_KEY`` in ``.env``. It is off when they are empty, when the
``prismtrace-sdk`` package is not installed, or when ``MONEY_TALKS_LLM_OFFLINE``
is set. Nothing in this module raises: the app must work the same whether or
not PRISM is reachable. The SDK sends in background threads, so a slow PRISM
does not slow the app down.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
import uuid
from typing import Any, Callable

DEFAULT_HOST = "https://api.prism.blockconvey.com"
AGENT_ID = "money-talks-stress-test"
AGENT_NAME = "Money Talks stress test"
OFFLINE_ENV_VAR = "MONEY_TALKS_LLM_OFFLINE"
SEND_TIMEOUT_SECONDS = 10

# One PRISM session per process, so a demo run shows up as one conversation.
SESSION_ID = uuid.uuid4().hex


@dataclasses.dataclass
class LLMCall:
    """One call to the LLM, remembered until the pipeline records it."""

    system_prompt: str
    user_prompt: str
    output: str = ""
    error: str = ""
    latency_ms: int = 0
    model: str = "unknown"


_local = threading.local()
_lock = threading.Lock()
_client: Any = None
_client_settings: tuple[str, str, str] | None = None


def _calls() -> list[LLMCall]:
    if not hasattr(_local, "calls"):
        _local.calls = []
    return _local.calls


def traced(
    call_llm: Callable[[str, str], str],
    model_env: str = "NICRRON_MODEL",
    model: Callable[[], str] | None = None,
) -> Callable[[str, str], str]:
    """Wrap ``call_llm(system_prompt, user_prompt) -> str`` so each call is remembered.

    ``model`` is asked, on the calling thread, which model the call used. Without
    it the name is read from the ``model_env`` environment variable.

    The wrapper behaves exactly like the original: same return value, same exceptions.
    """
    if getattr(call_llm, "_prism_traced", False):
        return call_llm

    def wrapper(system_prompt: str, user_prompt: str) -> str:
        call = LLMCall(system_prompt=str(system_prompt), user_prompt=str(user_prompt))
        started = time.perf_counter()
        try:
            call.output = call_llm(system_prompt, user_prompt)
            return call.output
        except Exception as exc:
            call.error = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            call.latency_ms = int((time.perf_counter() - started) * 1000)
            call.model = _model_name(model, model_env)
            _calls().append(call)

    wrapper._prism_traced = True  # type: ignore[attr-defined]
    wrapper.__wrapped__ = call_llm  # type: ignore[attr-defined]
    return wrapper


def _model_name(model: Callable[[], str] | None, model_env: str) -> str:
    if model is not None:
        try:
            name = str(model() or "").strip()
            if name:
                return name
        except Exception:  # noqa: BLE001 - tracing must never break the app
            pass
    return os.environ.get(model_env, "").strip() or "unknown"


def begin() -> None:
    """Forget any LLM calls remembered on this thread. Call before a pipeline stage."""
    _calls().clear()


def pending() -> list[LLMCall]:
    """The LLM calls made on this thread since ``begin()``."""
    return list(_calls())


def _settings() -> tuple[tuple[str, str, str] | None, str]:
    """Return ((host, project_id, api_key), "") or (None, why tracing is off)."""
    if os.environ.get(OFFLINE_ENV_VAR, "").strip().lower() not in ("", "0", "false", "no", "off"):
        return None, f"{OFFLINE_ENV_VAR} is set"
    host = os.environ.get("PRISMTRACE_HOST", "").strip().rstrip("/") or DEFAULT_HOST
    project_id = os.environ.get("PRISMTRACE_PROJECT_ID", "").strip()
    api_key = os.environ.get("PRISMTRACE_API_KEY", "").strip()
    missing = [
        name
        for name, value in (("PRISMTRACE_PROJECT_ID", project_id), ("PRISMTRACE_API_KEY", api_key))
        if not value
    ]
    if missing:
        return None, f"{', '.join(missing)} not set"
    return (host, project_id, api_key), ""


def _make_client(host: str, project_id: str, api_key: str) -> Any:
    from prismtrace import PRISMtrace  # imported late: the app runs without the package

    return PRISMtrace(api_key=api_key, host=host, project_id=project_id, timeout=SEND_TIMEOUT_SECONDS)


def _get_client() -> tuple[Any, str]:
    """Return (client, "") or (None, why tracing is off)."""
    global _client, _client_settings
    settings, reason = _settings()
    if settings is None:
        return None, reason
    with _lock:
        if _client is None or _client_settings != settings:
            try:
                _client = _make_client(*settings)
                _client_settings = settings
            except ImportError:
                return None, "the prismtrace-sdk package is not installed"
            except Exception as exc:  # noqa: BLE001 - tracing must never break the app
                return None, f"the PRISM client could not start ({type(exc).__name__})"
        return _client, ""


def status() -> str:
    """One line for the UI or the console saying whether traces are being sent."""
    try:
        client, reason = _get_client()
    except Exception:  # noqa: BLE001
        return "PRISM tracing is off."
    if client is None:
        return f"PRISM tracing is off: {reason}."
    return f"PRISM tracing is on (session {SESSION_ID[:8]})."


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def record(stage: str, **metadata: Any) -> int:
    """Send one PRISM trace per LLM call remembered on this thread, then forget them.

    ``metadata`` is attached to every trace; put the guardrail decision here.
    Returns the number of traces handed to PRISM. Never raises.
    """
    try:
        calls = pending()
        begin()
        if not calls:
            return 0
        client, _ = _get_client()
        if client is None:
            return 0
        sent = 0
        for call in calls:
            details = {"stage": stage, "llm_succeeded": not call.error, **metadata}
            if call.error:
                details["llm_error"] = call.error
            client.trace_llm(
                model=call.model,
                input_messages=[
                    {"role": "system", "content": call.system_prompt},
                    {"role": "user", "content": call.user_prompt},
                ],
                output=call.output,
                latency_ms=call.latency_ms,
                agent_id=AGENT_ID,
                agent_name=AGENT_NAME,
                session_id=SESSION_ID,
                metadata=_json_safe(details),
            )
            sent += 1
        return sent
    except Exception:  # noqa: BLE001 - tracing must never break the app
        return 0


def flush(timeout: float = 2.0) -> None:
    """Wait briefly for traces still being sent. Useful at the end of a script."""
    try:
        if _client is not None:
            _client.flush(timeout=timeout)
    except Exception:  # noqa: BLE001
        pass
