"""Non-blocking PRISM telemetry and guardrail audit logging (Role 1)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.request
import uuid
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_MAX_SPANS = 200
_SPAN_LOCK = threading.Lock()
_RECENT_SPANS: deque[dict[str, Any]] = deque(maxlen=_MAX_SPANS)


def _read_dotenv_value(name: str) -> str:
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for path in candidates:
        try:
            if not path.is_file():
                continue
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                if key.strip() == name:
                    return val.strip().strip('"').strip("'")
        except OSError:
            continue
    return ""


def _get_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if val:
        return val
    return _read_dotenv_value(name).strip()


def _evidence_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _emit_to_prism(span: dict[str, Any]) -> None:
    """Best-effort export to PRISM; failures are silently ignored."""
    host = _get_env("PRISMTRACE_HOST").rstrip("/")
    project_id = _get_env("PRISMTRACE_PROJECT_ID")
    api_key = _get_env("PRISMTRACE_API_KEY")
    if not (host and project_id and api_key):
        return

    try:
        body = json.dumps({"project_id": project_id, "spans": [span]}).encode("utf-8")
        req = urllib.request.Request(
            f"{host}/api/v1/traces",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "X-API-Key": api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0):
            return
    except Exception:  # noqa: BLE001
        return


def record_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    duration_ms: float = 0.0,
    status: str = "ok",
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Record a telemetry span locally and forward to PRISM if configured."""
    attrs = dict(attributes or {})
    span_core = {
        "trace_id": trace_id or uuid.uuid4().hex,
        "span_id": uuid.uuid4().hex[:16],
        "span_name": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_ms": round(float(duration_ms), 3),
        "status": status,
        "attributes": attrs,
    }
    span = {**span_core, "evidence_hash": _evidence_hash(span_core)}
    with _SPAN_LOCK:
        _RECENT_SPANS.append(span)
    _emit_to_prism(span)
    return span


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Time an operation and record a PRISM span without ever interrupting execution."""
    start = time.perf_counter()
    attrs = dict(attributes or {})
    status = "ok"
    try:
        yield attrs
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        attrs["error"] = str(exc)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        record_span(name=name, attributes=attrs, duration_ms=elapsed_ms, status=status)


def record_guardrail_decision(event_text: str, scenarios: list[Any]) -> dict[str, Any]:
    """Record a deterministic guardrail audit span for generated scenarios."""
    summary = []
    guardrail_triggered = False
    for scenario in scenarios:
        rationale = str(getattr(scenario, "rationale", "") or "")
        has_fix = "guardrail fixes:" in rationale.lower()
        if has_fix:
            guardrail_triggered = True
        summary.append(
            {
                "name": getattr(scenario, "name", ""),
                "source": getattr(scenario, "source", ""),
                "probability": getattr(scenario, "probability", 0.0),
                "severity": getattr(scenario, "severity", 1.0),
                "guardrail_modified": has_fix,
            }
        )
    return record_span(
        name="guardrail.decision",
        attributes={
            "event_length": len(event_text or ""),
            "scenario_count": len(scenarios),
            "guardrail_modified": guardrail_triggered,
            "scenarios": summary,
        },
    )


def get_recent_spans() -> list[dict[str, Any]]:
    """Return a copy of recent recorded spans."""
    with _SPAN_LOCK:
        return list(_RECENT_SPANS)


def clear_spans() -> None:
    """Clear recorded spans (for testing)."""
    with _SPAN_LOCK:
        _RECENT_SPANS.clear()
