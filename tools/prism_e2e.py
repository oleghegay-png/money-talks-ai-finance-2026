"""Prove that real AI calls made by the app arrive in PRISM, and write down the proof.

Run from the repository root:

    PYTHONPATH=src python tools/prism_e2e.py

Unlike tools/prism_check.py (which only checks the connection), this runs the
real thing once through money_talks.pipeline, the same functions app.py calls:

1. generates scenarios from a sentence with a live LLM call;
2. stress-tests a preset portfolio and asks the LLM to explain the result;
3. checks that PRISM acknowledged one trace for each of the two LLM calls;
4. appends what happened to docs/prism-evidence.md (use --no-evidence to skip).

It needs PRISMTRACE_PROJECT_ID, PRISMTRACE_API_KEY and NICRRON_API_KEY in the
environment or .env, and spends a small amount of LLM balance. It never writes
the keys anywhere, and it keeps its own cache and audit log in a temporary
folder so the repository's demo_cache.json and audit_log.jsonl are untouched.

Exit codes:
    0  both LLM calls were traced and PRISM acknowledged both traces
    1  the run finished but something was missing (see the printed lines)
    2  credentials are missing or PRISM could not be reached
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_FILE = REPO_ROOT / "docs" / "prism-evidence.md"
EVIDENCE_HEADER = """# PRISM tracing: evidence of live runs

Each section below was written by `tools/prism_e2e.py` after a real run: a live
LLM call through `money_talks.pipeline`, traced to PRISM (Block Convey), with
the trace ids PRISM returned. To add a run: `PYTHONPATH=src python tools/prism_e2e.py`.
To look a run up in the PRISM web app, open the project and search for its session id.

No API keys are stored here.
"""

# Keep this run's cache and audit log out of the repository. Must be set before
# money_talks is imported, because the scenario cache is loaded at import.
_TMP = tempfile.TemporaryDirectory()
os.environ["MONEY_TALKS_DEMO_CACHE"] = os.path.join(_TMP.name, "demo_cache.json")
os.environ["MONEY_TALKS_AUDIT_LOG"] = os.path.join(_TMP.name, "audit_log.jsonl")

from money_talks import pipeline, telemetry  # noqa: E402
from money_talks.ai import analyzer  # noqa: E402


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=15, check=False
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _live_count(client, project_id: str, api_key: str) -> int | None:
    """How many real traces PRISM says this project has received."""
    try:
        response = client.get(
            "/api/setup-doctor", params={"project_id": project_id}, headers={"X-PRISMtrace-Key": api_key}
        )
        return int(response.json()["live_trace_count"]) if response.status_code == 200 else None
    except Exception:  # noqa: BLE001
        return None


def _capture_acknowledgements(prism_client) -> list[dict]:
    """Record what PRISM answers to each trace this process sends."""
    seen: list[dict] = []
    original = prism_client._post

    def recording_post(path: str, payload: dict):
        answer = original(path, payload)
        seen.append({"payload": payload, "answer": answer})
        return answer

    prism_client._post = recording_post
    return seen


def _one_line(text: object, limit: int = 160) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-evidence", action="store_true", help="do not write docs/prism-evidence.md")
    parser.add_argument(
        "--event",
        default="Oil jumps 40% after a Gulf shipping blockade",
        help="the event sentence to turn into scenarios",
    )
    args = parser.parse_args(argv)

    analyzer._load_env_file()
    os.environ.pop(telemetry.OFFLINE_ENV_VAR, None)  # this check is meant to use the network
    settings, reason = telemetry._settings()
    if settings is None:
        print(f"Cannot run: {reason}.")
        return 2
    host, project_id, api_key = settings
    llm = pipeline.llm_status()
    if llm["key_source"] == "none":
        print("Cannot run: NICRRON_API_KEY not set, so there is no live LLM call to trace.")
        return 2
    prism_client, reason = telemetry._get_client()
    if prism_client is None:
        print(f"Cannot run: {reason}.")
        return 2

    import httpx

    started = datetime.now(timezone.utc)
    # A unique sentence, so neither the scenario cache nor PRISM has seen it before.
    event = f"{args.event} (PRISM end-to-end check {started:%Y-%m-%d %H:%M:%S} UTC)"
    analyzer.clear_cache()
    acknowledgements = _capture_acknowledgements(prism_client)

    with httpx.Client(base_url=host, timeout=30.0) as http:
        before = _live_count(http, project_id, api_key)
        if before is None:
            print("Cannot run: PRISM did not answer, or did not accept the API key.")
            return 2

        print(f"Model: {llm['model']}. Generating scenarios (live LLM call)...")
        scenarios = pipeline.scenarios_from_text(event)
        usable, _ = pipeline.safe_scenarios(scenarios)
        portfolio_name, portfolio = next(iter(pipeline.preset_portfolios().items()))
        report = pipeline.stress_test(portfolio, usable)
        print("Explaining the stress test (live LLM call)...")
        explanation = pipeline.explain(portfolio, report)
        telemetry.flush(timeout=30.0)

        after = before
        for _ in range(10):
            after = _live_count(http, project_id, api_key) or after
            if after >= before + 2:
                break
            import time

            time.sleep(2)

    stages = {}
    for item in acknowledgements:
        metadata = item["payload"].get("metadata") or {}
        answer = item["answer"] if isinstance(item["answer"], dict) else {}
        stages[metadata.get("stage", "?")] = {
            "acknowledged": item["answer"] is not None,
            "trace_id": answer.get("id") or answer.get("trace_id") or "",
            "model": item["payload"].get("model", ""),
            "latency_ms": item["payload"].get("latency_ms", 0),
            "llm_succeeded": bool(metadata.get("llm_succeeded")),
            "llm_error": metadata.get("llm_error", ""),
            "metadata": metadata,
        }

    scenario, explain = stages.get("scenario_generation"), stages.get("explanation")
    checks = [
        ("Scenario LLM call succeeded", bool(scenario and scenario["llm_succeeded"])),
        ("PRISM acknowledged the scenario trace", bool(scenario and scenario["acknowledged"])),
        ("Explanation LLM call was made", explain is not None),
        ("PRISM acknowledged the explanation trace", bool(explain and explain["acknowledged"])),
        ("PRISM's count of real traces rose by at least 2", after >= before + 2),
    ]
    passed = all(ok for _, ok in checks)

    print()
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print(f"\nResult: {'PASS' if passed else 'FAIL'}. PRISM session {telemetry.SESSION_ID}.")

    if not args.no_evidence:
        commit = _git("rev-parse", "--short", "HEAD") or "unknown"
        if _git("status", "--porcelain", "--untracked-files=no"):
            commit += " plus uncommitted changes"
        lines = [
            "",
            f"## {started:%Y-%m-%d %H:%M:%S} UTC: {'PASS' if passed else 'FAIL'}",
            "",
            f"- Code: commit `{commit}` on branch `{_git('rev-parse', '--abbrev-ref', 'HEAD') or 'unknown'}`",
            f"- PRISM host: `{host}`, project `{project_id}`",
            f"- PRISM session id: `{telemetry.SESSION_ID}`",
            f"- Real traces PRISM reported for the project: {before} before, {after} after",
            f"- Event sentence: {_one_line(event)}",
            f"- Portfolio: {portfolio_name}",
            "",
            "| Check | Result |",
            "|---|---|",
            *[f"| {label} | {'PASS' if ok else 'FAIL'} |" for label, ok in checks],
            "",
            "| Stage | PRISM trace id | Model | LLM call | Latency |",
            "|---|---|---|---|---|",
        ]
        for name in ("scenario_generation", "explanation"):
            stage = stages.get(name)
            if stage is None:
                lines.append(f"| {name} | not sent | | no LLM call was made | |")
                continue
            outcome = "succeeded" if stage["llm_succeeded"] else f"failed: {_one_line(stage['llm_error'], 80)}"
            trace_id = f"`{stage['trace_id']}`" if stage["trace_id"] else (
                "acknowledged, no id returned" if stage["acknowledged"] else "not acknowledged"
            )
            lines.append(f"| {name} | {trace_id} | {stage['model']} | {outcome} | {stage['latency_ms']} ms |")
        lines.append("")
        if scenario:
            lines.append("Guardrail decisions recorded in the scenario trace:")
            lines.append("")
            for accepted in scenario["metadata"].get("accepted_scenarios", []):
                lines.append(
                    f"- {_one_line(accepted.get('scenario'), 80)} "
                    f"(probability {float(accepted.get('probability', 0)):.1%}): "
                    f"{_one_line(accepted.get('guardrail_decision'), 220)}"
                )
            lines.append("")
        if explain:
            lines.append(
                f"Guardrail decision recorded in the explanation trace: "
                f"{_one_line(explain['metadata'].get('guardrail_decision'), 220)}"
            )
            lines.append("")
            lines.append(f"Explanation shown to the user: {_one_line(explanation, 400)}")
            lines.append("")
        EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing = EVIDENCE_FILE.read_text(encoding="utf-8") if EVIDENCE_FILE.exists() else EVIDENCE_HEADER
        EVIDENCE_FILE.write_text(existing.rstrip("\n") + "\n" + "\n".join(lines), encoding="utf-8")
        print(f"Evidence appended to {EVIDENCE_FILE.relative_to(REPO_ROOT)}.")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
