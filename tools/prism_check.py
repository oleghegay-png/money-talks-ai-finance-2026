"""Check that this project can really send traces to PRISM.

Run from the repository root:

    PYTHONPATH=src python tools/prism_check.py

It reads PRISMTRACE_HOST, PRISMTRACE_PROJECT_ID and PRISMTRACE_API_KEY from the
environment or from .env, then:

1. asks PRISM whether the API key is valid for the project;
2. sends one trace through money_talks.telemetry, the same code path the app uses;
3. asks PRISM whether a real trace has arrived.

Exit codes:
    0  PRISM accepted the key and has received a real trace
       (also 0 with --allow-missing when no credentials are configured)
    1  the key is valid but PRISM has not confirmed a real trace yet
    2  the key was rejected, PRISM is unreachable, or credentials are missing
"""

from __future__ import annotations

import argparse
import os
import sys

from money_talks import telemetry
from money_talks.ai import analyzer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="exit 0 instead of 2 when no PRISM credentials are configured (for CI)",
    )
    args = parser.parse_args(argv)

    analyzer._load_env_file()
    os.environ.pop(telemetry.OFFLINE_ENV_VAR, None)  # this check is meant to use the network
    settings, reason = telemetry._settings()
    if settings is None:
        print(f"PRISM check skipped: {reason}." if args.allow_missing else f"PRISM check failed: {reason}.")
        return 0 if args.allow_missing else 2
    host, project_id, api_key = settings

    try:
        import httpx
        from prismtrace import verify
    except ImportError:
        print("PRISM check failed: prismtrace-sdk is not installed. Run: pip install -r requirements.txt")
        return 2

    print(f"PRISM host: {host}")
    with httpx.Client(base_url=host, timeout=30.0) as client:
        ok, message, _ = verify.handshake(client, project_id, api_key, send_test_trace=False)
        if not ok:
            print(f"PRISM check failed: the API key was not accepted ({message}).")
            return 2
        print("1/3 API key accepted for this project.")

        ask = telemetry.traced(lambda system, user: "PRISM connection check: OK")
        ask("You are a connection check.", "Reply with OK.")
        sent = telemetry.record("connection_check", guardrail_decision="not applicable: connection check")
        telemetry.flush(timeout=30.0)
        if not sent:
            print(f"PRISM check failed: no trace was sent ({telemetry.status()})")
            return 2
        print("2/3 One trace sent through money_talks.telemetry.")

        result = verify.poll_doctor(client, project_id, api_key)

    if result and result.get("live_connected"):
        print("3/3 PRISM confirms it has received a real trace. Tracing works.")
        return 0
    print("3/3 PRISM has not confirmed a real trace yet. Look in the PRISM web app, then run this again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
