"""Offline tests for PRISM tracing. A fake client stands in for the PRISM SDK."""

import json
import os
import tempfile
import unittest
from unittest import mock as umock

from money_talks import pipeline, telemetry
from money_talks.ai import analyzer
from money_talks.contracts import ASSET_CLASSES

ENV_KEYS = (
    "MONEY_TALKS_LLM_OFFLINE",
    "MONEY_TALKS_AUDIT_LOG",
    "MONEY_TALKS_DEMO_CACHE",
    "PRISMTRACE_HOST",
    "PRISMTRACE_PROJECT_ID",
    "PRISMTRACE_API_KEY",
    "NICRRON_MODEL",
)


def _llm_reply() -> str:
    shocks = {asset: 0.0 for asset in ASSET_CLASSES}
    shocks["us_equity"] = -0.95  # outside the guardrail limit, so it gets clamped
    variants = [
        {
            "variant": variant,
            "description": f"Oil shock, {variant}.",
            "probability": probability,
            "shocks": shocks,
            "rationale": f"{variant} rationale.",
        }
        for variant, probability in (("mild", 0.10), ("base", 0.06), ("severe", 0.02))
    ]
    return json.dumps({"title": "Oil spike", "closest_historical_analogue": "1973", "scenarios": variants})


class FakeClient:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.traces: list[dict] = []

    def trace_llm(self, **kwargs) -> None:
        if self.fail:
            raise RuntimeError("PRISM is down")
        self.traces.append(kwargs)


class TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = {key: os.environ.get(key) for key in ENV_KEYS}
        os.environ.pop("MONEY_TALKS_LLM_OFFLINE", None)
        os.environ["MONEY_TALKS_AUDIT_LOG"] = os.path.join(self._tmp.name, "audit.jsonl")
        # Keep generated scenarios out of the repository's demo_cache.json.
        os.environ["MONEY_TALKS_DEMO_CACHE"] = os.path.join(self._tmp.name, "demo_cache.json")
        os.environ["PRISMTRACE_HOST"] = "https://prism.invalid"
        os.environ["PRISMTRACE_PROJECT_ID"] = "test-project"
        os.environ["PRISMTRACE_API_KEY"] = "test-key"
        os.environ["NICRRON_MODEL"] = "test-model"
        self._reset_client()
        analyzer.clear_cache()
        telemetry.begin()

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._reset_client()
        analyzer.clear_cache()
        self._tmp.cleanup()

    @staticmethod
    def _reset_client() -> None:
        telemetry._client = None
        telemetry._client_settings = None

    def _use(self, client: FakeClient, llm):
        """Patch in a fake PRISM client and a fake, traced LLM."""
        make = umock.patch.object(telemetry, "_make_client", return_value=client)
        call = umock.patch.object(analyzer, "_call_llm", telemetry.traced(llm))
        make.start()
        call.start()
        self.addCleanup(make.stop)
        self.addCleanup(call.stop)

    def test_scenario_trace_holds_the_proposal_and_the_guardrail_decision(self) -> None:
        client = FakeClient()
        self._use(client, lambda system, user: _llm_reply())
        scenarios = pipeline.scenarios_from_text("Oil jumps 40%")

        self.assertEqual(len(scenarios), 3)
        self.assertEqual(len(client.traces), 1)
        trace = client.traces[0]
        self.assertEqual(trace["model"], "test-model")
        self.assertEqual(trace["output"], _llm_reply())
        self.assertEqual([m["role"] for m in trace["input_messages"]], ["system", "user"])
        self.assertIn("Oil jumps 40%", trace["input_messages"][1]["content"])
        self.assertEqual(trace["session_id"], telemetry.SESSION_ID)
        metadata = trace["metadata"]
        self.assertEqual(metadata["stage"], "scenario_generation")
        self.assertTrue(metadata["llm_succeeded"])
        self.assertEqual(len(metadata["accepted_scenarios"]), 3)
        first = metadata["accepted_scenarios"][0]
        self.assertIn("Clamped us_equity shock", first["guardrail_decision"])
        self.assertEqual(first["shocks"]["us_equity"], -0.80)

    def test_cached_scenarios_send_no_second_trace(self) -> None:
        client = FakeClient()
        self._use(client, lambda system, user: _llm_reply())
        pipeline.scenarios_from_text("Oil jumps 40%")
        pipeline.scenarios_from_text("Oil jumps 40%")
        self.assertEqual(len(client.traces), 1)

    def test_failed_llm_call_is_traced_and_the_app_still_answers(self) -> None:
        client = FakeClient()

        def broken(system, user):
            raise TimeoutError("timed out")

        self._use(client, broken)
        scenarios = pipeline.scenarios_from_text("Oil jumps 40%")
        self.assertEqual(len(scenarios), 1)
        self.assertIn("FALLBACK", scenarios[0].name)
        metadata = client.traces[0]["metadata"]
        self.assertFalse(metadata["llm_succeeded"])
        self.assertIn("TimeoutError", metadata["llm_error"])
        self.assertEqual(client.traces[0]["output"], "")

    def test_explanation_trace_says_when_the_guardrail_rejects_the_reply(self) -> None:
        client = FakeClient()
        self._use(client, lambda system, user: "The portfolio loses exactly 98.7654% of its value.")
        portfolio = next(iter(pipeline.preset_portfolios().values()))
        report = pipeline.stress_test(portfolio, pipeline.historical_scenarios()[:2])
        text = pipeline.explain(portfolio, report)

        self.assertNotIn("98.7654", text)
        metadata = client.traces[0]["metadata"]
        self.assertEqual(metadata["stage"], "explanation")
        self.assertTrue(metadata["guardrail_decision"].startswith("rejected"))
        self.assertEqual(metadata["shown_text"], text)

    def test_prism_failure_never_breaks_the_app(self) -> None:
        self._use(FakeClient(fail=True), lambda system, user: _llm_reply())
        self.assertEqual(len(pipeline.scenarios_from_text("Oil jumps 40%")), 3)

    def test_tracing_is_off_without_credentials_or_when_offline(self) -> None:
        client = FakeClient()
        self._use(client, lambda system, user: _llm_reply())

        os.environ["PRISMTRACE_API_KEY"] = ""
        self.assertIn("PRISMTRACE_API_KEY not set", pipeline.prism_status())
        pipeline.scenarios_from_text("Oil jumps 40%")

        os.environ["PRISMTRACE_API_KEY"] = "test-key"
        os.environ["MONEY_TALKS_LLM_OFFLINE"] = "1"
        analyzer.clear_cache()
        self.assertIn("MONEY_TALKS_LLM_OFFLINE", pipeline.prism_status())
        pipeline.scenarios_from_text("Oil jumps 40%")

        self.assertEqual(client.traces, [])
        os.environ.pop("MONEY_TALKS_LLM_OFFLINE")
        self.assertIn("is on", pipeline.prism_status())

    def test_trace_names_the_model_chosen_for_the_session(self) -> None:
        client = FakeClient()
        make = umock.patch.object(telemetry, "_make_client", return_value=client)
        make.start()
        self.addCleanup(make.stop)
        # Wrap the fake the way the pipeline wraps the real call.
        wrapped = telemetry.traced(
            lambda system, user: _llm_reply(), model=lambda: analyzer.llm_status()["model"]
        )
        call = umock.patch.object(analyzer, "_call_llm", wrapped)
        call.start()
        self.addCleanup(call.stop)
        pipeline.set_llm_credentials(model="vendor/session-model")
        self.addCleanup(pipeline.set_llm_credentials)

        pipeline.scenarios_from_text("Oil jumps 40%")
        self.assertEqual(client.traces[0]["model"], "vendor/session-model")

    def test_model_lookup_failure_falls_back_to_the_environment(self) -> None:
        def broken() -> str:
            raise RuntimeError("no status")

        wrapped = telemetry.traced(lambda system, user: "ok", model=broken)
        wrapped("s", "u")
        self.assertEqual(telemetry.pending()[0].model, "test-model")

    def test_wrapping_twice_is_the_same_as_wrapping_once(self) -> None:
        wrapped = telemetry.traced(lambda system, user: "ok")
        self.assertIs(telemetry.traced(wrapped), wrapped)
        self.assertEqual(wrapped("s", "u"), "ok")
        self.assertEqual(len(telemetry.pending()), 1)


if __name__ == "__main__":
    unittest.main()
