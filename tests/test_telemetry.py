"""Unit tests for Role 1 non-blocking PRISM telemetry."""

import os
import unittest
from unittest.mock import patch

from money_talks import pipeline, telemetry
from money_talks.contracts import ASSET_CLASSES, Scenario


class TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        telemetry.clear_spans()

    def test_record_span_captures_evidence_hash_without_env(self) -> None:
        with patch.dict(
            os.environ,
            {"PRISMTRACE_HOST": "", "PRISMTRACE_PROJECT_ID": "", "PRISMTRACE_API_KEY": ""},
            clear=False,
        ):
            span = telemetry.record_span("test.span", {"key": "value"}, duration_ms=1.25)

        self.assertEqual(span["span_name"], "test.span")
        self.assertEqual(len(span["evidence_hash"]), 64)
        self.assertEqual(len(telemetry.get_recent_spans()), 1)

    def test_unreachable_prism_endpoint_never_raises(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PRISMTRACE_HOST": "https://prism.invalid.local",
                "PRISMTRACE_PROJECT_ID": "proj-1",
                "PRISMTRACE_API_KEY": "secret-key",
            },
            clear=False,
        ):
            with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
                sc = Scenario(
                    name="Test",
                    shocks={a: 0.0 for a in ASSET_CLASSES},
                    rationale="Fallback.\nGuardrail fixes: clamped us_equity shock from -0.95 to -0.80",
                )
                span = telemetry.record_guardrail_decision("oil shock", [sc])
                self.assertEqual(span["span_name"], "guardrail.decision")
                self.assertTrue(span["attributes"]["guardrail_modified"])

    def test_pipeline_emits_telemetry_spans(self) -> None:
        with patch("money_talks.ai.analyzer._call_llm", side_effect=OSError("offline")):
            scenarios = pipeline.scenarios_from_text("oil jumps 40%")
        self.assertEqual(len(scenarios), 3)
        names = [s["span_name"] for s in telemetry.get_recent_spans()]
        self.assertIn("llm.scenarios_from_text", names)
        self.assertIn("guardrail.decision", names)


if __name__ == "__main__":
    unittest.main()
