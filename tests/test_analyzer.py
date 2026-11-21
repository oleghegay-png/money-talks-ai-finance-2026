"""Offline unit tests for Role 4 deterministic guardrail, JSON parser, and fallbacks."""

import json
import os
import unittest
from unittest.mock import patch

from money_talks.ai import analyzer
from money_talks.contracts import ASSET_CLASSES, AllocationProposal, Portfolio, Scenario, StressReport


class AnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        analyzer.clear_caches()

    def test_validate_scenario_fixes_unknown_missing_and_out_of_bounds(self) -> None:
        dirty = Scenario(
            name="Extreme Oil Shock",
            description="Oil spikes sharply",
            shocks={
                "us_equity": -0.95,
                "commodities": 1.40,
                "crypto": -0.50,
            },
            probability=0.80,
            severity=0.10,
            source="llm",
            rationale="Original LLM rationale.",
        )
        clean, fixes = analyzer.validate_scenario(dirty)

        self.assertEqual(set(clean.shocks), set(ASSET_CLASSES))
        self.assertNotIn("crypto", clean.shocks)
        self.assertAlmostEqual(clean.shocks["us_equity"], -0.80, places=9)
        self.assertAlmostEqual(clean.shocks["commodities"], 1.00, places=9)
        self.assertAlmostEqual(clean.shocks["cash"], 0.0, places=9)
        self.assertAlmostEqual(clean.probability, 0.50, places=9)
        self.assertAlmostEqual(clean.severity, 0.25, places=9)
        self.assertGreaterEqual(len(fixes), 5)
        # Input object is not mutated
        self.assertIn("crypto", dirty.shocks)

    def test_validate_scenario_clean_input_returns_no_fixes(self) -> None:
        valid = Scenario(
            name="Clean",
            shocks={asset: -0.05 for asset in ASSET_CLASSES},
            probability=0.10,
            severity=1.0,
        )
        clean, fixes = analyzer.validate_scenario(valid)
        self.assertEqual(fixes, [])
        self.assertAlmostEqual(clean.probability, 0.10)

    def test_parse_llm_scenarios_json_scales_probabilities_and_records_fixes(self) -> None:
        canned_json = """```json
        {
          "scenarios": [
            {
              "name": "Oil Shock (mild)",
              "description": "Modest spike",
              "shocks": {"us_equity": -0.10, "commodities": 0.15},
              "probability": 0.30,
              "severity": 0.5,
              "historical_analogue": "2022 Inflation & Rate Shock",
              "rationale": "Mild supply disruption."
            },
            {
              "name": "Oil Shock (base)",
              "description": "Sustained shock",
              "shocks": {"us_equity": -0.90, "commodities": 0.35},
              "probability": 0.30,
              "severity": 1.0,
              "rationale": "Stagflation pressure."
            },
            {
              "name": "Oil Shock (severe)",
              "description": "Severe embargo",
              "shocks": {"us_equity": -0.45, "commodities": 0.50},
              "probability": 0.20,
              "severity": 1.5,
              "rationale": "Deep recession."
            }
          ]
        }
        ```"""
        scenarios = analyzer._parse_llm_scenarios_json(canned_json, "oil jumps 40%")
        self.assertEqual(len(scenarios), 3)
        self.assertLessEqual(sum(s.probability for s in scenarios), 0.50 + 1e-9)
        for s in scenarios:
            self.assertEqual(s.source, "llm")
            self.assertEqual(set(s.shocks), set(ASSET_CLASSES))
            self.assertIn("Guardrail fixes:", s.rationale)
        self.assertIn("2022 Inflation & Rate Shock", scenarios[0].rationale)
        self.assertAlmostEqual(scenarios[1].shocks["us_equity"], -0.80, places=9)

    def test_generate_scenarios_caches_successful_llm_calls(self) -> None:
        canned = json.dumps(
            {
                "scenarios": [
                    {
                        "name": f"Rate Hike ({lbl})",
                        "description": "Fed tightens",
                        "shocks": {a: -0.05 for a in ASSET_CLASSES},
                        "probability": p,
                        "severity": sev,
                        "rationale": "Tighter financial conditions.",
                    }
                    for lbl, sev, p in (("mild", 0.5, 0.15), ("base", 1.0, 0.10), ("severe", 1.5, 0.05))
                ]
            }
        )
        with patch.object(analyzer, "_call_llm", return_value=canned) as mock_call:
            first = analyzer.generate_scenarios("Fed hikes 100 bps")
            second = analyzer.generate_scenarios("Fed hikes 100 bps")
            self.assertEqual(mock_call.call_count, 1)
            self.assertEqual(len(first), 3)
            self.assertEqual(len(second), 3)

    def test_generate_scenarios_never_raises_on_errors(self) -> None:
        with patch.object(analyzer, "_call_llm", side_effect=OSError("network down")):
            res = analyzer.generate_scenarios("oil shock")
            self.assertEqual(len(res), 3)
            self.assertTrue(all(s.source == "llm" for s in res))
            self.assertTrue(all("unavailable" in s.rationale.lower() for s in res))
            self.assertLessEqual(sum(s.probability for s in res), 0.50)

        with patch.object(analyzer, "_call_llm", return_value="not valid json at all"):
            res_bad = analyzer.generate_scenarios("credit crunch")
            self.assertEqual(len(res_bad), 3)
            self.assertTrue(all("unavailable" in s.rationale.lower() for s in res_bad))

        with patch.dict(os.environ, {"NICRRON_API_KEY": ""}, clear=False):
            with patch.object(analyzer, "_read_dotenv_value", return_value=""):
                res_nokey = analyzer.generate_scenarios("tariff shock")
                self.assertEqual(len(res_nokey), 3)
                self.assertTrue(all("unavailable" in s.rationale.lower() for s in res_nokey))

    def test_explain_report_llm_and_fallback_template(self) -> None:
        portfolio = Portfolio(weights={a: (1.0 if a == "us_equity" else 0.0) for a in ASSET_CLASSES})
        report = StressReport(
            baseline_return=0.075,
            baseline_probability=0.90,
            expected_return=0.045,
            probability_weighted_loss=0.03,
            worst_case_scenario="2008 GFC",
            worst_case_return=-0.37,
        )
        proposal = AllocationProposal(
            before=report,
            after=StressReport(expected_return=0.052, worst_case_return=-0.15),
            feasible=True,
        )

        with patch.object(analyzer, "_call_llm", side_effect=TimeoutError("timed out")):
            fallback = analyzer.explain_report(portfolio, report, proposal)
            self.assertIn("2008 GFC", fallback)
            self.assertIn("-37.0%", fallback)
            self.assertIn("-15.0%", fallback)

        with patch.object(analyzer, "_call_llm", return_value="Canned LLM summary."):
            reply = analyzer.explain_report(portfolio, report, proposal)
            self.assertEqual(reply, "Canned LLM summary.")


if __name__ == "__main__":
    unittest.main()
