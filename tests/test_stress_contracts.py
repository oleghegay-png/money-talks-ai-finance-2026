"""Smoke tests for the stress-test contracts, mock data and pipeline facade."""

import os
import tempfile
import unittest

from money_talks import mock, pipeline
from money_talks.contracts import ASSET_CLASSES, Constraints, Scenario


class StressContractTests(unittest.TestCase):
    def setUp(self) -> None:
        # Never call the network or write into the repository from tests.
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = {k: os.environ.get(k) for k in ("MONEY_TALKS_LLM_OFFLINE", "MONEY_TALKS_AUDIT_LOG")}
        os.environ["MONEY_TALKS_LLM_OFFLINE"] = "1"
        os.environ["MONEY_TALKS_AUDIT_LOG"] = os.path.join(self._tmp.name, "audit.jsonl")

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def test_safe_scenarios_scales_only_when_needed(self) -> None:
        fine = [Scenario(name="a", probability=0.3), Scenario(name="b", probability=0.4)]
        same, message = pipeline.safe_scenarios(fine)
        self.assertIsNone(message)
        self.assertEqual([s.probability for s in same], [0.3, 0.4])

        too_much = [Scenario(name="a", probability=0.6), Scenario(name="b", probability=0.9)]
        scaled, message = pipeline.safe_scenarios(too_much)
        self.assertIn("150%", message)
        self.assertAlmostEqual(sum(s.probability for s in scaled), pipeline.MAX_TOTAL_PROBABILITY)
        self.assertAlmostEqual(scaled[0].probability / scaled[1].probability, 0.6 / 0.9)
        self.assertEqual(too_much[0].probability, 0.6)

        portfolio = next(iter(pipeline.preset_portfolios().values()))
        pipeline.stress_test(portfolio, scaled)  # must not raise

    def test_audit_trail_records_proposals_and_decisions(self) -> None:
        self.assertEqual(pipeline.audit_tail(), [])
        portfolio = next(iter(pipeline.preset_portfolios().values()))
        scenarios = pipeline.scenarios_from_text("oil shock")
        pipeline.prepare(portfolio, scenarios, Constraints())
        events = [record["event"] for record in pipeline.audit_tail()]
        self.assertIn("llm_scenario", events)
        self.assertEqual(events[-1], "allocation_decision")

    def test_mock_data_covers_every_asset_class(self) -> None:
        self.assertEqual(set(mock.get_baseline_returns()), set(ASSET_CLASSES))
        for portfolio in mock.get_preset_portfolios().values():
            self.assertAlmostEqual(sum(portfolio.weights.values()), 1.0)
        for scenario in mock.get_historical_scenarios():
            self.assertEqual(set(scenario.shocks), set(ASSET_CLASSES))

    def test_severity_scales_shocks_and_floors_at_total_loss(self) -> None:
        scenario = Scenario(name="x", shocks={"us_equity": -0.40}, severity=1.5)
        self.assertAlmostEqual(scenario.effective_shocks()["us_equity"], -0.60)
        scenario.severity = 5.0
        self.assertEqual(scenario.effective_shocks()["us_equity"], -1.0)
        self.assertEqual(scenario.effective_shocks()["cash"], 0.0)

    def test_pipeline_runs_end_to_end(self) -> None:
        portfolio = next(iter(pipeline.preset_portfolios().values()))
        scenarios = pipeline.historical_scenarios() + pipeline.scenarios_from_text("oil shock")
        report = pipeline.stress_test(portfolio, scenarios)
        self.assertEqual(len(report.results), len(scenarios))
        self.assertLess(report.worst_case_return, 0.0)

        proposal = pipeline.prepare(portfolio, scenarios, Constraints())
        self.assertAlmostEqual(sum(proposal.new_weights.values()), 1.0)
        self.assertIsInstance(pipeline.explain(portfolio, report, proposal), str)


if __name__ == "__main__":
    unittest.main()
