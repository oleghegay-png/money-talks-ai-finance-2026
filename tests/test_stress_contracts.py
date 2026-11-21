"""Smoke tests for the stress-test contracts, mock data and pipeline facade."""

import unittest

from money_talks import mock, pipeline
from money_talks.contracts import ASSET_CLASSES, Constraints, Scenario


class StressContractTests(unittest.TestCase):
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
