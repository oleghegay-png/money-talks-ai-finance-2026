"""Tests for the deterministic stress engine and allocation optimizer."""

import unittest

from money_talks.contracts import ASSET_CLASSES, Constraints, Portfolio, Scenario
from money_talks.finance.evaluator import propose_allocation, run_stress_test


def _weights(**overrides: float) -> dict[str, float]:
    weights = {asset: 0.0 for asset in ASSET_CLASSES}
    weights.update(overrides)
    return weights


def _returns(**overrides: float) -> dict[str, float]:
    values = {asset: 0.0 for asset in ASSET_CLASSES}
    values.update(overrides)
    return values


class StressEngineTests(unittest.TestCase):
    def test_hand_computed_stress_report(self) -> None:
        portfolio = Portfolio(_weights(us_equity=0.60, gov_bonds=0.40), total_value=1_000.0)
        scenario = Scenario(
            name="recession",
            shocks=_returns(us_equity=-0.20, gov_bonds=0.05),
            probability=0.25,
        )
        report = run_stress_test(
            portfolio,
            [scenario],
            _returns(us_equity=0.10, gov_bonds=0.02),
        )

        self.assertAlmostEqual(report.results[0].contribution_by_asset["us_equity"], -0.12)
        self.assertAlmostEqual(report.results[0].portfolio_return, -0.10)
        self.assertAlmostEqual(report.results[0].pnl_value, -100.0)
        self.assertAlmostEqual(report.baseline_return, 0.068)
        self.assertAlmostEqual(report.expected_return, 0.026)
        self.assertAlmostEqual(report.probability_weighted_loss, 0.025)
        self.assertEqual(report.worst_case_scenario, "recession")

    def test_cash_does_not_lose_when_its_shock_is_zero(self) -> None:
        portfolio = Portfolio(_weights(cash=1.0))
        scenario = Scenario(name="shock", shocks=_returns(us_equity=-0.50, cash=0.0))
        report = run_stress_test(portfolio, [scenario], _returns(cash=0.03))
        self.assertEqual(report.results[0].portfolio_return, 0.0)

    def test_empty_scenarios_use_the_baseline_only(self) -> None:
        portfolio = Portfolio(_weights(cash=1.0))
        report = run_stress_test(portfolio, [], _returns(cash=0.03))
        self.assertEqual(report.results, [])
        self.assertEqual(report.baseline_probability, 1.0)
        self.assertAlmostEqual(report.expected_return, 0.03)
        self.assertEqual(report.worst_case_scenario, "")

    def test_invalid_probability_total_is_rejected(self) -> None:
        portfolio = Portfolio(_weights(cash=1.0))
        scenarios = [
            Scenario(name="one", shocks=_returns(), probability=0.60),
            Scenario(name="two", shocks=_returns(), probability=0.41),
        ]
        with self.assertRaisesRegex(ValueError, "sum to 1.0 or less"):
            run_stress_test(portfolio, scenarios, _returns(cash=0.03))


class AllocationOptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.portfolio = Portfolio(_weights(us_equity=0.60, gov_bonds=0.40))
        self.scenarios = [
            Scenario(
                name="equity selloff",
                shocks=_returns(us_equity=-0.30, gov_bonds=0.0),
                probability=0.20,
            )
        ]
        self.baseline = _returns(us_equity=0.10, gov_bonds=0.03, cash=0.02)

    def test_optimizer_respects_loss_cap_and_turnover(self) -> None:
        constraints = Constraints(max_turnover=0.20, max_scenario_loss=0.12)
        proposal = propose_allocation(self.portfolio, self.scenarios, self.baseline, constraints)

        self.assertTrue(proposal.feasible)
        self.assertAlmostEqual(sum(proposal.new_weights.values()), 1.0)
        self.assertLessEqual(proposal.after.worst_case_return, 0.0)
        self.assertGreaterEqual(proposal.after.worst_case_return, -0.12 - 1e-8)
        turnover = sum(abs(trade) for trade in proposal.trades.values()) / 2
        self.assertLessEqual(turnover, constraints.max_turnover + 1e-8)
        self.assertTrue(all(0.0 <= weight <= 0.60 + 1e-8 for weight in proposal.new_weights.values()))

    def test_infeasible_loss_cap_returns_best_available_allocation(self) -> None:
        constraints = Constraints(max_turnover=0.05, max_scenario_loss=0.01)
        proposal = propose_allocation(self.portfolio, self.scenarios, self.baseline, constraints)

        self.assertFalse(proposal.feasible)
        self.assertAlmostEqual(sum(proposal.new_weights.values()), 1.0)
        self.assertLess(proposal.after.worst_case_return, -constraints.max_scenario_loss)
        self.assertIn("No allocation", proposal.explanation)


if __name__ == "__main__":
    unittest.main()
