"""Tests for the deterministic stress engine and allocation optimizer."""

import unittest

from money_talks.contracts import ASSET_CLASSES, ASSET_LABELS, Constraints, Portfolio, Scenario
from money_talks.finance.evaluator import propose_allocation, run_stress_test, severity_sensitivity


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
        self.assertIn("equity selloff", proposal.explanation)

    def test_default_max_weight_caps_new_holdings_at_40_pct_without_breaking_start(self) -> None:
        starting_high = Portfolio(_weights(us_equity=0.60, gov_bonds=0.40))
        constraints = Constraints(max_turnover=0.30, max_scenario_loss=0.12)
        proposal = propose_allocation(starting_high, self.scenarios, self.baseline, constraints)

        self.assertTrue(proposal.feasible)
        for asset, weight in proposal.new_weights.items():
            cap = 0.60 if asset == "us_equity" else 0.40
            self.assertLessEqual(weight, cap + 1e-8, asset)

        diversified = Portfolio(_weights(us_equity=0.40, intl_equity=0.30, gov_bonds=0.30))
        scenarios = [
            Scenario(
                name="global selloff",
                shocks=_returns(us_equity=-0.30, intl_equity=-0.30, gov_bonds=0.05),
                probability=0.20,
            )
        ]
        proposal_div = propose_allocation(
            diversified, scenarios, self.baseline, Constraints(max_turnover=0.50, max_scenario_loss=0.15)
        )
        self.assertTrue(proposal_div.feasible)
        self.assertTrue(
            all(weight <= 0.40 + 1e-8 for weight in proposal_div.new_weights.values())
        )

    def test_explicit_max_weight_overrides_default_40_pct_cap(self) -> None:
        portfolio = Portfolio(_weights(us_equity=0.35, gov_bonds=0.35, cash=0.30))
        constraints = Constraints(
            max_turnover=0.60,
            max_scenario_loss=0.05,
            max_weight={"cash": 0.80},
        )
        proposal = propose_allocation(portfolio, self.scenarios, self.baseline, constraints)
        self.assertTrue(proposal.feasible)
        self.assertGreater(proposal.new_weights["cash"], 0.40)
        self.assertLessEqual(proposal.new_weights["cash"], 0.80 + 1e-8)

    def test_proposal_explanation_names_largest_trades_and_binding_scenario(self) -> None:
        constraints = Constraints(max_turnover=0.30, max_scenario_loss=0.12)
        proposal = propose_allocation(self.portfolio, self.scenarios, self.baseline, constraints)

        self.assertIn(ASSET_LABELS["us_equity"], proposal.explanation)
        self.assertIn("equity selloff", proposal.explanation)
        self.assertIn("-12.0%", proposal.explanation)

        infeasible_proposal = propose_allocation(
            self.portfolio,
            self.scenarios,
            self.baseline,
            Constraints(max_turnover=0.05, max_scenario_loss=0.01),
        )
        self.assertIn(ASSET_LABELS["us_equity"], infeasible_proposal.explanation)
        self.assertIn("equity selloff", infeasible_proposal.explanation)

    def test_severity_sensitivity_scales_without_mutating_inputs(self) -> None:
        portfolio = Portfolio(_weights(us_equity=0.50, cash=0.50))
        scenario = Scenario(name="drop", shocks=_returns(us_equity=-0.40), probability=0.10, severity=1.0)
        sens = severity_sensitivity(portfolio, [scenario], self.baseline, multipliers=(0.5, 1.0, 1.5, 2.0))

        self.assertAlmostEqual(sens[0.5], -0.10)
        self.assertAlmostEqual(sens[1.0], -0.20)
        self.assertAlmostEqual(sens[1.5], -0.30)
        self.assertAlmostEqual(sens[2.0], -0.40)
        self.assertEqual(scenario.severity, 1.0)

        empty = severity_sensitivity(portfolio, [], self.baseline, multipliers=(0.5, 1.0))
        self.assertEqual(empty, {0.5: 0.0, 1.0: 0.0})

        with self.assertRaises(ValueError):
            severity_sensitivity(portfolio, [scenario], self.baseline, multipliers=(-0.5,))


if __name__ == "__main__":
    unittest.main()
