"""Unit tests for Role 3 deterministic stress engine and LP optimizer."""

import unittest

from money_talks.contracts import ASSET_CLASSES, Constraints, Portfolio, Scenario
from money_talks.finance.evaluator import (
    efficient_tradeoff_curve,
    propose_allocation,
    run_stress_test,
    severity_sensitivity,
)


def _zero_weights(**overrides: float) -> dict[str, float]:
    weights = {asset: 0.0 for asset in ASSET_CLASSES}
    weights.update(overrides)
    return weights


def _zero_shocks(**overrides: float) -> dict[str, float]:
    shocks = {asset: 0.0 for asset in ASSET_CLASSES}
    shocks.update(overrides)
    return shocks


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = {
            "us_equity": 0.08,
            "intl_equity": 0.07,
            "em_equity": 0.09,
            "gov_bonds": 0.04,
            "corp_credit": 0.05,
            "gold": 0.03,
            "commodities": 0.03,
            "real_estate": 0.06,
            "cash": 0.03,
        }

    def test_hand_computed_stress_test(self) -> None:
        portfolio = Portfolio(
            weights=_zero_weights(us_equity=0.60, gov_bonds=0.40),
            total_value=1_000_000.0,
        )
        s1 = Scenario(
            name="Crash",
            shocks=_zero_shocks(us_equity=-0.40, gov_bonds=0.10),
            probability=0.10,
            severity=1.0,
        )
        s2 = Scenario(
            name="Rates",
            shocks=_zero_shocks(us_equity=-0.10, gov_bonds=-0.10),
            probability=0.20,
            severity=1.5,
        )
        report = run_stress_test(portfolio, [s1, s2], self.baseline)

        # s1 return = 0.6*(-0.40) + 0.4*(0.10) = -0.20
        # s2 effective shocks = -0.15, -0.15 => return = -0.15
        # baseline return = 0.6*0.08 + 0.4*0.04 = 0.064, baseline_prob = 0.70
        expected = 0.10 * (-0.20) + 0.20 * (-0.15) + 0.70 * 0.064
        pwl = 0.10 * 0.20 + 0.20 * 0.15

        self.assertAlmostEqual(report.results[0].portfolio_return, -0.20, places=9)
        self.assertAlmostEqual(report.results[0].pnl_value, -200_000.0, places=4)
        self.assertAlmostEqual(report.results[1].portfolio_return, -0.15, places=9)
        self.assertAlmostEqual(report.baseline_return, 0.064, places=9)
        self.assertAlmostEqual(report.baseline_probability, 0.70, places=9)
        self.assertAlmostEqual(report.expected_return, expected, places=9)
        self.assertAlmostEqual(report.probability_weighted_loss, pwl, places=9)
        self.assertEqual(report.worst_case_scenario, "Crash")
        self.assertAlmostEqual(report.worst_case_return, -0.20, places=9)

    def test_all_cash_portfolio_loses_nothing_when_cash_shock_is_zero(self) -> None:
        portfolio = Portfolio(weights=_zero_weights(cash=1.0), total_value=500_000.0)
        crash = Scenario(name="Equity crash", shocks=_zero_shocks(us_equity=-0.50, cash=0.0), probability=0.20)
        report = run_stress_test(portfolio, [crash], self.baseline)
        self.assertAlmostEqual(report.results[0].portfolio_return, 0.0, places=9)
        self.assertAlmostEqual(report.probability_weighted_loss, 0.0, places=9)
        self.assertAlmostEqual(report.worst_case_return, 0.0, places=9)

    def test_empty_scenarios_uses_baseline(self) -> None:
        portfolio = Portfolio(weights=_zero_weights(us_equity=0.50, cash=0.50), total_value=100_000.0)
        report = run_stress_test(portfolio, [], self.baseline)
        self.assertEqual(report.results, [])
        self.assertAlmostEqual(report.baseline_probability, 1.0, places=9)
        self.assertAlmostEqual(report.expected_return, 0.50 * 0.08 + 0.50 * 0.03, places=9)
        self.assertEqual(report.worst_case_scenario, "")
        self.assertAlmostEqual(report.worst_case_return, 0.0, places=9)

    def test_validation_errors(self) -> None:
        bad_portfolio = Portfolio(weights=_zero_weights(us_equity=0.50, cash=0.40))
        s = Scenario(name="Mild", shocks=_zero_shocks(us_equity=-0.10), probability=0.20)
        with self.assertRaises(ValueError):
            run_stress_test(bad_portfolio, [s], self.baseline)

        good_portfolio = Portfolio(weights=_zero_weights(us_equity=0.50, cash=0.50))
        too_probable = [
            Scenario(name="A", shocks=_zero_shocks(us_equity=-0.10), probability=0.60),
            Scenario(name="B", shocks=_zero_shocks(us_equity=-0.20), probability=0.50),
        ]
        with self.assertRaises(ValueError):
            run_stress_test(good_portfolio, too_probable, self.baseline)

    def test_propose_allocation_feasible_respects_all_constraints(self) -> None:
        portfolio = Portfolio(
            weights=_zero_weights(us_equity=0.60, gov_bonds=0.35, cash=0.05),
            total_value=1_000_000.0,
        )
        scenarios = [
            Scenario(name="Crash", shocks=_zero_shocks(us_equity=-0.40, gov_bonds=0.10, cash=0.01), probability=0.10),
            Scenario(name="Inflation", shocks=_zero_shocks(us_equity=-0.20, gov_bonds=-0.15, cash=0.02), probability=0.15),
        ]
        constraints = Constraints(max_turnover=0.30, max_scenario_loss=0.12)
        proposal = propose_allocation(portfolio, scenarios, self.baseline, constraints)

        self.assertTrue(proposal.feasible)
        self.assertAlmostEqual(sum(proposal.new_weights.values()), 1.0, places=7)
        for asset, val in proposal.new_weights.items():
            self.assertGreaterEqual(val, -1e-9, asset)
            self.assertLessEqual(val, 0.60 + 1e-7, asset)
        turnover = sum(abs(proposal.trades[a]) for a in ASSET_CLASSES) / 2.0
        self.assertLessEqual(turnover, constraints.max_turnover + 1e-6)
        for res in proposal.after.results:
            self.assertGreaterEqual(res.portfolio_return, -constraints.max_scenario_loss - 1e-6)
        self.assertTrue(proposal.explanation)

    def test_propose_allocation_infeasible_returns_best_effort_fallback(self) -> None:
        portfolio = Portfolio(
            weights=_zero_weights(us_equity=0.60, intl_equity=0.35, cash=0.05),
            total_value=1_000_000.0,
        )
        severe = [
            Scenario(
                name="Deep Crash",
                shocks=_zero_shocks(us_equity=-0.50, intl_equity=-0.50, cash=0.0),
                probability=0.20,
            )
        ]
        constraints = Constraints(max_turnover=0.20, max_scenario_loss=0.05)
        proposal = propose_allocation(portfolio, severe, self.baseline, constraints)

        self.assertFalse(proposal.feasible)
        self.assertAlmostEqual(sum(proposal.new_weights.values()), 1.0, places=7)
        turnover = sum(abs(proposal.trades[a]) for a in ASSET_CLASSES) / 2.0
        self.assertLessEqual(turnover, constraints.max_turnover + 1e-6)
        self.assertGreater(proposal.after.worst_case_return, proposal.before.worst_case_return)

    def test_stretch_helpers(self) -> None:
        portfolio = Portfolio(weights=_zero_weights(us_equity=0.60, gov_bonds=0.40))
        scenarios = [
            Scenario(name="Shock", shocks=_zero_shocks(us_equity=-0.30, gov_bonds=0.05), probability=0.10)
        ]
        sens = severity_sensitivity(portfolio, scenarios, self.baseline, severities=(0.5, 1.0, 2.0))
        self.assertEqual(len(sens), 3)
        self.assertGreater(sens[0]["worst_case_return"], sens[2]["worst_case_return"])

        curve = efficient_tradeoff_curve(
            portfolio, scenarios, self.baseline, Constraints(max_turnover=0.50), loss_caps=(0.05, 0.15)
        )
        self.assertEqual(len(curve), 2)


if __name__ == "__main__":
    unittest.main()
