"""Unit tests for Role 2 scenario library, baseline returns, and preset portfolios."""

import unittest

from money_talks.contracts import ASSET_CLASSES
from money_talks.data.source import (
    get_baseline_returns,
    get_historical_scenarios,
    get_preset_portfolios,
)


class SourceTests(unittest.TestCase):
    def test_baseline_returns_cover_all_assets(self) -> None:
        baseline = get_baseline_returns()
        self.assertEqual(set(baseline), set(ASSET_CLASSES))
        for asset, ret in baseline.items():
            self.assertGreater(ret, -1.0, asset)
            self.assertLess(ret, 1.0, asset)

    def test_preset_portfolios_valid_and_complete(self) -> None:
        presets = get_preset_portfolios()
        self.assertGreaterEqual(len(presets), 4)
        self.assertIn("Balanced 60/40", presets)
        for name, portfolio in presets.items():
            self.assertEqual(set(portfolio.weights), set(ASSET_CLASSES), name)
            for asset, weight in portfolio.weights.items():
                self.assertGreaterEqual(weight, 0.0, f"{name}:{asset}")
                self.assertLessEqual(weight, 1.0, f"{name}:{asset}")
            self.assertAlmostEqual(sum(portfolio.weights.values()), 1.0, places=9, msg=name)
            self.assertGreater(portfolio.total_value, 0.0, name)

    def test_scenarios_valid_probabilities_and_shocks(self) -> None:
        scenarios = get_historical_scenarios()
        historical = [s for s in scenarios if s.source == "historical"]
        custom = [s for s in scenarios if s.source == "custom"]
        self.assertGreaterEqual(len(historical), 6)
        self.assertGreaterEqual(len(custom), 1)

        names = [s.name for s in scenarios]
        self.assertEqual(len(names), len(set(names)))

        for s in scenarios:
            self.assertEqual(set(s.shocks), set(ASSET_CLASSES), s.name)
            self.assertTrue(s.description.strip(), s.name)
            self.assertTrue(s.rationale.strip(), s.name)
            self.assertGreater(s.probability, 0.0, s.name)
            self.assertLess(s.probability, 1.0, s.name)
            for asset, shock in s.shocks.items():
                self.assertGreaterEqual(shock, -1.0, f"{s.name}:{asset}")
                self.assertLessEqual(shock, 1.0, f"{s.name}:{asset}")

        total_prob = sum(s.probability for s in scenarios)
        self.assertLess(total_prob, 1.0)
        self.assertLessEqual(total_prob, 0.50 + 1e-9)

    def test_returned_collections_are_isolated_copies(self) -> None:
        b1 = get_baseline_returns()
        b1["us_equity"] = -0.99
        self.assertNotEqual(get_baseline_returns()["us_equity"], -0.99)

        s1 = get_historical_scenarios()
        s1[0].shocks["us_equity"] = -0.99
        self.assertNotEqual(get_historical_scenarios()[0].shocks["us_equity"], -0.99)


if __name__ == "__main__":
    unittest.main()
