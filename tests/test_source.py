"""Tests for the scenario library, baseline returns and preset portfolios."""

import unittest

from money_talks import pipeline
from money_talks.contracts import ASSET_CLASSES
from money_talks.data import source


class SourceTests(unittest.TestCase):
    def test_baseline_covers_every_asset_class(self) -> None:
        baseline = source.get_baseline_returns()
        self.assertEqual(set(baseline), set(ASSET_CLASSES))
        for value in baseline.values():
            self.assertTrue(0.0 <= value <= 0.15)

    def test_scenarios_are_complete_and_in_range(self) -> None:
        library = source.get_scenario_library()
        self.assertGreaterEqual(len(source.get_historical_scenarios()), 6)
        self.assertEqual(len({s.name for s in library}), len(library))
        for scenario in library:
            self.assertEqual(set(scenario.shocks), set(ASSET_CLASSES), scenario.name)
            for shock in scenario.shocks.values():
                self.assertTrue(-1.0 <= shock <= 1.0, scenario.name)
            self.assertTrue(0.0 < scenario.probability < 1.0, scenario.name)
            self.assertTrue(scenario.description and scenario.rationale, scenario.name)
            self.assertIn(scenario.source, ("historical", "custom"))

    def test_library_probabilities_leave_room_for_the_baseline(self) -> None:
        self.assertLess(sum(s.probability for s in source.get_scenario_library()), 1.0)

    def test_presets_are_valid_portfolios(self) -> None:
        presets = source.get_preset_portfolios()
        self.assertGreaterEqual(len(presets), 4)
        for name, portfolio in presets.items():
            self.assertEqual(set(portfolio.weights), set(ASSET_CLASSES), name)
            self.assertAlmostEqual(sum(portfolio.weights.values()), 1.0, msg=name)
            self.assertTrue(all(w >= 0.0 for w in portfolio.weights.values()), name)

    def test_pipeline_serves_real_data(self) -> None:
        self.assertEqual(len(pipeline.historical_scenarios()), len(source.get_scenario_library()))
        self.assertEqual(pipeline.baseline_returns(), source.get_baseline_returns())
        self.assertIn("Balanced 60/40", pipeline.preset_portfolios())


if __name__ == "__main__":
    unittest.main()
