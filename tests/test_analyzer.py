"""Offline tests for role 4: guardrail, LLM JSON parsing, fallbacks and explanations.

No test touches the network: _call_llm is patched with canned replies, and
urllib is patched to fail loudly if anything tries to open a connection.
"""

import json
import os
import unittest
from unittest import mock as umock

from money_talks import mock as mock_data
from money_talks.ai import analyzer
from money_talks.contracts import ASSET_CLASSES, Constraints, Portfolio, Scenario


def _shocks(**overrides: float) -> dict[str, float]:
    shocks = {asset: 0.0 for asset in ASSET_CLASSES}
    shocks.update(overrides)
    return shocks


def _llm_reply(probabilities=(0.10, 0.06, 0.02)) -> str:
    variants = []
    for variant, probability in zip(("mild", "base", "severe"), probabilities):
        variants.append(
            {
                "variant": variant,
                "description": f"Oil shock, {variant}.",
                "probability": probability,
                "shocks": _shocks(us_equity=-0.10, commodities=0.30, gov_bonds=-0.03),
                "rationale": f"{variant} rationale.",
            }
        )
    return json.dumps(
        {"title": "Oil spike", "closest_historical_analogue": "1973 oil embargo", "scenarios": variants}
    )


def _no_network(*args, **kwargs):
    raise AssertionError("tests must not open network connections")


class AnalyzerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        analyzer.clear_cache()
        patcher = umock.patch("urllib.request.urlopen", side_effect=_no_network)
        patcher.start()
        self.addCleanup(patcher.stop)


class ValidateScenarioTests(AnalyzerTestCase):
    def test_clean_scenario_needs_no_fixes(self) -> None:
        scenario = Scenario(name="ok", shocks=_shocks(us_equity=-0.2), probability=0.1, severity=1.0)
        clean, fixes = analyzer.validate_scenario(scenario)
        self.assertEqual(fixes, [])
        self.assertEqual(clean.shocks, scenario.shocks)

    def test_fills_missing_and_drops_unknown_keys(self) -> None:
        scenario = Scenario(name="x", shocks={"us_equity": -0.2, "bitcoin": -0.5})
        clean, fixes = analyzer.validate_scenario(scenario)
        self.assertEqual(list(clean.shocks), list(ASSET_CLASSES))
        self.assertEqual(clean.shocks["gold"], 0.0)
        self.assertNotIn("bitcoin", clean.shocks)
        self.assertTrue(any("bitcoin" in fix for fix in fixes))
        self.assertTrue(any("gold" in fix for fix in fixes))

    def test_clamps_shocks_probability_and_severity(self) -> None:
        scenario = Scenario(
            name="x",
            shocks=_shocks(us_equity=-0.95, commodities=1.5),
            probability=0.9,
            severity=3.0,
        )
        clean, fixes = analyzer.validate_scenario(scenario)
        self.assertEqual(clean.shocks["us_equity"], -0.80)
        self.assertEqual(clean.shocks["commodities"], 1.00)
        self.assertEqual(clean.probability, 0.50)
        self.assertEqual(clean.severity, 2.0)
        self.assertEqual(len(fixes), 4)
        self.assertIn("Clamped us_equity shock from -95.0% to -80.0%.", fixes)

        low = Scenario(name="y", shocks=_shocks(), probability=0.0, severity=0.1)
        clean, _ = analyzer.validate_scenario(low)
        self.assertEqual(clean.probability, 0.001)
        self.assertEqual(clean.severity, 0.25)

    def test_does_not_mutate_input(self) -> None:
        shocks = {"us_equity": -0.95}
        scenario = Scenario(name="x", shocks=shocks, probability=0.9)
        analyzer.validate_scenario(scenario)
        self.assertEqual(scenario.shocks, {"us_equity": -0.95})
        self.assertEqual(scenario.probability, 0.9)

    def test_renames_labels_and_handles_bad_values(self) -> None:
        scenario = Scenario(
            name="x",
            shocks={"US Equity": "-35%", "gov-bonds": "abc", "cash": float("nan")},
            probability="n/a",
        )
        clean, fixes = analyzer.validate_scenario(scenario)
        self.assertAlmostEqual(clean.shocks["us_equity"], -0.35)
        self.assertEqual(clean.shocks["gov_bonds"], 0.0)
        self.assertEqual(clean.shocks["cash"], 0.0)
        self.assertEqual(clean.probability, 0.001)
        self.assertTrue(any("Renamed 'US Equity'" in fix for fix in fixes))


class GenerateScenariosTests(AnalyzerTestCase):
    def test_parses_three_guarded_llm_scenarios(self) -> None:
        data = json.loads(_llm_reply())
        data["scenarios"][0]["shocks"]["us_equity"] = -0.95
        with umock.patch.object(analyzer, "_call_llm", return_value=json.dumps(data)):
            scenarios = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertEqual(len(scenarios), 3)
        for scenario in scenarios:
            self.assertEqual(scenario.source, "llm")
            self.assertEqual(set(scenario.shocks), set(ASSET_CLASSES))
            self.assertIn(analyzer.GUARDRAIL_PREFIX, scenario.rationale)
            self.assertIn("1973 oil embargo", scenario.rationale)
        self.assertEqual(scenarios[0].name, "Oil spike (mild)")
        self.assertEqual(scenarios[0].shocks["us_equity"], -0.80)
        self.assertIn("Clamped us_equity shock", scenarios[0].rationale)
        self.assertIn("no corrections needed", scenarios[1].rationale)

    def test_scales_probabilities_to_fifty_percent(self) -> None:
        with umock.patch.object(analyzer, "_call_llm", return_value=_llm_reply((0.40, 0.30, 0.30))):
            scenarios = analyzer.generate_scenarios("Fed hikes 100bp")
        self.assertAlmostEqual(sum(s.probability for s in scenarios), 0.50)
        self.assertIn("Scaled probability", scenarios[0].rationale)

    def test_accepts_code_fences_and_percentage_points(self) -> None:
        data = json.loads(_llm_reply())
        data["scenarios"][0]["shocks"] = {"us_equity": -35, "gov_bonds": 5, "cash": 1}
        data["scenarios"][0]["probability"] = 10
        reply = "Here you go:\n```json\n" + json.dumps(data) + "\n```"
        with umock.patch.object(analyzer, "_call_llm", return_value=reply):
            scenarios = analyzer.generate_scenarios("Recession")
        mild = scenarios[0]
        self.assertAlmostEqual(mild.shocks["us_equity"], -0.35)
        self.assertAlmostEqual(mild.shocks["gov_bonds"], 0.05)
        self.assertAlmostEqual(mild.probability, 0.10)
        self.assertIn("percentage points", mild.rationale)

    def test_bad_json_returns_labelled_fallback(self) -> None:
        with umock.patch.object(analyzer, "_call_llm", return_value="Sorry, I cannot help."):
            scenarios = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertEqual(len(scenarios), 1)
        self.assertIn("FALLBACK", scenarios[0].name)
        self.assertIn("LLM was unavailable", scenarios[0].rationale)
        self.assertEqual(set(scenarios[0].shocks), set(ASSET_CLASSES))

    def test_network_errors_never_raise(self) -> None:
        for error in (TimeoutError("timed out"), OSError("connection refused"), ValueError("bad")):
            analyzer.clear_cache()
            with umock.patch.object(analyzer, "_call_llm", side_effect=error):
                scenarios = analyzer.generate_scenarios("Oil jumps 40%")
            self.assertIn("FALLBACK", scenarios[0].name)

    def test_missing_api_key_falls_back_without_network(self) -> None:
        env = {"NICRRON_API_KEY": "", analyzer.OFFLINE_ENV_VAR: ""}
        with umock.patch.dict(os.environ, env), umock.patch.object(analyzer, "_load_env_file"):
            scenarios = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertIn("NICRRON_API_KEY not set", scenarios[0].rationale)

    def test_offline_switch_skips_network(self) -> None:
        with umock.patch.dict(os.environ, {analyzer.OFFLINE_ENV_VAR: "1"}):
            scenarios = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertIn("FALLBACK", scenarios[0].name)

    def test_offline_switch_false_means_online(self) -> None:
        env = {analyzer.OFFLINE_ENV_VAR: "False", "NICRRON_API_KEY": ""}
        with umock.patch.dict(os.environ, env), umock.patch.object(analyzer, "_load_env_file"):
            with self.assertRaisesRegex(analyzer.LLMUnavailable, "NICRRON_API_KEY not set"):
                analyzer._call_llm("system", "user")

    def test_llm_severity_is_ignored_to_avoid_double_scaling(self) -> None:
        data = json.loads(_llm_reply())
        data["scenarios"][2]["severity"] = 1.5
        with umock.patch.object(analyzer, "_call_llm", return_value=json.dumps(data)):
            scenarios = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertEqual([s.severity for s in scenarios], [1.0, 1.0, 1.0])

    def test_duplicate_variant_labels_get_unique_names(self) -> None:
        data = json.loads(_llm_reply())
        for item in data["scenarios"]:
            item["variant"] = "base"
        with umock.patch.object(analyzer, "_call_llm", return_value=json.dumps(data)):
            scenarios = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertEqual(len({s.name for s in scenarios}), 3)

    def test_prose_with_brackets_before_json_still_parses(self) -> None:
        reply = "Note [1]: shocks are approximate.\n" + _llm_reply()
        with umock.patch.object(analyzer, "_call_llm", return_value=reply):
            scenarios = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertEqual(len(scenarios), 3)

    def test_fallback_name_is_cut_at_a_word_boundary(self) -> None:
        with umock.patch.object(analyzer, "_call_llm", side_effect=TimeoutError()):
            scenario = analyzer.generate_scenarios("oil jumps 40% and the Fed hikes 100 basis points")[0]
        self.assertTrue(scenario.name.startswith("oil jumps 40% and the Fed hikes 100..."))

    def test_results_are_cached_and_copied(self) -> None:
        with umock.patch.object(analyzer, "_call_llm", return_value=_llm_reply()) as call:
            first = analyzer.generate_scenarios("Oil jumps 40%")
            first[0].probability = 0.42  # e.g. a UI slider
            second = analyzer.generate_scenarios("  oil JUMPS   40% ")
        self.assertEqual(call.call_count, 1)
        self.assertNotEqual(second[0].probability, 0.42)

    def test_failures_are_not_cached(self) -> None:
        with umock.patch.object(analyzer, "_call_llm", side_effect=TimeoutError()):
            analyzer.generate_scenarios("Oil jumps 40%")
        with umock.patch.object(analyzer, "_call_llm", return_value=_llm_reply()):
            scenarios = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertEqual(len(scenarios), 3)


class ExplainReportTests(AnalyzerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.portfolio = mock_data.get_preset_portfolios()["Balanced 60/40"]
        self.scenarios = mock_data.get_historical_scenarios()
        self.report = mock_data.run_stress_test(
            self.portfolio, self.scenarios, mock_data.get_baseline_returns()
        )

    def test_template_used_when_llm_fails(self) -> None:
        with umock.patch.object(analyzer, "_call_llm", side_effect=TimeoutError()):
            text = analyzer.explain_report(self.portfolio, self.report)
        self.assertIn(self.report.worst_case_scenario, text)
        self.assertIn(f"{self.report.worst_case_return:.1%}", text)
        self.assertIn(f"{self.report.expected_return:.1%}", text)

    def test_llm_text_with_invented_number_is_rejected(self) -> None:
        invented = "Your portfolio could lose 99.9% in the worst case."
        with umock.patch.object(analyzer, "_call_llm", return_value=invented):
            text = analyzer.explain_report(self.portfolio, self.report)
        self.assertNotIn("99.9", text)
        self.assertIn(self.report.worst_case_scenario, text)

    def test_llm_text_with_verified_numbers_is_used(self) -> None:
        worst = f"{abs(self.report.worst_case_return):.1%}"
        good = f"In the worst case the portfolio would fall by {worst}. Diversification helps."
        with umock.patch.object(analyzer, "_call_llm", return_value=good):
            text = analyzer.explain_report(self.portfolio, self.report)
        self.assertEqual(text, good)

    def test_explains_proposal_and_empty_report(self) -> None:
        proposal = mock_data.propose_allocation(
            self.portfolio, self.scenarios, mock_data.get_baseline_returns(), Constraints()
        )
        with umock.patch.object(analyzer, "_call_llm", side_effect=TimeoutError()):
            text = analyzer.explain_report(self.portfolio, self.report, proposal)
            self.assertIn("re-allocation", text)
            empty = mock_data.run_stress_test(self.portfolio, [], mock_data.get_baseline_returns())
            self.assertIn("No shock scenarios", analyzer.explain_report(self.portfolio, empty))

    def test_never_raises_on_broken_input(self) -> None:
        text = analyzer.explain_report(Portfolio(), None)  # type: ignore[arg-type]
        self.assertIsInstance(text, str)


if __name__ == "__main__":
    unittest.main()
