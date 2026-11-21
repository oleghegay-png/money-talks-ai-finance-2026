"""Offline tests for the LLM stage: guardrail, JSON parsing, caching, fallbacks, explanations.

No test touches the network: _call_llm is patched with canned replies, and
urllib is patched to fail loudly if anything tries to open a connection.
No test writes to the repository: the persistent cache points at a temp file.
"""

import io
import json
import os
import socket
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock as umock

from money_talks import mock as mock_data
from money_talks.ai import analyzer
from money_talks.contracts import ASSET_CLASSES, Constraints, Portfolio, Scenario


def _shocks(**overrides: float) -> dict[str, float]:
    shocks = {asset: 0.0 for asset in ASSET_CLASSES}
    shocks.update(overrides)
    return shocks


def _llm_reply(probabilities=(0.07, 0.05, 0.02)) -> str:
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
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache_file = Path(tmp.name) / "demo_cache.json"
        env = umock.patch.dict(
            os.environ,
            {
                analyzer.CACHE_ENV_VAR: str(self.cache_file),
                analyzer.BAD_LLM_ENV_VAR: "",
                analyzer.OFFLINE_ENV_VAR: "",
                "NICRRON_MODEL": "test-model",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        analyzer.set_llm_override()  # no per-session key or model left over
        self.addCleanup(analyzer.set_llm_override)
        self.addCleanup(analyzer.clear_cache)
        self.key = "test-model::oil jumps 40%"


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

    def test_scales_probabilities_to_the_fifteen_percent_cap(self) -> None:
        self.assertEqual(analyzer.MAX_TOTAL_LLM_PROBABILITY, 0.15)
        # The live run that prompted the lower cap: 12%, 25% and 8%.
        with umock.patch.object(analyzer, "_call_llm", return_value=_llm_reply((0.12, 0.25, 0.08))):
            scenarios = analyzer.generate_scenarios("Fed hikes 100bp")
        self.assertAlmostEqual(sum(s.probability for s in scenarios), 0.15)
        self.assertAlmostEqual(scenarios[1].probability / scenarios[0].probability, 0.25 / 0.12)
        self.assertIn("Scaled probability", scenarios[0].rationale)
        self.assertIn("limit 15.0%", scenarios[0].rationale)

    def test_probabilities_exactly_at_the_cap_are_not_rescaled(self) -> None:
        # The live demo answer: 0.07 + 0.05 + 0.03 == 0.15000000000000002 in floating point.
        with umock.patch.object(analyzer, "_call_llm", return_value=_llm_reply((0.07, 0.05, 0.03))):
            scenarios = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertEqual([s.probability for s in scenarios], [0.07, 0.05, 0.03])
        for scenario in scenarios:
            self.assertNotIn("Scaled probability", scenario.rationale)
            self.assertIn("no corrections needed", scenario.rationale)

    def test_prompt_states_the_probability_cap(self) -> None:
        with umock.patch.object(analyzer, "_call_llm", return_value=_llm_reply()) as call:
            analyzer.generate_scenarios("Fed hikes 100bp")
        user_prompt = call.call_args.args[1]
        self.assertIn("at most 0.15 in total", user_prompt)
        self.assertIn("Fed hikes 100bp", user_prompt)

    def test_accepts_code_fences_and_percentage_points(self) -> None:
        data = json.loads(_llm_reply())
        data["scenarios"][0]["shocks"] = {"us_equity": -35, "gov_bonds": 5, "cash": 1}
        data["scenarios"][0]["probability"] = 7
        reply = "Here you go:\n```json\n" + json.dumps(data) + "\n```"
        with umock.patch.object(analyzer, "_call_llm", return_value=reply):
            scenarios = analyzer.generate_scenarios("Recession")
        mild = scenarios[0]
        self.assertAlmostEqual(mild.shocks["us_equity"], -0.35)
        self.assertAlmostEqual(mild.shocks["gov_bonds"], 0.05)
        self.assertAlmostEqual(mild.probability, 0.07)
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

    def test_llm_is_called_with_exactly_two_prompts(self) -> None:
        # pipeline.py wraps _call_llm with telemetry.traced(system_prompt, user_prompt).
        with umock.patch.object(analyzer, "_call_llm", side_effect=TimeoutError()) as call:
            analyzer.explain_report(self.portfolio, self.report)
        self.assertEqual(len(call.call_args.args), 2)
        self.assertEqual(call.call_args.kwargs, {})
        with umock.patch.object(analyzer, "_call_llm", return_value=_llm_reply()) as call:
            analyzer.generate_scenarios("Oil jumps 40%")
        self.assertEqual(len(call.call_args.args), 2)
        self.assertEqual(call.call_args.kwargs, {})

    def test_timeouts_reach_the_network_through_the_prism_wrapper(self) -> None:
        from money_talks import telemetry

        inner = getattr(analyzer._call_llm, "__wrapped__", analyzer._call_llm)
        worst = f"{abs(self.report.worst_case_return):.1%}"
        good = f"In the worst case the portfolio would fall by {worst}."
        timeouts = []

        def fake_urlopen(request, timeout):
            timeouts.append(timeout)
            reply = good if b"stress-test results" in request.data else _llm_reply()
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": reply}}]}).encode())

        env = {"NICRRON_BASE_URL": "http://llm.test/v1", "NICRRON_API_KEY": "k"}
        with umock.patch.dict(os.environ, env), umock.patch.object(analyzer, "_load_env_file"), umock.patch(
            "urllib.request.urlopen", side_effect=fake_urlopen
        ), umock.patch.object(analyzer, "_call_llm", telemetry.traced(inner)):
            text = analyzer.explain_report(self.portfolio, self.report)
            scenarios = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertEqual(text, good)
        self.assertEqual(len(scenarios), 3)
        self.assertEqual(timeouts, [analyzer.EXPLAIN_TIMEOUT_SECONDS, analyzer.LLM_TIMEOUT_SECONDS])

    def test_failure_pauses_llm_explanations(self) -> None:
        proposal = mock_data.propose_allocation(
            self.portfolio, self.scenarios, mock_data.get_baseline_returns(), Constraints()
        )
        with umock.patch.object(analyzer, "_call_llm", side_effect=TimeoutError()) as call:
            analyzer.explain_report(self.portfolio, self.report)
            text = analyzer.explain_report(self.portfolio, self.report, proposal)  # new numbers
        self.assertEqual(call.call_count, 1)
        self.assertIn("re-allocation", text)

    def test_rejected_llm_text_is_not_requested_again(self) -> None:
        with umock.patch.object(analyzer, "_call_llm", return_value="It loses 99.9%.") as call:
            first = analyzer.explain_report(self.portfolio, self.report)
            second = analyzer.explain_report(self.portfolio, self.report)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(first, second)
        self.assertNotIn("99.9", second)


class DiskCacheTests(AnalyzerTestCase):
    def _generate(self, text: str = "Oil jumps 40%"):
        with umock.patch.object(analyzer, "_call_llm", return_value=_llm_reply()):
            return analyzer.generate_scenarios(text)

    def _reload(self) -> int:
        analyzer.clear_cache()
        return analyzer.load_disk_cache()

    def test_success_is_written_and_survives_a_restart(self) -> None:
        first = self._generate()
        data = json.loads(self.cache_file.read_text(encoding="utf-8"))
        self.assertEqual(data["version"], analyzer._CACHE_VERSION)
        entry = data["entries"][self.key]
        self.assertEqual(entry["text"], "Oil jumps 40%")
        self.assertEqual(len(entry["scenarios"]), 3)

        self.assertEqual(self._reload(), 1)
        with umock.patch.object(analyzer, "_call_llm", side_effect=AssertionError("cache missed")):
            again = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertEqual([s.name for s in again], [s.name for s in first])
        self.assertEqual([s.shocks for s in again], [s.shocks for s in first])
        self.assertEqual([s.rationale for s in again], [s.rationale for s in first])

    def test_cache_ignores_case_spacing_and_trailing_punctuation(self) -> None:
        self._generate("The AI stock bubble bursts and big tech falls by half")
        self._reload()
        with umock.patch.object(analyzer, "_call_llm", side_effect=AssertionError("cache missed")):
            hit = analyzer.generate_scenarios("  the AI stock bubble  bursts and big tech falls by half. ")
        self.assertEqual(len(hit), 3)

    def test_cached_events_work_offline(self) -> None:
        self._generate()
        self._reload()
        with umock.patch.dict(os.environ, {analyzer.OFFLINE_ENV_VAR: "1"}):
            scenarios = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertEqual(len(scenarios), 3)
        self.assertNotIn("FALLBACK", scenarios[0].name)

    def test_missing_or_corrupt_file_never_fails(self) -> None:
        self.assertEqual(self._reload(), 0)  # missing
        self.cache_file.write_text("{not json", encoding="utf-8")
        self.assertEqual(self._reload(), 0)
        self.assertEqual(len(self._generate()), 3)  # and it repairs the file
        self.assertIn(self.key, json.loads(self.cache_file.read_text())["entries"])

    def test_other_versions_and_stale_entries_are_ignored(self) -> None:
        self._generate()
        data = json.loads(self.cache_file.read_text())
        data["version"] = 1
        self.cache_file.write_text(json.dumps(data))
        self.assertEqual(self._reload(), 0)

        data["version"] = analyzer._CACHE_VERSION
        for scenario in data["entries"][self.key]["scenarios"]:
            scenario["probability"] = 0.20  # written under the old 50% cap
        self.cache_file.write_text(json.dumps(data))
        self.assertEqual(self._reload(), 0)

    def test_edited_entries_are_guarded_again_on_load(self) -> None:
        self._generate()
        data = json.loads(self.cache_file.read_text())
        data["entries"][self.key]["scenarios"][0]["shocks"]["us_equity"] = -5.0
        data["entries"][self.key]["scenarios"][0]["bogus_field"] = "x"
        self.cache_file.write_text(json.dumps(data))
        self.assertEqual(self._reload(), 1)
        with umock.patch.object(analyzer, "_call_llm", side_effect=AssertionError("cache missed")):
            scenario = analyzer.generate_scenarios("Oil jumps 40%")[0]
        self.assertEqual(scenario.shocks["us_equity"], analyzer.SHOCK_MIN)

    def test_fallbacks_are_not_written(self) -> None:
        with umock.patch.object(analyzer, "_call_llm", side_effect=TimeoutError()):
            analyzer.generate_scenarios("Oil jumps 40%")
        self.assertFalse(self.cache_file.exists())

    def test_unwritable_cache_location_never_fails(self) -> None:
        blocker = self.cache_file.parent / "a_file"
        blocker.write_text("x")
        with umock.patch.dict(os.environ, {analyzer.CACHE_ENV_VAR: str(blocker / "cache.json")}):
            self.assertEqual(len(self._generate()), 3)
        self.assertEqual(list(self.cache_file.parent.glob(".demo_cache.*")), [])

    def test_warm_demo_cache(self) -> None:
        with umock.patch.object(analyzer, "_call_llm", return_value=_llm_reply()):
            first = analyzer.warm_demo_cache()
            second = analyzer.warm_demo_cache()
        self.assertEqual(set(first.values()), {"generated"})
        self.assertEqual(set(second.values()), {"already cached"})
        entries = json.loads(self.cache_file.read_text())["entries"]
        self.assertEqual(len(entries), 2)
        # Warmed for the app's default model, not the host's NICRRON_MODEL.
        self.assertTrue(all(k.startswith(f"{analyzer.DEFAULT_MODEL}::") for k in entries))
        self.assertEqual({e["model"] for e in entries.values()}, {analyzer.DEFAULT_MODEL})
        self.assertEqual(analyzer.llm_status()["model"], "test-model")  # override restored

        with umock.patch.object(analyzer, "_call_llm", side_effect=TimeoutError("slow")):
            forced = analyzer.warm_demo_cache(force=True)
        self.assertTrue(all(s.startswith("failed") for s in forced.values()))
        analyzer.set_llm_override(model=analyzer.DEFAULT_MODEL)
        with umock.patch.object(analyzer, "_call_llm", side_effect=AssertionError("cache missed")):
            kept = analyzer.generate_scenarios(analyzer.DEMO_SENTENCES[0])
        self.assertNotIn("FALLBACK", kept[0].name)  # the old answer survives a failed refresh
        self.assertNotIn("Saved answer from", kept[0].rationale)  # an exact hit, not a backup

    def test_warm_demo_cache_for_a_chosen_model(self) -> None:
        with umock.patch.object(analyzer, "_call_llm", return_value=_llm_reply()):
            analyzer.warm_demo_cache(model="anthropic/claude-haiku-5-5")
        entries = json.loads(self.cache_file.read_text())["entries"]
        self.assertTrue(all(k.startswith("anthropic/claude-haiku-5-5::") for k in entries))


class ModelAwareCacheTests(AnalyzerTestCase):
    def test_each_model_keeps_its_own_answer(self) -> None:
        with umock.patch.object(analyzer, "_call_llm", return_value=_llm_reply()) as call:
            analyzer.set_llm_override(model="model-a")
            analyzer.generate_scenarios("Oil jumps 40%")
            analyzer.generate_scenarios("Oil jumps 40%")
            analyzer.set_llm_override(model="model-b")
            analyzer.generate_scenarios("Oil jumps 40%")
        self.assertEqual(call.call_count, 2)
        entries = json.loads(self.cache_file.read_text())["entries"]
        self.assertEqual(set(entries), {"model-a::oil jumps 40%", "model-b::oil jumps 40%"})

    def test_failed_model_falls_back_to_another_models_saved_answer(self) -> None:
        analyzer.set_llm_override(model="model-a")
        with umock.patch.object(analyzer, "_call_llm", return_value=_llm_reply()):
            saved = analyzer.generate_scenarios("Oil jumps 40%")
        analyzer.set_llm_override(model="model-b")
        with umock.patch.object(analyzer, "_call_llm", side_effect=TimeoutError("slow")):
            backup = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertEqual([s.name for s in backup], [s.name for s in saved])
        self.assertIn("Saved answer from model-a", backup[0].rationale)
        self.assertNotIn("FALLBACK", backup[0].name)
        self.assertNotIn("Saved answer", saved[0].rationale)  # the cached original is untouched

    def test_cache_entries_load_under_their_model(self) -> None:
        analyzer.set_llm_override(model="model-a")
        with umock.patch.object(analyzer, "_call_llm", return_value=_llm_reply()):
            analyzer.generate_scenarios("Oil jumps 40%")
        analyzer.clear_cache()
        self.assertEqual(analyzer.load_disk_cache(), 1)
        self.assertIn("model-a::oil jumps 40%", analyzer._scenario_cache)

    def test_session_override_and_status(self) -> None:
        with umock.patch.dict(os.environ, {"NICRRON_API_KEY": ""}), umock.patch.object(analyzer, "_load_env_file"):
            self.assertEqual(analyzer.llm_status(), {"key_source": "none", "model": "test-model"})
        analyzer.set_llm_override(api_key="secret-key", model="model-x")
        status = analyzer.llm_status()
        self.assertEqual(status, {"key_source": "session", "model": "model-x"})
        self.assertNotIn("secret-key", json.dumps(status))
        self.assertEqual(analyzer._llm_config()[1:], ("model-x", "secret-key"))
        analyzer.set_llm_override()
        self.assertEqual(analyzer.llm_status()["model"], "test-model")


class BadLLMDemoTests(AnalyzerTestCase):
    def test_bad_reply_shows_every_guardrail_correction(self) -> None:
        with umock.patch.dict(os.environ, {analyzer.BAD_LLM_ENV_VAR: "1"}), umock.patch.object(
            analyzer, "_call_llm", side_effect=AssertionError("must not call the LLM")
        ):
            scenarios = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertEqual(len(scenarios), 3)
        mild, base, severe = scenarios
        for scenario in scenarios:
            self.assertIn(analyzer._DEMO_MARKER, scenario.rationale)
            self.assertEqual(set(scenario.shocks), set(ASSET_CLASSES))
        self.assertIn("percentage points", mild.rationale)
        self.assertIn("Dropped unknown asset class 'bitcoin'", mild.rationale)
        self.assertIn("Filled missing asset class cash", mild.rationale)
        self.assertIn("Clamped probability from 90.0% to 50.0%", mild.rationale)
        self.assertIn("Clamped us_equity shock from -95.0% to -80.0%", severe.rationale)
        self.assertIn("Filled missing asset class gold", severe.rationale)
        self.assertAlmostEqual(mild.shocks["us_equity"], -0.12)
        self.assertAlmostEqual(sum(s.probability for s in scenarios), 0.15)

    def test_bad_reply_is_never_cached_and_overrides_the_cache(self) -> None:
        with umock.patch.object(analyzer, "_call_llm", return_value=_llm_reply()):
            analyzer.generate_scenarios("Oil jumps 40%")
        with umock.patch.dict(os.environ, {analyzer.BAD_LLM_ENV_VAR: "1"}):
            bad = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertIn(analyzer._DEMO_MARKER, bad[0].rationale)
        good = analyzer.generate_scenarios("Oil jumps 40%")  # switch off: cached real answer
        self.assertNotIn(analyzer._DEMO_MARKER, good[0].rationale)
        cached = json.loads(self.cache_file.read_text())["entries"][self.key]
        self.assertNotIn(analyzer._DEMO_MARKER, json.dumps(cached))


class LLMClientTests(AnalyzerTestCase):
    def setUp(self) -> None:
        super().setUp()
        env = umock.patch.dict(
            os.environ,
            {"NICRRON_BASE_URL": "http://llm.test/v1", "NICRRON_MODEL": "m", "NICRRON_API_KEY": "k"},
        )
        env.start()
        self.addCleanup(env.stop)
        for target in ("_load_env_file",):
            patcher = umock.patch.object(analyzer, target)
            patcher.start()
            self.addCleanup(patcher.stop)
        sleep = umock.patch.object(analyzer.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    @staticmethod
    def _response(content) -> io.BytesIO:
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

    @staticmethod
    def _http_error(code: int) -> urllib.error.HTTPError:
        return urllib.error.HTTPError("http://llm.test/v1/chat/completions", code, "err", None, None)

    def test_scenario_timeout_is_sixty_seconds(self) -> None:
        self.assertEqual(analyzer.LLM_TIMEOUT_SECONDS, 60)
        with umock.patch("urllib.request.urlopen", return_value=self._response("hi")) as urlopen:
            self.assertEqual(analyzer._call_llm("s", "u"), "hi")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 60)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://llm.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer k")

    def test_retries_once_on_transient_errors(self) -> None:
        for first_error in (self._http_error(503), self._http_error(429), urllib.error.URLError("refused")):
            with umock.patch(
                "urllib.request.urlopen", side_effect=[first_error, self._response("ok")]
            ) as urlopen:
                self.assertEqual(analyzer._call_llm("s", "u"), "ok")
            self.assertEqual(urlopen.call_count, 2)

    def test_does_not_retry_auth_errors_or_timeouts(self) -> None:
        for error in (
            self._http_error(401),
            urllib.error.URLError(socket.timeout("timed out")),
            TimeoutError("timed out"),
        ):
            with umock.patch("urllib.request.urlopen", side_effect=[error, self._response("ok")]) as urlopen:
                with self.assertRaises(Exception):
                    analyzer._call_llm("s", "u")
            self.assertEqual(urlopen.call_count, 1)

    def test_gives_up_after_one_retry(self) -> None:
        errors = [self._http_error(503), self._http_error(503), self._response("late")]
        with umock.patch("urllib.request.urlopen", side_effect=errors) as urlopen:
            with self.assertRaises(urllib.error.HTTPError):
                analyzer._call_llm("s", "u")
        self.assertEqual(urlopen.call_count, 2)

    def test_handles_content_parts_and_reasoning_blocks(self) -> None:
        parts = [{"type": "text", "text": "<think>let me see</think>"}, {"type": "text", "text": '{"a": 1}'}]
        with umock.patch("urllib.request.urlopen", return_value=self._response(parts)):
            self.assertEqual(analyzer._call_llm("s", "u"), '{"a": 1}')
        with umock.patch("urllib.request.urlopen", return_value=self._response("<think>x</think>  ")):
            with self.assertRaises(analyzer.LLMUnavailable):
                analyzer._call_llm("s", "u")

    def test_generate_survives_a_non_json_http_body(self) -> None:
        with umock.patch("urllib.request.urlopen", return_value=io.BytesIO(b"<html>502</html>")):
            scenarios = analyzer.generate_scenarios("Oil jumps 40%")
        self.assertIn("FALLBACK", scenarios[0].name)


if __name__ == "__main__":
    unittest.main()
