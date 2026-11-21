"""Pipeline orchestration: data → AI/ML → financial evaluation."""

from money_talks.ai.analyzer import analyze_dataset
from money_talks.contracts import FinancialAssessment
from money_talks.data.source import load_dataset
from money_talks.finance.evaluator import evaluate_financially


def run_pipeline() -> FinancialAssessment:
    """Run the agreed stages in order and return the final assessment."""
    dataset = load_dataset()
    analysis = analyze_dataset(dataset)
    return evaluate_financially(dataset, analysis)


# ---------------------------------------------------------------------------
# Stress-test facade (ROLE 1 - Lead). See TEAM_PLAN.md.
# The UI imports ONLY from this module. Each function calls the real
# implementation and falls back to money_talks.mock while a role is unfinished,
# so the app works from minute one and upgrades itself as pull requests merge.
# ---------------------------------------------------------------------------

import dataclasses  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

from money_talks import mock  # noqa: E402
from money_talks.ai import analyzer  # noqa: E402
from money_talks.contracts import (  # noqa: E402
    AllocationProposal,
    Constraints,
    Portfolio,
    Scenario,
    StressReport,
)
from money_talks.data import source  # noqa: E402
from money_talks.finance import evaluator  # noqa: E402


def _real_or_mock(real, fallback, *args):
    try:
        return real(*args)
    except NotImplementedError:
        return fallback(*args)


def baseline_returns() -> dict[str, float]:
    return _real_or_mock(source.get_baseline_returns, mock.get_baseline_returns)


def historical_scenarios() -> list[Scenario]:
    """The full scenario library: historical crises plus forward-looking scenarios."""
    return _real_or_mock(source.get_scenario_library, mock.get_historical_scenarios)


def preset_portfolios() -> dict[str, Portfolio]:
    return _real_or_mock(source.get_preset_portfolios, mock.get_preset_portfolios)


def scenarios_from_text(text: str) -> list[Scenario]:
    scenarios = _real_or_mock(analyzer.generate_scenarios, mock.generate_scenarios, text)
    for scenario in scenarios:
        proposal, _, guardrail = scenario.rationale.partition("Guardrail:")
        _audit(
            "llm_scenario",
            event_text=text,
            scenario=scenario.name,
            source=scenario.source,
            probability=scenario.probability,
            shocks=scenario.shocks,
            model_rationale=proposal.strip(),
            guardrail_decision=guardrail.strip() or "not recorded",
            llm_available="FALLBACK" not in scenario.name,
        )
    return scenarios


def stress_test(portfolio: Portfolio, scenarios: list[Scenario]) -> StressReport:
    return _real_or_mock(
        evaluator.run_stress_test, mock.run_stress_test, portfolio, scenarios, baseline_returns()
    )


def prepare(
    portfolio: Portfolio, scenarios: list[Scenario], constraints: Constraints
) -> AllocationProposal:
    proposal = _real_or_mock(
        evaluator.propose_allocation,
        mock.propose_allocation,
        portfolio,
        scenarios,
        baseline_returns(),
        constraints,
    )
    _audit(
        "allocation_decision",
        feasible=proposal.feasible,
        max_scenario_loss=constraints.max_scenario_loss,
        max_turnover=constraints.max_turnover,
        worst_case_before=proposal.before.worst_case_return,
        worst_case_after=proposal.after.worst_case_return,
        expected_return_before=proposal.before.expected_return,
        expected_return_after=proposal.after.expected_return,
        new_weights=proposal.new_weights,
    )
    return proposal


def explain(
    portfolio: Portfolio, report: StressReport, proposal: AllocationProposal | None = None
) -> str:
    return _real_or_mock(analyzer.explain_report, mock.explain_report, portfolio, report, proposal)


# ---------------------------------------------------------------------------
# Probability safety net: the engine rejects totals above 100%.
# ---------------------------------------------------------------------------

MAX_TOTAL_PROBABILITY = 0.95


def safe_scenarios(scenarios: list[Scenario]) -> tuple[list[Scenario], str | None]:
    """Return scenarios whose probabilities fit under MAX_TOTAL_PROBABILITY.

    If the total is too high, every probability is scaled down by the same
    factor and a message for the user is returned. Inputs are not modified.
    """
    total = sum(max(0.0, s.probability) for s in scenarios)
    if total <= MAX_TOTAL_PROBABILITY:
        return list(scenarios), None
    factor = MAX_TOTAL_PROBABILITY / total
    scaled = [
        dataclasses.replace(s, probability=max(0.0, s.probability) * factor) for s in scenarios
    ]
    message = (
        f"The selected scenarios had probabilities adding up to {total:.0%}. Scenarios are "
        f"treated as mutually exclusive, so they were scaled down proportionally to "
        f"{MAX_TOTAL_PROBABILITY:.0%} in total."
    )
    return scaled, message


# ---------------------------------------------------------------------------
# Local audit trail: what the model proposed and what the code decided.
# One JSON object per line. Set MONEY_TALKS_AUDIT_LOG to change the location.
# Writing never raises: the app must work even if the file is not writable.
# ---------------------------------------------------------------------------


def _audit_path() -> Path:
    override = os.environ.get("MONEY_TALKS_AUDIT_LOG", "").strip()
    return Path(override) if override else Path(__file__).resolve().parents[2] / "audit_log.jsonl"


def _audit(event: str, **data) -> None:
    record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **data}
    try:
        with _audit_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


def audit_tail(n: int = 20) -> list[dict]:
    """The most recent audit records, newest last. Empty if there are none."""
    try:
        lines = _audit_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records = []
    for line in lines[-n:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
