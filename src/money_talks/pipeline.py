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
    return _real_or_mock(analyzer.generate_scenarios, mock.generate_scenarios, text)


def stress_test(portfolio: Portfolio, scenarios: list[Scenario]) -> StressReport:
    return _real_or_mock(
        evaluator.run_stress_test, mock.run_stress_test, portfolio, scenarios, baseline_returns()
    )


def prepare(
    portfolio: Portfolio, scenarios: list[Scenario], constraints: Constraints
) -> AllocationProposal:
    return _real_or_mock(
        evaluator.propose_allocation,
        mock.propose_allocation,
        portfolio,
        scenarios,
        baseline_returns(),
        constraints,
    )


def explain(
    portfolio: Portfolio, report: StressReport, proposal: AllocationProposal | None = None
) -> str:
    return _real_or_mock(analyzer.explain_report, mock.explain_report, portfolio, report, proposal)
