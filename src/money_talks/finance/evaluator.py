"""Financial interpretation and evaluation stage."""

from money_talks.contracts import AIAnalysis, Dataset, FinancialAssessment


def evaluate_financially(dataset: Dataset, analysis: AIAnalysis) -> FinancialAssessment:
    """Evaluate the analytical output using task-appropriate financial logic.

    Select metrics, assumptions, baselines, and caveats only after the prompt
    and scoring criteria are available. Do not invent financial significance.
    """
    raise NotImplementedError("Implement financial evaluation after the task and success criteria are known.")


# ---------------------------------------------------------------------------
# Stress engine and optimizer (ROLE 3). See TEAM_PLAN.md.
# Until these are implemented, money_talks.pipeline falls back to mock logic.
# ---------------------------------------------------------------------------

from money_talks.contracts import (  # noqa: E402
    AllocationProposal,
    Constraints,
    Portfolio,
    Scenario,
    StressReport,
)


def run_stress_test(
    portfolio: Portfolio,
    scenarios: list[Scenario],
    baseline_returns: dict[str, float],
) -> StressReport:
    """Apply every scenario to the portfolio and aggregate by probability."""
    raise NotImplementedError("Role 3: implement the stress engine.")


def propose_allocation(
    portfolio: Portfolio,
    scenarios: list[Scenario],
    baseline_returns: dict[str, float],
    constraints: Constraints,
) -> AllocationProposal:
    """Find new weights that maximise expected return within the constraints."""
    raise NotImplementedError("Role 3: implement the allocation optimizer.")
