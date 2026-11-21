"""AI/ML analysis stage."""

from money_talks.contracts import AIAnalysis, Dataset


def analyze_dataset(dataset: Dataset) -> AIAnalysis:
    """Analyze the input dataset and return structured findings.

    Define the model/provider and output schema only after the task is known.
    """
    raise NotImplementedError("Implement the task-specific AI/ML analysis after the task is announced.")


# ---------------------------------------------------------------------------
# LLM scenario generation and explanation (ROLE 4). See TEAM_PLAN.md.
# Until these are implemented, money_talks.pipeline falls back to mock output.
# ---------------------------------------------------------------------------

from money_talks.contracts import (  # noqa: E402
    AllocationProposal,
    Portfolio,
    Scenario,
    StressReport,
)


def generate_scenarios(text: str) -> list[Scenario]:
    """Turn a plain-English event into mild / base / severe Scenario objects.

    The LLM proposes shocks and probabilities; validate_scenario decides
    what is accepted. Must never raise on LLM or network failure.
    """
    raise NotImplementedError("Role 4: implement LLM scenario generation.")


def validate_scenario(scenario: Scenario) -> tuple[Scenario, list[str]]:
    """Deterministic guardrail: return a cleaned scenario and a list of fixes made."""
    raise NotImplementedError("Role 4: implement the scenario guardrail.")


def explain_report(
    portfolio: Portfolio,
    report: StressReport,
    proposal: AllocationProposal | None = None,
) -> str:
    """Short plain-English explanation. Numbers come from the report, never the LLM."""
    raise NotImplementedError("Role 4: implement the explanation.")
