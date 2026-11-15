"""Financial interpretation and evaluation stage."""

from money_talks.contracts import AIAnalysis, Dataset, FinancialAssessment


def evaluate_financially(dataset: Dataset, analysis: AIAnalysis) -> FinancialAssessment:
    """Evaluate the analytical output using task-appropriate financial logic.

    Select metrics, assumptions, baselines, and caveats only after the prompt
    and scoring criteria are available. Do not invent financial significance.
    """
    raise NotImplementedError("Implement financial evaluation after the task and success criteria are known.")
