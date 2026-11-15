"""AI/ML analysis stage."""

from money_talks.contracts import AIAnalysis, Dataset


def analyze_dataset(dataset: Dataset) -> AIAnalysis:
    """Analyze the input dataset and return structured findings.

    Define the model/provider and output schema only after the task is known.
    """
    raise NotImplementedError("Implement the task-specific AI/ML analysis after the task is announced.")
