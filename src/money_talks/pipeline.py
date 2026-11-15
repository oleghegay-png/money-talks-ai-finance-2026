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
