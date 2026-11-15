"""Smoke tests for the shared module contracts."""

import unittest

from money_talks.ai.analyzer import analyze_dataset
from money_talks.contracts import AIAnalysis, Dataset, FinancialAssessment
from money_talks.data.source import load_dataset
from money_talks.finance.evaluator import evaluate_financially
from money_talks.pipeline import run_pipeline


class ScaffoldTests(unittest.TestCase):
    def test_shared_contracts_can_be_created(self) -> None:
        dataset = Dataset(records=[{"id": "example-1"}], metadata={"source": "demo"})
        analysis = AIAnalysis(summary="example", findings=[{"label": "demo"}])
        assessment = FinancialAssessment(summary="example", metrics={"example_metric": 1.0})

        self.assertEqual(dataset.records[0]["id"], "example-1")
        self.assertEqual(analysis.findings[0]["label"], "demo")
        self.assertEqual(assessment.metrics["example_metric"], 1.0)

    def test_stage_functions_are_explicit_placeholders(self) -> None:
        dataset = Dataset()
        analysis = AIAnalysis()

        with self.assertRaises(NotImplementedError):
            load_dataset()
        with self.assertRaises(NotImplementedError):
            analyze_dataset(dataset)
        with self.assertRaises(NotImplementedError):
            evaluate_financially(dataset, analysis)
        with self.assertRaises(NotImplementedError):
            run_pipeline()


if __name__ == "__main__":
    unittest.main()
