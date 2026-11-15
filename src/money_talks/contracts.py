"""Small, shared interfaces between the team's modules.

Keep these structures task-neutral until the challenge prompt is known.
Agree on any changes in the team before changing function signatures.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Dataset:
    """Input data plus source/context metadata."""

    records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AIAnalysis:
    """Structured output from an AI/ML analysis stage."""

    summary: str = ""
    findings: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FinancialAssessment:
    """Financial interpretation, metrics, and caveats."""

    summary: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
