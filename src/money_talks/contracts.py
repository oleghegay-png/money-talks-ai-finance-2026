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


# ---------------------------------------------------------------------------
# Stress-test contracts (FROZEN - see TEAM_PLAN.md).
# Only the Lead (role 1) edits this section. Ask before changing anything.
# All returns, shocks, weights and probabilities are decimals: -0.35 = -35%.
# ---------------------------------------------------------------------------

ASSET_CLASSES: tuple[str, ...] = (
    "us_equity",
    "intl_equity",
    "em_equity",
    "gov_bonds",
    "corp_credit",
    "gold",
    "commodities",
    "real_estate",
    "cash",
)

ASSET_LABELS: dict[str, str] = {
    "us_equity": "US Equity",
    "intl_equity": "International Equity",
    "em_equity": "Emerging Markets Equity",
    "gov_bonds": "Government Bonds",
    "corp_credit": "Corporate Credit",
    "gold": "Gold",
    "commodities": "Commodities",
    "real_estate": "Real Estate",
    "cash": "Cash",
}


@dataclass
class Portfolio:
    """Weights per asset class (must sum to 1.0) and the portfolio value."""

    weights: dict[str, float] = field(default_factory=dict)
    total_value: float = 1_000_000.0


@dataclass
class Scenario:
    """One shock event.

    shocks       base-case return per asset class over the horizon (12 months).
    probability  chance the event happens within the horizon, 0..1.
    severity     multiplier on shocks: 0.5 = mild, 1.0 = base, 1.5 = severe.
    source       "historical", "llm" or "custom".
    rationale    why these numbers (LLM or analyst explanation).
    """

    name: str = ""
    description: str = ""
    shocks: dict[str, float] = field(default_factory=dict)
    probability: float = 0.10
    severity: float = 1.0
    source: str = "historical"
    rationale: str = ""

    def effective_shocks(self) -> dict[str, float]:
        """Shocks scaled by severity, floored at -100%, for every asset class."""
        return {
            asset: max(-1.0, self.shocks.get(asset, 0.0) * self.severity)
            for asset in ASSET_CLASSES
        }


@dataclass
class ScenarioResult:
    """What one scenario does to one portfolio."""

    scenario_name: str = ""
    probability: float = 0.0
    portfolio_return: float = 0.0  # e.g. -0.22
    pnl_value: float = 0.0  # portfolio_return * total_value (negative = loss)
    contribution_by_asset: dict[str, float] = field(default_factory=dict)  # weight * shock


@dataclass
class StressReport:
    """All scenarios plus the no-shock baseline, for one portfolio."""

    results: list[ScenarioResult] = field(default_factory=list)
    baseline_return: float = 0.0  # return if no scenario happens
    baseline_probability: float = 1.0  # 1 - sum(scenario probabilities)
    expected_return: float = 0.0  # probability-weighted, baseline included
    probability_weighted_loss: float = 0.0  # sum(p * max(0, -return)) over scenarios
    worst_case_scenario: str = ""
    worst_case_return: float = 0.0


@dataclass
class Constraints:
    """Limits the optimizer must respect."""

    max_turnover: float = 0.30  # sum(|new - old|) / 2
    max_scenario_loss: float = 0.15  # no scenario may lose more than this
    min_weight: dict[str, float] = field(default_factory=dict)  # default 0.0
    max_weight: dict[str, float] = field(default_factory=dict)  # default 0.60


@dataclass
class AllocationProposal:
    """Suggested re-allocation with before/after evidence."""

    new_weights: dict[str, float] = field(default_factory=dict)
    trades: dict[str, float] = field(default_factory=dict)  # new - old, per asset
    before: StressReport = field(default_factory=StressReport)
    after: StressReport = field(default_factory=StressReport)
    feasible: bool = True  # False = constraints could not all be met
    explanation: str = ""
