"""Mock data and stand-in logic so every role can start immediately.

Owned by the Lead (role 1). money_talks.pipeline falls back to these functions
while the real ones still raise NotImplementedError. Signatures here match the
real functions exactly.

All numbers are ILLUSTRATIVE placeholders, not calibrated market data.
"""

from money_talks.contracts import (
    ASSET_CLASSES,
    AllocationProposal,
    Constraints,
    Portfolio,
    Scenario,
    ScenarioResult,
    StressReport,
)


def get_baseline_returns() -> dict[str, float]:
    return {
        "us_equity": 0.07,
        "intl_equity": 0.065,
        "em_equity": 0.08,
        "gov_bonds": 0.035,
        "corp_credit": 0.05,
        "gold": 0.03,
        "commodities": 0.03,
        "real_estate": 0.055,
        "cash": 0.03,
    }


def get_preset_portfolios() -> dict[str, Portfolio]:
    return {
        "Balanced 60/40": Portfolio(
            weights={
                "us_equity": 0.35,
                "intl_equity": 0.15,
                "em_equity": 0.10,
                "gov_bonds": 0.25,
                "corp_credit": 0.10,
                "gold": 0.0,
                "commodities": 0.0,
                "real_estate": 0.0,
                "cash": 0.05,
            }
        ),
        "Conservative": Portfolio(
            weights={
                "us_equity": 0.15,
                "intl_equity": 0.05,
                "em_equity": 0.0,
                "gov_bonds": 0.40,
                "corp_credit": 0.15,
                "gold": 0.05,
                "commodities": 0.0,
                "real_estate": 0.05,
                "cash": 0.15,
            }
        ),
    }


def get_historical_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="Global financial crisis (2008-style)",
            description="Banking crisis, credit freeze, deep recession.",
            shocks={
                "us_equity": -0.40,
                "intl_equity": -0.43,
                "em_equity": -0.50,
                "gov_bonds": 0.12,
                "corp_credit": -0.10,
                "gold": 0.05,
                "commodities": -0.35,
                "real_estate": -0.38,
                "cash": 0.01,
            },
            probability=0.05,
            rationale="Illustrative placeholder values.",
        ),
        Scenario(
            name="Inflation and rate shock (2022-style)",
            description="Inflation surprise, fast central-bank tightening.",
            shocks={
                "us_equity": -0.19,
                "intl_equity": -0.16,
                "em_equity": -0.20,
                "gov_bonds": -0.13,
                "corp_credit": -0.15,
                "gold": 0.0,
                "commodities": 0.16,
                "real_estate": -0.25,
                "cash": 0.02,
            },
            probability=0.15,
            rationale="Illustrative placeholder values.",
        ),
        Scenario(
            name="Pandemic crash (2020-style)",
            description="Sudden global shutdown, sharp sell-off.",
            shocks={
                "us_equity": -0.34,
                "intl_equity": -0.33,
                "em_equity": -0.32,
                "gov_bonds": 0.05,
                "corp_credit": -0.12,
                "gold": -0.03,
                "commodities": -0.30,
                "real_estate": -0.40,
                "cash": 0.0,
            },
            probability=0.05,
            rationale="Illustrative placeholder values.",
        ),
    ]


def generate_scenarios(text: str) -> list[Scenario]:
    """Pretend-LLM: returns mild / base / severe variants of one canned shock."""
    base = get_historical_scenarios()[1]
    variants = (("mild", 0.5, 0.20), ("base", 1.0, 0.10), ("severe", 1.5, 0.03))
    return [
        Scenario(
            name=f"{text.strip()[:40] or 'Custom event'} ({label})",
            description=text,
            shocks=dict(base.shocks),
            probability=probability,
            severity=severity,
            source="llm",
            rationale="MOCK output - the real LLM is not connected yet.",
        )
        for label, severity, probability in variants
    ]


def run_stress_test(
    portfolio: Portfolio,
    scenarios: list[Scenario],
    baseline_returns: dict[str, float],
) -> StressReport:
    results = []
    for scenario in scenarios:
        shocks = scenario.effective_shocks()
        contribution = {a: portfolio.weights.get(a, 0.0) * shocks[a] for a in ASSET_CLASSES}
        total = sum(contribution.values())
        results.append(
            ScenarioResult(
                scenario_name=scenario.name,
                probability=scenario.probability,
                portfolio_return=total,
                pnl_value=total * portfolio.total_value,
                contribution_by_asset=contribution,
            )
        )
    baseline = sum(portfolio.weights.get(a, 0.0) * baseline_returns.get(a, 0.0) for a in ASSET_CLASSES)
    shock_probability = min(1.0, sum(r.probability for r in results))
    worst = min(results, key=lambda r: r.portfolio_return, default=None)
    return StressReport(
        results=results,
        baseline_return=baseline,
        baseline_probability=1.0 - shock_probability,
        expected_return=sum(r.probability * r.portfolio_return for r in results)
        + (1.0 - shock_probability) * baseline,
        probability_weighted_loss=sum(r.probability * max(0.0, -r.portfolio_return) for r in results),
        worst_case_scenario=worst.scenario_name if worst else "",
        worst_case_return=worst.portfolio_return if worst else 0.0,
    )


def propose_allocation(
    portfolio: Portfolio,
    scenarios: list[Scenario],
    baseline_returns: dict[str, float],
    constraints: Constraints,
) -> AllocationProposal:
    """Naive stand-in: move part of the largest holding into cash. NOT an optimizer."""
    weights = {a: portfolio.weights.get(a, 0.0) for a in ASSET_CLASSES}
    largest = max((a for a in ASSET_CLASSES if a != "cash"), key=lambda a: weights[a])
    shift = min(weights[largest], constraints.max_turnover, 0.10)
    new_weights = dict(weights)
    new_weights[largest] -= shift
    new_weights["cash"] += shift
    return AllocationProposal(
        new_weights=new_weights,
        trades={a: new_weights[a] - weights[a] for a in ASSET_CLASSES},
        before=run_stress_test(portfolio, scenarios, baseline_returns),
        after=run_stress_test(
            Portfolio(weights=new_weights, total_value=portfolio.total_value), scenarios, baseline_returns
        ),
        feasible=True,
        explanation="MOCK proposal - the real optimizer is not connected yet.",
    )


def explain_report(
    portfolio: Portfolio,
    report: StressReport,
    proposal: AllocationProposal | None = None,
) -> str:
    return (
        f"MOCK explanation. Worst case: {report.worst_case_scenario} "
        f"({report.worst_case_return:.1%}). Expected return: {report.expected_return:.1%}."
    )
