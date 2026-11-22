"""Financial interpretation and evaluation stage."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

from scipy.optimize import linprog

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
    ASSET_CLASSES,
    ASSET_LABELS,
    AllocationProposal,
    Constraints,
    Portfolio,
    Scenario,
    ScenarioResult,
    StressReport,
)


_SUM_TOLERANCE = 1e-8
_SOLVER_TOLERANCE = 1e-9
_DEFAULT_MAX_WEIGHT = 0.40


def _portfolio_weights(portfolio: Portfolio) -> dict[str, float]:
    """Return validated weights in the frozen asset-class order."""
    unknown_assets = set(portfolio.weights).difference(ASSET_CLASSES)
    if unknown_assets:
        raise ValueError(f"Portfolio contains unknown asset classes: {sorted(unknown_assets)}")

    weights = {asset: float(portfolio.weights.get(asset, 0.0)) for asset in ASSET_CLASSES}
    if not all(isfinite(weight) for weight in weights.values()):
        raise ValueError("Portfolio weights must be finite numbers.")
    if any(weight < -_SUM_TOLERANCE for weight in weights.values()):
        raise ValueError("Portfolio weights cannot be negative.")
    if abs(sum(weights.values()) - 1.0) > _SUM_TOLERANCE:
        raise ValueError("Portfolio weights must sum to 1.0.")
    return weights


def _scenario_probability(scenarios: list[Scenario]) -> float:
    """Validate mutually exclusive probabilities and return their total."""
    probabilities = [float(scenario.probability) for scenario in scenarios]
    if not all(isfinite(probability) for probability in probabilities):
        raise ValueError("Scenario probabilities must be finite numbers.")
    if any(probability < 0.0 for probability in probabilities):
        raise ValueError("Scenario probabilities cannot be negative.")
    total = sum(probabilities)
    if total > 1.0 + _SUM_TOLERANCE:
        raise ValueError("Scenario probabilities must sum to 1.0 or less.")
    return min(total, 1.0)


def _returns(weights: dict[str, float], values: dict[str, float]) -> float:
    return sum(weights[asset] * float(values.get(asset, 0.0)) for asset in ASSET_CLASSES)


def _clean_weights(values: list[float], bounds: list[tuple[float, float | None]]) -> dict[str, float]:
    """Remove harmless HiGHS floating-point noise without moving a real trade."""
    cleaned = []
    for value, (lower, upper) in zip(values, bounds):
        value = 0.0 if abs(value) < _SOLVER_TOLERANCE else value
        if upper is not None and abs(value - upper) < _SOLVER_TOLERANCE:
            value = upper
        if abs(value - lower) < _SOLVER_TOLERANCE:
            value = lower
        cleaned.append(float(value))

    residual = 1.0 - sum(cleaned)
    if abs(residual) < _SOLVER_TOLERANCE:
        for index, (_, upper) in enumerate(bounds):
            if residual > 0.0 and (upper is None or cleaned[index] + residual <= upper + _SOLVER_TOLERANCE):
                cleaned[index] += residual
                break
            if residual < 0.0 and cleaned[index] + residual >= bounds[index][0] - _SOLVER_TOLERANCE:
                cleaned[index] += residual
                break
    return dict(zip(ASSET_CLASSES, cleaned))


def _weight_bounds(
    constraints: Constraints, current_weights: dict[str, float] | None = None
) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for asset in ASSET_CLASSES:
        lower = float(constraints.min_weight.get(asset, 0.0))
        if asset in constraints.max_weight:
            upper = float(constraints.max_weight[asset])
        else:
            current = float((current_weights or {}).get(asset, 0.0))
            upper = max(_DEFAULT_MAX_WEIGHT, current)
        if not isfinite(lower) or not isfinite(upper) or lower < 0.0 or upper < lower:
            raise ValueError(f"Invalid weight bounds for {asset}.")
        bounds.append((lower, upper))
    return bounds


def _solver_inputs(
    weights: dict[str, float], scenarios: list[Scenario], constraints: Constraints
) -> tuple[list[tuple[float, float | None]], list[list[float]], list[float], list[float]]:
    """Build shared LP constraints for weights and absolute-trade variables."""
    n_assets = len(ASSET_CLASSES)
    weight_bounds = _weight_bounds(constraints, weights)
    if not isfinite(constraints.max_turnover) or constraints.max_turnover < 0.0:
        raise ValueError("max_turnover must be a non-negative finite number.")
    if not isfinite(constraints.max_scenario_loss) or constraints.max_scenario_loss < 0.0:
        raise ValueError("max_scenario_loss must be a non-negative finite number.")

    bounds: list[tuple[float, float | None]] = weight_bounds + [(0.0, None)] * n_assets
    a_ub: list[list[float]] = []
    b_ub: list[float] = []

    # sum(t) <= 2 * max_turnover
    a_ub.append([0.0] * n_assets + [1.0] * n_assets)
    b_ub.append(2.0 * constraints.max_turnover)

    # x - t <= w and -x - t <= -w encode t >= |x - w|.
    for index, asset in enumerate(ASSET_CLASSES):
        positive = [0.0] * (2 * n_assets)
        negative = [0.0] * (2 * n_assets)
        positive[index] = 1.0
        positive[n_assets + index] = -1.0
        negative[index] = -1.0
        negative[n_assets + index] = -1.0
        a_ub.extend((positive, negative))
        b_ub.extend((weights[asset], -weights[asset]))

    return bounds, a_ub, b_ub, [1.0] * n_assets + [0.0] * n_assets


def run_stress_test(
    portfolio: Portfolio,
    scenarios: list[Scenario],
    baseline_returns: dict[str, float],
) -> StressReport:
    """Apply every scenario to the portfolio and aggregate by probability."""
    weights = _portfolio_weights(portfolio)
    total_probability = _scenario_probability(scenarios)

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        shocks = scenario.effective_shocks()
        contribution = {
            asset: weights[asset] * shocks[asset]
            for asset in ASSET_CLASSES
        }
        portfolio_return = sum(contribution.values())
        results.append(
            ScenarioResult(
                scenario_name=scenario.name,
                probability=scenario.probability,
                portfolio_return=portfolio_return,
                pnl_value=portfolio_return * portfolio.total_value,
                contribution_by_asset=contribution,
            )
        )

    baseline_return = _returns(weights, baseline_returns)
    baseline_probability = 1.0 - total_probability
    worst = min(results, key=lambda result: result.portfolio_return, default=None)
    return StressReport(
        results=results,
        baseline_return=baseline_return,
        baseline_probability=baseline_probability,
        expected_return=(
            sum(result.probability * result.portfolio_return for result in results)
            + baseline_probability * baseline_return
        ),
        probability_weighted_loss=sum(
            result.probability * max(0.0, -result.portfolio_return) for result in results
        ),
        worst_case_scenario=worst.scenario_name if worst else "",
        worst_case_return=worst.portfolio_return if worst else 0.0,
    )


def _build_proposal_explanation(
    trades: dict[str, float],
    after: StressReport,
    constraints: Constraints,
    feasible: bool,
    fallback_succeeded: bool,
) -> str:
    """Build a concise factual summary naming the largest trades and binding scenario."""
    if not feasible and not fallback_succeeded:
        return (
            "No allocation satisfies the remaining weight and turnover limits, so the "
            "current allocation is retained."
        )

    ranked_trades = sorted(
        ((asset, trade) for asset, trade in trades.items() if abs(trade) >= 0.0005),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    formatted = [
        f"{'buy' if trade > 0 else 'sell'} {ASSET_LABELS[asset]} ({trade * 100:+.1f} pts)"
        for asset, trade in ranked_trades[:2]
    ]
    if len(formatted) >= 2:
        trade_phrase = f"largest trades: {formatted[0]} and {formatted[1]}"
    elif len(formatted) == 1:
        trade_phrase = f"largest trade: {formatted[0]}"
    else:
        trade_phrase = "no trades required"

    if after.results:
        worst_name = after.worst_case_scenario
        worst_ret = after.worst_case_return
        if feasible and abs(worst_ret + constraints.max_scenario_loss) <= 1e-4:
            scenario_phrase = f"bound by {worst_name} ({worst_ret:.1%})"
        else:
            scenario_phrase = f"worst scenario is {worst_name} ({worst_ret:.1%})"
        detail = f"{trade_phrase}; {scenario_phrase}"
    else:
        detail = trade_phrase

    if feasible:
        return (
            f"Maximises probability-weighted expected return within the "
            f"{constraints.max_scenario_loss:.1%} loss cap ({detail})."
        )
    return (
        f"No allocation meets the {constraints.max_scenario_loss:.1%} loss cap within the "
        f"remaining limits; best-effort rebalance minimises worst-case loss ({detail})."
    )


def propose_allocation(
    portfolio: Portfolio,
    scenarios: list[Scenario],
    baseline_returns: dict[str, float],
    constraints: Constraints,
) -> AllocationProposal:
    """Find new weights that maximise expected return within the constraints."""
    weights = _portfolio_weights(portfolio)
    total_probability = _scenario_probability(scenarios)
    before = run_stress_test(portfolio, scenarios, baseline_returns)
    n_assets = len(ASSET_CLASSES)
    bounds, common_a_ub, common_b_ub, equality = _solver_inputs(weights, scenarios, constraints)

    expected_returns = [
        baseline_returns.get(asset, 0.0) * (1.0 - total_probability)
        + sum(
            scenario.probability * scenario.effective_shocks()[asset] for scenario in scenarios
        )
        for asset in ASSET_CLASSES
    ]
    objective = [-expected_return for expected_return in expected_returns] + [0.0] * n_assets
    a_ub = list(common_a_ub)
    b_ub = list(common_b_ub)
    for scenario in scenarios:
        a_ub.append([-scenario.effective_shocks()[asset] for asset in ASSET_CLASSES] + [0.0] * n_assets)
        b_ub.append(constraints.max_scenario_loss)

    solution = linprog(
        c=objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=[equality],
        b_eq=[1.0],
        bounds=bounds,
        method="highs",
    )

    feasible = solution.success
    fallback_succeeded = False
    if feasible:
        chosen = solution.x[:n_assets]
    else:
        # With an unattainable loss cap, retain all other limits and minimise the
        # worst scenario return.  z is the final variable and bounds the loss of
        # each scenario: -return_s <= z.
        fallback_bounds = bounds + [(None, None)]
        fallback_a_ub = [row + [0.0] for row in common_a_ub]
        fallback_b_ub = list(common_b_ub)
        for scenario in scenarios:
            fallback_a_ub.append(
                [-scenario.effective_shocks()[asset] for asset in ASSET_CLASSES]
                + [0.0] * n_assets
                + [-1.0]
            )
            fallback_b_ub.append(0.0)
        fallback = linprog(
            c=[0.0] * (2 * n_assets) + [1.0],
            A_ub=fallback_a_ub,
            b_ub=fallback_b_ub,
            A_eq=[[value for value in equality] + [0.0]],
            b_eq=[1.0],
            bounds=fallback_bounds,
            method="highs",
        )
        fallback_succeeded = bool(fallback.success)
        if fallback_succeeded:
            chosen = fallback.x[:n_assets]
        else:
            chosen = [weights[asset] for asset in ASSET_CLASSES]

    new_weights = _clean_weights(list(chosen), bounds[:n_assets])
    trades = {
        asset: 0.0 if abs(new_weights[asset] - weights[asset]) < _SOLVER_TOLERANCE
        else new_weights[asset] - weights[asset]
        for asset in ASSET_CLASSES
    }
    after = run_stress_test(
        Portfolio(weights=new_weights, total_value=portfolio.total_value), scenarios, baseline_returns
    )
    explanation = _build_proposal_explanation(
        trades=trades,
        after=after,
        constraints=constraints,
        feasible=feasible,
        fallback_succeeded=fallback_succeeded,
    )
    return AllocationProposal(
        new_weights=new_weights,
        trades=trades,
        before=before,
        after=after,
        feasible=feasible,
        explanation=explanation,
    )


def severity_sensitivity(
    portfolio: Portfolio,
    scenarios: list[Scenario],
    baseline_returns: dict[str, float],
    multipliers: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
) -> dict[float, float]:
    """Worst-case portfolio return as every scenario's severity is scaled by each multiplier.

    Does not mutate input Scenario objects.
    """
    _portfolio_weights(portfolio)
    _scenario_probability(scenarios)
    results: dict[float, float] = {}
    for mult in multipliers:
        factor = float(mult)
        if not isfinite(factor) or factor < 0.0:
            raise ValueError("Severity multipliers must be non-negative finite numbers.")
        scaled = [replace(scenario, severity=scenario.severity * factor) for scenario in scenarios]
        report = run_stress_test(portfolio, scaled, baseline_returns)
        results[factor] = report.worst_case_return
    return results
