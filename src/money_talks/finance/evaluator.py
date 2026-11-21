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
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

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

_WEIGHT_SUM_TOL = 1e-4
_PROB_SUM_TOL = 1e-9
_NOISE_EPS = 1e-7
_DEFAULT_MAX_WEIGHT = 0.60


def _extract_and_validate_weights(portfolio: Portfolio) -> dict[str, float]:
    """Validate portfolio weights and return a complete 9-asset dict."""
    unknown = [
        asset
        for asset, val in portfolio.weights.items()
        if asset not in ASSET_CLASSES and abs(float(val)) > 1e-9
    ]
    if unknown:
        raise ValueError(f"Unknown asset classes in portfolio: {', '.join(sorted(unknown))}.")

    weights = {asset: float(portfolio.weights.get(asset, 0.0)) for asset in ASSET_CLASSES}
    negatives = [asset for asset, val in weights.items() if val < -1e-9]
    if negatives:
        raise ValueError(f"Portfolio weights cannot be negative: {', '.join(negatives)}.")

    weights = {asset: (0.0 if val < 0.0 else val) for asset, val in weights.items()}
    total = sum(weights.values())
    if abs(total - 1.0) > _WEIGHT_SUM_TOL:
        raise ValueError(f"Portfolio weights must sum to 1.0 (got {total:.4f}).")

    if total > 0.0 and abs(total - 1.0) <= 1e-6:
        weights = {asset: val / total for asset, val in weights.items()}
    return weights


def _validate_scenario_probabilities(scenarios: list[Scenario]) -> float:
    """Validate individual and cumulative scenario probabilities."""
    total_prob = 0.0
    for scenario in scenarios:
        prob = float(scenario.probability)
        if prob < -_PROB_SUM_TOL or prob > 1.0 + _PROB_SUM_TOL:
            raise ValueError(
                f"Scenario '{scenario.name}' has invalid probability {prob:.4f}; must be in [0, 1]."
            )
        total_prob += max(0.0, prob)

    if total_prob > 1.0 + _PROB_SUM_TOL:
        raise ValueError(
            f"Scenario probabilities sum to {total_prob:.4f}, which exceeds 1.0."
        )
    return min(1.0, total_prob)


def _clean_solver_weights(
    raw_x: np.ndarray,
    current_w: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    """Round tiny solver noise to zero while preserving bounds and sum == 1.0."""
    cleaned = np.clip(np.asarray(raw_x, dtype=float), lower, upper)

    for idx in range(len(ASSET_CLASSES)):
        if abs(cleaned[idx]) < _NOISE_EPS and lower[idx] <= 0.0 <= upper[idx]:
            cleaned[idx] = 0.0
        elif abs(cleaned[idx] - current_w[idx]) < _NOISE_EPS and lower[idx] <= current_w[idx] <= upper[idx]:
            cleaned[idx] = current_w[idx]

    diff = 1.0 - float(np.sum(cleaned))
    if abs(diff) > 1e-12:
        order = np.argsort(-cleaned)
        for idx in order:
            if diff > 0.0:
                room = upper[idx] - cleaned[idx]
                delta = min(diff, room)
            else:
                room = cleaned[idx] - lower[idx]
                delta = -min(-diff, room)
            if abs(delta) > 0.0:
                cleaned[idx] += delta
                diff -= delta
            if abs(diff) <= 1e-12:
                break

    cleaned = np.clip(cleaned, lower, upper)
    total = float(np.sum(cleaned))
    if total > 0.0 and abs(total - 1.0) > 1e-12:
        cleaned = cleaned / total

    return {asset: float(cleaned[idx]) for idx, asset in enumerate(ASSET_CLASSES)}


def run_stress_test(
    portfolio: Portfolio,
    scenarios: list[Scenario],
    baseline_returns: dict[str, float],
) -> StressReport:
    """Apply every scenario to the portfolio and aggregate by probability."""
    weights = _extract_and_validate_weights(portfolio)
    total_prob = _validate_scenario_probabilities(scenarios)
    total_value = float(portfolio.total_value)

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        shocks = scenario.effective_shocks()
        contribution = {asset: weights[asset] * float(shocks[asset]) for asset in ASSET_CLASSES}
        portfolio_return = sum(contribution.values())
        results.append(
            ScenarioResult(
                scenario_name=scenario.name,
                probability=max(0.0, float(scenario.probability)),
                portfolio_return=portfolio_return,
                pnl_value=portfolio_return * total_value,
                contribution_by_asset=contribution,
            )
        )

    baseline_return = sum(
        weights[asset] * float(baseline_returns.get(asset, 0.0)) for asset in ASSET_CLASSES
    )
    baseline_probability = max(0.0, 1.0 - total_prob)
    expected_return = (
        sum(r.probability * r.portfolio_return for r in results)
        + baseline_probability * baseline_return
    )
    probability_weighted_loss = sum(
        r.probability * max(0.0, -r.portfolio_return) for r in results
    )
    worst = min(results, key=lambda r: r.portfolio_return, default=None)

    return StressReport(
        results=results,
        baseline_return=baseline_return,
        baseline_probability=baseline_probability,
        expected_return=expected_return,
        probability_weighted_loss=probability_weighted_loss,
        worst_case_scenario=worst.scenario_name if worst is not None else "",
        worst_case_return=worst.portfolio_return if worst is not None else 0.0,
    )


def propose_allocation(
    portfolio: Portfolio,
    scenarios: list[Scenario],
    baseline_returns: dict[str, float],
    constraints: Constraints,
) -> AllocationProposal:
    """Find new weights that maximise expected return within the constraints."""
    weights = _extract_and_validate_weights(portfolio)
    total_prob = _validate_scenario_probabilities(scenarios)

    max_turnover = float(constraints.max_turnover)
    max_scenario_loss = float(constraints.max_scenario_loss)
    if max_turnover < -1e-9:
        raise ValueError(f"max_turnover cannot be negative (got {max_turnover:.4f}).")
    if max_scenario_loss < -1e-9:
        raise ValueError(f"max_scenario_loss cannot be negative (got {max_scenario_loss:.4f}).")

    n_assets = len(ASSET_CLASSES)
    lower = np.array(
        [float(constraints.min_weight.get(asset, 0.0)) for asset in ASSET_CLASSES],
        dtype=float,
    )
    upper = np.array(
        [float(constraints.max_weight.get(asset, _DEFAULT_MAX_WEIGHT)) for asset in ASSET_CLASSES],
        dtype=float,
    )
    if np.any(lower < -1e-9) or np.any(upper > 1.0 + 1e-9) or np.any(lower > upper + 1e-9):
        raise ValueError("Per-asset weight bounds must satisfy 0.0 <= min_weight <= max_weight <= 1.0.")
    if float(np.sum(lower)) > 1.0 + 1e-6 or float(np.sum(upper)) < 1.0 - 1e-6:
        raise ValueError("Per-asset min/max weight bounds cannot sum to a feasible portfolio of 1.0.")

    before = run_stress_test(portfolio, scenarios, baseline_returns)

    baseline_prob = max(0.0, 1.0 - total_prob)
    base_vec = np.array([float(baseline_returns.get(asset, 0.0)) for asset in ASSET_CLASSES], dtype=float)
    shock_matrix = (
        np.array(
            [[float(eff[asset]) for asset in ASSET_CLASSES] for eff in (s.effective_shocks() for s in scenarios)],
            dtype=float,
        )
        if scenarios
        else np.zeros((0, n_assets), dtype=float)
    )
    prob_vec = np.array([max(0.0, float(s.probability)) for s in scenarios], dtype=float)

    mu = baseline_prob * base_vec
    if len(scenarios) > 0:
        mu = mu + prob_vec @ shock_matrix

    current_w = np.array([weights[asset] for asset in ASSET_CLASSES], dtype=float)

    # Primary LP decision variables: [x_0..x_{N-1}, t_0..t_{N-1}]
    n_primary = 2 * n_assets
    c_primary = np.concatenate([-mu, np.full(n_assets, 1e-9, dtype=float)])

    a_eq_primary = np.zeros((1, n_primary), dtype=float)
    a_eq_primary[0, :n_assets] = 1.0
    b_eq = np.array([1.0], dtype=float)

    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []

    # Absolute turnover helper constraints: x_a - t_a <= w_a and -x_a - t_a <= -w_a
    for idx in range(n_assets):
        row_pos = np.zeros(n_primary, dtype=float)
        row_pos[idx] = 1.0
        row_pos[n_assets + idx] = -1.0
        ub_rows.append(row_pos)
        ub_rhs.append(current_w[idx])

        row_neg = np.zeros(n_primary, dtype=float)
        row_neg[idx] = -1.0
        row_neg[n_assets + idx] = -1.0
        ub_rows.append(row_neg)
        ub_rhs.append(-current_w[idx])

    # Total turnover cap: sum(t_a) <= 2 * max_turnover
    turnover_row = np.zeros(n_primary, dtype=float)
    turnover_row[n_assets:] = 1.0
    ub_rows.append(turnover_row)
    ub_rhs.append(2.0 * max(0.0, max_turnover))

    # Scenario loss cap: -shock_s . x <= max_scenario_loss
    for s_idx in range(len(scenarios)):
        s_row = np.zeros(n_primary, dtype=float)
        s_row[:n_assets] = -shock_matrix[s_idx]
        ub_rows.append(s_row)
        ub_rhs.append(max(0.0, max_scenario_loss))

    bounds_primary = [(float(lower[i]), float(upper[i])) for i in range(n_assets)] + [
        (0.0, None) for _ in range(n_assets)
    ]

    res = linprog(
        c_primary,
        A_ub=np.vstack(ub_rows),
        b_ub=np.array(ub_rhs, dtype=float),
        A_eq=a_eq_primary,
        b_eq=b_eq,
        bounds=bounds_primary,
        method="highs",
    )

    feasible = bool(res.success and res.x is not None)
    raw_x: np.ndarray | None = res.x[:n_assets] if feasible and res.x is not None else None

    if not feasible:
        # Secondary LP: [x_0..x_{N-1}, t_0..t_{N-1}, L] minimizing worst-case loss L
        n_secondary = n_primary + 1
        c_sec = np.concatenate([-1e-6 * mu, np.full(n_assets, 1e-9, dtype=float), [1.0]])
        a_eq_sec = np.zeros((1, n_secondary), dtype=float)
        a_eq_sec[0, :n_assets] = 1.0

        sec_rows: list[np.ndarray] = []
        sec_rhs: list[float] = []
        for idx in range(2 * n_assets + 1):
            padded = np.zeros(n_secondary, dtype=float)
            padded[:n_primary] = ub_rows[idx]
            sec_rows.append(padded)
            sec_rhs.append(ub_rhs[idx])

        for s_idx in range(len(scenarios)):
            s_row = np.zeros(n_secondary, dtype=float)
            s_row[:n_assets] = -shock_matrix[s_idx]
            s_row[n_primary] = -1.0
            sec_rows.append(s_row)
            sec_rhs.append(0.0)

        bounds_sec = bounds_primary + [(None, None)]
        res_sec = linprog(
            c_sec,
            A_ub=np.vstack(sec_rows),
            b_ub=np.array(sec_rhs, dtype=float),
            A_eq=a_eq_sec,
            b_eq=b_eq,
            bounds=bounds_sec,
            method="highs",
        )
        if res_sec.success and res_sec.x is not None:
            raw_x = res_sec.x[:n_assets]

    if raw_x is not None:
        new_weights = _clean_solver_weights(raw_x, current_w, lower, upper)
        if feasible and len(scenarios) > 0:
            worst_clean = min(
                sum(new_weights[asset] * float(eff[asset]) for asset in ASSET_CLASSES)
                for eff in (s.effective_shocks() for s in scenarios)
            )
            turnover_clean = sum(abs(new_weights[asset] - weights[asset]) for asset in ASSET_CLASSES) / 2.0
            if worst_clean < -max_scenario_loss - 1e-6 or turnover_clean > max_turnover + 1e-6:
                clipped = np.clip(raw_x, lower, upper)
                new_weights = {asset: float(clipped[i]) for i, asset in enumerate(ASSET_CLASSES)}
    else:
        new_weights = dict(weights)

    trades = {
        asset: (0.0 if abs(new_weights[asset] - weights[asset]) < _NOISE_EPS else new_weights[asset] - weights[asset])
        for asset in ASSET_CLASSES
    }
    after = run_stress_test(
        Portfolio(weights=new_weights, total_value=portfolio.total_value),
        scenarios,
        baseline_returns,
    )
    actual_turnover = sum(abs(trades[asset]) for asset in ASSET_CLASSES) / 2.0

    if raw_x is None:
        explanation = (
            "No feasible allocation exists within the specified weight and turnover limits, "
            "so current weights are retained."
        )
    elif feasible:
        explanation = (
            f"Rebalancing within {actual_turnover:.1%} turnover caps worst-case loss at "
            f"{after.worst_case_return:.1%} (from {before.worst_case_return:.1%}) with an "
            f"expected 12-month return of {after.expected_return:.1%}."
        )
    else:
        top_add = max(ASSET_CLASSES, key=lambda asset: trades[asset])
        explanation = (
            f"The {max_scenario_loss:.1%} loss cap is infeasible within {max_turnover:.1%} turnover; "
            f"the best-effort shift toward {ASSET_LABELS[top_add]} narrows worst-case return from "
            f"{before.worst_case_return:.1%} to {after.worst_case_return:.1%}."
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
    severities: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
) -> list[dict[str, float]]:
    """Stretch helper: evaluate worst-case and expected return across severity multipliers."""
    rows: list[dict[str, float]] = []
    for mult in severities:
        scaled = [
            Scenario(
                name=s.name,
                description=s.description,
                shocks=dict(s.shocks),
                probability=s.probability,
                severity=float(mult),
                source=s.source,
                rationale=s.rationale,
            )
            for s in scenarios
        ]
        report = run_stress_test(portfolio, scaled, baseline_returns)
        rows.append(
            {
                "severity": float(mult),
                "worst_case_return": report.worst_case_return,
                "expected_return": report.expected_return,
                "probability_weighted_loss": report.probability_weighted_loss,
            }
        )
    return rows


def efficient_tradeoff_curve(
    portfolio: Portfolio,
    scenarios: list[Scenario],
    baseline_returns: dict[str, float],
    constraints: Constraints,
    loss_caps: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
) -> list[dict[str, float | bool]]:
    """Stretch helper: sweep max_scenario_loss to map expected return vs. protection."""
    curve: list[dict[str, float | bool]] = []
    for cap in loss_caps:
        step_constraints = Constraints(
            max_turnover=constraints.max_turnover,
            max_scenario_loss=float(cap),
            min_weight=dict(constraints.min_weight),
            max_weight=dict(constraints.max_weight),
        )
        proposal = propose_allocation(portfolio, scenarios, baseline_returns, step_constraints)
        curve.append(
            {
                "max_scenario_loss": float(cap),
                "feasible": proposal.feasible,
                "expected_return": proposal.after.expected_return,
                "worst_case_return": proposal.after.worst_case_return,
            }
        )
    return curve
