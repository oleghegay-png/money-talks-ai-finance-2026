"""Streamlit UI for the Portfolio Stress Test and Allocation Tool (Role 5).

Imports ONLY from money_talks.pipeline and money_talks.contracts so the app
runs immediately on mock data and upgrades automatically as Roles 1-4 merge.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

# Ensure `src/` is importable when invoked as `streamlit run app.py`
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from money_talks import pipeline  # noqa: E402
from money_talks.contracts import (  # noqa: E402
    ASSET_CLASSES,
    ASSET_LABELS,
    AllocationProposal,
    Constraints,
    Portfolio,
    Scenario,
    StressReport,
)

DISCLAIMER = "Illustrative model, not investment advice."
_WEIGHT_TOLERANCE_PCT = 0.05

ASSET_COLORS: dict[str, str] = {
    "us_equity": "#1B4965",
    "intl_equity": "#2D6A4F",
    "em_equity": "#5FA8D3",
    "gov_bonds": "#40916C",
    "corp_credit": "#74C69D",
    "gold": "#E9C46A",
    "commodities": "#F4A261",
    "real_estate": "#E76F51",
    "cash": "#8D99AE",
}

_GUARDRAIL_LINE_RE = re.compile(
    r"^(?:guardrail(?:\s+(?:fixes|changes|adjustments|notes))?|fixes|adjustments)\s*:\s*(.*)$",
    re.IGNORECASE,
)
_GUARDRAIL_TAG_RE = re.compile(r"^\[guardrail[^\]]*\]\s*(.*)$", re.IGNORECASE)
_GUARDRAIL_INLINE_RE = re.compile(
    r"(?:guardrail(?:\s+(?:fixes|changes|adjustments|notes))?|fixes)\s*:\s*(.+)$",
    re.IGNORECASE,
)
_GUARDRAIL_KEYWORD_RE = re.compile(
    r"\b(?:clamped|filled missing|dropped unknown|scaled|fallback|unavailable)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Pure helper functions (UI-independent for clean unit testing)
# ---------------------------------------------------------------------------


def _key_slug(text: str) -> str:
    """Return a deterministic widget-safe slug with a short hash suffix."""
    prefix = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:24] or "item"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp numeric value into [low, high]."""
    return max(low, min(high, float(value)))


def _fmt_pct(value: float, signed: bool = False, decimals: int = 1) -> str:
    """Format a decimal fraction (e.g. -0.225) as a percentage string."""
    pct = float(value) * 100.0
    if abs(round(pct, decimals)) == 0.0:
        pct = 0.0
    if signed:
        return f"{pct:+.{decimals}f}%"
    return f"{pct:.{decimals}f}%"


def _fmt_currency(value: float, signed: bool = False) -> str:
    """Format a currency amount with the sign before the dollar symbol."""
    rounded = round(float(value))
    if rounded == 0:
        return "$0"
    if rounded < 0:
        return f"-${abs(rounded):,}"
    if signed:
        return f"+${rounded:,}"
    return f"${rounded:,}"


def _normalise_pct_dict(weights_pct: dict[str, float]) -> dict[str, float]:
    """Scale asset percentage weights so they sum to exactly 100.00%."""
    cleaned = {asset: max(0.0, float(weights_pct.get(asset, 0.0))) for asset in ASSET_CLASSES}
    total = sum(cleaned.values())
    if total <= 0.0:
        return {asset: (100.0 if asset == "cash" else 0.0) for asset in ASSET_CLASSES}

    scaled = {asset: round((cleaned[asset] / total) * 100.0, 2) for asset in ASSET_CLASSES}
    diff = round(100.0 - sum(scaled.values()), 2)
    if diff != 0.0:
        largest = max(ASSET_CLASSES, key=lambda asset: scaled[asset])
        scaled[largest] = round(scaled[largest] + diff, 2)
    return scaled


def _pct_to_portfolio(weights_pct: dict[str, float], total_value: float) -> tuple[Portfolio, float]:
    """Convert UI percentage weights into a Portfolio, removing float dust at 100%."""
    raw_sum_pct = sum(float(weights_pct.get(asset, 0.0)) for asset in ASSET_CLASSES)
    weights = {asset: float(weights_pct.get(asset, 0.0)) / 100.0 for asset in ASSET_CLASSES}
    weight_sum = sum(weights.values())
    if abs(raw_sum_pct - 100.0) <= _WEIGHT_TOLERANCE_PCT and weight_sum > 0.0:
        weights = {asset: val / weight_sum for asset, val in weights.items()}
    return Portfolio(weights=weights, total_value=float(total_value)), raw_sum_pct


def _compute_turnover(trades: dict[str, float]) -> float:
    """Compute one-way turnover sum(|trade|) / 2."""
    return sum(abs(float(trades.get(asset, 0.0))) for asset in ASSET_CLASSES) / 2.0


def _worst_case_pnl(report: StressReport, total_value: float) -> float:
    """Look up the P&L for the worst-case scenario in a StressReport."""
    for result in report.results:
        if result.scenario_name == report.worst_case_scenario:
            return float(result.pnl_value)
    return float(report.worst_case_return) * float(total_value)


def _split_fix_items(raw: str) -> list[str]:
    """Split a guardrail summary string into clean individual fix items."""
    parts = [part.strip(" \t-•*") for part in re.split(r";|\s*\|\s*", raw)]
    return [part for part in parts if part]


def _extract_guardrail_notes(rationale: str) -> tuple[str, list[str], bool]:
    """Separate base rationale from guardrail fix notes and detect mock output."""
    text = (rationale or "").strip()
    is_mock = "mock" in text.lower()
    if not text:
        return "", [], is_mock

    base_lines: list[str] = []
    fixes: list[str] = []
    in_guardrail_block = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            in_guardrail_block = False
            continue

        line_match = _GUARDRAIL_LINE_RE.match(line) or _GUARDRAIL_TAG_RE.match(line)
        if line_match:
            in_guardrail_block = True
            trailing = line_match.group(1).strip()
            if trailing:
                fixes.extend(_split_fix_items(trailing))
            continue

        if in_guardrail_block:
            if re.match(r"^(?:[-*•]|\d+[.)])\s+", line):
                item = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line).strip()
                if item:
                    fixes.append(item)
                continue
            in_guardrail_block = False

        inline_match = _GUARDRAIL_INLINE_RE.search(line)
        if inline_match:
            prefix = line[: inline_match.start()].strip(" .;:-")
            if prefix:
                base_lines.append(prefix)
            fixes.extend(_split_fix_items(inline_match.group(1)))
            continue

        if _GUARDRAIL_KEYWORD_RE.search(line):
            fixes.extend(_split_fix_items(line))
            base_lines.append(line)
            continue

        base_lines.append(line)

    deduped_fixes: list[str] = []
    seen: set[str] = set()
    for fix in fixes:
        if fix not in seen:
            seen.add(fix)
            deduped_fixes.append(fix)

    base_rationale = "\n".join(base_lines).strip() or text
    return base_rationale, deduped_fixes, is_mock


def _build_scenario_rows(scenarios: list[Scenario], report: StressReport) -> list[dict[str, Any]]:
    """Format StressReport scenario results for display in a dataframe."""
    by_name = {scenario.name: scenario for scenario in scenarios}
    rows: list[dict[str, Any]] = []
    for result in report.results:
        scenario = by_name.get(result.scenario_name)
        source_label = "—"
        severity_label = "—"
        if scenario is not None:
            source_label = "LLM" if scenario.source == "llm" else scenario.source.title()
            severity_label = f"{scenario.severity:.2f}x"

        detractor_label = "None"
        if result.contribution_by_asset:
            worst_asset, worst_contrib = min(
                result.contribution_by_asset.items(), key=lambda item: item[1]
            )
            if worst_contrib < -1e-6:
                asset_name = ASSET_LABELS.get(worst_asset, worst_asset)
                detractor_label = f"{asset_name} ({_fmt_pct(worst_contrib, signed=True)})"

        rows.append(
            {
                "Scenario": result.scenario_name,
                "Source": source_label,
                "Severity": severity_label,
                "Probability": _fmt_pct(result.probability),
                "Portfolio Return": _fmt_pct(result.portfolio_return, signed=True),
                "P&L ($)": _fmt_currency(result.pnl_value, signed=True),
                "Largest Detractor": detractor_label,
            }
        )
    return rows


def _build_trade_rows(portfolio: Portfolio, proposal: AllocationProposal) -> list[dict[str, Any]]:
    """Build current vs. proposed weight and trade rows for all asset classes."""
    rows: list[dict[str, Any]] = []
    for asset in ASSET_CLASSES:
        current = float(portfolio.weights.get(asset, 0.0))
        proposed = float(proposal.new_weights.get(asset, 0.0))
        trade = float(proposal.trades.get(asset, proposed - current))
        if trade > 1e-4:
            action = "Buy"
        elif trade < -1e-4:
            action = "Sell"
        else:
            action = "Hold"

        rows.append(
            {
                "Asset Class": ASSET_LABELS[asset],
                "Current": _fmt_pct(current),
                "Proposed": _fmt_pct(proposed),
                "Trade (%)": _fmt_pct(trade, signed=True),
                "Trade ($)": _fmt_currency(trade * portfolio.total_value, signed=True),
                "Action": action,
            }
        )
    return rows


def _state_fingerprint(
    portfolio: Portfolio,
    report: StressReport,
    proposal: AllocationProposal | None = None,
) -> str:
    """Create a deterministic cache key for pipeline.explain calls."""
    payload: dict[str, Any] = {
        "weights": {asset: round(float(portfolio.weights.get(asset, 0.0)), 6) for asset in ASSET_CLASSES},
        "total_value": round(float(portfolio.total_value), 2),
        "expected_return": round(float(report.expected_return), 6),
        "worst_case_scenario": report.worst_case_scenario,
        "worst_case_return": round(float(report.worst_case_return), 6),
        "results": [
            {
                "name": r.scenario_name,
                "prob": round(float(r.probability), 6),
                "ret": round(float(r.portfolio_return), 6),
            }
            for r in report.results
        ],
    }
    if proposal is not None:
        payload["proposal"] = {
            "feasible": bool(proposal.feasible),
            "new_weights": {
                asset: round(float(proposal.new_weights.get(asset, 0.0)), 6) for asset in ASSET_CLASSES
            },
            "after_expected_return": round(float(proposal.after.expected_return), 6),
            "after_worst_return": round(float(proposal.after.worst_case_return), 6),
            "explanation": proposal.explanation,
        }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _build_markdown_report(
    portfolio: Portfolio,
    scenarios: list[Scenario],
    report: StressReport,
    proposal: AllocationProposal | None,
    constraints: Constraints | None,
    explanation: str | None,
) -> str:
    """Render a downloadable Markdown summary of the stress test and rebalance."""
    lines = [
        "# Portfolio Stress Test Report",
        "",
        f"- **Portfolio Value:** {_fmt_currency(portfolio.total_value)}",
        f"- **Expected 12M Return (probability-weighted):** {_fmt_pct(report.expected_return, signed=True)}",
        f"- **Baseline Return (no shock, prob {_fmt_pct(report.baseline_probability)}):** {_fmt_pct(report.baseline_return, signed=True)}",
        f"- **Worst-Case Scenario:** {report.worst_case_scenario or 'N/A'} ({_fmt_pct(report.worst_case_return, signed=True)} / {_fmt_currency(_worst_case_pnl(report, portfolio.total_value), signed=True)})",
        f"- **Probability-Weighted Loss:** {_fmt_pct(report.probability_weighted_loss)}",
        "",
        "## Current Allocation",
        "",
        "| Asset Class | Weight | Value |",
        "|---|---:|---:|",
    ]
    for asset in ASSET_CLASSES:
        weight = float(portfolio.weights.get(asset, 0.0))
        lines.append(
            f"| {ASSET_LABELS[asset]} | {_fmt_pct(weight)} | {_fmt_currency(weight * portfolio.total_value)} |"
        )

    lines.extend(
        [
            "",
            "## Scenario Results",
            "",
            "| Scenario | Source | Severity | Probability | Return | P&L | Largest Detractor |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in _build_scenario_rows(scenarios, report):
        lines.append(
            f"| {row['Scenario']} | {row['Source']} | {row['Severity']} | {row['Probability']} | "
            f"{row['Portfolio Return']} | {row['P&L ($)']} | {row['Largest Detractor']} |"
        )

    if explanation:
        lines.extend(["", "## AI Commentary", "", explanation])

    if proposal is not None:
        status = "Feasible" if proposal.feasible else "Best-effort fallback (constraints not fully met)"
        lines.extend(["", "## Rebalance Proposal", "", f"- **Status:** {status}"])
        if constraints is not None:
            lines.append(
                f"- **Constraints:** Max turnover {_fmt_pct(constraints.max_turnover)}, "
                f"Max scenario loss {_fmt_pct(constraints.max_scenario_loss)}"
            )
        lines.extend(
            [
                f"- **Expected Return:** {_fmt_pct(proposal.before.expected_return, signed=True)} -> "
                f"{_fmt_pct(proposal.after.expected_return, signed=True)}",
                f"- **Worst-Case Return:** {_fmt_pct(proposal.before.worst_case_return, signed=True)} -> "
                f"{_fmt_pct(proposal.after.worst_case_return, signed=True)}",
                f"- **One-Way Turnover:** {_fmt_pct(_compute_turnover(proposal.trades))}",
                "",
                "| Asset Class | Current | Proposed | Trade (%) | Trade ($) | Action |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in _build_trade_rows(portfolio, proposal):
            lines.append(
                f"| {row['Asset Class']} | {row['Current']} | {row['Proposed']} | "
                f"{row['Trade (%)']} | {row['Trade ($)']} | {row['Action']} |"
            )
        if proposal.explanation:
            lines.extend(["", f"**Optimizer summary:** {proposal.explanation}"])

    lines.extend(["", "---", f"*{DISCLAIMER}*"])
    return "\n".join(lines)


def _build_json_report(
    portfolio: Portfolio,
    scenarios: list[Scenario],
    report: StressReport,
    proposal: AllocationProposal | None,
    constraints: Constraints | None,
    explanation: str | None,
) -> str:
    """Serialize the active portfolio, scenarios, stress report, and proposal as JSON."""
    data: dict[str, Any] = {
        "disclaimer": DISCLAIMER,
        "portfolio": {
            "total_value": portfolio.total_value,
            "weights": {asset: portfolio.weights.get(asset, 0.0) for asset in ASSET_CLASSES},
        },
        "scenarios": [
            {
                "name": s.name,
                "description": s.description,
                "source": s.source,
                "probability": s.probability,
                "severity": s.severity,
                "effective_shocks": s.effective_shocks(),
                "rationale": s.rationale,
            }
            for s in scenarios
        ],
        "stress_report": {
            "baseline_return": report.baseline_return,
            "baseline_probability": report.baseline_probability,
            "expected_return": report.expected_return,
            "probability_weighted_loss": report.probability_weighted_loss,
            "worst_case_scenario": report.worst_case_scenario,
            "worst_case_return": report.worst_case_return,
            "results": [
                {
                    "scenario_name": r.scenario_name,
                    "probability": r.probability,
                    "portfolio_return": r.portfolio_return,
                    "pnl_value": r.pnl_value,
                    "contribution_by_asset": r.contribution_by_asset,
                }
                for r in report.results
            ],
        },
        "explanation": explanation,
    }
    if proposal is not None:
        data["allocation_proposal"] = {
            "feasible": proposal.feasible,
            "constraints": (
                {
                    "max_turnover": constraints.max_turnover,
                    "max_scenario_loss": constraints.max_scenario_loss,
                    "min_weight": constraints.min_weight,
                    "max_weight": constraints.max_weight,
                }
                if constraints is not None
                else None
            ),
            "new_weights": proposal.new_weights,
            "trades": proposal.trades,
            "turnover": _compute_turnover(proposal.trades),
            "before": {
                "expected_return": proposal.before.expected_return,
                "worst_case_return": proposal.before.worst_case_return,
                "probability_weighted_loss": proposal.before.probability_weighted_loss,
            },
            "after": {
                "expected_return": proposal.after.expected_return,
                "worst_case_return": proposal.after.worst_case_return,
                "probability_weighted_loss": proposal.after.probability_weighted_loss,
            },
            "explanation": proposal.explanation,
        }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Plotly figures (graph_objects only)
# ---------------------------------------------------------------------------


def _make_allocation_chart(portfolio: Portfolio) -> go.Figure:
    """Build a donut chart of positive portfolio weights."""
    active_assets = [asset for asset in ASSET_CLASSES if portfolio.weights.get(asset, 0.0) > 1e-5]
    fig = go.Figure()
    if not active_assets:
        fig.add_annotation(
            text="No positive weights entered",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_layout(height=270, margin=dict(t=10, b=10, l=10, r=10))
        return fig

    labels = [ASSET_LABELS[asset] for asset in active_assets]
    values = [float(portfolio.weights[asset]) * 100.0 for asset in active_assets]
    dollars = [_fmt_currency(float(portfolio.weights[asset]) * portfolio.total_value) for asset in active_assets]
    colors = [ASSET_COLORS[asset] for asset in active_assets]

    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.52,
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=1.5)),
            customdata=dollars,
            textinfo="label+percent",
            textposition="inside",
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f}%<br>Value: %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
    )
    return fig


def _make_contribution_chart(report: StressReport) -> go.Figure:
    """Build a relative stacked bar chart of return contributions by asset per scenario."""
    fig = go.Figure()
    if not report.results:
        fig.add_annotation(
            text="Select at least one scenario to view contributions",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_layout(height=360, margin=dict(t=20, b=30, l=40, r=20))
        return fig

    scenario_names = [result.scenario_name for result in report.results]
    for asset in ASSET_CLASSES:
        values = [float(result.contribution_by_asset.get(asset, 0.0)) * 100.0 for result in report.results]
        if any(abs(val) > 1e-4 for val in values):
            fig.add_trace(
                go.Bar(
                    name=ASSET_LABELS[asset],
                    x=scenario_names,
                    y=values,
                    marker_color=ASSET_COLORS[asset],
                    hovertemplate=(
                        f"<b>{ASSET_LABELS[asset]}</b><br>"
                        "Scenario: %{x}<br>Contribution: %{y:+.2f}%<extra></extra>"
                    ),
                )
            )

    net_returns = [float(result.portfolio_return) * 100.0 for result in report.results]
    fig.add_trace(
        go.Scatter(
            name="Net portfolio return",
            x=scenario_names,
            y=net_returns,
            mode="markers+text",
            text=[f"{val:+.1f}%" for val in net_returns],
            textposition=["top center" if val >= 0 else "bottom center" for val in net_returns],
            marker=dict(color="#0F172A", size=9, symbol="diamond"),
            hovertemplate="<b>Net Return</b><br>Scenario: %{x}<br>Return: %{y:+.2f}%<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line_width=1, line_color="#475569")
    fig.update_layout(
        barmode="relative",
        height=390,
        margin=dict(t=25, b=40, l=45, r=15),
        yaxis=dict(title="Return contribution (%)", ticksuffix="%"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return fig


def _make_weights_comparison_chart(portfolio: Portfolio, proposal: AllocationProposal) -> go.Figure:
    """Compare current and proposed weights side by side across asset classes."""
    labels = [ASSET_LABELS[asset] for asset in ASSET_CLASSES]
    current_pct = [float(portfolio.weights.get(asset, 0.0)) * 100.0 for asset in ASSET_CLASSES]
    proposed_pct = [float(proposal.new_weights.get(asset, 0.0)) * 100.0 for asset in ASSET_CLASSES]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Current",
                x=labels,
                y=current_pct,
                marker_color="#94A3B8",
                hovertemplate="<b>%{x}</b><br>Current: %{y:.1f}%<extra></extra>",
            ),
            go.Bar(
                name="Proposed",
                x=labels,
                y=proposed_pct,
                marker_color="#1B4965",
                hovertemplate="<b>%{x}</b><br>Proposed: %{y:.1f}%<extra></extra>",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(t=30, b=40, l=45, r=15),
        yaxis=dict(title="Portfolio weight (%)", ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return fig


def _make_before_after_scenarios_chart(proposal: AllocationProposal, max_loss: float) -> go.Figure:
    """Compare scenario portfolio returns before and after rebalancing."""
    fig = go.Figure()
    if not proposal.before.results:
        return fig

    after_by_name = {res.scenario_name: res.portfolio_return for res in proposal.after.results}
    names = [res.scenario_name for res in proposal.before.results]
    before_pct = [float(res.portfolio_return) * 100.0 for res in proposal.before.results]
    after_pct = [
        float(
            after_by_name.get(
                res.scenario_name,
                proposal.after.results[idx].portfolio_return if idx < len(proposal.after.results) else 0.0,
            )
        )
        * 100.0
        for idx, res in enumerate(proposal.before.results)
    ]

    fig.add_trace(
        go.Bar(
            name="Before rebalance",
            x=names,
            y=before_pct,
            marker_color="#CBD5E1",
            hovertemplate="<b>%{x}</b><br>Before: %{y:+.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="After rebalance",
            x=names,
            y=after_pct,
            marker_color="#2D6A4F",
            hovertemplate="<b>%{x}</b><br>After: %{y:+.1f}%<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line_width=1, line_color="#475569")
    fig.add_hline(
        y=-float(max_loss) * 100.0,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"Loss cap ({_fmt_pct(-max_loss)})",
        annotation_position="bottom right",
    )
    fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(t=30, b=40, l=45, r=15),
        yaxis=dict(title="Scenario return (%)", ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return fig


# ---------------------------------------------------------------------------
# Session state management & callbacks
# ---------------------------------------------------------------------------


def _apply_preset_to_state(preset_name: str, presets: dict[str, Portfolio]) -> None:
    """Copy a named preset portfolio into st.session_state weight and value keys."""
    portfolio = presets.get(preset_name)
    if portfolio is None:
        return
    st.session_state["total_value"] = float(portfolio.total_value)
    for asset in ASSET_CLASSES:
        st.session_state[f"weight_pct_{asset}"] = round(float(portfolio.weights.get(asset, 0.0)) * 100.0, 2)


def _init_session_state(presets: dict[str, Portfolio], historical: list[Scenario]) -> None:
    """Initialize required st.session_state keys without mutating existing user choices."""
    preset_names = list(presets.keys())
    default_preset = "Balanced 60/40" if "Balanced 60/40" in presets else (preset_names[0] if preset_names else "")

    if "selected_preset" not in st.session_state or (
        preset_names and st.session_state["selected_preset"] not in presets
    ):
        st.session_state["selected_preset"] = default_preset

    active_preset = presets.get(st.session_state["selected_preset"])
    if "total_value" not in st.session_state:
        st.session_state["total_value"] = float(active_preset.total_value) if active_preset else 1_000_000.0

    for asset in ASSET_CLASSES:
        key = f"weight_pct_{asset}"
        if key not in st.session_state:
            default_weight = float(active_preset.weights.get(asset, 0.0)) if active_preset else (1.0 if asset == "cash" else 0.0)
            st.session_state[key] = round(default_weight * 100.0, 2)

    hist_names = [scenario.name for scenario in historical]
    if "selected_historical_names" not in st.session_state:
        st.session_state["selected_historical_names"] = hist_names[:3]
    else:
        st.session_state["selected_historical_names"] = [
            name for name in st.session_state["selected_historical_names"] if name in hist_names
        ]

    if "llm_scenarios" not in st.session_state:
        st.session_state["llm_scenarios"] = []
    if "explain_cache" not in st.session_state:
        st.session_state["explain_cache"] = {}
    if "prepare_ai_explain" not in st.session_state:
        st.session_state["prepare_ai_explain"] = None
    if "prepare_ai_explain_fp" not in st.session_state:
        st.session_state["prepare_ai_explain_fp"] = None


def _apply_preset_callback(presets: dict[str, Portfolio]) -> None:
    """Callback triggered when the user picks or resets a preset portfolio."""
    _apply_preset_to_state(st.session_state.get("selected_preset", ""), presets)


def _normalise_weights_callback() -> None:
    """Callback for 'Normalise to 100%' button."""
    current = {asset: float(st.session_state.get(f"weight_pct_{asset}", 0.0)) for asset in ASSET_CLASSES}
    normalised = _normalise_pct_dict(current)
    for asset, value in normalised.items():
        st.session_state[f"weight_pct_{asset}"] = value


def _remove_llm_scenario_callback(scenario_name: str) -> None:
    """Remove a single generated LLM scenario from session state."""
    existing: list[Scenario] = st.session_state.get("llm_scenarios", [])
    st.session_state["llm_scenarios"] = [s for s in existing if s.name != scenario_name]


def _clear_llm_scenarios_callback() -> None:
    """Clear all generated LLM scenarios from session state."""
    st.session_state["llm_scenarios"] = []


def _cached_explain(
    portfolio: Portfolio,
    report: StressReport,
    proposal: AllocationProposal | None = None,
) -> tuple[str | None, str | None]:
    """Return cached pipeline.explain output or compute and cache it safely."""
    cache: dict[str, str] = st.session_state.setdefault("explain_cache", {})
    key = _state_fingerprint(portfolio, report, proposal)
    if key in cache:
        return cache[key], None
    try:
        text = pipeline.explain(portfolio, report, proposal)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    cache[key] = text
    return text, None


# ---------------------------------------------------------------------------
# UI Panels
# ---------------------------------------------------------------------------


def _render_portfolio_sidebar(presets: dict[str, Portfolio]) -> tuple[Portfolio, float]:
    """Panel 1: Portfolio weights, total value, normalisation, and donut chart."""
    with st.sidebar:
        st.header("1. What do I own?")

        preset_names = list(presets.keys())
        col_preset, col_reset = st.columns([2.2, 1.0])
        with col_preset:
            st.selectbox(
                "Preset portfolio",
                options=preset_names,
                key="selected_preset",
                on_change=_apply_preset_callback,
                args=(presets,),
            )
        with col_reset:
            st.write("")
            st.button(
                "Reset",
                key="reset_preset_btn",
                on_click=_apply_preset_callback,
                args=(presets,),
                width="stretch",
                help="Reload the selected preset weights and value.",
            )

        st.number_input(
            "Total portfolio value ($)",
            min_value=1_000.0,
            step=50_000.0,
            format="%.0f",
            key="total_value",
        )

        st.subheader("Asset weights (%)")
        for asset in ASSET_CLASSES:
            st.number_input(
                ASSET_LABELS[asset],
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                format="%.2f",
                key=f"weight_pct_{asset}",
            )

        weights_pct = {asset: float(st.session_state.get(f"weight_pct_{asset}", 0.0)) for asset in ASSET_CLASSES}
        portfolio, raw_sum_pct = _pct_to_portfolio(weights_pct, float(st.session_state["total_value"]))

        if abs(raw_sum_pct - 100.0) <= _WEIGHT_TOLERANCE_PCT:
            st.success(f"Weight sum: {raw_sum_pct:.2f}%")
        else:
            st.warning(f"Weight sum: {raw_sum_pct:.2f}% (must equal 100.00%)")

        st.button(
            "Normalise to 100%",
            key="normalise_weights_btn",
            on_click=_normalise_weights_callback,
            width="stretch",
        )

        st.plotly_chart(_make_allocation_chart(portfolio), width="stretch")
        return portfolio, raw_sum_pct


def _render_guardrail_box(llm_scenarios: list[Scenario]) -> None:
    """Stretch panel: highlight guardrail adjustments on LLM-generated scenarios."""
    if not llm_scenarios:
        return

    with st.expander("What the guardrail changed (AI proposals)", expanded=True):
        all_mock = True
        for scenario in llm_scenarios:
            _, fixes, is_mock = _extract_guardrail_notes(scenario.rationale)
            all_mock = all_mock and is_mock
            if fixes:
                st.markdown(f"**{scenario.name}**")
                for fix in fixes:
                    st.markdown(f"- {fix}")
            elif is_mock:
                st.caption(
                    f"**{scenario.name}:** Using mock scenarios while Role 4 is pending integration."
                )
            else:
                st.caption(
                    f"**{scenario.name}:** Passed deterministic guardrail with no adjustments required."
                )
        if all_mock:
            st.info(
                "Mock mode active: once the live LLM + guardrail module merges, any clamped shocks or scaled probabilities will appear here."
            )


def _render_scenarios_panel(historical: list[Scenario]) -> list[Scenario]:
    """Panel 2: Historical multi-select, LLM text-to-scenario, and probability/severity controls."""
    st.subheader("2. What happens if things go wrong?")

    hist_by_name = {scenario.name: scenario for scenario in historical}
    hist_names = list(hist_by_name.keys())

    if hist_names:
        st.multiselect(
            "Historical crisis scenarios",
            options=hist_names,
            key="selected_historical_names",
            help="Pick historical shocks to include in the stress test.",
        )
    else:
        st.info("No historical scenarios available from the data source.")

    st.markdown("**Add custom event via AI**")
    event_text = st.text_area(
        "Describe a market shock in plain English",
        placeholder="e.g., Oil jumps 40% and the Fed hikes 100 basis points",
        height=76,
        key="llm_event_input",
    )

    col_gen, col_clear = st.columns([2.2, 1.0])
    with col_gen:
        generate_clicked = st.button(
            "Generate AI scenarios",
            type="primary",
            width="stretch",
            key="generate_llm_btn",
        )
    with col_clear:
        has_llm = bool(st.session_state.get("llm_scenarios"))
        st.button(
            "Clear AI",
            disabled=not has_llm,
            on_click=_clear_llm_scenarios_callback,
            width="stretch",
            key="clear_llm_btn",
        )

    if generate_clicked:
        cleaned_text = event_text.strip()
        if not cleaned_text:
            st.warning("Enter an event description before generating scenarios.")
        else:
            try:
                generated = pipeline.scenarios_from_text(cleaned_text)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not generate scenarios from text: {exc}")
            else:
                if not generated:
                    st.warning("No scenarios were returned for that event.")
                else:
                    existing: list[Scenario] = list(st.session_state.get("llm_scenarios", []))
                    by_name = {s.name: s for s in existing}
                    for sc in generated:
                        by_name[sc.name] = sc
                        slug = _key_slug(f"{sc.source}_{sc.name}")
                        st.session_state[f"prob_pct_{slug}"] = round(
                            _clamp(sc.probability * 100.0, 0.0, 100.0), 1
                        )
                        st.session_state[f"sev_{slug}"] = round(_clamp(sc.severity, 0.5, 2.0), 2)
                    st.session_state["llm_scenarios"] = list(by_name.values())

    llm_scenarios: list[Scenario] = list(st.session_state.get("llm_scenarios", []))
    _render_guardrail_box(llm_scenarios)

    selected_historical = [
        hist_by_name[name]
        for name in st.session_state.get("selected_historical_names", [])
        if name in hist_by_name
    ]
    base_scenarios = selected_historical + llm_scenarios

    if not base_scenarios:
        st.info("Select at least one historical scenario or generate an AI scenario above.")
        return []

    st.markdown("**Scenario parameters**")
    active_scenarios: list[Scenario] = []
    for scenario in base_scenarios:
        slug = _key_slug(f"{scenario.source}_{scenario.name}")
        prob_key = f"prob_pct_{slug}"
        sev_key = f"sev_{slug}"

        if prob_key not in st.session_state:
            st.session_state[prob_key] = round(_clamp(scenario.probability * 100.0, 0.0, 100.0), 1)
        else:
            st.session_state[prob_key] = _clamp(st.session_state[prob_key], 0.0, 100.0)

        if sev_key not in st.session_state:
            st.session_state[sev_key] = round(_clamp(scenario.severity, 0.5, 2.0), 2)
        else:
            st.session_state[sev_key] = _clamp(st.session_state[sev_key], 0.5, 2.0)

        badge = "AI / LLM" if scenario.source == "llm" else scenario.source.title()
        with st.container(border=True):
            header_col, remove_col = st.columns([4.0, 1.0])
            with header_col:
                st.markdown(f"**{scenario.name}** `{badge}`")
            with remove_col:
                if scenario.source == "llm":
                    st.button(
                        "Remove",
                        key=f"remove_{slug}",
                        on_click=_remove_llm_scenario_callback,
                        args=(scenario.name,),
                        width="stretch",
                    )

            if scenario.description:
                st.caption(scenario.description)

            s_col1, s_col2 = st.columns(2)
            with s_col1:
                prob_pct = st.slider(
                    "Probability (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    format="%.1f%%",
                    key=prob_key,
                )
            with s_col2:
                severity = st.slider(
                    "Severity multiplier",
                    min_value=0.5,
                    max_value=2.0,
                    step=0.05,
                    format="%.2fx",
                    key=sev_key,
                )

            base_rationale, _, _ = _extract_guardrail_notes(scenario.rationale)
            if base_rationale:
                st.caption(f"Rationale: {base_rationale}")

            updated = Scenario(
                name=scenario.name,
                description=scenario.description,
                shocks=dict(scenario.shocks),
                probability=float(prob_pct) / 100.0,
                severity=float(severity),
                source=scenario.source,
                rationale=scenario.rationale,
            )
            with st.expander("Effective asset shocks", expanded=False):
                shocks = updated.effective_shocks()
                shock_df = pd.DataFrame(
                    [
                        {"Asset Class": ASSET_LABELS[asset], "Shock": _fmt_pct(shocks[asset], signed=True)}
                        for asset in ASSET_CLASSES
                    ]
                )
                st.dataframe(shock_df, width="stretch", hide_index=True)

            active_scenarios.append(updated)

    total_prob = sum(s.probability for s in active_scenarios)
    if total_prob > 1.0 + 1e-9:
        st.warning(
            f"Scenario probabilities sum to {_fmt_pct(total_prob)} (> 100%). "
            "Scenarios are mutually exclusive — reduce probabilities so the total is 100% or less."
        )
    else:
        st.caption(
            f"Active scenarios probability: {_fmt_pct(total_prob)} | "
            f"No-shock baseline probability: {_fmt_pct(max(0.0, 1.0 - total_prob))}"
        )

    return active_scenarios


def _render_stress_panel(
    portfolio: Portfolio,
    raw_sum_pct: float,
    active_scenarios: list[Scenario],
) -> tuple[StressReport | None, str | None]:
    """Panel 3: Stress-test headline metrics, stacked chart, table, and AI explanation."""
    st.subheader("Stress test impact")

    if not active_scenarios:
        st.info("Select or generate scenarios on the left to compute portfolio stress results.")
        return None, None

    if abs(raw_sum_pct - 100.0) > _WEIGHT_TOLERANCE_PCT:
        st.warning("Portfolio weights do not sum to 100%. Click 'Normalise to 100%' in the sidebar.")
        return None, None

    if sum(s.probability for s in active_scenarios) > 1.0 + 1e-9:
        st.warning("Lower scenario probabilities so they sum to 100% or less before running the stress test.")
        return None, None

    try:
        report = pipeline.stress_test(portfolio, active_scenarios)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Stress test failed: {exc}")
        return None, None

    worst_pnl = _worst_case_pnl(report, portfolio.total_value)
    prob_loss_dollars = float(report.probability_weighted_loss) * float(portfolio.total_value)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Expected 12M Return",
        _fmt_pct(report.expected_return, signed=True),
        delta=f"{_fmt_pct(report.expected_return - report.baseline_return, signed=True)} vs baseline",
    )
    m2.metric(
        "Worst-Case Loss (%)",
        _fmt_pct(report.worst_case_return, signed=True),
    )
    m3.metric(
        "Worst-Case Loss ($)",
        _fmt_currency(worst_pnl, signed=True),
    )
    m4.metric(
        "Prob.-Weighted Loss",
        _fmt_pct(report.probability_weighted_loss),
        delta=_fmt_currency(-prob_loss_dollars, signed=True),
        delta_color="off",
    )

    st.caption(
        f"Worst case: **{report.worst_case_scenario or 'N/A'}** | "
        f"Baseline (no shock, {_fmt_pct(report.baseline_probability)} prob): "
        f"{_fmt_pct(report.baseline_return, signed=True)}"
    )

    st.markdown("**Loss contribution by asset class per scenario**")
    st.plotly_chart(_make_contribution_chart(report), width="stretch")

    st.markdown("**Scenario breakdown**")
    scenario_df = pd.DataFrame(_build_scenario_rows(active_scenarios, report))
    st.dataframe(scenario_df, width="stretch", hide_index=True)

    st.markdown("**Analyst explanation**")
    explanation, explain_err = _cached_explain(portfolio, report)
    if explain_err:
        st.error(f"Could not generate explanation: {explain_err}")
    elif explanation:
        st.info(explanation)

    return report, explanation


def _render_prepare_panel(
    portfolio: Portfolio,
    raw_sum_pct: float,
    active_scenarios: list[Scenario],
    report: StressReport | None,
    stress_explanation: str | None,
) -> None:
    """Panel 4: Optimizer constraints, before/after comparison, trades, and report download."""
    st.subheader("3. How should I prepare?")

    c1, c2 = st.columns(2)
    with c1:
        max_turnover_pct = st.slider(
            "Maximum one-way turnover (%)",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
            format="%.0f%%",
            key="max_turnover_pct",
            help="Maximum fraction of the portfolio allowed to trade (sum(|new - old|) / 2).",
        )
    with c2:
        max_loss_pct = st.slider(
            "Maximum loss in any scenario (%)",
            min_value=1.0,
            max_value=60.0,
            value=15.0,
            step=0.5,
            format="%.1f%%",
            key="max_scenario_loss_pct",
            help="Loss cap applied across every active scenario.",
        )

    min_weights: dict[str, float] = {}
    max_weights: dict[str, float] = {}
    with st.expander("Advanced: Per-asset weight limits (optional)", expanded=False):
        st.caption("Defaults match the optimizer specification: 0% minimum and 60% maximum per asset class.")
        cols = st.columns(3)
        for idx, asset in enumerate(ASSET_CLASSES):
            with cols[idx % 3]:
                st.markdown(f"**{ASSET_LABELS[asset]}**")
                low = st.number_input(
                    "Min (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=1.0,
                    key=f"min_w_{asset}",
                )
                high = st.number_input(
                    "Max (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=60.0,
                    step=1.0,
                    key=f"max_w_{asset}",
                )
                min_weights[asset] = float(low) / 100.0
                max_weights[asset] = max(float(low), float(high)) / 100.0

    if not active_scenarios:
        st.info("Select or generate scenarios in the 'Stress Test & Scenarios' tab first.")
        return

    if abs(raw_sum_pct - 100.0) > _WEIGHT_TOLERANCE_PCT:
        st.warning("Portfolio weights do not sum to 100%. Click 'Normalise to 100%' in the sidebar.")
        return

    if sum(s.probability for s in active_scenarios) > 1.0 + 1e-9:
        st.warning("Scenario probabilities sum to more than 100%. Adjust them in the Scenarios tab first.")
        return

    constraints = Constraints(
        max_turnover=float(max_turnover_pct) / 100.0,
        max_scenario_loss=float(max_loss_pct) / 100.0,
        min_weight=min_weights,
        max_weight=max_weights,
    )

    try:
        proposal = pipeline.prepare(portfolio, active_scenarios, constraints)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Allocation optimizer failed: {exc}")
        return

    if proposal.feasible:
        st.success(
            f"Feasible rebalance found: every scenario loss is capped within {_fmt_pct(constraints.max_scenario_loss)} "
            f"and turnover is within {_fmt_pct(constraints.max_turnover)}."
        )
    else:
        st.warning(
            "Constraints could not all be met simultaneously (infeasible). "
            "Showing the best-effort fallback allocation that minimizes worst-case loss within weight and turnover limits."
        )

    actual_turnover = _compute_turnover(proposal.trades)
    worst_delta = proposal.after.worst_case_return - proposal.before.worst_case_return
    expected_delta = proposal.after.expected_return - proposal.before.expected_return
    pwl_delta = proposal.after.probability_weighted_loss - proposal.before.probability_weighted_loss

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Worst-Case Return (After)",
        _fmt_pct(proposal.after.worst_case_return, signed=True),
        delta=f"{_fmt_pct(worst_delta, signed=True)} vs {_fmt_pct(proposal.before.worst_case_return, signed=True)}",
    )
    k2.metric(
        "Expected Return (After)",
        _fmt_pct(proposal.after.expected_return, signed=True),
        delta=f"{_fmt_pct(expected_delta, signed=True)} vs {_fmt_pct(proposal.before.expected_return, signed=True)}",
    )
    k3.metric(
        "Prob.-Weighted Loss (After)",
        _fmt_pct(proposal.after.probability_weighted_loss),
        delta=f"{_fmt_pct(pwl_delta, signed=True)} vs {_fmt_pct(proposal.before.probability_weighted_loss)}",
        delta_color="inverse",
    )
    k4.metric(
        "One-Way Turnover",
        _fmt_pct(actual_turnover),
        delta=f"Limit {_fmt_pct(constraints.max_turnover)}",
        delta_color="off",
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**Current vs. Proposed Weights**")
        st.plotly_chart(_make_weights_comparison_chart(portfolio, proposal), width="stretch")
    with chart_col2:
        st.markdown("**Scenario Returns: Before vs. After**")
        st.plotly_chart(
            _make_before_after_scenarios_chart(proposal, constraints.max_scenario_loss),
            width="stretch",
        )

    st.markdown("**Proposed Trade List**")
    trade_df = pd.DataFrame(_build_trade_rows(portfolio, proposal))
    st.dataframe(trade_df, width="stretch", hide_index=True)

    if proposal.explanation:
        st.info(proposal.explanation)

    if report is not None:
        proposal_fp = _state_fingerprint(portfolio, report, proposal)
        if st.button("Generate AI commentary on rebalance", key="rebalance_explain_btn"):
            text, err = _cached_explain(portfolio, report, proposal)
            if err:
                st.error(f"Could not generate rebalance commentary: {err}")
            else:
                st.session_state["prepare_ai_explain"] = text
                st.session_state["prepare_ai_explain_fp"] = proposal_fp

        if (
            st.session_state.get("prepare_ai_explain")
            and st.session_state.get("prepare_ai_explain_fp") == proposal_fp
        ):
            st.markdown("**AI commentary on rebalance**")
            st.write(st.session_state["prepare_ai_explain"])

        st.divider()
        st.markdown("**Export report**")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download Markdown report",
                data=_build_markdown_report(
                    portfolio, active_scenarios, report, proposal, constraints, stress_explanation
                ),
                file_name="stress_test_report.md",
                mime="text/markdown",
                width="stretch",
            )
        with dl2:
            st.download_button(
                "Download JSON data",
                data=_build_json_report(
                    portfolio, active_scenarios, report, proposal, constraints, stress_explanation
                ),
                file_name="stress_test_report.json",
                mime="application/json",
                width="stretch",
            )


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Money Talks — Portfolio Stress Test",
        layout="wide",
    )

    try:
        presets = pipeline.preset_portfolios()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load preset portfolios: {exc}")
        presets = {}

    if not presets:
        presets = {
            "All Cash": Portfolio(
                weights={asset: (1.0 if asset == "cash" else 0.0) for asset in ASSET_CLASSES},
                total_value=1_000_000.0,
            )
        }

    try:
        historical = pipeline.historical_scenarios()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load historical scenarios: {exc}")
        historical = []

    _init_session_state(presets, historical)

    portfolio, raw_sum_pct = _render_portfolio_sidebar(presets)

    st.title("Portfolio Stress Test & Allocation Tool")
    st.caption(
        "Answer three questions about your portfolio: what you own, what happens if markets break, and how to rebalance."
    )

    tab_stress, tab_prepare = st.tabs(["Stress Test & Scenarios", "Prepare & Rebalance"])

    report: StressReport | None = None
    stress_explanation: str | None = None
    active_scenarios: list[Scenario] = []

    with tab_stress:
        col_scenarios, col_stress = st.columns([1.0, 1.35], gap="large")
        with col_scenarios:
            active_scenarios = _render_scenarios_panel(historical)
        with col_stress:
            report, stress_explanation = _render_stress_panel(portfolio, raw_sum_pct, active_scenarios)

    with tab_prepare:
        _render_prepare_panel(portfolio, raw_sum_pct, active_scenarios, report, stress_explanation)

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
