"""Money Talks: portfolio stress test and allocation app.

Run with:  streamlit run app.py

The page imports only money_talks.pipeline and money_talks.contracts. Every
number on screen is computed by deterministic code; the LLM only proposes
scenarios and writes explanations.
"""

import dataclasses
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402
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

GUARDRAIL_MARK = "Guardrail:"
DEFAULT_SCENARIOS = (
    "Global Financial Crisis (2008)",
    "Inflation and Rate Shock (2022)",
    "AI / Mega-Cap Bubble Unwind",
)
EXAMPLE_EVENT = "Oil jumps 40% after a Gulf shipping blockade and the Fed hikes rates by 100 basis points"

st.set_page_config(page_title="Money Talks: Stress Test", layout="wide")
st.title("Portfolio Stress Test")
st.caption(
    "Test a portfolio against shock scenarios, then see how to prepare. "
    "The model proposes, the code decides. Illustrative model, not investment advice."
)

state = st.session_state
state.setdefault("llm_scenarios", [])
state.setdefault("explanations", {})

# Hosted deployments keep keys in Streamlit secrets; copy them to the environment once.
try:
    for _name in (
        "NICRRON_API_KEY",
        "NICRRON_BASE_URL",
        "NICRRON_MODEL",
        "PRISMTRACE_HOST",
        "PRISMTRACE_PROJECT_ID",
        "PRISMTRACE_API_KEY",
    ):
        if _name in st.secrets and not os.environ.get(_name):
            os.environ[_name] = str(st.secrets[_name])
except Exception:  # noqa: BLE001 - no secrets file is the normal local case
    pass

MODELS = {
    "GPT-5.4 mini (fast, about 3 seconds)": "openai/gpt-5.4-mini",
    "Claude Haiku 5.5 (fast)": "anthropic/claude-haiku-5-5",
    "Mistral Small 4 (fast)": "mistral/mistral-small-4",
    "GPT-6.1 Sol (strongest, about 17 seconds)": "openai/gpt-6.1-sol",
}
with st.sidebar:
    st.header("AI model")
    model_label = st.selectbox("Model for scenarios and explanations", list(MODELS))
    own_key = st.text_input(
        "Your own Nicrron API key (optional)",
        type="password",
        help="Used only for your session and never stored. Leave empty to use the host's key.",
    )
    pipeline.set_llm_credentials(api_key=own_key, model=MODELS[model_label])
    _status = pipeline.llm_status()
    if _status["key_source"] == "session":
        st.success("Using your key for this session.")
    elif _status["key_source"] == "host":
        st.info("Using the key configured by the host.")
    else:
        st.warning("No API key available. Generated scenarios will be labelled placeholders. Paste a key above to use the AI.")
    st.caption("The model only proposes scenarios and writes explanations. Every number is computed and checked by code.")
    st.header("PRISM tracing")
    _prism = pipeline.prism_status()
    if "is on" in _prism:
        st.success("On. Each AI call and the guardrail's decision are recorded in PRISM.")
    else:
        st.caption(_prism)


def pct(value: float) -> str:
    return f"{value:.1%}"


def money(value: float) -> str:
    return f"{value:,.0f}"


def _build_markdown_report(
    portfolio: Portfolio,
    scenarios: list[Scenario],
    report: StressReport,
    sensitivity: dict[float, float] | None = None,
    proposal: AllocationProposal | None = None,
    explanation: str | None = None,
) -> str:
    """Format a Markdown summary of the portfolio, stress test, sensitivity, and rebalance."""
    lines = [
        "# Portfolio Stress Test Report",
        "",
        f"- **Portfolio value:** {money(portfolio.total_value)}",
        f"- **Expected 12M return (all outcomes):** {pct(report.expected_return)}",
        f"- **Baseline return (no shock, {pct(report.baseline_probability)} prob):** {pct(report.baseline_return)}",
        (
            f"- **Worst-case scenario:** {report.worst_case_scenario or 'N/A'} "
            f"({pct(report.worst_case_return)} / {money(report.worst_case_return * portfolio.total_value)})"
        ),
        f"- **Probability-weighted loss:** {pct(report.probability_weighted_loss)}",
        "",
        "## Current Allocation",
        "",
        "| Asset Class | Weight | Value |",
        "|---|---:|---:|",
    ]
    for asset in ASSET_CLASSES:
        weight = float(portfolio.weights.get(asset, 0.0))
        lines.append(
            f"| {ASSET_LABELS[asset]} | {pct(weight)} | {money(weight * portfolio.total_value)} |"
        )

    lines.extend(
        [
            "",
            "## Scenario Results",
            "",
            "| Scenario | Source | Severity | Probability | Portfolio Return | Gain / Loss |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    by_name = {s.name: s for s in scenarios}
    for result in sorted(report.results, key=lambda r: r.portfolio_return):
        sc = by_name.get(result.scenario_name)
        src = sc.source if sc is not None else "—"
        sev = f"{sc.severity:.1f}x" if sc is not None else "—"
        lines.append(
            f"| {result.scenario_name} | {src} | {sev} | {pct(result.probability)} | "
            f"{pct(result.portfolio_return)} | {money(result.pnl_value)} |"
        )

    if sensitivity:
        lines.extend(
            [
                "",
                "## Severity Sensitivity",
                "",
                "| Severity Multiplier | Worst-Case Return | Worst-Case P&L |",
                "|---:|---:|---:|",
            ]
        )
        for mult, ret in sensitivity.items():
            lines.append(
                f"| {mult:.1f}x | {pct(ret)} | {money(ret * portfolio.total_value)} |"
            )

    if explanation:
        lines.extend(["", "## AI Explanation", "", explanation])

    if proposal is not None:
        status = "Feasible" if proposal.feasible else "Best-effort fallback (constraints not all met)"
        lines.extend(
            [
                "",
                "## Rebalance Proposal",
                "",
                f"- **Status:** {status}",
                f"- **Worst-case return:** {pct(proposal.before.worst_case_return)} -> {pct(proposal.after.worst_case_return)}",
                f"- **Expected return:** {pct(proposal.before.expected_return)} -> {pct(proposal.after.expected_return)}",
                "",
                "| Asset Class | Current | Proposed | Trade | Value Change |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for asset in ASSET_CLASSES:
            proposed = float(proposal.new_weights.get(asset, 0.0))
            trade = float(proposal.trades.get(asset, 0.0))
            current = proposed - trade
            lines.append(
                f"| {ASSET_LABELS[asset]} | {pct(current)} | {pct(proposed)} | "
                f"{trade * 100:+.1f} pts | {money(trade * portfolio.total_value)} |"
            )
        if proposal.explanation:
            lines.extend(["", f"**Optimizer summary:** {proposal.explanation}"])

    lines.extend(["", "---", "*Illustrative model, not investment advice.*"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Portfolio
# ---------------------------------------------------------------------------
st.header("1. Portfolio")
presets = pipeline.preset_portfolios()
left, right = st.columns([2, 1])
preset_name = left.selectbox("Starting portfolio", list(presets))
total_value = right.number_input("Portfolio value", min_value=1_000.0, value=1_000_000.0, step=50_000.0)


def weight_key(asset: str) -> str:
    return f"w::{preset_name}::{asset}"


def normalise_inputs() -> None:
    current = {a: float(state.get(weight_key(a), 0.0)) for a in ASSET_CLASSES}
    total = sum(current.values())
    if total > 0:
        for a in ASSET_CLASSES:
            state[weight_key(a)] = round(current[a] / total * 100, 2)


cols = st.columns(3)
raw_weights = {}
for i, asset in enumerate(ASSET_CLASSES):
    default = round(presets[preset_name].weights.get(asset, 0.0) * 100, 2)
    state.setdefault(weight_key(asset), default)
    raw_weights[asset] = cols[i % 3].number_input(
        f"{ASSET_LABELS[asset]} (%)", min_value=0.0, max_value=100.0, step=1.0, key=weight_key(asset)
    )

weight_sum = sum(raw_weights.values())
if weight_sum <= 0:
    st.error("Enter at least one weight above zero.")
    st.stop()
if abs(weight_sum - 100.0) > 0.01:
    st.warning(
        f"Weights add up to {weight_sum:.1f}%. Calculations use them scaled to 100%."
    )
    st.button("Normalise inputs to 100%", on_click=normalise_inputs)
weights = {a: raw_weights[a] / weight_sum for a in ASSET_CLASSES}
portfolio = Portfolio(weights=weights, total_value=total_value)

alloc_df = pd.DataFrame(
    {"Weight (%)": [weights[a] * 100 for a in ASSET_CLASSES if weights[a] > 1e-6]},
    index=[ASSET_LABELS[a] for a in ASSET_CLASSES if weights[a] > 1e-6],
)
st.bar_chart(alloc_df, y_label="Allocation (%)")

# ---------------------------------------------------------------------------
# 2. Scenarios
# ---------------------------------------------------------------------------
st.header("2. Scenarios")

with st.expander("Describe your own event (AI proposes, code checks)", expanded=True):
    event_text = st.text_area("Event in plain English", value=EXAMPLE_EVENT, height=70)
    add_col, clear_col = st.columns([1, 4])
    if add_col.button("Generate scenarios", type="primary"):
        with st.spinner("Asking the model, then running the guardrail..."):
            generated = pipeline.scenarios_from_text(event_text)
        known = {s.name for s in state.llm_scenarios}
        fresh = [s for s in generated if s.name not in known]
        state.llm_scenarios.extend(fresh)
        state["chosen"] = list(state.get("chosen", [])) + [s.name for s in fresh]
        if any("FALLBACK" in s.name for s in generated):
            st.warning("The model was unavailable, so a labelled placeholder scenario was added instead.")
    if state.llm_scenarios and clear_col.button("Remove generated scenarios"):
        removed = {s.name for s in state.llm_scenarios}
        state.llm_scenarios = []
        state["chosen"] = [n for n in state.get("chosen", []) if n not in removed]
        st.rerun()

library = {s.name: s for s in pipeline.historical_scenarios()}
for scenario in state.llm_scenarios:
    library[scenario.name] = scenario

if "chosen" not in state:
    state["chosen"] = [n for n in DEFAULT_SCENARIOS if n in library] or list(library)[:3]
state["chosen"] = [n for n in state["chosen"] if n in library]
chosen = st.multiselect("Scenarios to test", list(library), key="chosen")

SOURCE_LABEL = {"historical": "Historical", "custom": "Forward-looking", "llm": "AI-proposed"}
scenarios = []
for name in chosen:
    base = library[name]
    with st.expander(f"{name}  ·  {SOURCE_LABEL.get(base.source, base.source)}"):
        st.write(base.description)
        p_col, s_col = st.columns(2)
        probability = p_col.slider(
            "Probability within 12 months (%)", 0.0, 50.0, min(50.0, round(float(base.probability) * 100, 1)), 0.1,
            key=f"p::{name}",
        )
        severity = s_col.slider(
            "Severity (0.5 mild, 1.0 base, 2.0 extreme)", 0.5, 2.0, 1.0, 0.1, key=f"s::{name}"
        )
        scenario = dataclasses.replace(base, probability=probability / 100, severity=severity)
        shocks = scenario.effective_shocks()
        st.dataframe(
            pd.DataFrame({"Shock": [pct(shocks[a]) for a in ASSET_CLASSES]}, index=[ASSET_LABELS[a] for a in ASSET_CLASSES]).T,
            width="stretch",
        )
        proposal_text, mark, guardrail_text = base.rationale.partition(GUARDRAIL_MARK)
        if proposal_text.strip():
            st.caption(proposal_text.strip())
        if mark:
            st.info(f"Guardrail decision: {guardrail_text.strip()}")
    scenarios.append(scenario)

if not scenarios:
    st.info("Select at least one scenario to run the stress test.")
    st.stop()

scenarios, probability_warning = pipeline.safe_scenarios(scenarios)
if probability_warning:
    st.warning(probability_warning)

# ---------------------------------------------------------------------------
# 3. Stress test
# ---------------------------------------------------------------------------
st.header("3. Stress test")
try:
    report = pipeline.stress_test(portfolio, scenarios)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Expected return (all outcomes)", pct(report.expected_return))
m2.metric("If nothing happens", pct(report.baseline_return), f"probability {pct(report.baseline_probability)}", delta_color="off")
m3.metric("Worst case", pct(report.worst_case_return), report.worst_case_scenario, delta_color="off")
m4.metric("Worst-case loss", money(report.worst_case_return * total_value))

table = pd.DataFrame(
    [
        {
            "Scenario": r.scenario_name,
            "Probability": pct(r.probability),
            "Portfolio return": pct(r.portfolio_return),
            "Gain / loss": money(r.pnl_value),
        }
        for r in sorted(report.results, key=lambda r: r.portfolio_return)
    ]
)
st.dataframe(table, hide_index=True, width="stretch")

st.subheader("Where the loss comes from")
contributions = pd.DataFrame(
    {ASSET_LABELS[a]: [r.contribution_by_asset.get(a, 0.0) * 100 for r in report.results] for a in ASSET_CLASSES},
    index=[r.scenario_name for r in report.results],
)
st.bar_chart(contributions, y_label="Contribution to portfolio return (percentage points)")

sens: dict[float, float] | None = None
try:
    sens = pipeline.sensitivity(portfolio, scenarios)
except ValueError as exc:
    st.error(str(exc))
if sens:
    with st.expander("Severity sensitivity (0.5x to 2.0x)"):
        sens_df = pd.DataFrame(
            [
                {
                    "Severity multiplier": f"{m:.1f}x",
                    "Worst-case return": pct(ret),
                    "Worst-case P&L": money(ret * total_value),
                }
                for m, ret in sens.items()
            ]
        )
        st.dataframe(sens_df, hide_index=True, width="stretch")


def explanation_block(key: str, proposal=None) -> None:
    if st.button("Explain in plain English", key=f"btn::{key}"):
        with st.spinner("Writing the explanation (numbers come from the code)..."):
            state.explanations[key] = pipeline.explain(portfolio, proposal.before if proposal else report, proposal)
    if key in state.explanations:
        st.success(state.explanations[key])


explanation_block("stress")

# ---------------------------------------------------------------------------
# 4. Prepare
# ---------------------------------------------------------------------------
st.header("4. Prepare")
st.write(
    "The optimizer looks for the weights with the best expected return that keep the loss "
    "in every selected scenario under your limit."
)
c1, c2, c3 = st.columns(3)
max_loss = c1.slider("Maximum loss in any scenario (%)", 5, 40, 15) / 100
max_turnover = c2.slider("Maximum share of the portfolio to trade (%)", 5, 100, 30) / 100
max_weight = c3.slider("Maximum weight per asset class (%)", 20, 100, 40) / 100

constraints = Constraints(
    max_turnover=max_turnover,
    max_scenario_loss=max_loss,
    # Never make the starting portfolio illegal by itself.
    max_weight={a: max(max_weight, weights[a]) for a in ASSET_CLASSES},
)
signature = repr((sorted(weights.items()), total_value, [(s.name, s.probability, s.severity) for s in scenarios], max_loss, max_turnover, max_weight))

if st.button("Propose allocation", type="primary"):
    try:
        state["proposal"] = (signature, pipeline.prepare(portfolio, scenarios, constraints))
        state.explanations.pop("prepare", None)
    except ValueError as exc:
        st.error(str(exc))

if "proposal" in state:
    saved_signature, proposal = state["proposal"]
    if saved_signature != signature:
        st.warning("Inputs changed since this proposal was made. Click 'Propose allocation' to refresh it.")
    if proposal.feasible:
        st.success(proposal.explanation)
    else:
        st.warning(f"Not every limit could be met. {proposal.explanation}")

    b1, b2, b3 = st.columns(3)
    b1.metric(
        "Worst case",
        pct(proposal.after.worst_case_return),
        f"{(proposal.after.worst_case_return - proposal.before.worst_case_return) * 100:+.1f} pts vs {pct(proposal.before.worst_case_return)}",
    )
    b2.metric(
        "Expected return",
        pct(proposal.after.expected_return),
        f"{(proposal.after.expected_return - proposal.before.expected_return) * 100:+.1f} pts vs {pct(proposal.before.expected_return)}",
    )
    b3.metric(
        "Worst-case loss",
        money(proposal.after.worst_case_return * total_value),
        f"{money((proposal.after.worst_case_return - proposal.before.worst_case_return) * total_value)} saved",
    )

    w_col, t_col = st.columns([3, 2])
    comparison = pd.DataFrame(
        {
            "Current": [(proposal.new_weights.get(a, 0.0) - proposal.trades.get(a, 0.0)) * 100 for a in ASSET_CLASSES],
            "Proposed": [proposal.new_weights.get(a, 0.0) * 100 for a in ASSET_CLASSES],
        },
        index=[ASSET_LABELS[a] for a in ASSET_CLASSES],
    )
    try:
        w_col.bar_chart(comparison, stack=False, y_label="Weight (%)")
    except TypeError:
        w_col.bar_chart(comparison)
    trades = pd.DataFrame(
        [
            {
                "Asset class": ASSET_LABELS[a],
                "Action": "Buy" if t > 0 else "Sell",
                "Change": f"{t * 100:+.1f} pts",
                "Amount": money(abs(t) * total_value),
            }
            for a, t in sorted(proposal.trades.items(), key=lambda kv: kv[1])
            if abs(t) >= 0.0005
        ]
    )
    t_col.subheader("Trades")
    if trades.empty:
        t_col.write("No trades: the current portfolio is kept.")
    else:
        t_col.dataframe(trades, hide_index=True, width="stretch")

    explanation_block("prepare", proposal)

active_proposal = (
    state["proposal"][1]
    if "proposal" in state and state["proposal"][0] == signature
    else None
)
active_explanation = state.explanations.get("prepare") or state.explanations.get("stress")
st.download_button(
    "Download report (Markdown)",
    data=_build_markdown_report(
        portfolio,
        scenarios,
        report,
        sensitivity=sens,
        proposal=active_proposal,
        explanation=active_explanation,
    ),
    file_name="stress_test_report.md",
    mime="text/markdown",
)

# ---------------------------------------------------------------------------
# 5. Audit trail
# ---------------------------------------------------------------------------
st.header("5. Audit trail")
st.caption("What the model proposed and what the code decided. Newest last.")
records = pipeline.audit_tail(15)
if records:
    st.dataframe(pd.DataFrame(records).astype(str), hide_index=True, width="stretch")
else:
    st.write("Nothing recorded yet. Generate a scenario or propose an allocation.")
