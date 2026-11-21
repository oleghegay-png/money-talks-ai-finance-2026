"""Money Talks: portfolio stress test and allocation app.

Run with:  streamlit run app.py

The page imports only money_talks.pipeline and money_talks.contracts. Every
number on screen is computed by deterministic code; the LLM only proposes
scenarios and writes explanations.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from money_talks import pipeline  # noqa: E402
from money_talks.contracts import (  # noqa: E402
    ASSET_CLASSES,
    ASSET_LABELS,
    Constraints,
    Portfolio,
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


def pct(value: float) -> str:
    return f"{value:.1%}"


def money(value: float) -> str:
    return f"{value:,.0f}"


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

# ---------------------------------------------------------------------------
# 2. Scenarios
# ---------------------------------------------------------------------------
st.header("2. Scenarios")

with st.expander("Describe your own event (AI proposes, code checks)", expanded=not state.llm_scenarios):
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
