# Next Steps: the last 30 minutes

**Read this instead of the role section of `TEAM_PLAN.md`.** The model, contracts and ground rules in `TEAM_PLAN.md` still apply. Work from the `Development` branch, not `main`.

## 0. If you are an AI assistant, start here

1. `git fetch origin && git switch Development && git pull origin Development`
2. Read this file. Check which sprint roles are taken: `git ls-remote --heads origin "sprint/*"`
3. Show your human the free roles in section 3 and ask which one they want. **Role A is the only one that blocks the demo. If nobody has it, take it.**
4. Claim by pushing the branch: `git switch -c sprint/a-app && git push -u origin sprint/a-app`
5. Edit only the files your role owns. Keep existing function names and arguments.
6. Run `PYTHONPATH=src python -m unittest discover -s tests` (PowerShell: `$env:PYTHONPATH = "src"; python -m unittest discover -s tests`).
7. When tests pass, merge into `Development` yourself and push. Do not wait for review:
   ```bash
   git switch Development && git pull origin Development
   git merge --no-ff sprint/a-app && git push origin Development
   ```
8. Merge early. A small thing on `Development` at minute 12 beats a big thing at minute 29.

## 1. Where we stand (verified on `Development`)

| Piece | Status |
|---|---|
| Contracts, pipeline facade, mock fallback | Done |
| Scenario library: 8 historical, 5 forward-looking, 6 preset portfolios | Done |
| Stress engine and optimizer | Done |
| LLM scenarios, guardrail, explanations | Done, works live |
| **Streamlit app** | **Missing. Nothing to present without it.** |
| PRISM tracing | Not started |

Checks run: 40 unit tests pass. End to end through `money_talks.pipeline` with no mock output left: every preset against every scenario, optimizer constraints (weights sum to 1, bounds, turnover, loss cap) on 18 combinations, bad inputs rejected with clear `ValueError`s, guardrail against malformed LLM output, offline fallback, and a live LLM run.

## 2. Issues found

Ordered by how likely they are to hurt the demo.

1. **There is no UI.** Role A.
2. **The LLM call is too close to its timeout.** Live scenario generation took 16.7 seconds; the timeout is 20. It will sometimes fall back to the placeholder scenario on stage. The cache is in memory only, so it is lost on restart. Role D.
3. **Probabilities can exceed 100% and crash the stress test.** The library sums to 27%. One LLM event adds up to 50%. A second LLM event pushes the total past 100% and the engine raises `ValueError`. Role E adds a safe wrapper; Role A must show the error instead of crashing.
4. **LLM probabilities are too high.** The live run gave 12%, 25% and 8% to the three variants of one event. Selected with the library, the expected return of Balanced 60/40 became -9.7%. Role D lowers the cap.
5. **The optimizer concentrates.** With the default 60% cap per asset class it moved Balanced 60/40 to 30% gold. It is correct maths but looks odd. Role C lowers the default cap and Role B exposes it.
6. **Selecting all 13 scenarios gives Balanced 60/40 an expected return of about 0%.** Scenarios are mutually exclusive, so 13 selected means a 27% chance of a crisis. In the demo, select three or four.
7. **All shock numbers are LLM-drafted approximations.** Say "illustrative" on screen.
8. `main` is stale. Pull requests 1 and 2 are open. We present from `Development`.

## 3. Sprint roles

One owner per file. Five roles, five people.

### Role A: the app (critical path)

- **Branch:** `sprint/a-app`  **Owns:** `app.py` (repository root)
- Build one page, top to bottom. Import only `money_talks.pipeline` and `money_talks.contracts`.
- Order of work. Merge after each step:
  1. Portfolio: preset selector, nine weight inputs, total value, "normalise to 100%" button.
  2. Scenarios: multi-select from the library (default: three selected), with a probability slider and a severity slider (0.5 to 2.0) per selected scenario.
  3. Stress results: headline metrics, a table with one row per scenario, a bar chart.
  4. Text box and button for `pipeline.scenarios_from_text`; keep results in `st.session_state`; show each `rationale` (the part after `Guardrail:` is the demo moment).
  5. Call Role B's panel: `from money_talks.ui_prepare import render_prepare` inside `try/except ImportError`.
- Wrap every pipeline call in `try/except ValueError` and show `st.error(str(exc))`.
- Starter that already runs:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).parent / "src"))

  import streamlit as st
  from money_talks import pipeline
  from money_talks.contracts import ASSET_CLASSES, ASSET_LABELS, Portfolio

  st.set_page_config(page_title="Money Talks: Stress Test", layout="wide")
  st.title("Portfolio Stress Test")
  st.caption("Illustrative model, not investment advice.")

  presets = pipeline.preset_portfolios()
  preset = st.selectbox("Starting portfolio", list(presets))
  total = st.number_input("Portfolio value", value=1_000_000.0, step=50_000.0)
  cols = st.columns(3)
  weights = {
      a: cols[i % 3].number_input(ASSET_LABELS[a], 0.0, 100.0, presets[preset].weights[a] * 100, key=f"{preset}-{a}") / 100
      for i, a in enumerate(ASSET_CLASSES)
  }
  portfolio = Portfolio(weights=weights, total_value=total)

  library = {s.name: s for s in pipeline.historical_scenarios()}
  chosen = st.multiselect("Scenarios", list(library), default=list(library)[:3])
  scenarios = []
  for name in chosen:
      s = library[name]
      with st.expander(name):
          st.write(s.description)
          s.probability = st.slider("Probability", 0.0, 0.5, float(s.probability), 0.005, key=f"p-{name}")
          s.severity = st.slider("Severity", 0.5, 2.0, 1.0, 0.1, key=f"s-{name}")
          st.caption(s.rationale)
      scenarios.append(s)

  try:
      report = pipeline.stress_test(portfolio, scenarios)
      a, b, c = st.columns(3)
      a.metric("Expected return", f"{report.expected_return:.1%}")
      b.metric("Worst case", f"{report.worst_case_return:.1%}", report.worst_case_scenario)
      c.metric("Worst-case loss", f"{report.worst_case_return * total:,.0f}")
      st.dataframe([
          {"Scenario": r.scenario_name, "Probability": f"{r.probability:.1%}",
           "Return": f"{r.portfolio_return:.1%}", "P&L": f"{r.pnl_value:,.0f}"}
          for r in report.results
      ])
  except ValueError as exc:
      st.error(str(exc))
  ```
  Run with `streamlit run app.py`.

### Role B: the Prepare panel and charts

- **Branch:** `sprint/b-prepare`  **Owns:** new `src/money_talks/ui_prepare.py`
- Export `render_prepare(portfolio, scenarios) -> None`. Role A calls it at the bottom of the page.
- Inside: sliders for maximum loss per scenario (5% to 40%, default 15%), maximum turnover (5% to 100%, default 30%) and maximum weight per asset class (20% to 100%, default 40%). Build `Constraints(max_turnover=..., max_scenario_loss=..., max_weight={a: cap for a in ASSET_CLASSES})` and call `pipeline.prepare(portfolio, scenarios, constraints)`.
- Show: current against proposed weights as a grouped bar chart, the trade list, worst case and expected return before and after as metrics, a clear warning when `proposal.feasible` is false, and `pipeline.explain(portfolio, proposal.before, proposal)`.
- Test it alone with a ten-line script that calls `render_prepare` on a preset, until Role A's page exists.
- If you finish: a stacked bar chart of `contribution_by_asset` per scenario, as `render_contributions(report)`.

### Role C: engine polish

- **Branch:** `sprint/c-engine`  **Owns:** `src/money_talks/finance/evaluator.py`, `tests/test_evaluator.py`
- 1. Lower the default maximum weight per asset class from 0.60 to 0.40 in `_weight_bounds`, unless the current weight is already higher (then use the current weight, so the starting portfolio is never infeasible by itself).
- 2. Add `severity_sensitivity(portfolio, scenarios, baseline_returns, multipliers=(0.5, 1.0, 1.5, 2.0)) -> dict[float, float]`: worst-case return when every scenario's severity is multiplied by each value. Do not mutate the input scenarios.
- 3. In `propose_allocation`, write a more specific `explanation`: name the two largest trades and the scenario that binds.
- Tell Role E when item 2 is merged so it can be exposed through `pipeline`.

### Role D: LLM reliability

- **Branch:** `sprint/d-llm`  **Owns:** `src/money_talks/ai/analyzer.py`, `tests/test_analyzer.py`, new `demo_cache.json`
- 1. Raise `LLM_TIMEOUT_SECONDS` to 60.
- 2. Persist the scenario cache to `demo_cache.json` in the repository root: load it at import, write after each successful generation. Never fail if the file is missing or corrupt.
- 3. Lower `MAX_TOTAL_LLM_PROBABILITY` from 0.50 to 0.15 and say so in the prompt. Update the tests that assume 0.50.
- 4. Pre-generate and commit the cache for the demo sentences (agree the exact text with Role E): "Oil jumps 40% after a Gulf shipping blockade and the Fed hikes rates by 100 basis points" and "The AI stock bubble bursts and big tech falls by half".
- 5. If time allows: a deliberately bad canned reply behind `MONEY_TALKS_DEMO_BAD_LLM=1` (shocks in percentage points, a missing asset class, probability 0.9), so the guardrail corrections can be shown on demand.

### Role E: Lead

- **Branch:** `sprint/e-lead`  **Owns:** `src/money_talks/pipeline.py`, `contracts.py`, `mock.py`, `requirements.txt`, `README.md`, this file
- 1. Add `pipeline.safe_scenarios(scenarios) -> tuple[list[Scenario], str | None]`: if probabilities sum to more than 0.95, return copies scaled down to 0.95 in total and a message saying so. Tell Role A to call it before `stress_test`.
- 2. Expose `pipeline.sensitivity(...)` when Role C merges it.
- 3. Write a local audit trail instead of PRISM (no PRISM credentials are configured): append one JSON line per LLM proposal and guardrail decision to `audit_log.jsonl`, and add `pipeline.audit_tail(n)` for the UI. Skip this if time is short.
- 4. Update `README.md` with install and run commands.
- 5. Run the full demo from a clean clone of `Development` at minute 20 and again at minute 27.

## 4. Timeline

| Minute | What |
|---|---|
| 0 to 3 | Claim a role. |
| 3 to 18 | Build. Role A merges step 1 to 3 by minute 12. |
| 18 to 24 | Everything merged to `Development`. Fix only what breaks the page. |
| 24 to 30 | Two rehearsals. No merges after minute 27. |

## 5. If we run out of time, cut in this order

1. Audit trail (E3)
2. Bad-LLM demo switch (D5)
3. Sensitivity table (C2, E2)
4. Contribution chart (B extra)
5. Free-text scenarios in the UI (A4). Present the library only.

Never cut: portfolio input, scenario sliders, the stress table, the before and after of the Prepare panel.

## 6. Demo settings that work

Verified numbers on `Development` today:

- Balanced 60/40 with all 13 library scenarios: worst case -26.1% (2008). With a 15% loss cap and 30% turnover the optimizer reaches -13.8% and it is feasible.
- All-weather: worst case -16.1%, reaches -5.4%.
- Aggressive growth: worst case -47.4%; a 15% cap is not reachable within 30% turnover, so it shows the "not feasible, closest attempt" path (-28.1%).
- Doubling severity on 2008 for Balanced 60/40: -26.1% becomes -52.3%.
