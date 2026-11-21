# Team Plan: Portfolio Stress Test and Allocation Tool

> **Update:** the roles in section 5 are finished or superseded. For the final sprint, read [NEXT_STEPS.md](NEXT_STEPS.md) on the `Development` branch.

**Build window: 2 hours. Five roles. One person per role.**

This file is the single source of truth for the build. It is written so that each teammate can hand it to their AI coding assistant.

---

## 0. If you are an AI assistant, start here

1. Read this whole file before writing any code.
2. See which roles are already taken:
   ```bash
   git fetch origin
   git ls-remote --heads origin "role/*"
   ```
   A role is taken if its branch exists on the remote.
3. Show your human the roles that are still free (section 5) and ask which one they want. Do not choose for them.
4. Claim the role by pushing its branch. The push is the claim:
   ```bash
   git switch main && git pull origin main
   git switch -c role/3-engine          # use the branch name of the chosen role
   git push -u origin role/3-engine
   ```
   If the push is rejected because the branch already exists, someone else got there first. Pick another role.
5. Work only in the files your role owns. Never edit a file owned by another role. If you need a change in `contracts.py`, `pipeline.py`, `mock.py` or `requirements.txt`, ask the Lead (role 1).
6. Build against the function signatures that already exist as stubs in the repo. Do not rename them or change their arguments.
7. Push small commits often. Open a pull request into `main` as soon as the first deliverable works, not at the end. Open follow-up pull requests for later deliverables.
8. Before every pull request, run:
   ```bash
   PYTHONPATH=src python -m unittest discover -s tests -v
   ```
   Windows PowerShell: `$env:PYTHONPATH = "src"; python -m unittest discover -s tests -v`
9. Never commit secrets. API keys live in a local `.env` file, which Git ignores.
10. Write everything in English.

---

## 1. What we are building

A Streamlit app that answers three questions about a portfolio:

1. **What do I own?** The user sets weights across nine asset classes.
2. **What happens if things go wrong?** The user picks several shock scenarios. Each scenario has a **probability** and a **severity**. The app shows the loss in every scenario and the probability-weighted outcome.
3. **How should I prepare?** An optimizer proposes new weights that keep the best expected return while capping the loss in every scenario. The app shows before and after.

Scenarios come from two places: a built-in library of historical crises, and free text ("oil jumps 40% and the Fed hikes 100 basis points") that an LLM turns into structured scenarios.

### The rule we follow: the model proposes, the code decides

This matches the principle in `requirements-AG.txt`.

- The LLM only **proposes** scenarios (shock sizes, probabilities, a rationale) and **writes explanations**.
- Deterministic Python **validates** every LLM proposal, computes every number, and runs the optimizer.
- The LLM never does arithmetic that reaches the screen. Numbers in explanations are inserted by code.

This is also our demo story: show an LLM proposal being corrected by the guardrail.

---

## 2. The model

All values are decimals: `-0.35` means minus 35%. The horizon is 12 months.

**Asset classes** (fixed list, in `contracts.ASSET_CLASSES`):
`us_equity`, `intl_equity`, `em_equity`, `gov_bonds`, `corp_credit`, `gold`, `commodities`, `real_estate`, `cash`

**Scenario.** Each scenario has:
- `shocks`: the base-case return of every asset class if the event happens.
- `severity`: a multiplier on the shocks. 0.5 is mild, 1.0 is base, 1.5 is severe.
- `probability`: the chance the event happens within the horizon.

Always read shocks through `scenario.effective_shocks()`. It applies severity and floors losses at minus 100%.

**Baseline.** If no scenario happens, each asset class earns its baseline expected return. The baseline has probability `1 - sum(scenario probabilities)`. The scenario probabilities must sum to 1.0 or less; scenarios are treated as mutually exclusive.

**Stress test.** For portfolio weights `w` and scenario `s`:

```
contribution[asset]  = w[asset] * effective_shock_s[asset]
portfolio_return_s   = sum(contribution)
pnl_value_s          = portfolio_return_s * total_value
baseline_return      = sum(w[asset] * baseline_return[asset])
expected_return      = sum(p_s * portfolio_return_s) + p_baseline * baseline_return
probability_weighted_loss = sum(p_s * max(0, -portfolio_return_s))
worst case           = scenario with the lowest portfolio_return
```

**Optimizer.** A linear program over the new weights `x`:

```
maximise    expected_return(x)
subject to  sum(x) = 1
            min_weight[a] <= x[a] <= max_weight[a]        (defaults 0.0 and 0.60)
            portfolio_return_s(x) >= -max_scenario_loss   for every scenario s
            sum(|x - w|) / 2 <= max_turnover
```

If the constraints cannot all be met, solve a second problem instead (minimise the worst-case loss within the turnover and weight limits) and return it with `feasible=False`.

Why this shape: minimising loss in one scenario alone would just move everything to cash. Probabilities give the cost of hedging, and the loss cap gives the protection.

---

## 3. Contracts (frozen)

The dataclasses live in `src/money_talks/contracts.py`. Read that file; it is the source of truth.

| Contract | Purpose |
|---|---|
| `Portfolio` | `weights`, `total_value` |
| `Scenario` | `name`, `description`, `shocks`, `probability`, `severity`, `source`, `rationale`, `effective_shocks()` |
| `ScenarioResult` | one scenario applied to one portfolio |
| `StressReport` | all `ScenarioResult`s plus baseline, expected return, worst case |
| `Constraints` | `max_turnover`, `max_scenario_loss`, `min_weight`, `max_weight` |
| `AllocationProposal` | `new_weights`, `trades`, `before`, `after`, `feasible`, `explanation` |

Only the Lead edits `contracts.py`. If you think a contract is wrong, tell the Lead in the team chat first.

---

## 4. How the pieces connect

```
app.py (role 5)
   |  imports only money_talks.pipeline
   v
pipeline.py (role 1)  -- tries the real function, falls back to mock.py --
   |-- data/source.py        (role 2)  baseline returns, scenario library, presets
   |-- finance/evaluator.py  (role 3)  stress engine, optimizer
   |-- ai/analyzer.py        (role 4)  LLM scenarios, guardrail, explanation
```

`pipeline.py` already works today using `mock.py`. Every real function currently raises `NotImplementedError`; as soon as a role replaces the stub and the pull request merges, the app uses the real version automatically. Nobody has to wait for anybody.

Mock numbers are illustrative placeholders. The mock "optimizer" is not an optimizer.

---

## 5. The five roles

### Role 1: Lead and integrator

- **Branch:** `role/1-lead`
- **Owns:** `src/money_talks/contracts.py`, `src/money_talks/pipeline.py`, `src/money_talks/mock.py`, `requirements.txt`, `README.md`, `tests/test_stress_contracts.py`, this file
- **Deliverables:**
  1. Review and merge pull requests quickly. Aim for under five minutes each. Keep `main` runnable at all times.
  2. Answer contract questions. Change contracts only when truly needed, and announce every change.
  3. Add PRISM tracing around the LLM calls and the guardrail decision (new file `src/money_talks/telemetry.py`), using the `PRISMTRACE_*` variables in `.env.example`. Tracing must never break the app if PRISM is unreachable.
  4. Write the two-minute demo script (section 8) and run the rehearsal.
  5. After each merge, run the app end to end and report what broke.
- **Done when:** `main` runs the full demo with no mock output left.

### Role 2: Scenario library and data

- **Branch:** `role/2-scenarios`
- **Owns:** `src/money_talks/data/source.py`, new `tests/test_source.py`
- **Deliverables** (replace the three stubs at the bottom of `source.py`):
  1. `get_baseline_returns()`: one expected 12-month return for each of the nine asset classes.
  2. `get_historical_scenarios()`: at least six `Scenario` objects with `source="historical"`. Suggested: 2008 global financial crisis, 2020 pandemic crash, 2022 inflation and rate shock, 1970s stagflation, 2000 dot-com bust, 2011 euro sovereign debt crisis, 1997 emerging markets crisis. Each needs shocks for all nine asset classes, a one-line description, a default probability, and a rationale that says where the numbers come from.
  3. `get_preset_portfolios()`: at least four named portfolios, for example Balanced 60/40, Conservative, Aggressive growth, All-weather.
- **Rules:** Every `shocks` dict has exactly the nine keys. Preset weights sum to 1.0. Default probabilities across the whole library sum to less than 1.0. Be honest in `rationale`: if a number is an approximation, say so. Do not fetch data from the network.
- **Tests:** keys are complete, weights sum to 1.0, shocks are within -1.0 and +1.0, probabilities are within 0 and 1.
- **Stretch:** add forward-looking scenarios (for example an AI equity bubble unwinding, a Taiwan conflict, a US debt crisis) with `source="custom"`.

### Role 3: Stress engine and optimizer

- **Branch:** `role/3-engine`
- **Owns:** `src/money_talks/finance/evaluator.py`, new `tests/test_evaluator.py`
- **Deliverables** (replace the two stubs at the bottom of `evaluator.py`):
  1. `run_stress_test(portfolio, scenarios, baseline_returns) -> StressReport`, using the formulas in section 2. Open a pull request as soon as this works.
  2. `propose_allocation(portfolio, scenarios, baseline_returns, constraints) -> AllocationProposal`, using the linear program in section 2. Use `scipy.optimize.linprog`. Model turnover with helper variables `t[a] >= |x[a] - w[a]|`. Fill `before`, `after`, `trades`, `feasible`, and a one-sentence factual `explanation`.
- **Rules:** Pure functions, no network, no LLM. Handle an empty scenario list, probabilities that sum to more than 1.0 (raise `ValueError` with a clear message), and weights that do not sum to 1.0 (raise `ValueError`). Round tiny solver noise to zero.
- **Tests:** a hand-computed stress example; an all-cash portfolio loses nothing in a scenario where the cash shock is zero; optimizer output sums to 1.0 and respects every constraint; the infeasible case returns `feasible=False`.
- **Stretch:** sensitivity table (worst-case loss as severity goes from 0.5 to 2.0); the efficient trade-off curve of expected return against `max_scenario_loss`.

### Role 4: LLM scenarios, guardrail and explanations

- **Branch:** `role/4-llm`
- **Owns:** `src/money_talks/ai/analyzer.py`, new `tests/test_analyzer.py`
- **LLM access:** an OpenAI-compatible chat endpoint. Read `NICRRON_BASE_URL`, `NICRRON_MODEL` and `NICRRON_API_KEY` from the environment (local `.env`). Ask the Lead for the values. Use the standard library (`urllib`) or ask the Lead to add a client package to `requirements.txt`.
- **Deliverables** (replace the three stubs at the bottom of `analyzer.py`):
  1. `validate_scenario(scenario) -> (clean_scenario, fixes)`: the deterministic guardrail. Fill missing asset classes with 0.0, drop unknown keys, clamp each shock to between -0.80 and +1.00, clamp probability to between 0.001 and 0.50, clamp severity to between 0.25 and 2.0. Return a human-readable list of every fix made. Do this first; it needs no LLM.
  2. `generate_scenarios(text) -> list[Scenario]`: ask the LLM for strict JSON describing the event as three scenarios (mild, base, severe), each with shocks for all nine asset classes, a probability, and a short rationale. Parse, pass each through `validate_scenario`, set `source="llm"`, and append the fixes to `rationale` so the UI can show them. The three probabilities together must not exceed 0.50; scale them down if they do.
  3. `explain_report(portfolio, report, proposal=None) -> str`: three or four plain-English sentences. Build the numbers into the prompt from the report and tell the model not to invent any. If the LLM fails, return a template sentence built from the report.
- **Rules:** These functions must never raise on a network error, timeout, or bad JSON. On failure, `generate_scenarios` returns a clearly labelled fallback scenario with `rationale` explaining that the LLM was unavailable. Use a timeout of 20 seconds. Cache results in memory by input text so the demo can replay instantly. Tests must not call the network; test the guardrail and the JSON parsing with canned strings.
- **Stretch:** ask the LLM which historical scenario is the closest analogue and include it in the rationale.

### Role 5: Streamlit app

- **Branch:** `role/5-ui`
- **Owns:** new `app.py` at the repository root, new `.streamlit/` folder if needed
- **Run:** `streamlit run app.py`. At the top of `app.py`, add `src` to `sys.path` so it works without setting `PYTHONPATH`.
- **Import only from `money_talks.pipeline` and `money_talks.contracts`.** Never import `mock`, `source`, `evaluator` or `analyzer` directly. The pipeline works today with mock data, so start building immediately.
- **Deliverables:**
  1. **Portfolio panel:** choose a preset (`pipeline.preset_portfolios()`), adjust nine weights, enter the total value. Show the sum and offer a "normalise to 100%" button. Show a pie or bar chart.
  2. **Scenarios panel:** multi-select from `pipeline.historical_scenarios()`. A text box plus button calls `pipeline.scenarios_from_text(text)` and adds the results. For every selected scenario, show a probability slider and a severity slider (0.5 to 2.0), and show the rationale. Warn if probabilities sum to more than 100%.
  3. **Stress test panel:** call `pipeline.stress_test(portfolio, scenarios)`. Show headline numbers (expected return, worst-case loss in percent and in money, probability-weighted loss), a table with one row per scenario, and a stacked bar chart of loss contribution by asset class per scenario. Show `pipeline.explain(...)` underneath.
  4. **Prepare panel:** sliders for maximum turnover and maximum loss per scenario. Call `pipeline.prepare(portfolio, scenarios, constraints)`. Show current against proposed weights side by side, the trade list, and before against after for worst-case loss and expected return. If `feasible` is false, say so clearly.
- **Rules:** Use `st.session_state` to keep scenarios between reruns. Wrap pipeline calls in `try/except` and show errors with `st.error` rather than crashing. Show percentages as percentages. Add one line of small print: "Illustrative model, not investment advice."
- **Stretch:** a "what the guardrail changed" box for LLM scenarios; a download button for the report.

---

## 6. Timeline

| Time | What happens |
|---|---|
| 0:00 to 0:10 | Everyone reads this file, claims a role, creates the branch. |
| 0:10 to 0:45 | First deliverable of every role. Open the first pull request by 0:45. |
| 0:45 to 1:15 | Second and third deliverables. Lead merges continuously. |
| 1:15 to 1:35 | Integration on `main`. Fix what breaks. No new features. |
| 1:35 to 1:50 | Rehearse the demo twice. Fix only what the rehearsal exposes. |
| 1:50 to 2:00 | Freeze. Nothing merges. |

## 7. If we run out of time, cut in this order

1. PRISM tracing
2. LLM explanations (keep text to scenarios)
3. Per-asset minimum and maximum weights in the UI
4. The infeasible-case fallback in the optimizer
5. Extra historical scenarios beyond three

Never cut: the stress test table, the probability and severity sliders, the before and after comparison.

## 8. Demo script (two minutes)

1. **Context (20 seconds).** "Most portfolios are built for normal times. We ask what happens when times are not normal, and what to do about it."
2. **Stress (40 seconds).** Load Balanced 60/40. Select three historical crises. Move a severity slider and a probability slider and watch the numbers change.
3. **AI proposes, code decides (30 seconds).** Type an event in plain English. The LLM proposes mild, base and severe scenarios. Show a value the guardrail corrected.
4. **Prepare (30 seconds).** Set the maximum loss to 15%. The optimizer proposes new weights. Show worst-case loss before and after, and what the protection costs in expected return.

## 9. Ground rules

- One owner per file. If two people need the same file, talk first.
- `main` must always run. Do not merge a pull request with failing tests.
- Small pull requests, merged often.
- Ask in the team chat the moment you are blocked for more than five minutes.
