# Money Talks — Portfolio Stress Test and Allocation Tool

A Streamlit application that answers three questions about a multi-asset portfolio following **"the model proposes, the code decides"**:

1. **What do I own?** Configure portfolio weights and dollar value across 9 asset classes (`us_equity`, `intl_equity`, `em_equity`, `gov_bonds`, `corp_credit`, `gold`, `commodities`, `real_estate`, `cash`) or load a preset.
2. **What happens if things go wrong?** Stress-test against historical crises, forward-looking scenarios, or plain-English shocks parsed by an LLM and checked by a deterministic guardrail.
3. **How should I prepare?** Run a linear-program optimizer (`scipy.optimize.linprog`) to maximize expected return while capping worst-case scenario loss and turnover (with a best-effort fallback when constraints are infeasible).

---

## Project Structure

```text
app.py                              # Role 5: Streamlit UI (imports only pipeline & contracts)
.streamlit/config.toml              # Role 5: Streamlit theme config
src/money_talks/
  contracts.py                      # Role 1: Frozen shared dataclasses and asset constants
  pipeline.py                       # Role 1: Facade connecting modules + PRISM tracing
  telemetry.py                      # Role 1: Non-blocking PRISM/local span & evidence hash logger
  mock.py                           # Role 1: Fallback stand-ins
  data/source.py                    # Role 2: Baseline returns, historical & custom scenarios, presets
  finance/evaluator.py              # Role 3: Deterministic stress test & HiGHS LP optimizer
  ai/analyzer.py                    # Role 4: OpenAI-compatible LLM adapter, guardrail & explanations
tests/
  test_scaffold.py                  # Scaffold placeholder smoke tests
  test_stress_contracts.py          # End-to-end contract & pipeline smoke tests
  test_source.py                    # Role 2 data & library validation tests
  test_evaluator.py                 # Role 3 stress math & LP optimizer tests
  test_analyzer.py                  # Role 4 guardrail, JSON parsing & offline fallback tests
  test_telemetry.py                 # Role 1 non-blocking telemetry tests
```

---

## Local Setup

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Environment Variables (optional for live LLM & PRISM)

Copy `.env.example` to `.env` (git-ignored) and fill in credentials if running with live services:

- **LLM (`NICRRON_*`):** `NICRRON_BASE_URL`, `NICRRON_MODEL`, `NICRRON_API_KEY` (if unset or unreachable, the app falls back safely to deterministic scenarios and template explanations).
- **PRISM (`PRISMTRACE_*`):** `PRISMTRACE_HOST`, `PRISMTRACE_PROJECT_ID`, `PRISMTRACE_API_KEY` (tracing never interrupts the app if PRISM is unset or unreachable).

---

## Run & Verify

Run the full test suite (all tests are offline and deterministic):

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m money_talks.main
```

Launch the Streamlit app:

```bash
streamlit run app.py
```

---

## Two-Minute Demo Flow

1. **Context:** Load **Balanced 60/40** in the sidebar and review current asset weights.
2. **Stress:** Select historical crises (e.g., 2008 GFC, 2022 Inflation & Rate Shock, 2020 Pandemic Crash) and adjust severity and probability sliders to inspect expected return, worst-case drawdown, and stacked asset contributions.
3. **AI proposes, code decides:** Enter an event in plain English (e.g., `"Oil jumps 40% and the Fed hikes 100 basis points"`), inspect mild/base/severe proposals, and view **"What the guardrail changed"**.
4. **Prepare:** Switch to **Prepare & Rebalance**, set maximum loss per scenario to `15%`, and compare before/after worst-case loss, expected return, and proposed trades.

---

## Security & Event Rules

- Never commit API keys, `.env`, or secrets (`.gitignore` excludes `.env`, `.venv`, and `.streamlit/secrets.toml`).
- Write code, comments, commit messages, and documentation in **English**.
