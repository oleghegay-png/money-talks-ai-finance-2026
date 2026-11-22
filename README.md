# Money Talks — Portfolio Stress Test and Allocation Tool

**Live demo:** https://money-talks-ai-finance-2026-ci3sbjwkq29hw8z6ce4udp.streamlit.app/

A Streamlit application that answers three questions about a multi-asset portfolio following **"the model proposes, the code decides"**:

1. **What do I own?** Configure portfolio weights and total value across 9 asset classes (`us_equity`, `intl_equity`, `em_equity`, `gov_bonds`, `corp_credit`, `gold`, `commodities`, `real_estate`, `cash`) or load a preset.
2. **What happens if things go wrong?** Stress-test against historical crises, forward-looking scenarios, or plain-English shocks parsed by an LLM and validated by a deterministic guardrail.
3. **How should I prepare?** Run a linear-program optimizer (`scipy.optimize.linprog`) to maximize expected return while capping worst-case scenario loss and turnover (with a best-effort fallback when constraints are infeasible).

For architecture, frozen contracts, and original role assignments, see [TEAM_PLAN.md](TEAM_PLAN.md).

---

## Local setup

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the stress-test app

Both `main` and `Development` are kept in sync with the live deployment:

```bash
streamlit run app.py
```

For live LLM scenarios, copy `.env.example` to `.env` and fill in `NICRRON_API_KEY` (or paste your key in the app sidebar). Without a key, the app still runs using pre-warmed demo scenarios (`demo_cache.json`) and clearly labelled deterministic fallback scenarios. Set `MONEY_TALKS_LLM_OFFLINE=1` to skip the network on purpose.

The app writes `audit_log.jsonl` (ignored by Git): one line for every scenario the model proposed, what the guardrail changed, and every allocation decision.

### PRISM tracing

Every LLM call is also sent to PRISM (Block Convey) as one trace holding the prompts, the model's reply, the latency, and what the deterministic guardrail decided. To switch it on, fill in `PRISMTRACE_PROJECT_ID` and `PRISMTRACE_API_KEY` in `.env` (create both in the PRISM web app). Tracing is off when either is empty or when `MONEY_TALKS_LLM_OFFLINE=1`, and the app behaves the same either way: a PRISM outage only prints a warning. `pipeline.prism_status()` returns one line saying whether traces are being sent. The code is in `src/money_talks/telemetry.py`.

To prove it works end to end, run `PYTHONPATH=src python tools/prism_check.py` (PowerShell: `$env:PYTHONPATH = "src"; python tools/prism_check.py`). It checks the key, sends one trace through the app's own code, and asks PRISM whether the trace arrived. For proof with real AI calls, run `PYTHONPATH=src python tools/prism_e2e.py`: it generates scenarios and an explanation with a live LLM call, checks that PRISM acknowledged both traces, and appends the trace ids to [docs/prism-evidence.md](docs/prism-evidence.md).

All shock numbers are illustrative approximations. This is not investment advice.

## Host it on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with a GitHub account that can read this repository.
2. Create an app: repository `oleghegay-png/money-talks-ai-finance-2026`, branch `Development` (or `main`), main file `app.py`.
3. Optional, under Advanced settings > Secrets, add `NICRRON_API_KEY = "..."` so visitors can use the AI without their own key. Anyone with the link then spends that key's balance. Leave it out and visitors paste their own key in the sidebar.
4. Optional, in the same Secrets box, add `PRISMTRACE_PROJECT_ID = "..."` and `PRISMTRACE_API_KEY = "..."` to send traces to PRISM from the hosted app. The sidebar then shows "PRISM tracing: On". Visitors' event descriptions are sent to that PRISM project.

Everything except live AI scenario generation works with no key at all.

## Two-minute demo flow

1. **Context:** Load **Balanced 60/40** in Section 1 and inspect the portfolio allocation chart.
2. **Stress:** Select three crises (e.g., 2008 GFC, 2022 Inflation & Rate Shock, AI Bubble Unwind), move the probability and severity sliders, and review expected return, worst-case drawdown, asset contributions, and the 0.5x–2.0x severity sensitivity table.
3. **AI proposes, code decides:** Enter a plain-English event, generate mild/base/severe scenarios, and inspect the guardrail decision and audit trail.
4. **Prepare:** Set maximum scenario loss to `15%`, click **Propose allocation**, compare before/after worst-case loss and expected return alongside proposed trades, and optionally download the Markdown report.

## Verify & test

macOS / Linux:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m money_talks.main
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m money_talks.main
```

## Project structure

```text
app.py                          # Streamlit UI (imports only pipeline & contracts)
demo_cache.json                 # Pre-warmed LLM scenario cache for offline/live demo
requirements.txt                # Runtime dependencies
TEAM_PLAN.md                    # Team roles, model math, and frozen contract spec
src/money_talks/
  contracts.py                  # Shared dataclasses and frozen asset constants
  pipeline.py                   # Pipeline facade, probability safety net, audit log
  telemetry.py                  # PRISM tracing of LLM calls and guardrail decisions
  mock.py                       # Deterministic fallback stand-ins
  main.py                       # Local CLI entry point
  data/source.py                # Baseline returns, scenario library, preset portfolios
  ai/analyzer.py                # OpenAI-compatible LLM adapter, guardrail & explanations
  finance/evaluator.py          # Stress engine, severity sensitivity & HiGHS LP optimizer
tools/
  prism_check.py                # PRISM connectivity verification script
  prism_e2e.py                  # End-to-end live LLM + PRISM verification script
docs/
  prism-evidence.md             # Recorded live PRISM trace verification evidence
tests/
  test_analyzer.py              # Guardrail, JSON parsing, cache & offline fallback tests
  test_evaluator.py             # Stress engine, LP optimizer & sensitivity tests
  test_scaffold.py              # Initial scaffold interface tests
  test_source.py                # Scenario library, baseline & preset validation tests
  test_stress_contracts.py      # End-to-end pipeline, safety net & audit trail tests
  test_telemetry.py             # Non-blocking PRISM telemetry tests
```

## Security and event rules

- Never commit API keys, `.env`, or personal credentials (`.gitignore` excludes `.env`, `.venv`, `.streamlit/secrets.toml`, and `audit_log.jsonl`); share only variable names in `.env.example`.
- Do not commit private, licensed, or restricted datasets unless event rules explicitly permit it.
- Write all code, comments, commit messages, and documentation in **English**.
