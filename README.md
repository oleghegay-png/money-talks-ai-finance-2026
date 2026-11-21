# Money Talks — AI × Finance Hackathon

**Live demo:** https://money-talks-ai-finance-2026-ci3sbjwkq29hw8z6ce4udp.streamlit.app/

> **Hackathon build in progress.** We are building a portfolio stress-test and allocation tool.
> Everyone (and their AI assistant) starts with [TEAM_PLAN.md](TEAM_PLAN.md): it holds the plan,
> the frozen contracts, and the five roles.

Shared starter workspace for the three-person team. The repository, documentation, code comments, commit messages, issues, and pull requests should be written in **English** so every teammate can follow the work.

This is a neutral technical scaffold, not a pre-built solution to the challenge. We will adapt the modules after the organizers announce the task, permitted data sources, tools, and PRISM requirements.

## Run the stress-test app

Work from the `Development` branch.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

For live LLM scenarios, copy `.env.example` to `.env` and fill in `NICRRON_API_KEY`. Without it the app still runs and uses a clearly labelled fallback scenario. Set `MONEY_TALKS_LLM_OFFLINE=1` to skip the network on purpose.

The app writes `audit_log.jsonl` (ignored by Git): one line for every scenario the model proposed, what the guardrail changed, and every allocation decision.

### PRISM tracing

Every LLM call is also sent to PRISM (Block Convey) as one trace holding the prompts, the model's reply, the latency, and what the deterministic guardrail decided. To switch it on, fill in `PRISMTRACE_PROJECT_ID` and `PRISMTRACE_API_KEY` in `.env` (create both in the PRISM web app). Tracing is off when either is empty or when `MONEY_TALKS_LLM_OFFLINE=1`, and the app behaves the same either way: a PRISM outage only prints a warning. `pipeline.prism_status()` returns one line saying whether traces are being sent. The code is in `src/money_talks/telemetry.py`.

To prove it works end to end, run `PYTHONPATH=src python tools/prism_check.py` (PowerShell: `$env:PYTHONPATH = "src"; python tools/prism_check.py`). It checks the key, sends one trace through the app's own code, and asks PRISM whether the trace arrived.

All shock numbers are illustrative approximations. This is not investment advice.

## Host it on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with a GitHub account that can read this repository.
2. Create an app: repository `oleghegay-png/money-talks-ai-finance-2026`, branch `Development`, main file `app.py`.
3. Optional, under Advanced settings > Secrets, add `NICRRON_API_KEY = "..."` so visitors can use the AI without their own key. Anyone with the link then spends that key's balance. Leave it out and visitors paste their own key in the sidebar.
4. Optional, in the same Secrets box, add `PRISMTRACE_PROJECT_ID = "..."` and `PRISMTRACE_API_KEY = "..."` to send traces to PRISM from the hosted app. The sidebar then shows "PRISM tracing: On". Visitors' event descriptions are sent to that PRISM project.

Everything except the AI scenario generator works with no key at all.

## Project structure

```text
src/money_talks/
  contracts.py          # shared data structures passed between modules
  pipeline.py           # connects the project stages
  telemetry.py          # PRISM tracing of LLM calls and guardrail decisions
  main.py               # local entry point
  data/source.py        # data input adapter
  ai/analyzer.py        # AI / ML analysis adapter
  finance/evaluator.py  # financial evaluation and metrics

tests/
  test_scaffold.py      # smoke tests for the shared interfaces
```

## Local setup

Use Python 3.11 or newer unless the event specifies a different version.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

There are no third-party dependencies yet. Add only packages required by the chosen solution, and agree on versions before merging changes.

## Verify the starter project

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

The data, AI / ML, and finance stage functions intentionally raise `NotImplementedError` until the team chooses an approach after seeing the challenge. The smoke tests verify that the shared contracts and module interfaces are connected correctly; they do not claim to validate a final solution.

## Shared module contracts

The initial interfaces are defined in `src/money_talks/contracts.py`:

- `Dataset`: input records and source metadata.
- `AIAnalysis`: structured findings produced by an AI / ML stage.
- `FinancialAssessment`: financial interpretation, metrics, and caveats.

The exact fields and function signatures may change once the challenge is known. Agree on interface changes together before implementing dependent modules.

## Team workflow

1. Clone the repository once you have accepted the GitHub invitation: `git clone https://github.com/oleghegay-png/money-talks-ai-finance-2026.git`.
2. Before starting work, update your local `main`: `git switch main` followed by `git pull origin main`.
3. Create a focused branch, for example `feat/data-input`, `feat/ai-analysis`, or `feat/financial-evaluation`.
4. Work mainly in your assigned module. Coordinate before changing shared contracts or another teammate's files.
5. Run the smoke tests, then commit and push your branch.
6. Open a Pull Request into `main`. Another teammate reviews it; merge only after the integrated project still runs.

For a one-day hackathon, keep changes small and communicate early when an interface needs to change. Avoid having two people edit the same file at the same time.

## Security and event rules

- Never commit API keys, passwords, access tokens, or personal credentials. Put local secrets in `.env` (already ignored by Git); share only variable names in `.env.example`.
- Do not commit private, licensed, or restricted datasets unless the event rules explicitly permit it.
- Follow the organizers' rules for pre-existing code, external APIs, AI tools, attribution, data access, and repository visibility.
- This scaffold is preparation infrastructure only. Do not add a solution, dataset, or proprietary research before the task and event rules are known.
