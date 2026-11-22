# PRISM tracing: evidence of live runs

Each section below was written by `tools/prism_e2e.py` after a real run: a live
LLM call through `money_talks.pipeline`, traced to PRISM (Block Convey), with
the trace ids PRISM returned. To add a run: `PYTHONPATH=src python tools/prism_e2e.py`.
To look a run up in the PRISM web app, open the project and search for its session id.

No API keys are stored here.

## 2026-11-21 21:55:10 UTC: PASS

- Code: commit `3df7e76` on branch `sprint/prism-fixes`
- PRISM host: `https://prism.blockconvey.com`, project `e8de2772-5216-42aa-aafa-f5d9a097da46`
- PRISM session id: `f1ce1fe1d83543129c95408033eeae19`
- Real traces PRISM reported for the project: 9 before, 11 after
- Event sentence: Oil jumps 40% after a Gulf shipping blockade (PRISM end-to-end check 2026-11-21 21:55:10 UTC)
- Portfolio: Balanced 60/40

| Check | Result |
|---|---|
| Scenario LLM call succeeded | PASS |
| PRISM acknowledged the scenario trace | PASS |
| Explanation LLM call was made | PASS |
| PRISM acknowledged the explanation trace | PASS |
| PRISM's count of real traces rose by at least 2 | PASS |

| Stage | PRISM trace id | Model | LLM call | Latency |
|---|---|---|---|---|
| scenario_generation | `79d7a727-0d63-4b22-8161-6fb59c3b9b69` | openai/gpt-6.1-sol | succeeded | 9470 ms |
| explanation | `56facf75-59bc-4468-9353-af49752abb05` | openai/gpt-6.1-sol | succeeded | 3165 ms |

Guardrail decisions recorded in the scenario trace:

- Gulf Blockade Oil Shock (mild) (probability 7.0%): no corrections needed.
- Gulf Blockade Oil Shock (base) (probability 5.0%): no corrections needed.
- Gulf Blockade Oil Shock (severe) (probability 2.0%): no corrections needed.

Guardrail decision recorded in the explanation trace: accepted: every number in the reply matches the computed facts

Explanation shown to the user: The worst case is the Gulf Blockade Oil Shock (severe), which would produce a return of -22.4% and a loss of 224,000 on the portfolio value of 1,000,000. US Equity is the largest contributor to that decline, with an impact of -12.8% of the portfolio; this scenario has a probability of 2.0%. Across all outcomes, the probability-weighted expected 12-month return is 4.1%, compared with 6.1% if no ...
