"""AI/ML analysis stage."""

from money_talks.contracts import AIAnalysis, Dataset


def analyze_dataset(dataset: Dataset) -> AIAnalysis:
    """Analyze the input dataset and return structured findings.

    Define the model/provider and output schema only after the task is known.
    """
    raise NotImplementedError("Implement the task-specific AI/ML analysis after the task is announced.")


# ---------------------------------------------------------------------------
# LLM scenario generation and explanation (ROLE 4). See TEAM_PLAN.md.
#
# The model proposes, the code decides:
#   * the LLM only proposes scenarios and writes prose;
#   * validate_scenario() is a deterministic guardrail that every LLM scenario
#     passes through, and every fix it makes is appended to the rationale;
#   * explanations may only use numbers that code computed and formatted.
#
# None of the public functions raise on network errors, timeouts or bad JSON.
# Every network call goes through _call_llm(), so tests can stub it and the
# Lead can wrap it with PRISM tracing.
# ---------------------------------------------------------------------------

import copy  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from money_talks.contracts import (  # noqa: E402
    ASSET_CLASSES,
    ASSET_LABELS,
    AllocationProposal,
    Portfolio,
    Scenario,
    StressReport,
)

# Guardrail limits (TEAM_PLAN.md, role 4).
SHOCK_MIN, SHOCK_MAX = -0.80, 1.00
PROBABILITY_MIN, PROBABILITY_MAX = 0.001, 0.50
SEVERITY_MIN, SEVERITY_MAX = 0.25, 2.0
MAX_TOTAL_LLM_PROBABILITY = 0.50  # mild + base + severe together

LLM_TIMEOUT_SECONDS = 20
GUARDRAIL_PREFIX = "Guardrail:"  # marks the fixes inside Scenario.rationale
VARIANTS = ("mild", "base", "severe")

# Set MONEY_TALKS_LLM_OFFLINE=1 to skip the network entirely (tests, offline demo).
OFFLINE_ENV_VAR = "MONEY_TALKS_LLM_OFFLINE"

_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"

_scenario_cache: dict[str, list[Scenario]] = {}
_explanation_cache: dict[str, str] = {}


class LLMUnavailable(Exception):
    """The LLM could not be reached or returned something unusable."""


def clear_cache() -> None:
    """Forget cached LLM results (used by tests)."""
    _scenario_cache.clear()
    _explanation_cache.clear()


# ---------------------------------------------------------------------------
# 1. Deterministic guardrail
# ---------------------------------------------------------------------------


def _normalise_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


# "US Equity", "us-equity", "International Equity" ... -> canonical key.
_KEY_ALIASES: dict[str, str] = {}
for _asset in ASSET_CLASSES:
    _KEY_ALIASES[_normalise_key(_asset)] = _asset
    _KEY_ALIASES[_normalise_key(ASSET_LABELS[_asset])] = _asset


def _to_float(value: Any) -> float | None:
    """Parse numbers and strings such as "-35%" or "0.12". None if unusable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip().replace("−", "-").replace(" ", "")
        is_percent = text.endswith("%")
        try:
            number = float(text.rstrip("%"))
        except ValueError:
            return None
        if is_percent:
            number /= 100.0
    else:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _pct(value: float) -> str:
    return f"{value:.1%}"


def validate_scenario(scenario: Scenario) -> tuple[Scenario, list[str]]:
    """Deterministic guardrail: return a cleaned scenario and a list of fixes made.

    Fills missing asset classes with 0.0, drops unknown keys, clamps shocks to
    [-80%, +100%], probability to [0.1%, 50%] and severity to [0.25, 2.0].
    The input scenario is not modified.
    """
    fixes: list[str] = []
    raw_shocks = scenario.shocks if isinstance(scenario.shocks, dict) else {}
    if not isinstance(scenario.shocks, dict):
        fixes.append("Shocks were not a mapping; replaced them with zeros.")

    shocks: dict[str, float] = {}
    for key, value in raw_shocks.items():
        asset = _KEY_ALIASES.get(_normalise_key(key))
        if asset is None:
            fixes.append(f"Dropped unknown asset class '{key}'.")
            continue
        if asset in shocks:
            fixes.append(f"Dropped duplicate entry '{key}' for {asset}.")
            continue
        if key != asset:
            fixes.append(f"Renamed '{key}' to '{asset}'.")
        number = _to_float(value)
        if number is None:
            fixes.append(f"Replaced non-numeric {asset} shock {value!r} with 0.0%.")
            number = 0.0
        clamped = _clamp(number, SHOCK_MIN, SHOCK_MAX)
        if clamped != number:
            fixes.append(f"Clamped {asset} shock from {_pct(number)} to {_pct(clamped)}.")
        shocks[asset] = clamped

    for asset in ASSET_CLASSES:
        if asset not in shocks:
            fixes.append(f"Filled missing asset class {asset} with 0.0%.")
            shocks[asset] = 0.0
    shocks = {asset: shocks[asset] for asset in ASSET_CLASSES}

    probability = _to_float(scenario.probability)
    if probability is None:
        fixes.append(
            f"Replaced non-numeric probability {scenario.probability!r} with {_pct(PROBABILITY_MIN)}."
        )
        probability = PROBABILITY_MIN
    clamped = _clamp(probability, PROBABILITY_MIN, PROBABILITY_MAX)
    if clamped != probability:
        fixes.append(f"Clamped probability from {_pct(probability)} to {_pct(clamped)}.")
    probability = clamped

    severity = _to_float(scenario.severity)
    if severity is None:
        fixes.append(f"Replaced non-numeric severity {scenario.severity!r} with 1.0.")
        severity = 1.0
    clamped = _clamp(severity, SEVERITY_MIN, SEVERITY_MAX)
    if clamped != severity:
        fixes.append(f"Clamped severity from {severity:.2f} to {clamped:.2f}.")
    severity = clamped

    clean = dataclasses.replace(
        scenario, shocks=shocks, probability=probability, severity=severity
    )
    return clean, fixes


# ---------------------------------------------------------------------------
# 2. LLM client (OpenAI-compatible chat completions, standard library only)
# ---------------------------------------------------------------------------


def _load_env_file() -> None:
    """Read KEY=VALUE pairs from the repo's .env without overriding real env vars."""
    try:
        lines = _ENV_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _llm_config() -> tuple[str, str, str]:
    _load_env_file()
    base_url = os.environ.get("NICRRON_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("NICRRON_MODEL", "").strip()
    api_key = os.environ.get("NICRRON_API_KEY", "").strip()
    missing = [
        name
        for name, value in (
            ("NICRRON_BASE_URL", base_url),
            ("NICRRON_MODEL", model),
            ("NICRRON_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        raise LLMUnavailable(f"{', '.join(missing)} not set")
    return base_url, model, api_key


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Send one chat request and return the reply text. Raises on any failure."""
    if os.environ.get(OFFLINE_ENV_VAR, "").strip() not in ("", "0", "false"):
        raise LLMUnavailable(f"{OFFLINE_ENV_VAR} is set")
    base_url, model, api_key = _llm_config()
    body = json.dumps(
        {
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "money-talks/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMUnavailable("unexpected response shape") from exc
    if not isinstance(content, str) or not content.strip():
        raise LLMUnavailable("empty response")
    return content


def _failure_reason(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text if len(text) <= 120 else text[:117] + "..."


# ---------------------------------------------------------------------------
# 3. Scenario generation
# ---------------------------------------------------------------------------

_SCENARIO_SYSTEM_PROMPT = (
    "You are a cautious macro risk analyst. You translate a described event into "
    "12-month total-return shocks for nine asset classes. "
    "Reply with one strict JSON object only: no prose, no markdown, no code fences."
)

_SCENARIO_USER_TEMPLATE = """Event described by the user:
<<<
{event}
>>>

Describe this event as three mutually exclusive variants: mild, base and severe.

Return exactly this JSON shape:
{{
  "title": "short name of the event, at most 6 words",
  "closest_historical_analogue": "closest past episode and year, or 'none'",
  "scenarios": [
    {{
      "variant": "mild",
      "description": "one sentence describing what happens",
      "probability": 0.10,
      "shocks": {{{shock_keys}}},
      "rationale": "one or two sentences on why these shocks"
    }},
    {{"variant": "base", "...": "same fields"}},
    {{"variant": "severe", "...": "same fields"}}
  ]
}}

Rules:
- Every number is a decimal fraction: -0.35 means -35%. Never use percent signs.
- Shocks are 12-month total returns if the variant happens.
- "shocks" must contain exactly these nine keys: {key_list}.
- "probability" is the chance that this variant happens within 12 months.
  The three probabilities must add up to at most 0.50. Severe is usually the least likely.
- Keep shocks plausible versus history. Cash is usually between 0.00 and 0.05.
"""


_FALLBACK_SHOCKS = {
    "us_equity": -0.15,
    "intl_equity": -0.15,
    "em_equity": -0.20,
    "gov_bonds": 0.03,
    "corp_credit": -0.05,
    "gold": 0.05,
    "commodities": -0.10,
    "real_estate": -0.12,
    "cash": 0.0,
}


def _fallback_scenario(text: str, reason: str) -> Scenario:
    label = text.strip()[:40] or "Custom event"
    return Scenario(
        name=f"{label} (FALLBACK: LLM unavailable)",
        description=text.strip(),
        shocks=dict(_FALLBACK_SHOCKS),
        probability=0.05,
        severity=1.0,
        source="custom",
        rationale=(
            f"The LLM was unavailable ({reason}), so this is a generic risk-off "
            "placeholder, NOT an analysis of the event. Try again when the LLM is reachable."
        ),
    )


def _extract_json(raw: str) -> Any:
    """Parse JSON from an LLM reply, tolerating code fences and stray prose."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON found in LLM reply")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        raise ValueError("no JSON found in LLM reply")
    return json.loads(text[start : end + 1])


def _looks_like_percent_points(values: list[float]) -> bool:
    """True when most shocks are above 100% in size, e.g. -35 instead of -0.35."""
    nonzero = [abs(v) for v in values if v != 0.0]
    return bool(nonzero) and sum(v > 1.0 for v in nonzero) * 2 >= len(nonzero)


def _scenario_from_item(
    item: dict[str, Any], index: int, title: str, event_text: str
) -> tuple[Scenario, list[str]]:
    """Turn one JSON scenario into a Scenario plus parsing fixes (before the guardrail)."""
    fixes: list[str] = []
    variant = str(item.get("variant") or item.get("name") or "").strip().lower()
    if variant not in VARIANTS:
        variant = VARIANTS[index] if index < len(VARIANTS) else f"variant {index + 1}"

    shocks = item.get("shocks")
    shocks = dict(shocks) if isinstance(shocks, dict) else {}
    numbers = [n for n in (_to_float(v) for v in shocks.values()) if n is not None]
    if _looks_like_percent_points(numbers):
        fixes.append("Shocks looked like percentage points; divided them by 100.")
        rescaled = {}
        for key, value in shocks.items():
            number = _to_float(value)
            rescaled[key] = number / 100.0 if number is not None else value
        shocks = rescaled

    probability = item.get("probability")
    number = _to_float(probability)
    if number is not None and 1.0 < number <= 100.0:
        fixes.append(f"Probability {number:g} looked like a percentage; divided it by 100.")
        probability = number / 100.0

    scenario = Scenario(
        name=f"{title} ({variant})",
        description=str(item.get("description") or event_text).strip(),
        shocks=shocks,
        probability=probability if probability is not None else PROBABILITY_MIN,
        severity=item.get("severity", 1.0),
        source="llm",
        rationale=str(item.get("rationale") or "").strip(),
    )
    return scenario, fixes


def parse_llm_scenarios(raw: str, event_text: str) -> list[Scenario]:
    """Parse an LLM reply into guarded Scenario objects. Raises ValueError if unusable.

    Every scenario passes through validate_scenario, and the probabilities of all
    variants together are scaled down to at most 50%. Every fix is appended to
    the scenario's rationale after GUARDRAIL_PREFIX.
    """
    data = _extract_json(raw)
    if isinstance(data, list):
        data = {"scenarios": data}
    if not isinstance(data, dict):
        raise ValueError("LLM reply is not a JSON object")
    items = [item for item in data.get("scenarios") or [] if isinstance(item, dict)]
    if not items:
        raise ValueError("LLM reply contains no scenarios")
    items = items[: len(VARIANTS)]

    title = str(data.get("title") or "").strip()[:60] or event_text.strip()[:40] or "Custom event"
    analogue = str(data.get("closest_historical_analogue") or "").strip()

    guarded: list[tuple[Scenario, list[str]]] = []
    for index, item in enumerate(items):
        scenario, fixes = _scenario_from_item(item, index, title, event_text)
        clean, guard_fixes = validate_scenario(scenario)
        guarded.append((clean, fixes + guard_fixes))

    total = sum(scenario.probability for scenario, _ in guarded)
    if total > MAX_TOTAL_LLM_PROBABILITY:
        factor = MAX_TOTAL_LLM_PROBABILITY / total
        for scenario, fixes in guarded:
            scaled = scenario.probability * factor
            fixes.append(
                f"Scaled probability from {_pct(scenario.probability)} to {_pct(scaled)} "
                f"because the variants added up to {_pct(total)} (limit {_pct(MAX_TOTAL_LLM_PROBABILITY)})."
            )
            scenario.probability = scaled

    result = []
    for scenario, fixes in guarded:
        parts = [scenario.rationale] if scenario.rationale else []
        if analogue and analogue.lower() != "none":
            parts.append(f"Closest historical analogue: {analogue}.")
        if fixes:
            parts.append(f"{GUARDRAIL_PREFIX} " + " ".join(fixes))
        else:
            parts.append(f"{GUARDRAIL_PREFIX} no corrections needed.")
        scenario.rationale = " ".join(parts)
        result.append(scenario)
    return result


def generate_scenarios(text: str) -> list[Scenario]:
    """Turn a plain-English event into mild / base / severe Scenario objects.

    The LLM proposes shocks and probabilities; validate_scenario decides
    what is accepted. Must never raise on LLM or network failure.
    """
    text = text if isinstance(text, str) else str(text or "")
    cache_key = " ".join(text.split()).lower()
    if not cache_key:
        return [_fallback_scenario(text, "no event description was given")]
    if cache_key in _scenario_cache:
        return copy.deepcopy(_scenario_cache[cache_key])

    user_prompt = _SCENARIO_USER_TEMPLATE.format(
        event=text.strip()[:2000],
        shock_keys=", ".join(f'"{asset}": 0.0' for asset in ASSET_CLASSES),
        key_list=", ".join(ASSET_CLASSES),
    )
    try:
        raw = _call_llm(_SCENARIO_SYSTEM_PROMPT, user_prompt)
        scenarios = parse_llm_scenarios(raw, text)
    except Exception as exc:  # noqa: BLE001 - must never raise to the UI
        return [_fallback_scenario(text, _failure_reason(exc))]

    _scenario_cache[cache_key] = scenarios
    return copy.deepcopy(scenarios)


# ---------------------------------------------------------------------------
# 4. Explanations: numbers come from code, prose may come from the LLM
# ---------------------------------------------------------------------------


def _money(value: float) -> str:
    return f"{abs(value):,.0f}"


def _report_facts(portfolio: Portfolio, report: StressReport, proposal: AllocationProposal | None) -> list[str]:
    """Pre-formatted facts. These strings are the only numbers allowed in an explanation."""
    facts = [
        f"Portfolio value: {_money(portfolio.total_value)}.",
        f"Probability-weighted expected 12-month return: {_pct(report.expected_return)}.",
        f"Return if no scenario happens: {_pct(report.baseline_return)} "
        f"(probability {_pct(report.baseline_probability)}).",
    ]
    if report.results:
        worst = min(report.results, key=lambda r: r.portfolio_return)
        facts.append(f"Number of shock scenarios: {len(report.results)}.")
        facts.append(
            f"Worst case: {worst.scenario_name}, return {_pct(worst.portfolio_return)}, "
            f"{'loss' if worst.pnl_value < 0 else 'gain'} of {_money(worst.pnl_value)}, "
            f"probability {_pct(worst.probability)}."
        )
        if worst.contribution_by_asset:
            asset, value = min(worst.contribution_by_asset.items(), key=lambda kv: kv[1])
            if value < 0:
                facts.append(
                    f"Largest drag in the worst case: {ASSET_LABELS.get(asset, asset)} "
                    f"({_pct(value)} of the portfolio)."
                )
        facts.append(f"Probability-weighted loss across scenarios: {_pct(report.probability_weighted_loss)}.")
        for result in sorted(report.results, key=lambda r: r.portfolio_return)[:5]:
            facts.append(
                f"Scenario {result.scenario_name}: return {_pct(result.portfolio_return)}, "
                f"probability {_pct(result.probability)}."
            )
    else:
        facts.append("No shock scenarios are selected.")

    if proposal is not None:
        before, after = proposal.before, proposal.after
        facts.append(
            "Proposed re-allocation "
            + ("meets all constraints." if proposal.feasible else "could NOT meet all constraints; it is the closest attempt.")
        )
        facts.append(
            f"Worst-case return before: {_pct(before.worst_case_return)}, after: {_pct(after.worst_case_return)}."
        )
        facts.append(
            f"Expected return before: {_pct(before.expected_return)}, after: {_pct(after.expected_return)} "
            f"(change {after.expected_return - before.expected_return:+.1%})."
        )
        trades = sorted(
            ((a, t) for a, t in proposal.trades.items() if abs(t) >= 0.005),
            key=lambda kv: -abs(kv[1]),
        )[:3]
        if trades:
            facts.append(
                "Largest trades: "
                + ", ".join(f"{ASSET_LABELS.get(a, a)} {t:+.1%}" for a, t in trades)
                + "."
            )
    return facts


def _template_explanation(portfolio: Portfolio, report: StressReport, proposal: AllocationProposal | None) -> str:
    """Deterministic explanation used when the LLM is unavailable or untrustworthy."""
    if not report.results:
        text = (
            f"No shock scenarios are selected, so the expected 12-month return is the baseline "
            f"of {_pct(report.baseline_return)}."
        )
    else:
        worst = min(report.results, key=lambda r: r.portfolio_return)
        text = (
            f"The worst of the {len(report.results)} selected scenarios is {worst.scenario_name}, "
            f"where the portfolio returns {_pct(worst.portfolio_return)} "
            f"({'a loss' if worst.pnl_value < 0 else 'a gain'} of {_money(worst.pnl_value)}). "
            f"Weighting every scenario and the no-shock baseline by its probability, the expected "
            f"12-month return is {_pct(report.expected_return)}, and the probability-weighted loss is "
            f"{_pct(report.probability_weighted_loss)}."
        )
    if proposal is not None:
        before, after = proposal.before, proposal.after
        text += (
            f" The proposed re-allocation moves the worst-case return from {_pct(before.worst_case_return)} "
            f"to {_pct(after.worst_case_return)} and the expected return from {_pct(before.expected_return)} "
            f"to {_pct(after.expected_return)}."
        )
        if not proposal.feasible:
            text += " It could not meet every constraint, so treat it as the closest available attempt."
    return text


_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _number_tokens(text: str) -> set[str]:
    tokens = set()
    for match in _NUMBER_RE.findall(text):
        token = match.replace(",", "").rstrip(".")
        tokens.add(token)
        if token.endswith(".0"):
            tokens.add(token[:-2])  # allow "22%" for "22.0%"
    return tokens


def unverified_numbers(explanation: str, facts_text: str) -> set[str]:
    """Numbers in an LLM explanation that do not appear in the facts given to it."""
    return _number_tokens(explanation) - _number_tokens(facts_text)


_EXPLAIN_SYSTEM_PROMPT = (
    "You explain portfolio stress-test results to a non-expert client in plain English. "
    "You never calculate, round, estimate or invent numbers: you may only copy numbers "
    "exactly as they appear in the facts you are given. You give no investment advice."
)


def explain_report(
    portfolio: Portfolio,
    report: StressReport,
    proposal: AllocationProposal | None = None,
) -> str:
    """Short plain-English explanation. Numbers come from the report, never the LLM."""
    try:
        facts = _report_facts(portfolio, report, proposal)
        template = _template_explanation(portfolio, report, proposal)
    except Exception:  # noqa: BLE001 - must never raise to the UI
        return "The stress test ran, but an explanation could not be produced for these results."

    facts_text = "\n".join(f"- {fact}" for fact in facts)
    if facts_text in _explanation_cache:
        return _explanation_cache[facts_text]

    user_prompt = (
        "Facts (12-month horizon):\n"
        f"{facts_text}\n\n"
        "Write three or four sentences: what the worst case does to the portfolio and why, "
        "what the probability-weighted picture looks like"
        + (", and what the proposed re-allocation buys and what it costs in expected return" if proposal else "")
        + ". Copy numbers exactly as written in the facts. Do not add any other number. "
        "Plain text only, no lists, no headings."
    )
    try:
        explanation = " ".join(_call_llm(_EXPLAIN_SYSTEM_PROMPT, user_prompt).split())
    except Exception:  # noqa: BLE001
        return template
    if not explanation or unverified_numbers(explanation, facts_text):
        # The model invented or altered a number: the code decides, use the template.
        return template

    _explanation_cache[facts_text] = explanation
    return explanation
