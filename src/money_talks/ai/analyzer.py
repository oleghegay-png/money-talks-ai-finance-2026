"""AI/ML analysis stage and LLM stress-scenario generator / guardrail."""

from money_talks.contracts import AIAnalysis, Dataset


def analyze_dataset(dataset: Dataset) -> AIAnalysis:
    """Analyze the input dataset and return structured findings.

    Define the model/provider and output schema only after the task is known.
    """
    raise NotImplementedError("Implement the task-specific AI/ML analysis after the task is announced.")


# ---------------------------------------------------------------------------
# LLM scenario generation and explanation (ROLE 4). See TEAM_PLAN.md.
# ---------------------------------------------------------------------------

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from money_talks.contracts import (  # noqa: E402
    ASSET_CLASSES,
    AllocationProposal,
    Portfolio,
    Scenario,
    StressReport,
)

_REQUEST_TIMEOUT_SEC = 20.0
_MIN_SHOCK = -0.80
_MAX_SHOCK = 1.00
_MIN_PROB = 0.001
_MAX_PROB = 0.50
_MIN_SEVERITY = 0.25
_MAX_SEVERITY = 2.0
_MAX_COMBINED_LLM_PROB = 0.50

_SCENARIO_CACHE: dict[str, list[Scenario]] = {}
_EXPLAIN_CACHE: dict[str, str] = {}


def clear_caches() -> None:
    """Clear in-memory LLM scenario and explanation caches."""
    _SCENARIO_CACHE.clear()
    _EXPLAIN_CACHE.clear()


def _read_dotenv_value(name: str) -> str:
    """Best-effort lookup of a variable from a local .env file without third-party deps."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[3] / ".env",
    ]
    for path in candidates:
        try:
            if not path.is_file():
                continue
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                if key.strip() == name:
                    return val.strip().strip('"').strip("'")
        except OSError:
            continue
    return ""


def _get_env(name: str, default: str = "") -> str:
    """Read an environment variable, falling back to local .env then default."""
    val = os.environ.get(name, "").strip()
    if val:
        return val
    dot_val = _read_dotenv_value(name).strip()
    return dot_val or default


def _to_finite_float(value: Any, default: float) -> tuple[float, bool]:
    """Convert value to a finite float; return (number, converted_cleanly)."""
    if isinstance(value, bool):
        return default, False
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default, False
    if not math.isfinite(num):
        return default, False
    return num, True


def validate_scenario(scenario: Scenario) -> tuple[Scenario, list[str]]:
    """Deterministic guardrail: return a cleaned scenario and a list of fixes made."""
    fixes: list[str] = []
    raw_shocks = dict(scenario.shocks or {})

    unknown_keys = sorted(key for key in raw_shocks if key not in ASSET_CLASSES)
    if unknown_keys:
        fixes.append(f"dropped unknown asset keys: {', '.join(unknown_keys)}")

    clean_shocks: dict[str, float] = {}
    missing_keys: list[str] = []
    for asset in ASSET_CLASSES:
        if asset not in raw_shocks:
            clean_shocks[asset] = 0.0
            missing_keys.append(asset)
            continue

        val, ok = _to_finite_float(raw_shocks[asset], 0.0)
        if not ok:
            clean_shocks[asset] = 0.0
            fixes.append(f"replaced non-numeric {asset} shock with 0.0")
            continue

        clamped = max(_MIN_SHOCK, min(_MAX_SHOCK, val))
        if abs(clamped - val) > 1e-12:
            fixes.append(f"clamped {asset} shock from {val:+.2f} to {clamped:+.2f}")
        clean_shocks[asset] = float(clamped)

    if missing_keys:
        fixes.append(f"filled missing asset shocks with 0.0: {', '.join(missing_keys)}")

    prob_val, prob_ok = _to_finite_float(scenario.probability, 0.10)
    if not prob_ok:
        fixes.append("replaced invalid probability with 0.100")
    clamped_prob = max(_MIN_PROB, min(_MAX_PROB, prob_val))
    if prob_ok and abs(clamped_prob - prob_val) > 1e-12:
        fixes.append(f"clamped probability from {prob_val:.3f} to {clamped_prob:.3f}")

    sev_val, sev_ok = _to_finite_float(scenario.severity, 1.0)
    if not sev_ok:
        fixes.append("replaced invalid severity with 1.00")
    clamped_sev = max(_MIN_SEVERITY, min(_MAX_SEVERITY, sev_val))
    if sev_ok and abs(clamped_sev - sev_val) > 1e-12:
        fixes.append(f"clamped severity from {sev_val:.2f} to {clamped_sev:.2f}")

    cleaned = Scenario(
        name=str(scenario.name or "Custom scenario").strip() or "Custom scenario",
        description=str(scenario.description or "").strip(),
        shocks=clean_shocks,
        probability=float(clamped_prob),
        severity=float(clamped_sev),
        source=str(scenario.source or "llm").strip() or "llm",
        rationale=str(scenario.rationale or "").strip(),
    )
    return cleaned, fixes


def _call_llm(messages: list[dict[str, str]], temperature: float = 0.2) -> str:
    """Call an OpenAI-compatible chat completions endpoint via stdlib urllib."""
    api_key = _get_env("NICRRON_API_KEY")
    if not api_key:
        raise RuntimeError("NICRRON_API_KEY is not configured.")

    base_url = _get_env("NICRRON_BASE_URL", "https://api.nicrron.ai/v1").rstrip("/")
    model = _get_env("NICRRON_MODEL", "openai/gpt-6.1-sol")
    url = f"{base_url}/chat/completions"

    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SEC) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    return str(parsed["choices"][0]["message"]["content"])


def _extract_json_payload(raw_content: str) -> Any:
    """Extract JSON array or object from raw LLM text (including fenced code blocks)."""
    text = raw_content.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])

    raise json.JSONDecodeError("No valid JSON payload found in LLM response.", text, 0)


def _scale_probabilities_to_cap(
    scenarios: list[Scenario],
    fixes_per_scenario: list[list[str]],
) -> None:
    """Scale combined LLM scenario probabilities down if they exceed 0.50."""
    total_prob = sum(s.probability for s in scenarios)
    if total_prob <= _MAX_COMBINED_LLM_PROB + 1e-12 or total_prob <= 0.0:
        return

    scale = _MAX_COMBINED_LLM_PROB / total_prob
    for scenario, fixes in zip(scenarios, fixes_per_scenario):
        old_p = scenario.probability
        new_p = max(_MIN_PROB, round(old_p * scale, 4))
        scenario.probability = new_p
        fixes.append(f"scaled probability from {old_p:.3f} to {new_p:.3f} so combined probability <= 0.50")

    # Guard against rounding dust slightly pushing the sum above 0.50
    overflow = sum(s.probability for s in scenarios) - _MAX_COMBINED_LLM_PROB
    if overflow > 1e-9:
        largest = max(scenarios, key=lambda s: s.probability)
        largest.probability = max(_MIN_PROB, round(largest.probability - overflow - 1e-4, 4))


def _parse_llm_scenarios_json(raw_content: str, event_text: str) -> list[Scenario]:
    """Parse strict JSON scenarios, apply validate_scenario, and attach guardrail fixes."""
    payload = _extract_json_payload(raw_content)
    items = payload.get("scenarios") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError("LLM JSON did not contain a non-empty list of scenarios.")

    default_meta = (("mild", 0.5, 0.15), ("base", 1.0, 0.10), ("severe", 1.5, 0.05))
    cleaned_list: list[Scenario] = []
    fixes_list: list[list[str]] = []

    for idx, item in enumerate(items[:3]):
        if not isinstance(item, dict):
            raise ValueError("Each scenario entry must be a JSON object.")
        label, default_sev, default_prob = default_meta[min(idx, len(default_meta) - 1)]
        raw_name = str(item.get("name") or f"{event_text.strip()[:36] or 'Custom event'} ({label})").strip()
        raw_desc = str(item.get("description") or event_text.strip()).strip()
        raw_shocks = item.get("shocks")
        if not isinstance(raw_shocks, dict):
            raw_shocks = {}

        raw_prob = item.get("probability", default_prob)
        raw_sev = item.get("severity", default_sev)
        raw_rationale = str(item.get("rationale") or f"LLM-estimated {label} impact.").strip()
        analogue = str(item.get("historical_analogue") or "").strip()
        if analogue and analogue.lower() not in raw_rationale.lower():
            raw_rationale = f"{raw_rationale} Closest historical analogue: {analogue}."

        candidate = Scenario(
            name=raw_name,
            description=raw_desc,
            shocks=raw_shocks,
            probability=raw_prob,
            severity=raw_sev,
            source="llm",
            rationale=raw_rationale,
        )
        cleaned, fixes = validate_scenario(candidate)
        cleaned.source = "llm"
        cleaned_list.append(cleaned)
        fixes_list.append(fixes)

    _scale_probabilities_to_cap(cleaned_list, fixes_list)

    for scenario, fixes in zip(cleaned_list, fixes_list):
        if fixes:
            scenario.rationale = (
                f"{scenario.rationale}\nGuardrail fixes: {'; '.join(fixes)}"
                if scenario.rationale
                else f"Guardrail fixes: {'; '.join(fixes)}"
            )

    return cleaned_list


def _fallback_scenarios(text: str, reason: str) -> list[Scenario]:
    """Return three deterministic fallback scenarios when the LLM is unavailable."""
    lower = text.lower()
    if any(word in lower for word in ("oil", "energy", "inflation", "fed", "rate", "hike", "tariff")):
        base_shocks = {
            "us_equity": -0.18,
            "intl_equity": -0.15,
            "em_equity": -0.20,
            "gov_bonds": -0.11,
            "corp_credit": -0.13,
            "gold": 0.06,
            "commodities": 0.18,
            "real_estate": -0.20,
            "cash": 0.03,
        }
        analogue = "2022 Inflation & Rate Shock"
    elif any(word in lower for word in ("tech", "ai", "bubble", "equity")):
        base_shocks = {
            "us_equity": -0.28,
            "intl_equity": -0.18,
            "em_equity": -0.22,
            "gov_bonds": 0.08,
            "corp_credit": -0.07,
            "gold": 0.04,
            "commodities": -0.08,
            "real_estate": -0.08,
            "cash": 0.03,
        }
        analogue = "2000 Dot-Com Bust"
    else:
        base_shocks = {
            "us_equity": -0.24,
            "intl_equity": -0.25,
            "em_equity": -0.30,
            "gov_bonds": 0.06,
            "corp_credit": -0.11,
            "gold": 0.05,
            "commodities": -0.15,
            "real_estate": -0.22,
            "cash": 0.02,
        }
        analogue = "2008 Global Financial Crisis"

    short_label = text.strip()[:38] or "Custom macro shock"
    variants = (("mild", 0.5, 0.15), ("base", 1.0, 0.10), ("severe", 1.5, 0.05))
    results: list[Scenario] = []
    for label, severity, probability in variants:
        candidate = Scenario(
            name=f"{short_label} ({label})",
            description=text.strip() or "Custom macro stress event.",
            shocks=dict(base_shocks),
            probability=probability,
            severity=severity,
            source="llm",
            rationale=(
                f"LLM unavailable ({reason}); using deterministic macro fallback "
                f"(closest historical analogue: {analogue}).\n"
                f"Guardrail fixes: fallback scenario validated within deterministic shock and probability bounds"
            ),
        )
        cleaned, _ = validate_scenario(candidate)
        cleaned.source = "llm"
        results.append(cleaned)
    return results


def _copy_scenarios(scenarios: list[Scenario]) -> list[Scenario]:
    return [
        Scenario(
            name=s.name,
            description=s.description,
            shocks=dict(s.shocks),
            probability=s.probability,
            severity=s.severity,
            source=s.source,
            rationale=s.rationale,
        )
        for s in scenarios
    ]


def generate_scenarios(text: str) -> list[Scenario]:
    """Turn a plain-English event into mild / base / severe Scenario objects.

    The LLM proposes shocks and probabilities; validate_scenario decides
    what is accepted. Must never raise on LLM or network failure.
    """
    cache_key = (text or "").strip()
    if cache_key in _SCENARIO_CACHE:
        return _copy_scenarios(_SCENARIO_CACHE[cache_key])

    if not cache_key:
        return _fallback_scenarios("Custom macro shock", "empty event description")

    system_prompt = (
        "You are a quantitative macro risk analyst. Convert the user's event description into "
        "STRICT JSON containing an object {\"scenarios\": [...]} with exactly 3 scenarios "
        "(mild, base, severe). Each scenario must include: "
        "\"name\" (string), \"description\" (string), "
        f"\"shocks\" (object with exact decimal 12-month returns for all 9 keys: {list(ASSET_CLASSES)}), "
        "\"probability\" (decimal between 0.001 and 0.50; sum of all 3 <= 0.50), "
        "\"severity\" (0.5 for mild, 1.0 for base, 1.5 for severe), "
        "\"historical_analogue\" (closest past crisis name), and \"rationale\" (1-2 sentences). "
        "Return ONLY raw JSON."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Event: {cache_key}"},
    ]

    try:
        raw_reply = _call_llm(messages, temperature=0.2)
        parsed = _parse_llm_scenarios_json(raw_reply, cache_key)
    except (
        RuntimeError,
        OSError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return _fallback_scenarios(cache_key, str(exc))

    _SCENARIO_CACHE[cache_key] = _copy_scenarios(parsed)
    return parsed


def _template_explanation(
    portfolio: Portfolio,
    report: StressReport,
    proposal: AllocationProposal | None,
) -> str:
    """Deterministic fallback explanation using exact numbers from the report."""
    worst_name = report.worst_case_scenario or "no active shock"
    worst_pnl = report.worst_case_return * float(portfolio.total_value)
    sentences = [
        (
            f"Across the active scenarios, the portfolio's probability-weighted 12-month expected "
            f"return is {report.expected_return:+.1%} compared with a no-shock baseline return of "
            f"{report.baseline_return:+.1%}."
        ),
        (
            f"The largest modeled drawdown occurs in {worst_name} at {report.worst_case_return:+.1%} "
            f"(${worst_pnl:+,.0f}), while probability-weighted scenario loss is "
            f"{report.probability_weighted_loss:.1%}."
        ),
    ]
    if proposal is not None:
        status_text = "meets all risk and turnover limits" if proposal.feasible else "uses the best-effort fallback"
        sentences.append(
            f"The proposed rebalance {status_text}, shifting worst-case scenario return from "
            f"{proposal.before.worst_case_return:+.1%} to {proposal.after.worst_case_return:+.1%} "
            f"with an expected return of {proposal.after.expected_return:+.1%}."
        )
    else:
        sentences.append(
            "Review the Prepare & Rebalance tab to test how capping worst-case loss affects expected return."
        )
    return " ".join(sentences)


def explain_report(
    portfolio: Portfolio,
    report: StressReport,
    proposal: AllocationProposal | None = None,
) -> str:
    """Short plain-English explanation. Numbers come from the report, never the LLM."""
    fp_payload = {
        "total_value": round(float(portfolio.total_value), 2),
        "expected_return": round(float(report.expected_return), 6),
        "baseline_return": round(float(report.baseline_return), 6),
        "worst_case_scenario": report.worst_case_scenario,
        "worst_case_return": round(float(report.worst_case_return), 6),
        "pwl": round(float(report.probability_weighted_loss), 6),
        "proposal": (
            {
                "feasible": proposal.feasible,
                "after_worst": round(float(proposal.after.worst_case_return), 6),
                "after_exp": round(float(proposal.after.expected_return), 6),
            }
            if proposal is not None
            else None
        ),
    }
    cache_key = hashlib.sha1(json.dumps(fp_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    if cache_key in _EXPLAIN_CACHE:
        return _EXPLAIN_CACHE[cache_key]

    worst_pnl = report.worst_case_return * float(portfolio.total_value)
    facts = [
        f"Portfolio total value: ${portfolio.total_value:,.0f}",
        f"Baseline return (no shock, probability {report.baseline_probability:.1%}): {report.baseline_return:+.1%}",
        f"Probability-weighted expected 12M return: {report.expected_return:+.1%}",
        f"Worst-case scenario: {report.worst_case_scenario or 'N/A'} ({report.worst_case_return:+.1%}, ${worst_pnl:+,.0f})",
        f"Probability-weighted loss: {report.probability_weighted_loss:.1%}",
    ]
    if proposal is not None:
        facts.extend(
            [
                f"Rebalance feasible: {proposal.feasible}",
                f"Before vs after worst-case return: {proposal.before.worst_case_return:+.1%} -> {proposal.after.worst_case_return:+.1%}",
                f"Before vs after expected return: {proposal.before.expected_return:+.1%} -> {proposal.after.expected_return:+.1%}",
            ]
        )

    messages = [
        {
            "role": "system",
            "content": (
                "Write 3 to 4 clear, plain-English sentences summarizing this portfolio stress test. "
                "Use ONLY the exact numbers provided below; do not invent or recalculate any numbers."
            ),
        },
        {"role": "user", "content": "\n".join(facts)},
    ]

    try:
        reply = _call_llm(messages, temperature=0.2).strip()
        if not reply:
            raise ValueError("Empty explanation from LLM.")
    except (
        RuntimeError,
        OSError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return _template_explanation(portfolio, report, proposal)

    _EXPLAIN_CACHE[cache_key] = reply
    return reply
