"""Load the dataset selected for the hackathon task and provide stress-test reference data."""

from money_talks.contracts import Dataset


def load_dataset() -> Dataset:
    """Load data from the approved source and return the shared Dataset type.

    Implement this after the organizers specify the data source and access rules.
    Keep credentials out of source code; read them from environment variables.
    """
    raise NotImplementedError("Choose and implement the approved data source after the task is announced.")


# ---------------------------------------------------------------------------
# Stress-test data (ROLE 2 - Scenario library). See TEAM_PLAN.md.
# ---------------------------------------------------------------------------

from money_talks.contracts import ASSET_CLASSES, Portfolio, Scenario  # noqa: E402

_BASELINE_RETURNS: dict[str, float] = {
    "us_equity": 0.075,
    "intl_equity": 0.070,
    "em_equity": 0.085,
    "gov_bonds": 0.038,
    "corp_credit": 0.052,
    "gold": 0.035,
    "commodities": 0.040,
    "real_estate": 0.060,
    "cash": 0.035,
}

_PRESET_PORTFOLIOS: dict[str, dict[str, float]] = {
    "Balanced 60/40": {
        "us_equity": 0.35,
        "intl_equity": 0.15,
        "em_equity": 0.10,
        "gov_bonds": 0.25,
        "corp_credit": 0.10,
        "gold": 0.00,
        "commodities": 0.00,
        "real_estate": 0.00,
        "cash": 0.05,
    },
    "Conservative": {
        "us_equity": 0.15,
        "intl_equity": 0.05,
        "em_equity": 0.00,
        "gov_bonds": 0.40,
        "corp_credit": 0.15,
        "gold": 0.05,
        "commodities": 0.00,
        "real_estate": 0.05,
        "cash": 0.15,
    },
    "Aggressive Growth": {
        "us_equity": 0.45,
        "intl_equity": 0.20,
        "em_equity": 0.15,
        "gov_bonds": 0.05,
        "corp_credit": 0.05,
        "gold": 0.00,
        "commodities": 0.03,
        "real_estate": 0.05,
        "cash": 0.02,
    },
    "All-Weather": {
        "us_equity": 0.20,
        "intl_equity": 0.05,
        "em_equity": 0.05,
        "gov_bonds": 0.40,
        "corp_credit": 0.10,
        "gold": 0.075,
        "commodities": 0.075,
        "real_estate": 0.00,
        "cash": 0.05,
    },
}

# Total default probability across all 10 scenarios is 0.50 (0.37 historical + 0.13 custom),
# leaving headroom for up to 0.50 combined probability from LLM-generated scenarios.
_SCENARIO_LIBRARY: tuple[Scenario, ...] = (
    Scenario(
        name="2008 Global Financial Crisis",
        description="Systemic banking crisis, global credit freeze, and deep recession.",
        shocks={
            "us_equity": -0.37,
            "intl_equity": -0.43,
            "em_equity": -0.53,
            "gov_bonds": 0.14,
            "corp_credit": -0.09,
            "gold": 0.05,
            "commodities": -0.36,
            "real_estate": -0.38,
            "cash": 0.015,
        },
        probability=0.05,
        severity=1.0,
        source="historical",
        rationale=(
            "Approximated 12-month calendar 2008 asset-class returns (S&P 500, MSCI EAFE/EM, "
            "US Treasuries, IG/HY credit, spot gold, broad commodities, REITs, T-bills)."
        ),
    ),
    Scenario(
        name="2022 Inflation & Rate Shock",
        description="Persistent inflation surprise triggers rapid global central-bank tightening.",
        shocks={
            "us_equity": -0.18,
            "intl_equity": -0.14,
            "em_equity": -0.20,
            "gov_bonds": -0.13,
            "corp_credit": -0.15,
            "gold": 0.00,
            "commodities": 0.16,
            "real_estate": -0.25,
            "cash": 0.02,
        },
        probability=0.10,
        severity=1.0,
        source="historical",
        rationale=(
            "Approximated 2022 full-year returns when stocks and long-duration bonds sold off "
            "together while energy-heavy commodities rallied."
        ),
    ),
    Scenario(
        name="2020 Pandemic Crash",
        description="Sudden global lockdowns, liquidity dash for cash, and sharp GDP contraction.",
        shocks={
            "us_equity": -0.34,
            "intl_equity": -0.33,
            "em_equity": -0.32,
            "gov_bonds": 0.06,
            "corp_credit": -0.12,
            "gold": -0.03,
            "commodities": -0.30,
            "real_estate": -0.35,
            "cash": 0.005,
        },
        probability=0.04,
        severity=1.0,
        source="historical",
        rationale=(
            "Approximated peak-to-trough February-March 2020 drawdown shock across risk assets "
            "with a flight-to-safety rally in sovereign Treasuries."
        ),
    ),
    Scenario(
        name="1970s Stagflation Shock",
        description="Oil embargo supply shock, entrenched inflation, and stagnant growth.",
        shocks={
            "us_equity": -0.26,
            "intl_equity": -0.24,
            "em_equity": -0.22,
            "gov_bonds": -0.07,
            "corp_credit": -0.11,
            "gold": 0.35,
            "commodities": 0.28,
            "real_estate": -0.18,
            "cash": 0.06,
        },
        probability=0.05,
        severity=1.0,
        source="historical",
        rationale=(
            "Stylized 12-month real/nominal stress approximation based on the 1973-1974 OPEC oil "
            "embargo regime where gold and commodities outperformed financial assets."
        ),
    ),
    Scenario(
        name="2000 Dot-Com Bust",
        description="Tech valuation unwind and prolonged US capex recession.",
        shocks={
            "us_equity": -0.28,
            "intl_equity": -0.22,
            "em_equity": -0.30,
            "gov_bonds": 0.11,
            "corp_credit": -0.04,
            "gold": 0.02,
            "commodities": -0.08,
            "real_estate": 0.06,
            "cash": 0.04,
        },
        probability=0.05,
        severity=1.0,
        source="historical",
        rationale=(
            "Approximated 12-month unwind phase of the 2000-2002 equity bubble: growth equities "
            "fell sharply while Treasuries and value/real-estate exposures held up."
        ),
    ),
    Scenario(
        name="2011 Euro Sovereign Debt Crisis",
        description="Peripheral European sovereign stress spills into global banks and credit.",
        shocks={
            "us_equity": -0.12,
            "intl_equity": -0.24,
            "em_equity": -0.22,
            "gov_bonds": 0.09,
            "corp_credit": -0.07,
            "gold": 0.10,
            "commodities": -0.14,
            "real_estate": -0.10,
            "cash": 0.005,
        },
        probability=0.04,
        severity=1.0,
        source="historical",
        rationale=(
            "Approximated mid-2011 European banking and sovereign spread shock; non-US developed "
            "equities suffered more than US markets while core government bonds and gold rallied."
        ),
    ),
    Scenario(
        name="1997 Emerging Markets Crisis",
        description="Asian currency devaluations, EM capital flight, and commodity demand drop.",
        shocks={
            "us_equity": -0.08,
            "intl_equity": -0.14,
            "em_equity": -0.45,
            "gov_bonds": 0.08,
            "corp_credit": -0.06,
            "gold": -0.10,
            "commodities": -0.20,
            "real_estate": -0.12,
            "cash": 0.05,
        },
        probability=0.04,
        severity=1.0,
        source="historical",
        rationale=(
            "Approximated 1997-1998 Asian/EM contagion profile featuring severe EM equity and "
            "commodity losses alongside US Treasury gains."
        ),
    ),
    Scenario(
        name="AI Equity Bubble Unwind",
        description="Forward scenario: hyperscaler capex slowdown triggers a US mega-cap de-rating.",
        shocks={
            "us_equity": -0.32,
            "intl_equity": -0.18,
            "em_equity": -0.22,
            "gov_bonds": 0.09,
            "corp_credit": -0.08,
            "gold": 0.06,
            "commodities": -0.10,
            "real_estate": -0.08,
            "cash": 0.035,
        },
        probability=0.06,
        severity=1.0,
        source="custom",
        rationale=(
            "Forward-looking illustrative scenario assuming concentrated US tech valuation "
            "compression with moderate spillover to global risk assets and lower policy rates."
        ),
    ),
    Scenario(
        name="Taiwan Strait Conflict",
        description="Forward scenario: severe semiconductor supply disruption and geopolitical risk-off.",
        shocks={
            "us_equity": -0.30,
            "intl_equity": -0.34,
            "em_equity": -0.48,
            "gov_bonds": 0.07,
            "corp_credit": -0.14,
            "gold": 0.18,
            "commodities": 0.22,
            "real_estate": -0.20,
            "cash": 0.03,
        },
        probability=0.03,
        severity=1.0,
        source="custom",
        rationale=(
            "Forward-looking illustrative supply-shock scenario: heavy EM/global equity sell-off, "
            "energy/shipping inflation spike, and strong safe-haven bid for gold."
        ),
    ),
    Scenario(
        name="US Sovereign Debt & Term-Premium Shock",
        description="Forward scenario: fiscal deficit concerns drive a sharp spike in long Treasury yields.",
        shocks={
            "us_equity": -0.20,
            "intl_equity": -0.15,
            "em_equity": -0.24,
            "gov_bonds": -0.18,
            "corp_credit": -0.16,
            "gold": 0.14,
            "commodities": 0.08,
            "real_estate": -0.22,
            "cash": 0.045,
        },
        probability=0.04,
        severity=1.0,
        source="custom",
        rationale=(
            "Forward-looking illustrative term-premium shock where long sovereign bonds fail to "
            "hedge equities, favoring cash and real stores of value like gold."
        ),
    ),
)


def get_baseline_returns() -> dict[str, float]:
    """Expected 12-month return per asset class when no shock happens."""
    return {asset: float(_BASELINE_RETURNS[asset]) for asset in ASSET_CLASSES}


def get_historical_scenarios() -> list[Scenario]:
    """Library of historical crises and forward-looking scenarios."""
    return [
        Scenario(
            name=scenario.name,
            description=scenario.description,
            shocks={asset: float(scenario.shocks[asset]) for asset in ASSET_CLASSES},
            probability=float(scenario.probability),
            severity=float(scenario.severity),
            source=scenario.source,
            rationale=scenario.rationale,
        )
        for scenario in _SCENARIO_LIBRARY
    ]


def get_preset_portfolios() -> dict[str, Portfolio]:
    """Named starting portfolios for the UI, e.g. 'Balanced 60/40'."""
    return {
        name: Portfolio(
            weights={asset: float(weights[asset]) for asset in ASSET_CLASSES},
            total_value=1_000_000.0,
        )
        for name, weights in _PRESET_PORTFOLIOS.items()
    }
