"""Load the dataset selected for the hackathon task."""

from money_talks.contracts import Dataset


def load_dataset() -> Dataset:
    """Load data from the approved source and return the shared Dataset type.

    Implement this after the organizers specify the data source and access rules.
    Keep credentials out of source code; read them from environment variables.
    """
    raise NotImplementedError("Choose and implement the approved data source after the task is announced.")


# ---------------------------------------------------------------------------
# Stress-test data (ROLE 2 - Scenario library). See TEAM_PLAN.md.
#
# PROVENANCE: every number below was drafted by an LLM (GPT-6.1 via Nicrron)
# from memory and then sanity-checked by a reviewer. They are ILLUSTRATIVE
# approximations, not index data. Each scenario's rationale says which figures
# are well documented and which are estimates. Do not present as exact.
#
# Default probabilities are the drafted per-event estimates scaled by 0.4 so
# that the whole library sums to about 0.27. The events overlap in type, and
# the model treats scenarios as mutually exclusive, so unscaled values would
# imply a 68% chance of a crisis within 12 months.
# ---------------------------------------------------------------------------

from money_talks.contracts import Portfolio, Scenario  # noqa: E402

BASELINE_RATIONALE = 'Assumptions are nominal USD total returns with income reinvested, unhedged foreign exposures, fully collateralized commodity futures exposure, and no macroeconomic shock. These illustrative long-run annual expectations reflect equity and credit risk premiums over cash, moderate bond income, and modest real-asset appreciation rather than a current tactical forecast.'


def get_baseline_returns() -> dict[str, float]:
    """Expected 12-month return per asset class when no shock happens."""
    return {"us_equity": 0.07, "intl_equity": 0.075, "em_equity": 0.085, "gov_bonds": 0.04, "corp_credit": 0.05, "gold": 0.035, "commodities": 0.04, "real_estate": 0.07, "cash": 0.03}


def get_historical_scenarios() -> list[Scenario]:
    """Historical crises. Shocks are for the worst ~12 months or peak-to-trough."""
    return [
        Scenario(
            name='Global Financial Crisis (2008)',
            description='Housing losses and banking-system failures triggered a global credit contraction, forced deleveraging, and a collapse in risky assets.',
            shocks={"us_equity": -0.433, "intl_equity": -0.473, "em_equity": -0.486, "gov_bonds": 0.1, "corp_credit": -0.12, "gold": -0.03, "commodities": -0.47, "real_estate": -0.6, "cash": 0.015},
            probability=0.012,
            source='historical',
            rationale='Window: 2008-03-01 to 2009-02-28; approximately the worst month-end-to-month-end 12-month window for US equities. The equity losses are close to documented USD total-index returns, while Treasury, investment-grade credit, gold, commodity, REIT, and bill returns are rounded estimates for this common window, assuming unhedged foreign equities and representative investable benchmarks. Funding stress and widening credit spreads overwhelmed risky assets, while safe-haven demand supported Treasuries; this probability is a subjective, non-additive estimate for a comparable systemic crisis starting within 12 months. Confidence: medium.',
        ),
        Scenario(
            name='COVID Pandemic Crash (2020)',
            description='Pandemic shutdowns caused an exceptionally rapid equity crash and a dash for cash before massive monetary and fiscal intervention stabilized markets.',
            shocks={"us_equity": -0.338, "intl_equity": -0.335, "em_equity": -0.315, "gov_bonds": 0.055, "corp_credit": -0.15, "gold": -0.035, "commodities": -0.32, "real_estate": -0.42, "cash": 0.001},
            probability=0.016,
            source='historical',
            rationale='Window: 2020-02-19 to 2020-03-23; S&P 500 closing peak to trough, with every asset measured over the same dates. The S&P 500 total-return drawdown is well documented, while the other shocks are approximate same-window USD total returns, with particularly material benchmark sensitivity for corporate bonds, commodities, and listed REITs. Shutdown expectations, dollar funding shortages, and forced selling depressed risky assets and temporarily gold, while falling yields supported intermediate Treasuries; the probability refers to a comparable abrupt global shutdown or liquidity shock rather than necessarily another pandemic. Confidence: medium.',
        ),
        Scenario(
            name='Inflation and Rate Shock (2022)',
            description="Persistent inflation, aggressive central-bank tightening, and the energy shock following Russia's invasion of Ukraine caused simultaneous stock and bond losses.",
            shocks={"us_equity": -0.1811, "intl_equity": -0.1445, "em_equity": -0.2009, "gov_bonds": -0.155, "corp_credit": -0.158, "gold": -0.003, "commodities": 0.161, "real_estate": -0.2495, "cash": 0.015},
            probability=0.032,
            source='historical',
            rationale='Window: 2022-01-01 to 2022-12-31; calendar-year total returns. S&P 500, MSCI EAFE net, MSCI EM net, Bloomberg Commodity, and broad US equity REIT calendar-year returns are well documented, while intermediate Treasury, investment-grade corporate, gold, and three-month bill returns are rounded benchmark-dependent approximations. Rising real yields and discount rates hurt equities, property securities, and duration together, while supply-driven commodity gains provided diversification and dollar strength reduced unhedged foreign-equity returns. Confidence: high.',
        ),
        Scenario(
            name='Dot-Com Bust (2000-2002)',
            description='Technology valuations collapsed as earnings expectations deteriorated, recession developed, and the September 11 attacks intensified risk aversion.',
            shocks={"us_equity": -0.266, "intl_equity": -0.28, "em_equity": -0.35, "gov_bonds": 0.13, "corp_credit": 0.12, "gold": 0.08, "commodities": -0.2, "real_estate": 0.11, "cash": 0.048},
            probability=0.024,
            source='historical',
            rationale='Window: 2000-10-01 to 2001-09-30; approximately the worst month-end-to-month-end 12-month S&P 500 window within the bust, not the full multi-year drawdown. The S&P 500 loss is close to its documented trailing-year total return through September 2001, while all other figures are reasoned approximate same-window USD total returns, with lower precision for emerging markets, commodities, and REITs and no claim that each asset experienced its own worst year simultaneously. Valuation compression and recession damaged equities, but policy easing supported high-quality bonds and relatively inexpensive income-producing property securities, unlike the later banking-centered crisis. Confidence: medium.',
        ),
        Scenario(
            name='Oil Embargo and Stagflation (1973-74)',
            description='The oil embargo, accelerating inflation and recession produced a severe equity bear market while undermining nominal bonds.',
            shocks={"us_equity": -0.39, "intl_equity": -0.4, "em_equity": -0.35, "gov_bonds": -0.02, "corp_credit": -0.07, "gold": 0.5, "commodities": 0.35, "real_estate": -0.5, "cash": 0.08},
            probability=0.016,
            source='historical',
            rationale='Window: 1973-09-30 to 1974-09-30; approximately the worst 12-month US equity window. The approximate S&P 500 loss and gold gain are supported by historical observations, while EAFE, Treasury, corporate, REIT and bill returns are rounded estimates; MSCI EM did not exist, so its shock is a hypothetical emerging-market proxy, and commodities assume an energy-heavy GSCI-like backcast rather than a precise investable index. Assumptions are nominal USD total returns with income reinvested, a common window for every asset and no inflation adjustment: the oil supply shock raised inflation and discount rates while compressing profits, benefiting gold and energy-heavy commodities but damaging equities, property and nominal credit. Confidence: low.',
        ),
        Scenario(
            name='Black Monday Crash (1987)',
            description='An abrupt global equity crash, amplified by portfolio insurance and strained market liquidity, erased roughly one-third of US equity value.',
            shocks={"us_equity": -0.33, "intl_equity": -0.18, "em_equity": -0.3, "gov_bonds": 0.04, "corp_credit": 0.02, "gold": 0.1, "commodities": -0.04, "real_estate": -0.2, "cash": 0.02},
            probability=0.012,
            source='historical',
            rationale='Window: 1987-08-25 to 1987-12-04; US equity peak to closing trough within the August-December episode. The S&P 500 peak-to-trough decline is well documented, while its income adjustment and the EAFE, bond, gold, commodity, REIT and bill shocks are rounded common-window estimates; MSCI EM began in 1988, making the emerging-market shock a hypothetical proxy rather than an observed index return. Assuming nominal USD total returns with income reinvested, forced equity selling and liquidity withdrawal drove the losses, while subsequent monetary easing supported Treasuries and dollar weakness cushioned unhedged foreign equities. Confidence: medium.',
        ),
        Scenario(
            name='Asian Crisis, Russia and LTCM (1997-98)',
            description="Asian currency collapses spread into a broader emerging-market funding crisis, culminating in Russia's default and the liquidity dislocation surrounding LTCM.",
            shocks={"us_equity": 0.085, "intl_equity": -0.06, "em_equity": -0.48, "gov_bonds": 0.13, "corp_credit": 0.095, "gold": -0.14, "commodities": -0.24, "real_estate": -0.14, "cash": 0.053},
            probability=0.024,
            source='historical',
            rationale="Window: 1997-08-31 to 1998-08-31; approximately the worst 12-month emerging-equity window, ending with Russia's default and early LTCM distress. The roughly halving of emerging equities, positive trailing-year S&P 500 return and falling gold prices are historically well supported, while all figures are rounded and EAFE, duration-specific bonds, broad commodities and REITs are estimates rather than verified endpoint calculations. Assuming unhedged nominal USD total returns with income reinvested, currency devaluations, dollar funding pressure and collapsing commodity demand devastated emerging assets while a flight to quality lifted Treasuries; this common window excludes the September LTCM rescue and includes earlier US equity gains. Confidence: medium.",
        ),
        Scenario(
            name='Euro Sovereign Debt Crisis (2011)',
            description='Fears of euro-area sovereign defaults and banking contagion, compounded by the US debt-ceiling confrontation, triggered a global risk-asset selloff.',
            shocks={"us_equity": -0.188, "intl_equity": -0.25, "em_equity": -0.3, "gov_bonds": 0.105, "corp_credit": 0.015, "gold": 0.08, "commodities": -0.22, "real_estate": -0.22, "cash": 0.0002},
            probability=0.028,
            source='historical',
            rationale="Window: 2011-04-29 to 2011-10-03; spring US equity closing peak to autumn closing trough. The approximately 19.4% S&P 500 price decline and gold's approximate endpoint gain are well documented, while the dividend-adjusted equity shock and EAFE, EM, intermediate Treasury, corporate, commodity, REIT and bill returns are rounded common-window estimates. Assuming nominal USD total returns with income reinvested, sovereign-bank contagion fears and weaker growth expectations depressed equities and commodities, while falling Treasury yields offset wider investment-grade credit spreads and supported safe-haven assets. Confidence: medium.",
        ),
    ]


def get_forward_scenarios() -> list[Scenario]:
    """Hypothetical forward-looking scenarios (source='custom')."""
    return [
        Scenario(
            name='AI / Mega-Cap Bubble Unwind',
            description='Disappointing AI monetization triggers a collapse in mega-cap valuations, a capital-spending retrenchment and a broader equity bear market.',
            shocks={"us_equity": -0.36, "intl_equity": -0.25, "em_equity": -0.28, "gov_bonds": 0.1, "corp_credit": 0.01, "gold": 0.14, "commodities": -0.18, "real_estate": -0.18, "cash": 0.03},
            probability=0.032,
            source='custom',
            rationale='Window: hypothetical, next 12 months. The closest analogue is the 2000-2002 dot-com bust, whose equity losses and Treasury gains are well documented; all nine shocks are forward-looking estimates, and throughout these scenarios returns assume unhedged USD exposure with income reinvested over a stressed roughly 12-month window. Concentrated valuation losses and weaker technology investment depress global earnings, while disinflation and rate cuts lift Treasuries and approximately offset investment-grade spread widening; the REIT decline assumes greater financing vulnerability than in the historical analogue. Confidence: medium.',
        ),
        Scenario(
            name='Taiwan Conflict or Blockade',
            description='A sustained blockade or military conflict around Taiwan disrupts semiconductor production and Asian shipping while triggering sanctions and global risk aversion.',
            shocks={"us_equity": -0.35, "intl_equity": -0.32, "em_equity": -0.48, "gov_bonds": 0.04, "corp_credit": -0.11, "gold": 0.24, "commodities": 0.15, "real_estate": -0.27, "cash": 0.03},
            probability=0.012,
            source='custom',
            rationale="Window: hypothetical, next 12 months. The closest analogue is Russia's 2022 invasion of Ukraine, whose regional equity losses and initial commodity surge are well documented, but all nine returns are estimates because a Taiwan semiconductor and trade disruption has no close historical equivalent. Supply shortages and sanctions combine recessionary demand destruction with inflation, limiting Treasury protection while lifting gold and commodities; emerging-market losses assume severe Taiwan and China markdowns with impaired but ultimately realizable market access. Confidence: low.",
        ),
        Scenario(
            name='US Treasury Confidence Crisis',
            description='A fiscal credibility shock drives Treasury yields sharply higher and the dollar lower despite weakening US growth.',
            shocks={"us_equity": -0.3, "intl_equity": -0.12, "em_equity": -0.17, "gov_bonds": -0.22, "corp_credit": -0.24, "gold": 0.35, "commodities": 0.14, "real_estate": -0.34, "cash": 0.045},
            probability=0.012,
            source='custom',
            rationale='Window: hypothetical, next 12 months. The closest analogue is the September-October 2022 UK gilt and sterling confidence crisis, whose yield spike and currency weakness are well documented; all nine US-centered returns are estimates, including an assumed roughly 350-basis-point intermediate Treasury yield increase and substantial dollar depreciation. Higher term premiums and forced deleveraging hurt duration, credit and property together, while dollar weakness cushions foreign equities in USD and supports gold; cash assumes Treasury bills continue paying principal and interest without default. Confidence: low.',
        ),
        Scenario(
            name='Middle East Oil Supply Shock',
            description='A major Middle East supply interruption raises oil prices by 60% and produces a global stagflationary shock.',
            shocks={"us_equity": -0.23, "intl_equity": -0.26, "em_equity": -0.22, "gov_bonds": -0.055, "corp_credit": -0.1, "gold": 0.16, "commodities": 0.28, "real_estate": -0.24, "cash": 0.045},
            probability=0.024,
            source='custom',
            rationale='Window: hypothetical, next 12 months. The closest analogue is the 1990 Iraq-Kuwait oil shock, whose oil-price surge and equity drawdown are well documented; all nine returns are estimates, with Treasury losses assuming more persistent inflation and less policy easing than in that episode. Assuming oil remains about 60% above its starting level, higher input costs squeeze consumption and margins while restrictive rates hurt bonds and REITs, and a diversified commodity basket rises less than oil because non-energy components and futures carry dilute the gain. Confidence: medium.',
        ),
        Scenario(
            name='Disorderly Global Deflationary Recession',
            description='A synchronized global recession and credit contraction trigger forced deleveraging, collapsing commodity demand and a deflation scare.',
            shocks={"us_equity": -0.45, "intl_equity": -0.48, "em_equity": -0.52, "gov_bonds": 0.16, "corp_credit": -0.12, "gold": 0.08, "commodities": -0.4, "real_estate": -0.55, "cash": 0.02},
            probability=0.028,
            source='custom',
            rationale='Window: hypothetical, next 12 months. The closest analogue is the 2008-2009 global financial crisis, whose severe equity, commodity and REIT losses and Treasury rally are well documented; all nine shocks are rounded forward-looking estimates rather than returns copied from different historical peak-to-trough dates. Forced selling, falling earnings and widening credit spreads overwhelm lower risk-free yields outside Treasuries, while aggressive rate cuts reduce bill income and gold finishes modestly higher despite an assumed initial liquidation-driven selloff. Confidence: medium.',
        ),
    ]


def get_scenario_library() -> list[Scenario]:
    """Everything the UI offers: historical first, then forward-looking."""
    return get_historical_scenarios() + get_forward_scenarios()


def get_preset_portfolios() -> dict[str, Portfolio]:
    """Named starting portfolios for the UI."""
    return {
        'Balanced 60/40': Portfolio(weights={"us_equity": 0.4, "intl_equity": 0.15, "em_equity": 0.05, "gov_bonds": 0.25, "corp_credit": 0.15, "gold": 0.0, "commodities": 0.0, "real_estate": 0.0, "cash": 0.0}),
        'Conservative income': Portfolio(weights={"us_equity": 0.18, "intl_equity": 0.07, "em_equity": 0.0, "gov_bonds": 0.4, "corp_credit": 0.25, "gold": 0.0, "commodities": 0.0, "real_estate": 0.05, "cash": 0.05}),
        'Aggressive growth': Portfolio(weights={"us_equity": 0.5, "intl_equity": 0.2, "em_equity": 0.15, "gov_bonds": 0.0, "corp_credit": 0.0, "gold": 0.0, "commodities": 0.05, "real_estate": 0.1, "cash": 0.0}),
        'All-weather / risk-parity style': Portfolio(weights={"us_equity": 0.2, "intl_equity": 0.05, "em_equity": 0.05, "gov_bonds": 0.35, "corp_credit": 0.1, "gold": 0.1, "commodities": 0.1, "real_estate": 0.0, "cash": 0.05}),
        'Endowment style (real-asset heavy)': Portfolio(weights={"us_equity": 0.25, "intl_equity": 0.15, "em_equity": 0.1, "gov_bonds": 0.1, "corp_credit": 0.05, "gold": 0.1, "commodities": 0.1, "real_estate": 0.15, "cash": 0.0}),
        'Capital preservation': Portfolio(weights={"us_equity": 0.05, "intl_equity": 0.0, "em_equity": 0.0, "gov_bonds": 0.2, "corp_credit": 0.05, "gold": 0.05, "commodities": 0.0, "real_estate": 0.0, "cash": 0.65}),
    }
