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
# Until these are implemented, money_talks.pipeline falls back to mock data.
# ---------------------------------------------------------------------------

from money_talks.contracts import Portfolio, Scenario  # noqa: E402


def get_baseline_returns() -> dict[str, float]:
    """Expected 12-month return per asset class when no shock happens."""
    raise NotImplementedError("Role 2: implement baseline expected returns.")


def get_historical_scenarios() -> list[Scenario]:
    """Library of historical crises as Scenario objects (source='historical')."""
    raise NotImplementedError("Role 2: implement the historical scenario library.")


def get_preset_portfolios() -> dict[str, Portfolio]:
    """Named starting portfolios for the UI, e.g. 'Balanced 60/40'."""
    raise NotImplementedError("Role 2: implement preset portfolios.")
