"""Load the dataset selected for the hackathon task."""

from money_talks.contracts import Dataset


def load_dataset() -> Dataset:
    """Load data from the approved source and return the shared Dataset type.

    Implement this after the organizers specify the data source and access rules.
    Keep credentials out of source code; read them from environment variables.
    """
    raise NotImplementedError("Choose and implement the approved data source after the task is announced.")
