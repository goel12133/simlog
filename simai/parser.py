import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Assumes first row is header. Raises a clear error if empty.
    """
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"CSV at {path} is empty.")

    # Try to ensure there is at least one numeric column
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        raise ValueError(
            "No numeric columns detected. "
            "Simulation data should have at least one numeric column."
        )

    return df
