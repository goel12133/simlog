import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_columns(
    df: pd.DataFrame,
    save_dir: str = "plots",
    x_col: Optional[str] = None,
) -> None:
    """
    Plot each numeric column against x_col (if provided) or against the index.

    Saves PNGs into save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include=["number"]).columns

    # Choose x-axis column
    if x_col is not None and x_col in df.columns:
        x = df[x_col]
    else:
        # Use index as fallback
        x = df.index
        x_col = "index"

    for col in numeric_cols:
        plt.figure()
        plt.plot(x, df[col])
        plt.xlabel(x_col)
        plt.ylabel(col)
        plt.title(f"{col} vs {x_col}")
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{col}.png")
        plt.savefig(out_path)
        plt.close()
