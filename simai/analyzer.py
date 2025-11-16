from typing import Dict, Any

import numpy as np
import pandas as pd


def _column_analysis(series: pd.Series) -> Dict[str, Any]:
    """Compute basic statistics for a single numeric column."""
    diff = series.diff()

    std = float(series.std()) if len(series) > 1 else 0.0

    trend = float(series.iloc[-1] - series.iloc[0]) if len(series) > 1 else 0.0

    is_increasing = bool(np.all(diff.dropna() >= 0)) if len(series) > 1 else False
    is_decreasing = bool(np.all(diff.dropna() <= 0)) if len(series) > 1 else False

    return {
        "mean": float(series.mean()),
        "min": float(series.min()),
        "max": float(series.max()),
        "std": std,
        "first_val": float(series.iloc[0]),
        "last_val": float(series.iloc[-1]),
        "trend": trend,
        "is_monotonic_increasing": is_increasing,
        "is_monotonic_decreasing": is_decreasing,
        "count": int(series.count()),
    }


def analyze(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze all numeric columns in the DataFrame.

    Returns a dict like:
    {
      "meta": {...},
      "columns": {
        "temperature": {...stats...},
        "pressure": {...stats...},
      }
    }
    """
    numeric_df = df.select_dtypes(include=["number"])
    column_stats: Dict[str, Any] = {}

    for col in numeric_df.columns:
        column_stats[col] = _column_analysis(numeric_df[col])

    meta = {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "numeric_columns": list(numeric_df.columns),
        "all_columns": list(df.columns),
    }

    return {
        "meta": meta,
        "columns": column_stats,
    }
