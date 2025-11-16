import argparse
import json
import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import pandas as pd

from simai.parser import load_csv
from simai.analyzer import analyze


def _diff_stats(stats1: Dict[str, Any], stats2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute per-column differences between two sets of stats.
    Includes absolute and relative differences.
    """
    diff: Dict[str, Any] = {}

    numeric_keys = [
        "mean",
        "min",
        "max",
        "std",
        "first_val",
        "last_val",
        "trend",
    ]

    for key in numeric_keys:
        v1 = stats1.get(key, 0.0)
        v2 = stats2.get(key, 0.0)

        abs_diff = v2 - v1
        if v1 != 0 and v1 is not None:
            rel_diff = abs_diff / v1 * 100.0
        else:
            rel_diff = None

        diff[key] = {
            "run1": v1,
            "run2": v2,
            "abs_diff": abs_diff,
            "rel_diff_percent": rel_diff,
        }

    # monotonicity comparison
    mono1 = (
        "increasing"
        if stats1.get("is_monotonic_increasing")
        else ("decreasing" if stats1.get("is_monotonic_decreasing") else "none")
    )
    mono2 = (
        "increasing"
        if stats2.get("is_monotonic_increasing")
        else ("decreasing" if stats2.get("is_monotonic_decreasing") else "none")
    )
    diff["monotonic"] = {"run1": mono1, "run2": mono2}

    return diff


def _print_diff(col_name: str, diffs: Dict[str, Any]) -> None:
    print(f"\n=== Column: {col_name} ===")

    for key, values in diffs.items():
        if key == "monotonic":
            print(f"  monotonic: run1={values['run1']}, run2={values['run2']}")
        else:
            v1 = values["run1"]
            v2 = values["run2"]
            ad = values["abs_diff"]
            rd = values["rel_diff_percent"]

            rd_fmt = f"{rd:.2f}%" if rd is not None else "N/A"

            print(
                f"  {key}: "
                f"{v1:.3f} → {v2:.3f}   "
                f"(Δ={ad:.3f}, {rd_fmt})"
            )


def _plot_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    x_col: str,
    outdir: str = "plots_compare",
) -> None:
    """
    Save overlay comparison plots for all shared numeric columns
    (excluding the x-axis column).
    """
    os.makedirs(outdir, exist_ok=True)

    numeric1 = df1.select_dtypes(include=["number"]).columns
    numeric2 = df2.select_dtypes(include=["number"]).columns
    shared_numeric = [c for c in numeric1 if c in numeric2]

    for col in shared_numeric:
        if col == x_col:
            continue

        plt.figure(figsize=(8, 4))
        plt.plot(df1[x_col], df1[col], label="Run 1")
        plt.plot(df2[x_col], df2[col], label="Run 2")
        plt.xlabel(x_col)
        plt.ylabel(col)
        plt.title(f"{col}: Run 1 vs Run 2")
        plt.legend()
        plt.tight_layout()

        outpath = os.path.join(outdir, f"compare_{col}.png")
        plt.savefig(outpath)
        plt.close()

        print(f"Saved plot: {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two simulation CSV runs numerically."
    )
    parser.add_argument("csv1", type=str, help="Path to first CSV (Run 1)")
    parser.add_argument("csv2", type=str, help="Path to second CSV (Run 2)")
    parser.add_argument(
        "--x-col",
        type=str,
        required=True,
        help="Name of the column to use as x-axis (must exist in both CSVs).",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots_compare",
        help="Directory to save comparison plots.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to save the comparison result as a JSON file.",
    )

    args = parser.parse_args()

    # Load data
    df1 = load_csv(args.csv1)
    df2 = load_csv(args.csv2)

    if args.x_col not in df1.columns or args.x_col not in df2.columns:
        raise ValueError(
            f"x-col '{args.x_col}' must exist in both CSVs. "
            f"Found columns: run1={list(df1.columns)}, run2={list(df2.columns)}"
        )

    # Analyze both
    analysis1 = analyze(df1)
    analysis2 = analyze(df2)

    cols1 = analysis1["columns"]
    cols2 = analysis2["columns"]

    shared_cols = sorted(set(cols1.keys()) & set(cols2.keys()))

    print("\n=== SimAI Run Comparison ===")
    print(f"Run 1: {args.csv1}")
    print(f"Run 2: {args.csv2}")
    print(f"Shared numeric columns: {', '.join(shared_cols) if shared_cols else 'None'}")

    comparison_json: Dict[str, Any] = {
        "meta": {
            "run1_path": args.csv1,
            "run2_path": args.csv2,
            "x_col": args.x_col,
            "shared_columns": shared_cols,
        },
        "columns": {},
    }

    # Print diffs and build JSON
    for col in shared_cols:
        diffs = _diff_stats(cols1[col], cols2[col])
        _print_diff(col, diffs)
        comparison_json["columns"][col] = diffs

    # Plots
    _plot_comparison(df1, df2, args.x_col, outdir=args.plots_dir)

    # JSON output
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(comparison_json, f, indent=2)
        print(f"\nJSON comparison saved to: {args.json_out}")

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
