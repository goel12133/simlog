
import argparse

from simai.parser import load_csv
from simai.analyzer import analyze
from simai.visualizer import plot_columns
from simai.summarizer import summarize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimAI CSV analysis tool: analyze and summarize simulation CSV data."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the CSV file to analyze.",
    )
    parser.add_argument(
        "--x-col",
        type=str,
        default=None,
        help="Optional name of the column to use as x-axis (e.g. 'time').",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help="Directory to save generated plots.",
    )

    args = parser.parse_args()

    # 1. Load
    df = load_csv(args.csv_path)

    # 2. Analyze
    analysis = analyze(df)

    # 3. Visualize
    plot_columns(df, save_dir=args.plots_dir, x_col=args.x_col)

    # 4. Summarize (AI or fallback)
    summary_text = summarize(analysis)

    print("\n=== SimAI Analysis Summary ===\n")
    print(summary_text)
    print("\nPlots saved to:", args.plots_dir)


if __name__ == "__main__":
    main()
