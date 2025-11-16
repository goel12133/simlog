
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
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Optional path to save the AI summary to a file.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to save the analysis summary as a JSON file.",
    )

    args = parser.parse_args()

    # 1. Load
    df = load_csv(args.csv_path)

    # 2. Analyze
    analysis = analyze(df)
    if args.json_out:
        import json
        with open(args.json_out, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"JSON analysis saved to: {args.json_out}")

    # 3. Visualize
    plot_columns(df, save_dir=args.plots_dir, x_col=args.x_col)

    # 4. Summarize (AI or fallback)
    summary_text = summarize(analysis)

    print("\n=== SimAI Analysis Summary ===\n")
    print(summary_text)
    print("\nPlots saved to:", args.plots_dir)

    # Save summary to file if requested
    if args.summary_file:
        with open(args.summary_file, "w", encoding="utf-8") as f:
            f.write("=== SimAI Analysis Summary ===\n\n")
            f.write(summary_text)
            f.write("\n\nPlots saved to: " + args.plots_dir + "\n")
        print(f"Summary saved to: {args.summary_file}")


if __name__ == "__main__":
    main()
