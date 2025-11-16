import argparse
from pprint import pprint
from .storage import load_runs

def cmd_runs(args: argparse.Namespace) -> None:
    runs = load_runs()
    for r in runs[-args.n:]:
        print(f"{r['run_id']} | {r['created_at']} | {r['project']} | {r['func_name']} | {r['status']}")

def cmd_show(args: argparse.Namespace) -> None:
    runs = load_runs()
    for r in runs:
        if r["run_id"] == args.run_id:
            pprint(r)
            return
    print("Run not found")

def cmd_compare(args: argparse.Namespace) -> None:
    runs = load_runs()
    r1 = next((r for r in runs if r["run_id"] == args.run1), None)
    r2 = next((r for r in runs if r["run_id"] == args.run2), None)
    if not r1 or not r2:
        print("Run(s) not found")
        return

    print(f"Comparing {r1['run_id']} vs {r2['run_id']}")
    print("\nParams diff:")
    for k in sorted(set(r1["params"].keys()) | set(r2["params"].keys())):
        v1 = r1["params"].get(k)
        v2 = r2["params"].get(k)
        if v1 != v2:
            print(f"  {k}: {v1}  ->  {v2}")

    print("\nMetrics diff:")
    for k in sorted(set(r1["metrics"].keys()) | set(r2["metrics"].keys())):
        v1 = r1["metrics"].get(k)
        v2 = r2["metrics"].get(k)
        if v1 != v2:
            print(f"  {k}: {v1}  ->  {v2}")

def main() -> None:
    parser = argparse.ArgumentParser(prog="simlog", description="Simulation logging + comparison")
    sub = parser.add_subparsers(dest="command", required=True)

    p_runs = sub.add_parser("runs", help="List recent runs")
    p_runs.add_argument("-n", type=int, default=20, help="Number of runs to show")
    p_runs.set_defaults(func=cmd_runs)

    p_show = sub.add_parser("show", help="Show a single run")
    p_show.add_argument("run_id")
    p_show.set_defaults(func=cmd_show)

    p_cmp = sub.add_parser("compare", help="Compare two runs")
    p_cmp.add_argument("run1")
    p_cmp.add_argument("run2")
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)
