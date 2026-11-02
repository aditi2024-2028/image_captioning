"""
scripts/compare.py
==================
Diff any run's metrics against the frozen E0-R baseline targets.

Usage
-----
    python scripts/compare.py --metrics outputs/e1a/metrics.json
    python scripts/compare.py --metrics outputs/e1a/metrics.json --csv

Frozen E0-R targets (never edit these — they define the baseline)
-----------------------------------------------------------------
  BLEU-1   77.60 %
  BLEU-2   61.70 %
  BLEU-3   47.50 %
  BLEU-4   36.30 %
  METEOR   28.00 %
  ROUGE-L  57.20 %
  CIDEr   118.30
  CHAIR_s   7.98 %
  CHAIR_i   5.3451 %
"""

import argparse
import json
import sys

E0R_TARGETS: dict[str, float] = {
    "bleu_1"  : 77.60,
    "bleu_2"  : 61.70,
    "bleu_3"  : 47.50,
    "bleu_4"  : 36.30,
    "meteor"  : 28.00,
    "rouge_l" : 57.20,
    "cider"   : 118.30,
    "chair_s" : 7.98,
    "chair_i" : 5.3451,
}

# For CHAIR lower is better; for everything else higher is better
LOWER_IS_BETTER = {"chair_s", "chair_i"}


def compare(metrics_path: str, csv: bool = False) -> None:
    with open(metrics_path) as f:
        run = json.load(f)

    headers = ["metric", "e0r_target", "this_run", "delta", "direction"]
    rows: list[dict] = []

    for metric, target in E0R_TARGETS.items():
        actual = run.get(metric)
        if actual is None:
            rows.append({
                "metric"    : metric,
                "e0r_target": target,
                "this_run"  : "N/A",
                "delta"     : "N/A",
                "direction" : "—",
            })
            continue

        delta = actual - target
        if metric in LOWER_IS_BETTER:
            direction = "✓ better" if delta < 0 else ("✗ worse" if delta > 0 else "= same")
        else:
            direction = "✓ better" if delta > 0 else ("✗ worse" if delta < 0 else "= same")

        rows.append({
            "metric"    : metric,
            "e0r_target": target,
            "this_run"  : actual,
            "delta"     : round(delta, 4),
            "direction" : direction,
        })

    if csv:
        print(",".join(headers))
        for r in rows:
            print(f"{r['metric']},{r['e0r_target']},{r['this_run']},{r['delta']},{r['direction']}")
    else:
        col_w = [12, 14, 14, 10, 12]
        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
        sep = "─" * len(header_line)
        print(f"\n{'E0-R Comparison':^{len(header_line)}}")
        print(f"Run file: {metrics_path}\n")
        print(sep)
        print(header_line)
        print(sep)
        for r in rows:
            row_vals = [
                str(r["metric"]).ljust(col_w[0]),
                str(r["e0r_target"]).ljust(col_w[1]),
                str(r["this_run"]).ljust(col_w[2]),
                str(r["delta"]).ljust(col_w[3]),
                str(r["direction"]).ljust(col_w[4]),
            ]
            print("  ".join(row_vals))
        print(sep + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a run's metrics against E0-R targets")
    parser.add_argument("--metrics", required=True, help="Path to metrics.json from evaluate.py")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    args = parser.parse_args()
    compare(args.metrics, csv=args.csv)


if __name__ == "__main__":
    main()
