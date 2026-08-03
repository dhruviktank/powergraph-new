from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_summaries(results_dir: str | Path):
    results_dir = Path(results_dir)
    for path in results_dir.glob("*_summary.json"):
        with path.open() as f:
            yield json.load(f)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def std(values):
    values = list(values)
    if not values:
        return 0.0
    mu = mean(values)
    return (sum((value - mu) ** 2 for value in values) / len(values)) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate PowerGraph-Node run summaries")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="aggregate_summary.csv")
    args = parser.parse_args()

    grouped = defaultdict(list)
    for summary in load_summaries(args.results_dir):
        key = (
            summary.get("problem"),
            summary.get("model_name"),
            summary.get("hidden_dim"),
            summary.get("num_layers"),
            summary.get("heads"),
        )
        grouped[key].append(summary)

    rows = []
    for (problem, model_name, hidden_dim, num_layers, heads), items in grouped.items():
        rows.append(
            {
                "problem": problem,
                "model_name": model_name,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "heads": heads,
                "test_loss_mean": mean(item["test_loss"] for item in items),
                "test_loss_std": std(item["test_loss"] for item in items),
                "test_r2_mean": mean(item["test_r2"] for item in items),
                "best_val_loss_mean": mean(item["best_val_loss"] for item in items),
                "best_val_r2_mean": mean(item["best_val_r2"] for item in items),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved aggregate summary to {output_path}")


if __name__ == "__main__":
    main()
