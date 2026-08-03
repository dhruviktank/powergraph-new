from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .train import TrainConfig, run_experiment


def parse_list(values: str, cast):
    return [cast(item) for item in values.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PowerGraph-Node benchmark sweeps")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--processed_dir", type=str, default="processed_powergraph")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="transformer", choices=["gat", "transformer"])
    parser.add_argument("--problem", type=str, default="pf", choices=["pf", "opf"])
    parser.add_argument("--seeds", type=str, default="0,100,300,700,1000")
    parser.add_argument("--hidden_dims", type=str, default="8,16,32")
    parser.add_argument("--num_layers_list", type=str, default="1,2,3")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--norm_type", type=str, default="none")
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--ffn_dim", type=int, default=None)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--layer_norm", action="store_true")
    parser.add_argument("--concat_heads", action="store_true")
    parser.add_argument("--fit_on_train_only", action="store_true")
    parser.add_argument("--symmetrize_edges", action="store_true")
    args = parser.parse_args()

    seeds = parse_list(args.seeds, int)
    hidden_dims = parse_list(args.hidden_dims, int)
    num_layers_list = parse_list(args.num_layers_list, int)

    summaries = []
    for seed in seeds:
        for hidden_dim in hidden_dims:
            for num_layers in num_layers_list:
                cfg = TrainConfig(
                    data_dir=args.data_dir,
                    processed_dir=args.processed_dir,
                    results_dir=args.results_dir,
                    model_name=args.model_name,
                    problem=args.problem,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    heads=args.heads,
                    dropout=args.dropout,
                    activation=args.activation,
                    norm_type=args.norm_type,
                    attention_dropout=args.attention_dropout,
                    ffn_dim=args.ffn_dim,
                    residual=args.residual,
                    layer_norm=args.layer_norm,
                    concat_heads=args.concat_heads,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    epochs=args.epochs,
                    patience=10,
                    num_early_stop=10,
                    batch_size=args.batch_size,
                    seed=seed,
                    fit_on_train_only=args.fit_on_train_only,
                    symmetrize_edges=args.symmetrize_edges,
                )
                result = run_experiment(cfg)
                summaries.append(
                    {
                        "model_name": args.model_name,
                        "problem": args.problem,
                        "seed": seed,
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "best_val_loss": result.best_val_loss,
                        "best_val_r2": result.best_val_r2,
                        "test_loss": result.test_loss,
                        "test_r2": result.test_r2,
                    }
                )

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated_path = out_dir / f"{args.problem}_{args.model_name}_aggregate.csv"
    with aggregated_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()) if summaries else [])
        if summaries:
            writer.writeheader()
            writer.writerows(summaries)

    grouped = defaultdict(list)
    for row in summaries:
        key = (row["model_name"], row["problem"], row["hidden_dim"], row["num_layers"])
        grouped[key].append(row)

    stats_rows = []
    for (model_name, problem, hidden_dim, num_layers), rows in grouped.items():
        stats_rows.append(
            {
                "model_name": model_name,
                "problem": problem,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "test_loss_mean": sum(item["test_loss"] for item in rows) / len(rows),
                "test_loss_std": (sum((item["test_loss"] - sum(item["test_loss"] for item in rows) / len(rows)) ** 2 for item in rows) / len(rows)) ** 0.5,
                "test_r2_mean": sum(item["test_r2"] for item in rows) / len(rows),
                "best_val_loss_mean": sum(item["best_val_loss"] for item in rows) / len(rows),
                "best_val_r2_mean": sum(item["best_val_r2"] for item in rows) / len(rows),
            }
        )

    stats_path = out_dir / f"{args.problem}_{args.model_name}_aggregate_stats.csv"
    with stats_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()) if stats_rows else [])
        if stats_rows:
            writer.writeheader()
            writer.writerows(stats_rows)

    print(f"Saved per-run results to {aggregated_path}")
    print(f"Saved aggregated stats to {stats_path}")


if __name__ == "__main__":
    main()
