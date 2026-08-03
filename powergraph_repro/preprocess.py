from __future__ import annotations

import argparse
from pathlib import Path

from .data import build_powergraph_bundle, save_powergraph_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PowerGraph-Node datasets for GAT/Transformer experiments")
    parser.add_argument("--data_dir", type=str, required=True, help="folder containing the raw .mat files")
    parser.add_argument("--out_dir", type=str, default="processed", help="folder where processed tensors are stored")
    parser.add_argument("--problem", type=str, default="pf", choices=["pf", "opf"])
    parser.add_argument("--train_frac", type=float, default=0.85)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fit_on_train_only", action="store_true")
    parser.add_argument("--symmetrize_edges", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_powergraph_bundle(
        data_dir=Path(args.data_dir),
        problem=args.problem,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        fit_on_train_only=args.fit_on_train_only,
        symmetrize_edges=args.symmetrize_edges,
    )
    save_powergraph_bundle(bundle, args.out_dir)
    print(
        f"Saved {len(bundle.train)} train / {len(bundle.val)} val / {len(bundle.test)} test graphs to {args.out_dir}"
    )
    print(bundle.meta)


if __name__ == "__main__":
    main()
