"""
train.py
--------
Train the Graph Transformer on preprocessed power-flow / OPF data.

Loss/metrics are computed ONLY on entries where `mask` is True (mask =
target != 0 before normalization), matching the reference PowerGrid paper.
If you instead average over every entry (including structurally-zero
targets like reactive power at certain bus types or a fixed slack angle),
your numbers will not match the benchmark and will typically look off by
roughly an order of magnitude, since those zero entries are trivially
"easy" and dilute the average error.

Example
-------
    python preprocessing.py --data_dir raw_data --out_dir processed_pf --problem pf
    python train.py --data_dir processed_pf --backend pyg --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from preprocessing import load_dataset
from model import build_model


def get_loader(dataset_list, batch_size, shuffle):
    from torch_geometric.loader import DataLoader
    return DataLoader(dataset_list, batch_size=batch_size, shuffle=shuffle)


def masked_mse(pred, target, mask):
    """Mean squared error computed only over entries where mask is True."""
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask
    denom = mask.sum().clamp_min(1)
    return diff2.sum() / denom


def masked_mae_sum(pred, target, mask):
    """Sum of absolute error over masked entries, and the count of those entries
    (so callers can accumulate a running mean across batches correctly)."""
    err = (pred - target).abs() * mask
    return err.sum(dim=0), mask.sum(dim=0).clamp_min(1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_sq, total_count = 0.0, 0.0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        mask = batch.mask.float()
        diff2 = ((pred - batch.y) ** 2) * mask
        total_sq += diff2.sum().item()
        total_count += mask.sum().item()
    mse = total_sq / max(total_count, 1)
    return mse


def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    running_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = masked_mse(pred, batch.y, batch.mask.float())
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_denormalized(model, loader, device, num_targets):
    """
    Per-feature MAE in physical units, masked, using each graph's own
    `maxs` (max-abs scaling vector) to invert normalization -- matches
    the reference paper's `Y_norm = Y / maxsY` scheme (NOT z-score, so
    de-normalization here is a pure multiply, no mean to add back).
    """
    model.eval()
    sums = torch.zeros(num_targets, device=device)
    counts = torch.zeros(num_targets, device=device)
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # batch.maxs is (num_graphs_in_batch, F); expand to per-node via batch.batch
        maxs_per_node = batch.maxs[batch.batch]  # (N_total_in_batch, F)
        pred_phys = pred * maxs_per_node
        target_phys = batch.y * maxs_per_node
        mask = batch.mask.float()
        err = (pred_phys - target_phys).abs() * mask
        sums += err.sum(dim=0)
        counts += mask.sum(dim=0)
    return (sums / counts.clamp_min(1)).cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="Train Graph Transformer for PF/OPF regression")
    parser.add_argument("--data_dir", type=str, required=True, help="processed data dir from preprocessing.py")
    parser.add_argument("--backend", type=str, default="pyg", choices=["pyg", "custom"])
    parser.add_argument("--model_variant", type=str, default="current", choices=["current", "paper"],
                        help="'current' keeps the existing implementation; 'paper' uses a paper-style TransformerConv stack")
    parser.add_argument("--hidden_dim", type=int, default=32, help="paper sweep uses 8, 16, or 32")
    parser.add_argument("--num_layers", type=int, default=3, help="paper sweep uses 1, 2, or 3 message-passing layers")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="prelu", choices=["prelu", "gelu", "relu"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15, help="early stopping patience")
    parser.add_argument("--out", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_list, val_list, test_list, meta = load_dataset(args.data_dir)
    train_loader = get_loader(train_list, args.batch_size, shuffle=True)
    val_loader = get_loader(val_list, args.batch_size, shuffle=False)
    test_loader = get_loader(test_list, args.batch_size, shuffle=False)

    model = build_model(
        backend=args.backend,
        num_node_features=meta["num_node_features"],
        num_edge_features=meta["num_edge_features"],
        num_targets=meta["num_targets"],
        model_variant=args.model_variant,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        activation=args.activation,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.backend}/{args.model_variant} | Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | train_loss(masked MSE, norm) {train_loss:.6f} | "
              f"val_loss(masked MSE, norm) {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, out_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # final test evaluation using the best checkpoint
    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss = evaluate(model, test_loader, device)
    print(f"\nBest val_loss (masked MSE, normalized): {best_val_loss:.6f}")
    print(f"Test  loss (masked MSE, normalized): {test_loss:.6f}")

    denorm_mae = evaluate_denormalized(model, test_loader, device, meta["num_targets"])
    print("Test MAE in physical units (masked, de-normalized via max-abs scaling):")
    for name, val in zip(meta["y_feature_names"], denorm_mae):
        print(f"  {name:>10s}: MAE = {val:.6f}")


if __name__ == "__main__":
    main()