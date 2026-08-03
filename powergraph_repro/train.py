from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .data import PowerGraphBundle, build_powergraph_bundle, load_powergraph_bundle, save_powergraph_bundle
from .models import build_model


@dataclass
class TrainConfig:
    data_dir: str
    processed_dir: str
    results_dir: str
    model_name: str = "transformer"
    problem: str = "pf"
    hidden_dim: int = 128
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.0
    activation: str = "relu"
    norm_type: str = "none"
    attention_dropout: float = 0.0
    ffn_dim: int | None = None
    residual: bool = False
    layer_norm: bool = False
    concat_heads: bool = False
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 50
    patience: int = 10
    num_early_stop: int = 10
    batch_size: int = 32
    seed: int = 0
    train_frac: float = 0.85
    val_frac: float = 0.05
    test_frac: float = 0.10
    fit_on_train_only: bool = False
    symmetrize_edges: bool = False
    save_checkpoints: bool = True


@dataclass
class RunResult:
    config: dict[str, Any]
    best_val_loss: float
    best_val_r2: float
    test_loss: float
    test_r2: float
    node_averaged_mse: dict[str, float]
    node_averaged_mae: dict[str, float]
    history: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = (pred - target) ** 2
    masked = squared_error * mask
    return masked.sum() / mask.sum().clamp_min(1)


def masked_r2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    masked_pred = pred[mask.bool()].reshape(-1).float()
    masked_target = target[mask.bool()].reshape(-1).float()
    if masked_target.numel() == 0:
        return 0.0
    target_mean = masked_target.mean()
    ss_res = torch.sum((masked_target - masked_pred) ** 2)
    ss_tot = torch.sum((masked_target - target_mean) ** 2)
    if ss_tot.item() == 0:
        return 0.0
    return (1.0 - ss_res / ss_tot).item()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    r2_scores: list[float] = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        mask = batch.mask.float()
        loss = masked_mse(pred, batch.y, mask)
        losses.append(loss.item())
        r2_scores.append(masked_r2(pred, batch.y, mask))
    return float(np.mean(losses)), float(np.mean(r2_scores))


@torch.no_grad()
def evaluate_physical_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    feature_names: list[str],
) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]]]:
    model.eval()
    mse_sums = torch.zeros(len(feature_names), device=device)
    mae_sums = torch.zeros(len(feature_names), device=device)
    counts = torch.zeros(len(feature_names), device=device)
    rows: list[dict[str, Any]] = []
    graph_offset = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        maxs = batch.maxs[batch.batch]
        pred_phys = pred * maxs
        target_phys = batch.y * maxs
        mask = batch.mask.float()

        err = pred_phys - target_phys
        mse_sums += ((err ** 2) * mask).sum(dim=0)
        mae_sums += (err.abs() * mask).sum(dim=0)
        counts += mask.sum(dim=0)

        num_graphs = int(batch.batch.max().item()) + 1
        num_nodes = batch.x.size(0) // num_graphs
        node_ids = torch.arange(num_nodes, device=device).repeat(num_graphs)
        graph_ids = torch.arange(graph_offset, graph_offset + num_graphs, device=device).repeat_interleave(num_nodes)
        graph_offset += num_graphs

        for idx in range(pred_phys.size(0)):
            row = {"graph_id": int(graph_ids[idx].item()), "node_id": int(node_ids[idx].item())}
            for feat_idx, feat_name in enumerate(feature_names):
                t_val = float(target_phys[idx, feat_idx].item())
                p_val = float(pred_phys[idx, feat_idx].item())
                row[f"{feat_name}_target"] = t_val
                row[f"{feat_name}_pred"] = p_val
                row[f"{feat_name}_abs_err"] = abs(p_val - t_val)
                row[f"{feat_name}_sq_err"] = (p_val - t_val) ** 2
                row[f"{feat_name}_mask"] = int(mask[idx, feat_idx].item())
            rows.append(row)

    mse = (mse_sums / counts.clamp_min(1)).detach().cpu().tolist()
    mae = (mae_sums / counts.clamp_min(1)).detach().cpu().tolist()
    return (
        {name: float(val) for name, val in zip(feature_names, mse)},
        {name: float(val) for name, val in zip(feature_names, mae)},
        rows,
    )


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_or_prepare_bundle(cfg: TrainConfig) -> PowerGraphBundle:
    processed_dir = Path(cfg.processed_dir)
    if (processed_dir / "train.pt").exists():
        return load_powergraph_bundle(processed_dir)
    bundle = build_powergraph_bundle(
        data_dir=cfg.data_dir,
        problem=cfg.problem,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
        fit_on_train_only=cfg.fit_on_train_only,
        symmetrize_edges=cfg.symmetrize_edges,
    )
    save_powergraph_bundle(bundle, processed_dir)
    return bundle


def build_loaders(bundle: PowerGraphBundle, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(bundle.train, batch_size=batch_size, shuffle=True),
        DataLoader(bundle.val, batch_size=batch_size, shuffle=False),
        DataLoader(bundle.test, batch_size=batch_size, shuffle=False),
    )


def run_experiment(cfg: TrainConfig) -> RunResult:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_or_prepare_bundle(cfg)
    train_loader, val_loader, test_loader = build_loaders(bundle, cfg.batch_size)

    model = build_model(
        model_name=cfg.model_name,
        num_node_features=bundle.meta["num_node_features"],
        num_edge_features=bundle.meta["num_edge_features"],
        num_targets=bundle.meta["num_targets"],
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        heads=cfg.heads,
        dropout=cfg.dropout,
        activation=cfg.activation,
        norm_type=cfg.norm_type,
        attention_dropout=cfg.attention_dropout,
        ffn_dim=cfg.ffn_dim,
        residual=cfg.residual,
        layer_norm=cfg.layer_norm,
        concat_heads=cfg.concat_heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

    run_id = f"{bundle.meta['problem']}_{cfg.model_name}_h{cfg.hidden_dim}_l{cfg.num_layers}_s{cfg.seed}"
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / f"{run_id}_latest.pt"
    best_ckpt = ckpt_dir / f"{run_id}_best.pt"

    best_val_loss = float("inf")
    best_val_r2 = float("-inf")
    epochs_no_improve = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = masked_mse(pred, batch.y, batch.mask.float())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss, val_r2 = evaluate(model, val_loader, device)
        previous_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_r2": val_r2,
                "lr": current_lr,
                "best_val_loss": best_val_loss if best_val_loss != float("inf") else val_loss,
                "best_val_r2": best_val_r2 if best_val_r2 != float("-inf") else val_r2,
            }
        )

        print(
            f"Epoch:{epoch}, Training_loss:{train_loss:.8f}, Eval_loss:{val_loss:.8f}, Eval_r2:{val_r2:.8f}, lr:{current_lr:.8g}"
        )
        if current_lr < previous_lr:
            print(
                f"  ReduceLROnPlateau lowered lr from {previous_lr:.8g} to {current_lr:.8g} because val_loss stopped improving"
            )

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "meta": bundle.meta}, best_ckpt)

        torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "meta": bundle.meta}, latest_ckpt)

        if epoch > cfg.epochs / 2 and epochs_no_improve > cfg.num_early_stop:
            print(f"Early stopping at epoch {epoch} (no improvement for {cfg.num_early_stop} epochs)")
            break

    checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_r2 = evaluate(model, test_loader, device)
    node_mse, node_mae, test_rows = evaluate_physical_metrics(
        model=model,
        loader=test_loader,
        device=device,
        feature_names=bundle.meta["y_feature_names"],
    )

    summary = {
        "run_id": run_id,
        "problem": bundle.meta["problem"],
        "model_name": cfg.model_name,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "heads": cfg.heads,
        "seed": cfg.seed,
        "best_val_loss": best_val_loss,
        "best_val_r2": best_val_r2,
        "test_loss": test_loss,
        "test_r2": test_r2,
        "node_averaged_mse": node_mse,
        "node_averaged_mae": node_mae,
        "split": bundle.meta["split"],
        "train_frac": bundle.meta["train_frac"],
        "val_frac": bundle.meta["val_frac"],
        "test_frac": bundle.meta["test_frac"],
    }

    save_json(results_dir / f"{run_id}_summary.json", summary)
    save_csv(results_dir / f"{run_id}_history.csv", history)
    save_csv(results_dir / f"{run_id}_test_predictions.csv", test_rows)

    paper_data_rows = []
    for row in test_rows:
        paper_row = {}
        for feat_name in bundle.meta["y_feature_names"]:
            paper_row[f"{feat_name} target"] = row[f"{feat_name}_target"]
            paper_row[f"{feat_name} pred"] = row[f"{feat_name}_pred"]
        paper_data_rows.append(paper_row)
    save_csv(results_dir / f"{run_id}_paper_data.csv", paper_data_rows)
    save_csv(
        results_dir / f"{run_id}_paper_metrics.csv",
        [
            {"Metric": "MSE loss", "Value": test_loss},
            {"Metric": "R2 score", "Value": test_r2},
        ],
    )

    print(f"\nBest val_loss (masked MSE, normalized): {best_val_loss:.8f}")
    print(f"Test  loss (masked MSE, normalized): {test_loss:.8f}")
    print(f"Test  r2score: {test_r2:.8f}")
    print("Node-averaged Mean Squared Errors on the predicted physical quantities:")
    for name, value in node_mse.items():
        print(f"  {name:>10s}: MSE = {value:.8f}")
    print("Node-averaged Mean Absolute Errors on the predicted physical quantities:")
    for name, value in node_mae.items():
        print(f"  {name:>10s}: MAE = {value:.8f}")

    return RunResult(
        config=asdict(cfg),
        best_val_loss=best_val_loss,
        best_val_r2=best_val_r2,
        test_loss=test_loss,
        test_r2=test_r2,
        node_averaged_mse=node_mse,
        node_averaged_mae=node_mae,
        history=history,
        test_rows=test_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PowerGraph-Node reproduction models")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--processed_dir", type=str, default="processed_powergraph")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="transformer", choices=["gat", "transformer"])
    parser.add_argument("--problem", type=str, default="pf", choices=["pf", "opf"])
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--norm_type", type=str, default="none")
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--ffn_dim", type=int, default=None)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--layer_norm", action="store_true")
    parser.add_argument("--concat_heads", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_early_stop", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_frac", type=float, default=0.85)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.10)
    parser.add_argument("--fit_on_train_only", action="store_true")
    parser.add_argument("--symmetrize_edges", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        model_name=args.model_name,
        problem=args.problem,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
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
        patience=args.patience,
        num_early_stop=args.num_early_stop,
        batch_size=args.batch_size,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        fit_on_train_only=args.fit_on_train_only,
        symmetrize_edges=args.symmetrize_edges,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
