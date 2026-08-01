"""
preprocessing.py
----------------
Faithful reproduction of the original PowerGrid dataset paper's preprocessing
for the node-level Power Flow (PF) and Optimal Power Flow (OPF) regression
tasks (adapted from the authors' `PowerGrid(InMemoryDataset)`, datatype
'node' / 'nodeopf').

If you're comparing against this, note the specific things that make it
different from a "generic" .mat -> PyG pipeline -- getting any one of these
wrong is enough to reproduce the "order of magnitude off" symptom:

1. X.mat / Y_polar.mat (and the *_opf variants) are MATLAB CELL ARRAYS,
   one cell per scenario -- not a dense (S, N, F) tensor. They're normally
   saved as v7.3, so they must be loaded with `mat73` (or h5py), NOT
   `scipy.io.loadmat`. scipy will load a v7.3 cell-array file without
   erroring, but hands back a 1-D *object* array (one opaque blob per
   scenario) instead of stacked numeric data -- this is what caused
   `ValueError: not enough values to unpack (expected 3, got 1)`.

2. edge_index.mat / edge_attr.mat ARE small dense numeric matrices and
   load fine with scipy.io.loadmat, same as the reference code.

3. Normalization is MAX-ABS scaling (`x / max(|x|)`), computed once over
   ALL scenarios concatenated together -- not z-score, and not fit on the
   train split only. (This does mean train stats "see" val/test data,
   same as the reference code; there's a `fit_on_train_only` flag below
   if you want to avoid that leakage, but it will no longer match
   published numbers exactly.)

4. A boolean `mask = (Y != 0)` is stored per node/feature and should be
   used to restrict the loss/metrics to only the entries where mask is
   True. Certain target entries are structurally zero (e.g. reactive
   power at some bus types, a slack bus's fixed angle). Averaging over
   ALL entries including these dilutes/distorts the metric relative to
   the benchmark number -- this is the most common cause of the
   order-of-magnitude mismatch.

5. edge_attr is L2-normalized per column (`F.normalize(edge_attr, dim=0)`),
   not z-scored.

6. edge_index is used exactly as given in the file for the node/nodeopf
   tasks -- NOT made bidirectional here. (The reference code only adds
   reverse edges for the separate binary/regression/multiclass
   graph-level tasks.) If your edge_index.mat only lists one direction
   per branch and you want a bidirectional graph, add reverse edges
   yourself -- but check `edge_index.mat`'s shape/semantics first rather
   than silently doubling edges, since that changes node degree and can
   itself shift GNN output scale.

7. OPF drops 3 of the 6 X columns by default (keeps only indices [0,1,3]
   = Pg-Pd, Qg-Qd, theta), matching `X['X'][i][:, [0, 1, 3]]` in the
   reference code. PF keeps all 6 columns. Override with `x_cols=` if
   your data layout differs.

Usage
-----
    from preprocessing import build_dataset, save_dataset

    train_list, val_list, test_list, meta = build_dataset(
        data_dir="raw_data", problem="pf",
    )
    save_dataset(train_list, val_list, test_list, meta, out_dir="processed_pf")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from mat_io import load_numeric_mat, load_cell_mat


# --------------------------------------------------------------------------
# Config: raw file name + the MATLAB variable name(s) to try inside it.
# The paper's own code is inconsistent about variable names across files
# (e.g. 'Xpf'/'Y_polarpf' inside X.mat/Y_polar.mat for PF, but plain
# 'X'/'Y_polar' inside Xopf.mat/Y_polar_opf.mat for OPF) -- candidate_keys
# lists are tried in order, then we fall back to autodetection.
# --------------------------------------------------------------------------

FILE_MAP = {
    "pf": {
        "edge_index": ("edge_index.mat", ["edge_index", "bList"]),
        "edge_attr":  ("edge_attr.mat", ["edge_attr"]),
        "X":          ("X.mat", ["Xpf", "X"]),
        "Y":          ("Y_polar.mat", ["Y_polarpf", "Y_polar"]),
        "x_cols": None,            # keep all input columns
    },
    "opf": {
        "edge_index": ("edge_index_opf.mat", ["edge_index", "bList"]),
        "edge_attr":  ("edge_attr_opf.mat", ["edge_attr"]),
        "X":          ("Xopf.mat", ["X", "Xopf"]),
        "Y":          ("Y_polar_opf.mat", ["Y_polar", "Y_polar_opf"]),
        "x_cols": [0, 1, 3],       # Pg-Pd, Qg-Qd, theta (matches reference)
    },
}

X_FEATURE_NAMES_FULL = ["Pg-Pd", "Qg-Qd", "V", "theta", "N_loads", "N_gen"]
Y_FEATURE_NAMES = ["Pg-Pd", "Qg-Qd", "V", "theta"]


# --------------------------------------------------------------------------
# Core build function
# --------------------------------------------------------------------------

def build_dataset(
    data_dir: str,
    problem: str = "pf",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    x_cols: Optional[List[int]] = None,
    fit_on_train_only: bool = False,
    symmetrize_edges: bool = False,
):
    """
    Load raw .mat files and build a list of PyG `Data` objects (one per
    scenario), split into train/val/test, exactly matching the reference
    PowerGrid paper's node/nodeopf preprocessing.

    Parameters
    ----------
    fit_on_train_only : if True, computes the max-abs scaling stats using
        only the train split (avoids train/val/test leakage). Default
        False to match the reference code's behavior (fit on all
        scenarios) -- set True only if you're not trying to reproduce
        published numbers exactly.
    symmetrize_edges : if True, adds the reverse of every edge. Default
        False to match the reference code for node/nodeopf tasks. Only
        turn this on if you've checked edge_index.mat and confirmed it's
        one-directional per branch and your model needs both directions.

    Returns
    -------
    train_list, val_list, test_list : List[torch_geometric.data.Data]
        Each Data has: x, edge_index, edge_attr, y, mask (bool, same
        shape as y, True where y != 0), maxs (the maxsY scaling vector,
        for de-normalizing predictions back to physical units).
    meta : dict with num_node_features, num_edge_features, num_targets,
        num_nodes, maxsX, maxsY.
    """
    assert problem in FILE_MAP, f"problem must be one of {list(FILE_MAP)}"
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
    cfg = FILE_MAP[problem]
    data_dir = Path(data_dir)
    x_cols = x_cols if x_cols is not None else cfg["x_cols"]

    ei_file, ei_keys = cfg["edge_index"]
    ea_file, ea_keys = cfg["edge_attr"]
    x_file, x_keys = cfg["X"]
    y_file, y_keys = cfg["Y"]

    edge_index_raw, used_ei_key = load_numeric_mat(data_dir / ei_file, candidate_keys=ei_keys)
    edge_attr_raw, used_ea_key = load_numeric_mat(data_dir / ea_file, candidate_keys=ea_keys)
    X_scenarios, used_x_key = load_cell_mat(data_dir / x_file, candidate_keys=x_keys)
    Y_scenarios, used_y_key = load_cell_mat(data_dir / y_file, candidate_keys=y_keys)

    print(f"[{problem}] X: '{x_file}' key='{used_x_key}' ({len(X_scenarios)} scenarios, "
          f"each shape {np.asarray(X_scenarios[0]).shape})")
    print(f"[{problem}] Y: '{y_file}' key='{used_y_key}' ({len(Y_scenarios)} scenarios, "
          f"each shape {np.asarray(Y_scenarios[0]).shape})")
    print(f"[{problem}] edge_index: '{ei_file}' key='{used_ei_key}' shape {edge_index_raw.shape}")
    print(f"[{problem}] edge_attr:  '{ea_file}' key='{used_ea_key}' shape {edge_attr_raw.shape}")

    assert len(X_scenarios) == len(Y_scenarios), (
        f"X has {len(X_scenarios)} scenarios but Y has {len(Y_scenarios)} -- mismatched files?"
    )

    # --- edge_index: (E,2) or (2,E), 1-indexed MATLAB -> 0-indexed torch, (2,E) ---
    ei = np.asarray(edge_index_raw)
    if ei.shape[0] == 2 and ei.shape[1] != 2:
        edge_index = torch.as_tensor(ei, dtype=torch.long)
    else:
        edge_index = torch.as_tensor(ei, dtype=torch.long).t().contiguous()
    if edge_index.min() >= 1:
        edge_index = edge_index - 1

    if symmetrize_edges:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # --- edge_attr: L2-normalize per column, matching F.normalize(edge_attr, dim=0) ---
    edge_attr = torch.as_tensor(np.asarray(edge_attr_raw), dtype=torch.float32)
    if symmetrize_edges:
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    edge_attr = F.normalize(edge_attr, dim=0)

    # --- per-scenario X / Y, with optional column selection on X ---
    fullX, fullY = [], []
    for xi, yi in zip(X_scenarios, Y_scenarios):
        xi = np.asarray(xi, dtype=np.float32)
        yi = np.asarray(yi, dtype=np.float32)
        if x_cols is not None:
            xi = xi[:, x_cols]
        fullX.append(torch.as_tensor(xi))
        fullY.append(torch.as_tensor(yi))

    num_scenarios = len(fullX)
    num_nodes = fullX[0].shape[0]
    num_node_features = fullX[0].shape[-1]
    num_targets = fullY[0].shape[-1]

    # --- split scenario indices first (needed either way; used for
    #     fit_on_train_only, and always used for the final split) ---
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_scenarios)
    n_train = int(train_frac * num_scenarios)
    n_val = int(val_frac * num_scenarios)
    n_test = num_scenarios - n_train - n_val
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:n_train + n_val + n_test]

    # --- max-abs scaling stats (matches reference: torch.max(torch.abs(cat), dim=0)) ---
    if fit_on_train_only:
        fit_X = torch.cat([fullX[i] for i in train_idx], dim=0)
        fit_Y = torch.cat([fullY[i] for i in train_idx], dim=0)
    else:
        fit_X = torch.cat(fullX, dim=0)
        fit_Y = torch.cat(fullY, dim=0)

    maxsX, _ = torch.max(torch.abs(fit_X), dim=0)
    maxsY, _ = torch.max(torch.abs(fit_Y), dim=0)
    maxsX = maxsX.clamp_min(1e-12)
    maxsY = maxsY.clamp_min(1e-12)

    # --- build Data objects ---
    data_list = []
    for xi, yi in zip(fullX, fullY):
        mask = yi != 0
        x_norm = xi / maxsX
        y_norm = yi / maxsY
        data = Data(
            x=x_norm,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_norm,
            mask=mask,
            maxs=maxsY.unsqueeze(0),   # keep as (1, F) so PyG batches it cleanly
        )
        data_list.append(data)

    train_list = [data_list[i] for i in train_idx]
    val_list = [data_list[i] for i in val_idx]
    test_list = [data_list[i] for i in test_idx]

    meta = dict(
        num_node_features=num_node_features,
        num_edge_features=edge_attr.shape[-1],
        num_targets=num_targets,
        num_nodes=num_nodes,
        maxsX=maxsX,
        maxsY=maxsY,
        x_cols=x_cols,
        x_feature_names=(
            X_FEATURE_NAMES_FULL if x_cols is None
            else [X_FEATURE_NAMES_FULL[c] for c in x_cols]
        ),
        y_feature_names=Y_FEATURE_NAMES,
    )
    return train_list, val_list, test_list, meta


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_dataset(train_list, val_list, test_list, meta: dict, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_list, out_dir / "train.pt")
    torch.save(val_list, out_dir / "val.pt")
    torch.save(test_list, out_dir / "test.pt")
    torch.save(meta, out_dir / "meta.pt")
    print(f"Saved {len(train_list)} train / {len(val_list)} val / {len(test_list)} test graphs to {out_dir}")


def load_dataset(out_dir: str):
    out_dir = Path(out_dir)
    train_list = torch.load(out_dir / "train.pt", weights_only=False)
    val_list = torch.load(out_dir / "val.pt", weights_only=False)
    test_list = torch.load(out_dir / "test.pt", weights_only=False)
    meta = torch.load(out_dir / "meta.pt", weights_only=False)
    return train_list, val_list, test_list, meta


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess power-flow / OPF .mat data (paper-faithful pipeline)")
    parser.add_argument("--data_dir", type=str, required=True, help="folder containing the raw .mat files")
    parser.add_argument("--out_dir", type=str, default="processed", help="where to save processed tensors")
    parser.add_argument("--problem", type=str, default="pf", choices=["pf", "opf"])
    parser.add_argument("--train_frac", type=float, default=0.85)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x_cols", type=int, nargs="*", default=None,
                         help="override which X columns to keep, e.g. --x_cols 0 1 3")
    parser.add_argument("--fit_on_train_only", action="store_true",
                         help="fit max-abs scaling on train split only instead of all scenarios "
                              "(deviates from the reference paper but avoids leakage)")
    parser.add_argument("--symmetrize_edges", action="store_true",
                         help="add reverse edges (off by default to match reference node/nodeopf code)")
    args = parser.parse_args()

    train_list, val_list, test_list, meta = build_dataset(
        data_dir=args.data_dir,
        problem=args.problem,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        x_cols=args.x_cols,
        fit_on_train_only=args.fit_on_train_only,
        symmetrize_edges=args.symmetrize_edges,
    )
    save_dataset(train_list, val_list, test_list, meta, out_dir=args.out_dir)
    print("Node features:", meta["num_node_features"], "| Edge features:", meta["num_edge_features"],
          "| Targets:", meta["num_targets"], "| Nodes/graph:", meta["num_nodes"])
    print("x_feature_names:", meta["x_feature_names"])