from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

try:
    import mat73
except Exception:  # pragma: no cover - optional dependency
    mat73 = None


PF_X_KEYS: tuple[str, ...] = ("Xpf", "X")
PF_Y_KEYS: tuple[str, ...] = ("Y_polarpf", "Y_polar")
OPF_X_KEYS: tuple[str, ...] = ("X", "Xopf")
OPF_Y_KEYS: tuple[str, ...] = ("Y_polar", "Y_polar_opf")
EDGE_KEYS: tuple[str, ...] = ("edge_index", "bList")
EDGE_ATTR_KEYS: tuple[str, ...] = ("edge_attr",)


@dataclass
class PowerGraphBundle:
    train: list[Data]
    val: list[Data]
    test: list[Data]
    meta: dict


def _read_any_mat(path: Path) -> dict:
    path = Path(path)
    try:
        return scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
    except Exception:
        if mat73 is None:
            raise
        return mat73.loadmat(str(path))


def _pick_key(mapping: dict, candidates: Iterable[str]) -> tuple[str, object]:
    for key in candidates:
        if key in mapping:
            return key, mapping[key]
    available = ", ".join(sorted(mapping.keys()))
    raise KeyError(f"None of {tuple(candidates)} were found. Available keys: {available}")


def _load_numeric_mat(path: Path, candidates: Iterable[str]) -> tuple[str, np.ndarray]:
    mapping = _read_any_mat(path)
    key, value = _pick_key(mapping, candidates)
    array = np.asarray(value)
    return key, array


def _load_cell_mat(path: Path, candidates: Iterable[str]) -> tuple[str, list[np.ndarray]]:
    if mat73 is not None:
        try:
            mapping = mat73.loadmat(str(path))
            key, value = _pick_key(mapping, candidates)
            if isinstance(value, list):
                return key, [np.asarray(item) for item in value]
            if isinstance(value, np.ndarray) and value.dtype == object:
                return key, [np.asarray(item) for item in value.tolist()]
            return key, [np.asarray(item) for item in list(value)]
        except Exception:
            pass

    mapping = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
    key, value = _pick_key(mapping, candidates)
    array = np.asarray(value)
    if array.dtype == object:
        return key, [np.asarray(item) for item in array.reshape(-1).tolist()]
    if array.ndim == 1:
        return key, [np.asarray(item) for item in array.tolist()]
    return key, [np.asarray(item) for item in array]


def _ensure_edge_index(edge_index: np.ndarray) -> torch.Tensor:
    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape {edge_index.shape}")
    if edge_index.shape[0] == 2:
        tensor = torch.as_tensor(edge_index, dtype=torch.long)
    else:
        tensor = torch.as_tensor(edge_index, dtype=torch.long).t().contiguous()
    if tensor.min().item() >= 1:
        tensor = tensor - 1
    return tensor


def _normalize_edge_attr(edge_attr: np.ndarray, symmetrize_edges: bool) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(edge_attr), dtype=torch.float32)
    if symmetrize_edges:
        tensor = torch.cat([tensor, tensor], dim=0)
    return F.normalize(tensor, dim=0)


def _select_x_columns(problem: str, x_item: np.ndarray) -> np.ndarray:
    x_item = np.asarray(x_item, dtype=np.float32)
    if problem == "opf":
        return x_item[:, [0, 1, 3]]
    return x_item


def build_powergraph_bundle(
    data_dir: str | Path,
    problem: str = "pf",
    train_frac: float = 0.85,
    val_frac: float = 0.05,
    test_frac: float = 0.10,
    seed: int = 0,
    fit_on_train_only: bool = False,
    symmetrize_edges: bool = False,
) -> PowerGraphBundle:
    if problem not in {"pf", "opf"}:
        raise ValueError("problem must be 'pf' or 'opf'")
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    data_dir = Path(data_dir)
    if problem == "pf":
        edge_file = data_dir / "edge_index.mat"
        edge_attr_file = data_dir / "edge_attr.mat"
        x_file = data_dir / "X.mat"
        y_file = data_dir / "Y_polar.mat"
        x_keys = PF_X_KEYS
        y_keys = PF_Y_KEYS
    else:
        edge_file = data_dir / "edge_index_opf.mat"
        edge_attr_file = data_dir / "edge_attr_opf.mat"
        x_file = data_dir / "Xopf.mat"
        y_file = data_dir / "Y_polar_opf.mat"
        x_keys = OPF_X_KEYS
        y_keys = OPF_Y_KEYS

    edge_key, edge_index_raw = _load_numeric_mat(edge_file, EDGE_KEYS)
    edge_attr_key, edge_attr_raw = _load_numeric_mat(edge_attr_file, EDGE_ATTR_KEYS)
    x_key, x_scenarios = _load_cell_mat(x_file, x_keys)
    y_key, y_scenarios = _load_cell_mat(y_file, y_keys)

    if len(x_scenarios) != len(y_scenarios):
        raise ValueError(f"X scenarios ({len(x_scenarios)}) and Y scenarios ({len(y_scenarios)}) do not match")

    edge_index = _ensure_edge_index(edge_index_raw)
    if symmetrize_edges:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_attr = _normalize_edge_attr(edge_attr_raw, symmetrize_edges=symmetrize_edges)

    full_x: list[torch.Tensor] = []
    full_y: list[torch.Tensor] = []
    for x_item, y_item in zip(x_scenarios, y_scenarios):
        x_tensor = torch.as_tensor(_select_x_columns(problem, x_item), dtype=torch.float32)
        y_tensor = torch.as_tensor(np.asarray(y_item), dtype=torch.float32)
        full_x.append(x_tensor)
        full_y.append(y_tensor)

    num_scenarios = len(full_x)
    num_nodes = full_x[0].shape[0]
    num_node_features = full_x[0].shape[-1]
    num_targets = full_y[0].shape[-1]

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_scenarios, generator=generator).tolist()
    num_train = int(train_frac * num_scenarios)
    num_val = int(val_frac * num_scenarios)
    train_idx = permutation[:num_train]
    val_idx = permutation[num_train:num_train + num_val]
    test_idx = permutation[num_train + num_val:]

    if fit_on_train_only:
        fit_x = torch.cat([full_x[idx] for idx in train_idx], dim=0)
        fit_y = torch.cat([full_y[idx] for idx in train_idx], dim=0)
    else:
        fit_x = torch.cat(full_x, dim=0)
        fit_y = torch.cat(full_y, dim=0)

    maxs_x = torch.max(torch.abs(fit_x), dim=0).values.clamp_min(1e-12)
    maxs_y = torch.max(torch.abs(fit_y), dim=0).values.clamp_min(1e-12)

    data_list: list[Data] = []
    for x_tensor, y_tensor in zip(full_x, full_y):
        mask = y_tensor != 0
        data_list.append(
            Data(
                x=x_tensor / maxs_x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y_tensor / maxs_y,
                mask=mask,
                maxs=maxs_y.unsqueeze(0),
            )
        )

    meta = {
        "problem": problem,
        "num_node_features": num_node_features,
        "num_edge_features": edge_attr.shape[-1],
        "num_targets": num_targets,
        "num_nodes": num_nodes,
        "maxsX": maxs_x,
        "maxsY": maxs_y,
        "x_feature_names": ["Pg-Pd", "Qg-Qd", "V", "theta", "N_loads", "N_gen"][:num_node_features]
        if problem == "pf"
        else ["Pg-Pd", "Qg-Qd", "theta"],
        "y_feature_names": ["Pg-Pd", "Qg-Qd", "V", "theta"],
        "raw_files": {
            "edge_index": edge_file.name,
            "edge_attr": edge_attr_file.name,
            "X": x_file.name,
            "Y": y_file.name,
        },
        "mat_keys": {
            "edge_index": edge_key,
            "edge_attr": edge_attr_key,
            "X": x_key,
            "Y": y_key,
        },
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "fit_on_train_only": fit_on_train_only,
        "symmetrize_edges": symmetrize_edges,
    }

    return PowerGraphBundle(
        train=[data_list[idx] for idx in train_idx],
        val=[data_list[idx] for idx in val_idx],
        test=[data_list[idx] for idx in test_idx],
        meta=meta,
    )


def save_powergraph_bundle(bundle: PowerGraphBundle, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.train, out_dir / "train.pt")
    torch.save(bundle.val, out_dir / "val.pt")
    torch.save(bundle.test, out_dir / "test.pt")
    torch.save(bundle.meta, out_dir / "meta.pt")


def load_powergraph_bundle(out_dir: str | Path) -> PowerGraphBundle:
    out_dir = Path(out_dir)
    train = torch.load(out_dir / "train.pt", weights_only=False)
    val = torch.load(out_dir / "val.pt", weights_only=False)
    test = torch.load(out_dir / "test.pt", weights_only=False)
    meta = torch.load(out_dir / "meta.pt", weights_only=False)
    return PowerGraphBundle(train=train, val=val, test=test, meta=meta)
