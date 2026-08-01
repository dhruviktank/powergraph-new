"""
mat_io.py
---------
Robust .mat file loading. MATLAB files come in two flavors:

  * v5 / v7 / v7.2  -> readable with scipy.io.loadmat
  * v7.3            -> actually an HDF5 file, needs h5py

This module tries scipy first and transparently falls back to h5py,
so you don't need to know ahead of time which format your files are in.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Any, Dict


def _load_with_scipy(path: str) -> Dict[str, Any]:
    from scipy.io import loadmat
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def _load_with_h5py(path: str) -> Dict[str, Any]:
    import h5py

    out = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            item = f[key]
            arr = np.array(item)
            # MATLAB stores arrays column-major / transposed relative to
            # how scipy.io.loadmat would hand them back -> transpose so the
            # array shape/orientation matches the scipy.io.loadmat convention.
            if arr.ndim >= 2:
                arr = arr.T
            out[key] = arr
    return out


def load_mat(path: str) -> Dict[str, Any]:
    """Load a .mat file (any MATLAB version) into a dict of numpy arrays."""
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(path)

    try:
        return _load_with_scipy(path)
    except NotImplementedError:
        # scipy raises this specifically for v7.3 (HDF5-based) files
        return _load_with_h5py(path)
    except Exception as e:
        # Some v7.3 files raise a generic ValueError instead; try h5py too
        try:
            return _load_with_h5py(path)
        except Exception:
            raise e


def _pick_key(d: Dict[str, Any], candidate_keys=None) -> str:
    """Pick the MATLAB variable name to use out of a loaded dict."""
    if candidate_keys:
        lower_map = {k.lower(): k for k in d.keys()}
        for name in candidate_keys:
            if name and name.lower() in lower_map:
                return lower_map[name.lower()]
    # fall back: the only non-metadata key, or the largest array
    keys = [k for k in d.keys() if not k.startswith("__")]
    if len(keys) == 1:
        return keys[0]
    sizes = [(np.asarray(d[k]).size if hasattr(d[k], "__len__") else 0, k) for k in keys]
    sizes.sort(reverse=True)
    return sizes[0][1]


def load_numeric_mat(path: str, candidate_keys=None) -> np.ndarray:
    """
    Load a small, dense numeric .mat file (e.g. edge_index.mat, edge_attr.mat)
    with scipy.io.loadmat, exactly like the reference PowerGrid preprocessing.
    Returns a plain numpy array.
    """
    from scipy.io import loadmat
    d = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    d = {k: v for k, v in d.items() if not k.startswith("__")}
    key = _pick_key(d, candidate_keys)
    return np.asarray(d[key]), key


def load_cell_mat(path: str, candidate_keys=None):
    """
    Load a MATLAB cell-array-of-scenarios file (e.g. X.mat, Y_polar.mat,
    Xopf.mat, Y_polar_opf.mat). These are typically saved as v7.3 (HDF5),
    which is why the reference code uses `mat73.loadmat` for these files
    specifically (NOT scipy.io.loadmat -- scipy will load a v7.3 cell array
    but give you back a 1-D object array of opaque scenario blobs instead
    of the numeric data you want).

    Returns
    -------
    scenarios : list[np.ndarray]   one (N, F) array per scenario
    key       : str                which MATLAB variable name was used
    """
    d = None
    try:
        import mat73
        d = mat73.loadmat(str(path))
    except ImportError:
        raise ImportError(
            "This file looks like a MATLAB v7.3 cell array (one cell per "
            "scenario). Install mat73 to load it: pip install mat73"
        )
    except Exception:
        # not actually v7.3 -> fall back to scipy, which returns an object array
        from scipy.io import loadmat
        raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        d = {k: v for k, v in raw.items() if not k.startswith("__")}

    key = _pick_key(d, candidate_keys)
    cell = d[key]
    scenarios = [np.asarray(c) for c in cell]
    return scenarios, key


def get_main_array(mat_dict: Dict[str, Any], preferred_names=None) -> np.ndarray:
    """
    Pull the "real" data array out of a loaded .mat dict.

    MATLAB files usually contain exactly one meaningful variable plus some
    metadata keys. If `preferred_names` is given, those keys are tried
    first (case-insensitive); otherwise we fall back to the single
    largest array in the dict.
    """
    if preferred_names:
        lower_map = {k.lower(): k for k in mat_dict.keys()}
        for name in preferred_names:
            if name.lower() in lower_map:
                return np.asarray(mat_dict[lower_map[name.lower()]])

    # fall back: largest ndarray by element count
    candidates = [(np.asarray(v).size, k) for k, v in mat_dict.items()
                  if isinstance(v, np.ndarray)]
    if not candidates:
        raise ValueError(f"No array found in mat file. Keys: {list(mat_dict.keys())}")
    candidates.sort(reverse=True)
    return np.asarray(mat_dict[candidates[0][1]])