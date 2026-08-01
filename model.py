"""
model.py
--------
Graph Transformer for node-level regression on power-flow / OPF graphs.

Two implementations are provided:

1. `GraphTransformer` - built on PyTorch Geometric's `TransformerConv`
   (Shi et al., "Masked Label Prediction: Unified Message Passing Model
   for Semi-Supervised Classification", 2020), which natively supports
   edge features (branch G_ij, B_ij) inside the attention score. This
   is the recommended, best-tested option if `torch_geometric` is
   available.

2. `EdgeAwareGraphTransformerLayer` / `CustomGraphTransformer` - a
   from-scratch multi-head sparse-attention layer, useful if you want
   full control or don't want a torch_geometric dependency for the
   model itself (preprocessing.py still benefits from it for the Data
   object, but the model here only needs torch + torch_scatter-free
   scatter-softmax done manually).

Both models:
    input  : node features (S, N, Fx) via batched PyG graphs,
             edge_index (2, E), edge_attr (E, Fe)
    output : node-level regression targets (N, Fy)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "prelu":
        return nn.PReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation '{name}', expected one of: prelu, gelu, relu")


# ==========================================================================
# 1) PyG-based Graph Transformer (recommended)
# ==========================================================================

class GraphTransformer(nn.Module):
    """
    Stack of TransformerConv layers with edge-feature conditioning,
    residual connections, layer norm, and a final MLP readout head.

    Requires: torch_geometric >= 2.0
        pip install torch_geometric
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_targets: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        edge_dim: Optional[int] = None,
        beta: bool = True,
    ):
        super().__init__()
        from torch_geometric.nn import TransformerConv

        edge_dim = edge_dim if edge_dim is not None else num_edge_features
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        head_dim = hidden_dim // num_heads

        self.input_proj = nn.Linear(num_node_features, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=head_dim,
                    heads=num_heads,
                    concat=True,
                    beta=beta,          # gated residual (learns a mixing beta)
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.activations.append(_make_activation(activation))

        self.dropout = nn.Dropout(dropout)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            _make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        h = self.input_proj(x)
        for conv, norm, activation in zip(self.convs, self.norms, self.activations):
            h_new = conv(h, edge_index, edge_attr=edge_attr)
            h = norm(h + self.dropout(h_new))   # residual + norm
            h = activation(h)
        out = self.readout(h)
        return out


class PaperGraphTransformer(nn.Module):
    """Closer match to the paper's Transformer baseline.

    This version keeps the architecture lightweight and mirrors the codebase
    pattern you shared more literally:
    - no input projection layer
    - TransformerConv with concat=False
    - ReLU after every layer
    - no residual connection
    - no LayerNorm
    - a single linear readout head
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_targets: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.0,
        activation: str = "relu",
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        from torch_geometric.nn import TransformerConv

        edge_dim = edge_dim if edge_dim is not None else num_edge_features
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        current_dim = num_node_features
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=current_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )
            current_dim = hidden_dim

        self.readout = nn.Linear(hidden_dim, num_targets)

    def forward(self, x, edge_index, edge_attr, batch=None):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = F.relu(h)
            h = self.dropout(h)
        return self.readout(h)


# ==========================================================================
# 2) From-scratch edge-aware sparse Graph Transformer layer
# ==========================================================================

def _segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Softmax of `scores` (E,) grouped by destination node `index` (E,),
    i.e. softmax over incoming edges per target node. Implemented with
    scatter ops only (no torch_scatter dependency).
    """
    # subtract per-group max for numerical stability
    max_per_node = torch.full((num_nodes,), float("-inf"), device=scores.device, dtype=scores.dtype)
    max_per_node = max_per_node.scatter_reduce(0, index, scores, reduce="amax", include_self=True)
    scores = scores - max_per_node[index]

    exp_scores = scores.exp()
    denom = torch.zeros(num_nodes, device=scores.device, dtype=scores.dtype)
    denom = denom.scatter_add(0, index, exp_scores)
    denom = denom.clamp_min(1e-16)
    return exp_scores / denom[index]


class EdgeAwareGraphTransformerLayer(nn.Module):
    """
    Single multi-head self-attention layer over graph edges, with edge
    features injected into both the attention logits and the messages
    (in the spirit of Graphormer / TransformerConv / GATv2 + edge bias).

    x         : (N, d_model)
    edge_index: (2, E)  [row = source, col = target]
    edge_attr : (E, d_edge)
    returns   : (N, d_model)
    """

    def __init__(self, d_model: int, num_heads: int, d_edge: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # project edge features into a per-head bias term and a value gate
        self.edge_bias_proj = nn.Linear(d_edge, num_heads)
        self.edge_value_proj = nn.Linear(d_edge, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x, edge_index, edge_attr):
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]  # messages flow src -> dst

        q = self.q_proj(x).view(N, self.num_heads, self.d_head)
        k = self.k_proj(x).view(N, self.num_heads, self.d_head)
        v = self.v_proj(x).view(N, self.num_heads, self.d_head)

        q_dst = q[dst]                              # (E, H, d_head)
        k_src = k[src]                               # (E, H, d_head)
        v_src = v[src]                               # (E, H, d_head)

        edge_bias = self.edge_bias_proj(edge_attr)    # (E, H)
        edge_value = self.edge_value_proj(edge_attr).view(-1, self.num_heads, self.d_head)  # (E,H,d_head)

        attn_logits = (q_dst * k_src).sum(-1) * self.scale + edge_bias  # (E, H)

        messages = v_src + edge_value  # inject edge info into the value/message

        out = torch.zeros(N, self.num_heads, self.d_head, device=x.device, dtype=x.dtype)
        for h in range(self.num_heads):
            alpha_h = _segment_softmax(attn_logits[:, h], dst, N)     # (E,)
            alpha_h = self.dropout(alpha_h)
            weighted = messages[:, h, :] * alpha_h.unsqueeze(-1)      # (E, d_head)
            out[:, h, :] = out[:, h, :].scatter_add(
                0, dst.unsqueeze(-1).expand(-1, self.d_head), weighted
            )

        out = out.reshape(N, self.d_model)
        return self.out_proj(out)


class CustomGraphTransformer(nn.Module):
    """Pure PyTorch (no torch_geometric) Graph Transformer stack for node regression."""

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_targets: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_node_features, hidden_dim)

        self.layers = nn.ModuleList([
            EdgeAwareGraphTransformerLayer(hidden_dim, num_heads, num_edge_features, dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.activations = nn.ModuleList([_make_activation(activation) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            _make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        h = self.input_proj(x)
        for attn, norm, activation, ffn, ffn_norm in zip(self.layers, self.norms, self.activations, self.ffns, self.ffn_norms):
            h = norm(h + self.dropout(attn(h, edge_index, edge_attr)))
            h = ffn_norm(h + self.dropout(ffn(h)))
            h = activation(h)
        return self.readout(h)


# ==========================================================================
# Factory
# ==========================================================================

def build_model(
    backend: str,
    num_node_features: int,
    num_edge_features: int,
    num_targets: int,
    model_variant: str = "current",
    **kwargs,
) -> nn.Module:
    """backend: 'pyg' (TransformerConv-based) or 'custom' (pure PyTorch).

    model_variant:
        - 'current': the existing residual/norm/MLP transformer in this repo
        - 'paper': a lighter transformer that mirrors the paper code more closely
    """
    if backend == "pyg":
        if model_variant == "paper":
            return PaperGraphTransformer(num_node_features, num_edge_features, num_targets, **kwargs)
        return GraphTransformer(num_node_features, num_edge_features, num_targets, **kwargs)
    elif backend == "custom":
        return CustomGraphTransformer(num_node_features, num_edge_features, num_targets, **kwargs)
    else:
        raise ValueError(f"Unknown backend '{backend}', expected 'pyg' or 'custom'")