from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def make_activation(name: str | None) -> nn.Module:
    if name is None:
        return Identity()
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "prelu":
        return nn.PReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU(0.2)
    raise ValueError(f"Unknown activation '{name}'")


def make_norm(norm_type: str | None, num_features: int) -> nn.Module:
    if norm_type is None or norm_type.lower() in {"none", "identity"}:
        return Identity()
    norm_type = norm_type.lower()
    if norm_type == "layernorm":
        return nn.LayerNorm(num_features)
    if norm_type == "batchnorm":
        return nn.BatchNorm1d(num_features)
    raise ValueError(f"Unknown norm '{norm_type}'")


class GATRegressor(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_targets: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        activation: str = "leakyrelu",
        norm_type: str = "layernorm",
        attention_dropout: float = 0.0,
        residual: bool = True,
        readout_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_node_features, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.act = make_activation(activation)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=attention_dropout,
                    edge_dim=num_edge_features,
                )
            )
            self.norms.append(make_norm(norm_type, hidden_dim))

        readout_hidden_dim = readout_hidden_dim or hidden_dim
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, readout_hidden_dim),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(readout_hidden_dim, num_targets),
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr=edge_attr)
            h = h + self.dropout(h_new) if self.residual else h_new
            h = norm(h)
            h = self.act(h)
        return self.readout(h)


class GraphTransformerRegressor(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_targets: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_type: str = "none",
        attention_dropout: float = 0.0,
        ffn_dim: Optional[int] = None,
        residual: bool = False,
        layer_norm: bool = False,
        concat_heads: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_node_features, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.layer_norm = layer_norm
        self.act = make_activation(activation)
        self.concat_heads = concat_heads
        self.ffn_dim = ffn_dim or (hidden_dim * 2)

        self.attn_layers = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.ffns = nn.ModuleList()

        for _ in range(num_layers):
            self.attn_layers.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim if not concat_heads else hidden_dim // heads,
                    heads=heads,
                    concat=concat_heads,
                    dropout=attention_dropout,
                    edge_dim=num_edge_features,
                )
            )
            self.norm1.append(nn.LayerNorm(hidden_dim) if layer_norm else Identity())
            self.norm2.append(nn.LayerNorm(hidden_dim) if layer_norm else Identity())
            self.ffns.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, self.ffn_dim),
                    make_activation(activation),
                    nn.Dropout(dropout),
                    nn.Linear(self.ffn_dim, hidden_dim),
                )
            )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        h = self.input_proj(x)
        for attn, n1, ffn, n2 in zip(self.attn_layers, self.norm1, self.ffns, self.norm2):
            attn_out = attn(h, edge_index, edge_attr=edge_attr)
            h = h + self.dropout(attn_out) if self.residual else attn_out
            h = n1(h)
            ffn_out = ffn(h)
            h = h + self.dropout(ffn_out) if self.residual else ffn_out
            h = n2(h)
            h = self.act(h)
        return self.readout(h)

class GraphTransformerOriginal(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_targets: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        current_dim = num_node_features
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=current_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=num_edge_features,
                )
            )
            current_dim = hidden_dim

        self.dropout = dropout

        # Same prediction head as original GNN_basic
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        edge_weight=None,
        batch=None,
    ):
        # Original implementation multiplies edge attributes by edge weights
        if edge_weight is not None:
            edge_attr = edge_attr * edge_weight[:, None]

        for conv in self.convs:
            x = conv(
                x,
                edge_index,
                edge_attr=edge_attr,
            )
            x = F.relu(x)
            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )

        return self.readout(x)

@dataclass
class ModelConfig:
    model_name: str
    num_node_features: int
    num_edge_features: int
    num_targets: int
    hidden_dim: int
    num_layers: int
    dropout: float = 0.0
    activation: str = "relu"
    norm_type: str = "none"
    heads: int = 4
    attention_dropout: float = 0.0
    ffn_dim: Optional[int] = None
    residual: bool = False
    layer_norm: bool = False
    concat_heads: bool = False


def build_model(**kwargs) -> nn.Module:
    config = ModelConfig(**kwargs)
    model_name = config.model_name.lower()
    if model_name == "gat":
        return GATRegressor(
            num_node_features=config.num_node_features,
            num_edge_features=config.num_edge_features,
            num_targets=config.num_targets,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            heads=config.heads,
            dropout=config.dropout,
            activation=config.activation,
            norm_type=config.norm_type,
            attention_dropout=config.attention_dropout,
            residual=config.residual,
            readout_hidden_dim=config.ffn_dim,
        )
    if model_name in {"transformer", "graph_transformer"}:
        return GraphTransformerRegressor(
            num_node_features=config.num_node_features,
            num_edge_features=config.num_edge_features,
            num_targets=config.num_targets,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            heads=config.heads,
            dropout=config.dropout,
            activation=config.activation,
            attention_dropout=config.attention_dropout,
            ffn_dim=config.ffn_dim,
            residual=config.residual,
            layer_norm=config.layer_norm,
            concat_heads=config.concat_heads,
        )
    if model_name in {"transformer_original", "original_transformer"}:
        return GraphTransformerOriginal(
            num_node_features=config.num_node_features,
            num_edge_features=config.num_edge_features,
            num_targets=config.num_targets,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            heads=config.heads,
            dropout=config.dropout,
        )
    raise ValueError("model_name must be 'gat' or 'transformer'")
