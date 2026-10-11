"""Graph neural network baselines (GCN and GAT).

Implemented in plain PyTorch against a dense adjacency matrix, so no
torch_geometric install is required. Topologies here have 14-300 buses, for
which an (N, N) dense adjacency is small and fast.

GNNs are the current state-of-the-art family for FDIA detection, which is
why their absence from Table III is the single most damaging gap Reviewer 1
identified (comment 8).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines._common import SparseEmbeddingFrontend, resolve_emb_dim
from baselines._sequence import pool_sequence
from recad.detector import DataSpec, DetectorBase


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """Symmetric-normalised graph convolution: H' = sigma(A_hat H W)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F), adj_norm: (N, N)
        h = torch.einsum("nm,bmf->bnf", adj_norm, x)
        h = self.linear(h)
        h = F.relu(h)
        return F.dropout(h, p=self.dropout, training=self.training)


class GATLayer(nn.Module):
    """Multi-head graph attention with additive attention coefficients."""

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 heads: int = 4,
                 dropout: float = 0.0,
                 negative_slope: float = 0.2) -> None:
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.heads, self.out_dim)

        e_src = torch.einsum("bnhd,hd->bnh", h, self.a_src)
        e_dst = torch.einsum("bnhd,hd->bnh", h, self.a_dst)
        e = e_src.unsqueeze(2) + e_dst.unsqueeze(1)          # (B, N, N, H)
        e = F.leaky_relu(e, self.negative_slope)

        # Self-loops are baked into `adj` by the model, so no row is empty.
        mask = (adj > 0).unsqueeze(0).unsqueeze(-1)          # (1, N, N, 1)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.einsum("bnmh,bmhd->bnhd", alpha, h)      # (B, N, H, D)
        return F.elu(out.reshape(B, N, self.heads * self.out_dim))


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class GNNDetector(DetectorBase):
    """GCN or GAT encoder + mean readout + linear head."""

    needs_node_feat = True
    needs_adj = True

    def __init__(self,
                 spec: DataSpec,
                 cfg: Dict,
                 name: str = "GNN",
                 variant: str = "gcn",
                 emb_dim: Optional[int] = None) -> None:
        super().__init__(spec, cfg, name)
        if emb_dim is None:
            emb_dim = resolve_emb_dim(cfg)
        bcfg = (cfg.get("baselines", {}) or {}).get("gnn", {}) or {}
        hidden = int(bcfg.get("hidden", 64))
        layers = int(bcfg.get("layers", 2))
        heads = int(bcfg.get("heads", 4))
        dropout = float(bcfg.get("dropout", 0.1))

        if variant not in ("gcn", "gat"):
            raise ValueError(f"unknown GNN variant {variant!r}")
        self.variant = variant

        self.embedding = SparseEmbeddingFrontend(spec.sparse_cardinalities, emb_dim)
        self.emb_dim = emb_dim

        # GAT concatenates attention heads, so each head must produce
        # hidden/heads features for the layer output to stay `hidden` wide.
        # Without this the second GAT layer receives heads*hidden inputs and
        # the matmul shape check fails.
        if variant == "gat":
            per_head = max(1, hidden // heads)
            hidden = per_head * heads
            if hidden % heads != 0:
                raise ValueError("for GAT, hidden must be divisible by heads")
        self.hidden = hidden
        self.per_head = hidden // heads if variant == "gat" else None

        dims: List[int] = [spec.node_feat_dim] + [hidden] * layers
        blocks: List[nn.Module] = []
        for i in range(layers):
            if variant == "gcn":
                blocks.append(GCNLayer(dims[i], dims[i + 1], dropout))
            else:
                blocks.append(GATLayer(dims[i], self.per_head, heads=heads,
                                       dropout=dropout))
        self.layers = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            # 3x because mean, max and std pooling are concatenated.
            nn.Linear(3 * hidden + emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._adj_norm_cache: Dict[int, torch.Tensor] = {}

    # -- graph preprocessing ----------------------------------------------

    def _normalised_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Symmetric normalisation of (A + I). Cached per adjacency tensor."""
        key = int(adj.data_ptr())
        if key in self._adj_norm_cache:
            return self._adj_norm_cache[key]

        n = adj.shape[0]
        a = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)
        deg = a.sum(dim=1)
        d_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        norm = d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)
        self._adj_norm_cache[key] = norm
        return norm

    # -- forward -----------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["node_feat"]
        adj = batch["adj"]
        if adj.dim() == 3:                      # (B, N, N) -> (N, N)
            adj = adj[0]

        if self.variant == "gcn":
            adj_used = self._normalised_adj(adj)
            for layer in self.layers:
                x = layer(x, adj_used)
        else:
            # Self-loops so every node attends to at least itself.
            adj_used = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
            for layer in self.layers:
                x = layer(x, adj_used)

        graph = pool_sequence(x)
        sparse = self.embedding(batch["sparse"])
        return self.head(torch.cat([graph, sparse], dim=1)).squeeze(-1)

    def extra_metrics(self, y_true, y_score, attack_type=None) -> Dict[str, float]:
        return {
            "embedding_rows": float(self.embedding.embedding_rows()),
            "embedding_bytes": float(self.embedding.embedding_bytes()),
        }
