"""Shared building blocks for the baseline detectors."""

from __future__ import annotations

import warnings
import numpy as np

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from recad.detector import DataSpec, DetectorBase


def resolve_emb_dim(cfg: Dict) -> int:
    """The embedding dimension Rec-AD uses, so baselines cannot diverge from it.

    Previously the baselines defaulted to 16 in four places while Rec-AD resolved 64
    from `cfg['tt']['split']`. Every cross-family comparison was therefore also a
    comparison of capacity. The module docstring says the only intended difference is
    the representation, so both sides now read one source.
    """
    from recad.model import RecADConfig
    return int(RecADConfig.from_cfg(cfg).embedding_dim)


class SparseEmbeddingFrontend(nn.Module):
    """Standard dense embedding tables + mean pooling over sparse columns.

    This is the *uncompressed* counterpart of Rec-AD's Eff-TT table. Using a
    plain `nn.Embedding` per column here is deliberate: it makes the memory
    comparison in the paper meaningful, because the only difference between a
    baseline and Rec-AD is the embedding representation, not the feature set.
    """

    def __init__(self, cardinalities: List[int], dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.tables = nn.ModuleList([nn.Embedding(c, dim) for c in cardinalities])
        self._init_weights()

    def _init_weights(self) -> None:
        for table in self.tables:
            nn.init.normal_(table.weight, mean=0.0, std=0.02)

    def forward(self, sparse: torch.Tensor) -> torch.Tensor:
        """sparse: (B, S) int64 -> (B, dim) pooled embedding."""
        if sparse.dim() == 3:            # (B, S, k) bags
            sparse = sparse.reshape(sparse.shape[0], -1)
        cols = [self.tables[i](sparse[:, i]) for i in range(sparse.shape[1])]
        return torch.stack(cols, dim=1).mean(dim=1)

    def embedding_bytes(self) -> int:
        return int(sum(t.weight.numel() * t.weight.element_size() for t in self.tables))

    def embedding_rows(self) -> int:
        return int(sum(t.num_embeddings for t in self.tables))


class FlatFeatureDetector(DetectorBase):
    """Base for detectors that consume a single flat feature vector.

    Feature vector = [dense measurements | pooled sparse embedding].
    Autoencoder, LSTM-as-MLP and other tabular baselines inherit this.
    """

    def __init__(self, spec: DataSpec, cfg: Dict, name: str,
                 emb_dim: Optional[int] = None,
                 h_matrix: Optional[np.ndarray] = None) -> None:
        super().__init__(spec, cfg, name)
        if emb_dim is None:
            emb_dim = resolve_emb_dim(cfg)
        self.embedding = SparseEmbeddingFrontend(spec.sparse_cardinalities, emb_dim)
        self.in_dim = spec.num_dense + emb_dim

        # Optional residual input. See the module docstring of
        # `_add_residual_input.py` for why this exists.
        bcfg = (cfg.get("baselines", {}) or {})
        self.input_mode = str(bcfg.get("input_mode", "dense")).lower()
        self._proj = None
        if self.input_mode == "residual":
            if h_matrix is None:
                warnings.warn(
                    f"{name}: input_mode='residual' requested but the contract has no "
                    "h_matrix; falling back to raw measurements. The run is NOT the "
                    "residual-input condition.",
                    RuntimeWarning, stacklevel=2,
                )
                self.input_mode = "dense"
            else:
                H = np.asarray(h_matrix, dtype=np.float64)
                q, r = np.linalg.qr(H)
                keep = np.abs(np.diag(r)) > 1e-10
                self._proj = q[:, keep]          # orthonormal basis of col(H)

    def _to_residual(self, dense: torch.Tensor) -> torch.Tensor:
        """z - H (H^T H)^-1 H^T z, i.e. the component outside col(H)."""
        if self._proj is None:
            return dense
        P = torch.as_tensor(self._proj, dtype=dense.dtype, device=dense.device)
        return dense - (dense @ P) @ P.T

    def features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        dense = batch["dense"]
        if self.input_mode == "residual":
            dense = self._to_residual(dense)
        pooled = self.embedding(batch["sparse"])
        return torch.cat([dense, pooled], dim=1)

    def extra_metrics(self, y_true, y_score, attack_type=None):
        return {
            "embedding_rows": float(self.embedding.embedding_rows()),
            "embedding_bytes": float(self.embedding.embedding_bytes()),
        }


def make_mlp(in_dim: int,
             hidden: List[int],
             out_dim: int = 1,
             dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)
