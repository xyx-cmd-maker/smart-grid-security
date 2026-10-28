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
        # Per-dimension standardisation of the assembled feature vector, the same procedure the
        # sequence detectors apply to their channel blocks. Without it the dense block and the
        # pooled embedding differ in scale by four orders of magnitude (see check_flat_scale.py),
        # and the larger block dominates the first linear layer.
        #
        # DEFAULT FALSE ON PURPOSE: the reported autoencoder numbers were produced without
        # this step, so a fresh clone must reproduce them. Enabling it moves that baseline
        # from about 0.49 to 0.52 AUROC, which is a different result and must be reported.
        self.normalise_features = bool(
            ((cfg.get("baselines", {}) or {}).get("autoencoder", {}) or {})
            .get("normalise_features", False))
        self.register_buffer("_flat_mu", torch.zeros(0), persistent=False)
        self.register_buffer("_flat_sd", torch.ones(0), persistent=False)

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
        # Both halves must sit on the module's device. The loader puts the batch on
        # CPU; the tables and projections move with the model. Moving here keeps the
        # lookup and the concatenation consistent without either side assuming.
        dev = next(self.parameters()).device
        dense = batch["dense"].to(dev)
        if self.input_mode == "residual":
            dense = self._to_residual(dense)
        pooled = self.embedding(batch["sparse"].to(dev))
        out = torch.cat([dense, pooled], dim=1)
        # Inert until `fit` has estimated the statistics; see __init__.
        if self.normalise_features and self._flat_sd.numel() == out.shape[1]:
            out = (out - self._flat_mu.to(out.dtype)) / self._flat_sd.to(out.dtype)
        return out

    def _estimate_feature_statistics(self, train_loader) -> None:
        """Per-dimension mean and standard deviation on the TRAIN split.

        One pass before any gradient step, so the transform is fixed preprocessing the
        optimiser cannot game. Inert when normalisation is off.
        """
        if not self.normalise_features:
            return
        # Markers either side of the statistics pass, flushed so they survive a native crash.
        # Their purpose is to separate "crash inside this pass" from "crash later" using evidence
        # rather than timing.
        import sys as _sys
        if __import__('os').environ.get('_DSH_STATS_MARKERS'):
            print('STATS-PASS-BEGIN', flush=True)
            _sys.stdout.flush()
        total_sum = None
        total_sq = None
        count = 0
        with torch.no_grad():
            for batch in train_loader:
                f = self._raw_features(batch)
                total_sum = f.sum(dim=0) if total_sum is None else total_sum + f.sum(dim=0)
                total_sq = (f ** 2).sum(dim=0) if total_sq is None else total_sq + (f ** 2).sum(dim=0)
                count += int(f.shape[0])
        if count > 0 and total_sum is not None:
            mean = total_sum / count
            var = torch.clamp(total_sq / count - mean ** 2, min=1e-12)
            self._flat_mu = mean.detach()
            self._flat_sd = var.sqrt().detach().clamp_min(1e-6)
        if __import__('os').environ.get('_DSH_STATS_MARKERS'):
            print(f'STATS-PASS-END count={count}', flush=True)
            _sys.stdout.flush()
        # release the accumulators before training: on the largest system they are what the
        # optimiser then has no room for
        total_sum = None
        total_sq = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def _raw_features(self, batch):
        """The assembled vector before standardisation, used to estimate the statistics."""
        dev = next(self.parameters()).device
        dense = self._to_residual(batch["dense"].to(dev)) if self.input_mode == "residual" else batch["dense"].to(dev)
        pooled = self.embedding(batch["sparse"].to(dev))
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
