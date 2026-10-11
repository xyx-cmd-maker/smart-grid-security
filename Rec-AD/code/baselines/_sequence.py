"""Shared base for detectors that read a sequence.

Two token orders are possible, and the contract decides which one is used:

* **Temporal** (preferred when the contract carries a `window` array): one
  token per TIMESTEP, each carrying the full measurement vector. This is the
  order that gives the model any chance against a stealthy injection, because
  `a = Hc` is invisible in a single snapshot but shows up as a state jump
  between consecutive snapshots.
* **Spatial** (fallback): one token per BUS, as before.

Both are just `(B, L, F)` to the encoder, so LSTM/CNN/Transformer differ only
in the encoder and the comparison isolates architecture rather than input.
An input projection maps `F` to `hidden` so the encoders never need to know
which order they were handed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from baselines._common import SparseEmbeddingFrontend, resolve_emb_dim
from recad.detector import DataSpec, DetectorBase


def pool_sequence(seq: torch.Tensor) -> torch.Tensor:
    """Mean + max + std pooling over the token axis.

    Mean pooling alone discards exactly the information a sparse FDIA
    carries: an injection displaces a subset of buses, which changes the
    cross-token dispersion while barely moving the average. Pooling the
    dispersion back in is standard practice for sequence anomaly detection
    and keeps the baselines genuinely competitive rather than convenient.

    For the temporal order the max over timesteps is what catches a stepped
    injection: the jump appears in exactly one inter-step interval.
    """
    mean = seq.mean(dim=1)
    maximum = seq.amax(dim=1)
    if seq.shape[1] > 1:
        std = seq.std(dim=1, unbiased=False)
    else:
        std = torch.zeros_like(mean)
    return torch.cat([mean, maximum, std], dim=1)


class SequenceDetectorBase(DetectorBase):
    """Encoder over a token sequence + pooled sparse embedding -> logit."""

    needs_node_feat = True
    needs_adj = False

    def __init__(self,
                 spec: DataSpec,
                 cfg: Dict[str, Any],
                 name: str,
                 hidden: int,
                 emb_dim: Optional[int] = None,
                 h_matrix: Any = None) -> None:
        super().__init__(spec, cfg, name)
        if emb_dim is None:
            emb_dim = resolve_emb_dim(cfg)

        # Prefer the temporal order whenever the contract provides it: the
        # whole point of the trajectory is that a stealthy injection is only
        # visible across time. The spatial order remains for contracts
        # generated without `--window` (window=1), which keeps the old
        # comparison available as an ablation.
        self.uses_window = bool(getattr(spec, "has_window", False))
        self.needs_window = self.uses_window
        self.needs_node_feat = not self.uses_window

        # ---- physics-informed input --------------------------------
        # On RAW measurements a generic sequence model cannot find the attack.
        # Measured on this project's data: LSTM AUC 0.485 and CNN AUC 0.502
        # against the stealthy family, i.e. chance, while a deterministic
        # temporal innovation test reaches AUC 0.98 on the same input.
        #
        # The reason is dynamic range. The injection perturbs each measurement
        # by ~4% of the natural spread of that measurement, so the legitimate
        # state variation dominates whatever the encoder sees. The classical
        # test works because it first projects the measurement onto col(H) and
        # looks only at what is left, which removes ~99% of the variation.
        #
        # So the same projection is applied here, as a FIXED (non-learned)
        # preprocessing layer, and every sequence baseline gets it. That keeps
        # the architecture comparison fair -- what differs between LSTM, CNN
        # and Transformer is still only the encoder -- while no longer asking
        # them to rediscover a projection the physics already hands us.
        self.projector_: Optional[torch.Tensor] = None
        self.physics_channels = 0
        use_physics = bool(
            ((cfg.get("baselines", {}) or {}).get("sequence", {}) or {})
            .get("physics_features", True))
        self.physics_enabled = use_physics
        if self.uses_window and h_matrix is not None and use_physics:
            h = np.asarray(h_matrix, dtype=np.float64)
            proj = h @ np.linalg.pinv(h)
            self.register_buffer("_proj", torch.from_numpy(proj).float())
            self.projector_ = self._proj
            # [z_t , residual_t , innovation_t]  -> 3 x m channels
            self.physics_channels = 3

        raw_dim = (spec.num_measurements if self.uses_window
                   else spec.node_feat_dim)
        self.in_dim = raw_dim if self.physics_channels == 0 else (
            self.physics_channels * spec.num_measurements)

        # Per-channel standardisation. Without it the three channel blocks differ in
        # scale by ~1600x (raw z sd 67.9, residual 0.483, innovation 1.486) and the
        # largest dominates the first Linear layer, which measurably destroys
        # detection on every family: same model and seed, unnormalised vs normalised
        # AUC was random 0.5194 -> 1.0000, load_redist 0.5112 -> 0.8396,
        # stealthy 0.5073 -> 0.7402. The buffers are filled by `fit` from the TRAIN
        # split only; until then the transform is inert, so nothing silently changes
        # meaning for a model that was never fitted.
        self.normalise_channels = bool(
            ((cfg.get("baselines", {}) or {}).get("sequence", {}) or {})
            .get("normalise_channels", True))
        self.register_buffer("_ch_mu", torch.zeros(0), persistent=False)
        self.register_buffer("_ch_sd", torch.ones(0), persistent=False)

        self.embedding = SparseEmbeddingFrontend(spec.sparse_cardinalities, emb_dim)
        self.emb_dim = emb_dim
        self.hidden = hidden
        # Tokens are projected to `hidden` first, so every encoder works on
        # (B, L, hidden) regardless of which order it was given.
        self.input_proj = (nn.Identity() if self.in_dim == hidden
                           else nn.Linear(self.in_dim, hidden))
        self.head = self._make_head(hidden, emb_dim)

    @staticmethod
    def _make_head(hidden: int, emb_dim: int) -> nn.Sequential:
        # 3x because mean, max and std are concatenated.
        return nn.Sequential(
            nn.Linear(3 * hidden + emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def token_sequence(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, L, in_dim) tokens, temporal order preferred.

        When a Jacobian is available the token carries three views of each
        timestep rather than the raw measurement alone:

            z_t                        raw measurement
            r_t = z_t - P z_t          residual -- what BDD tests
            nu_t = z_t - P z_{t-1}     innovation -- what a Kalman filter tests

        All three are linear and computed from the physics, so no baseline gets
        an advantage the others lack. The first innovation has no predecessor
        and falls back to the residual.
        """
        if not self.uses_window:
            return batch["node_feat"]

        win = batch["window"]                                  # (B, L, m)
        if self.physics_channels == 0 or self.projector_ is None:
            return win

        proj = self.projector_.to(win.dtype)
        residual = win - win @ proj.T                          # (B, L, m)
        pred = win[:, :-1, :] @ proj.T
        innovation = torch.cat(
            [residual[:, :1, :], win[:, 1:, :] - pred], dim=1)
        tokens = torch.cat([win, residual, innovation], dim=2)
        # Applied only once `fit` has estimated the statistics; see __init__.
        if self.normalise_channels and self._ch_sd.numel() == tokens.shape[2]:
            tokens = (tokens - self._ch_mu.to(tokens.dtype)) / self._ch_sd.to(tokens.dtype)
        return tokens

    def fit(self, train_loader, val_loader=None, **kwargs):
        """Estimate per-channel statistics on the TRAIN split, then train normally.

        One pass over the training data before any gradient step, so the transform is
        fixed preprocessing the optimiser cannot game. Skipped when the physics
        channels are off or normalisation is disabled, in which case behaviour is
        byte-identical to before this change.
        """
        if (self.normalise_channels and self.physics_channels > 0
                and self.projector_ is not None):
            total_sum = None
            total_sq = None
            count = 0
            with torch.no_grad():
                for batch in train_loader:
                    t = self.token_sequence(batch)
                    if t.dim() != 3:
                        continue
                    s = t.sum(dim=(0, 1))
                    q = (t ** 2).sum(dim=(0, 1))
                    total_sum = s if total_sum is None else total_sum + s
                    total_sq = q if total_sq is None else total_sq + q
                    count += int(t.shape[0]) * int(t.shape[1])
            if count > 0 and total_sum is not None:
                mean = total_sum / count
                var = torch.clamp(total_sq / count - mean ** 2, min=1e-12)
                self._ch_mu = mean.detach()
                self._ch_sd = var.sqrt().detach().clamp_min(1e-6)
        return super().fit(train_loader, val_loader, **kwargs)

    # Subclasses implement the sequence encoder over (B, L, hidden).
    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L, hidden) -> (B, L, hidden)"""
        raise NotImplementedError

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = self.input_proj(self.token_sequence(batch))
        seq = self.encode(tokens)
        pooled = pool_sequence(seq)
        sparse = self.embedding(batch["sparse"])
        return self.head(torch.cat([pooled, sparse], dim=1)).squeeze(-1)

    def extra_metrics(self, y_true, y_score, attack_type=None) -> Dict[str, float]:
        return {
            "embedding_rows": float(self.embedding.embedding_rows()),
            "embedding_bytes": float(self.embedding.embedding_bytes()),
            "token_order": 1.0 if self.uses_window else 0.0,
            "n_tokens": float(self.spec.window_len if self.uses_window
                              else self.spec.num_bus),
            "physics_channels": float(self.physics_channels),
            "physics_enabled": 1.0 if self.physics_enabled else 0.0,
        }
