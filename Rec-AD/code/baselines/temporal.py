"""Temporal innovation detector — the classical answer to a stepped FDIA.

Why this baseline is necessary
------------------------------
A stealthy FDIA satisfies ``a = Hc``: the attacked measurement vector is a
legitimate measurement of the state ``x + c``. No detector that sees a single
snapshot can distinguish it from a legitimate state change -- that is the
definition, not an implementation gap. Measured on this project's data, a
single-snapshot residual test scores AUC 0.494 against the stealthy family,
i.e. chance.

What breaks the tie is TIME. Real power-system states drift; they do not
teleport. If the injection arrives as a step at some ``t0``, the apparent state
jumps by ``c`` in one step, and that jump is inconsistent with the dynamics.

The detector here is the textbook one for that situation:

    predict   z_hat_t = H x_hat_{t-1}
    innovate  nu_t    = z_t - z_hat_t
    whiten    d_t     = nu_t / s          (s from clean training data)
    test      J       = max_t ||d_t||^2   ~ chi-square(m)

Properties that make it a fair control:

* A CONSTANT injection ``a = Hc`` is absorbed into the state estimate and
  leaves ``nu_t`` unchanged -- it stays invisible, same as for BDD.
* A STEPPED injection produces a spike in ``nu_t`` at ``t0`` only.

This is the "why not just use a Kalman filter?" baseline. Without it, a
reviewer is entitled to ask why the paper needed a learned model at all.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from data.contract import ATTACK_NORMAL
from recad.detector import DataSpec, DetectorBase


class TemporalInnovationDetector(DetectorBase):
    """Predict the next snapshot from the previous one; test the innovation."""

    needs_node_feat = False
    needs_adj = False
    needs_window = True

    def __init__(self,
                 spec: DataSpec,
                 cfg: Dict[str, Any],
                 name: str = "TemporalResidual",
                 h_matrix: Optional[np.ndarray] = None) -> None:
        super().__init__(spec, cfg, name)

        tcfg = (cfg.get("baselines", {}) or {}).get("temporal", {}) or {}
        self.alpha = float(tcfg.get("alpha", 0.01))
        self.aggregate = str(tcfg.get("aggregate", "max"))   # "max" | "mean"

        if not getattr(spec, "has_window", False):
            raise ValueError(
                "TemporalInnovationDetector needs a contract with a `window` "
                "array; regenerate the data with --window > 1. A single "
                "snapshot carries no temporal information to test."
            )

        self.h_matrix = (None if h_matrix is None
                         else np.asarray(h_matrix, dtype=np.float64))
        self.projector_: Optional[np.ndarray] = None   # (m, m) onto col(H)
        self.scale_: Optional[np.ndarray] = None       # (m,) per-dim innovation std
        self.dof_: int = 1
        self.threshold_: float = np.inf
        self._fitted = False

        self.register_buffer("_dummy", torch.zeros(1))

    # -- fitting -----------------------------------------------------------

    def fit(self, train_loader, val_loader=None, **kwargs: Any) -> Dict[str, Any]:
        """Estimate the projection and per-dimension innovation scale on CLEAN data.

        The prediction model is deliberately the simplest one that is still
        physics-aware: the state is assumed to persist between consecutive
        samples, so the predicted measurement is the projection of the previous
        measurement onto col(H). Fitting only on normal samples matches how an
        operator would deploy this.
        """
        win_train, normal_mask = self._collect(train_loader)
        win_clean = win_train[normal_mask]
        if len(win_clean) < 10:
            raise RuntimeError("needs at least 10 clean training trajectories")

        m = win_clean.shape[2]
        if self.h_matrix is not None:
            pinv = np.linalg.pinv(self.h_matrix)
            self.projector_ = self.h_matrix @ pinv
        else:
            # No Jacobian: learn the legitimate measurement subspace by PCA on
            # the current snapshot, exactly as BDD's `subspace` variant does.
            flat = win_clean[:, -1, :]
            flat = flat - flat.mean(axis=0)
            _, _, vt = np.linalg.svd(flat, full_matrices=False)
            keep = min(m - 1, max(1, vt.shape[0]))
            basis = vt[:keep].T
            self.projector_ = basis @ basis.T

        nu = self._innovation(win_clean)
        # Per-dimension scale so the whitened statistic has unit variance in
        # every direction; H H^T is far from isotropic, and without this the
        # test would be dominated by whichever measurements H happens to weight
        # most.
        self.scale_ = np.sqrt((nu ** 2).mean(axis=(0, 1))) + 1e-12
        self.dof_ = int(m)

        ref = win_clean
        if val_loader is not None:
            win_val, val_normal = self._collect(val_loader)
            if val_normal.any():
                ref = win_val[val_normal]

        stat_ref = self._statistic(ref)
        self.threshold_ = float(np.quantile(stat_ref, 1.0 - self.alpha))

        self._fitted = True
        return {
            "aggregate": self.aggregate,
            "dof": int(self.dof_),
            "alpha": self.alpha,
            "threshold": self.threshold_,
            "n_clean_train": int(len(win_clean)),
            "window_len": int(win_clean.shape[1]),
        }

    @staticmethod
    def _collect(loader) -> tuple[np.ndarray, np.ndarray]:
        wins, is_normal = [], []
        for batch in loader:
            if "window" not in batch:
                raise KeyError(
                    "batch has no `window`; the loader was built from a "
                    "contract generated without --window"
                )
            wins.append(batch["window"].cpu().numpy().astype(np.float64))
            is_normal.append(batch["attack_type"].cpu().numpy() == ATTACK_NORMAL)
        return np.concatenate(wins), np.concatenate(is_normal)

    # -- scoring -----------------------------------------------------------

    def _innovation(self, win: np.ndarray) -> np.ndarray:
        """nu_t = z_t - P z_{t-1}, shape (n, W-1, m)."""
        predicted = np.einsum("nwm,km->nwk", win[:, :-1, :], self.projector_)
        return win[:, 1:, :] - predicted

    def _statistic(self, win: np.ndarray) -> np.ndarray:
        """J = max_t ||nu_t / s||^2, one value per trajectory."""
        nu = self._innovation(win) / self.scale_
        per_step = np.sum(nu ** 2, axis=2)              # (n, W-1)
        if per_step.shape[1] == 0:
            return np.zeros(len(win))
        return (per_step.max(axis=1) if self.aggregate == "max"
                else per_step.mean(axis=1))

    def _score(self, win: np.ndarray) -> np.ndarray:
        """Map the statistic to P(attacked) through the chi-square survival."""
        stat = self._statistic(win)
        try:
            from scipy.stats import chi2
            p_value = chi2.sf(stat, self.dof_)
        except ImportError:                              # pragma: no cover
            from math import erf, sqrt
            zsc = ((stat / self.dof_) ** (1.0 / 3.0)
                   - (1.0 - 2.0 / (9.0 * self.dof_))) / np.sqrt(2.0 / (9.0 * self.dof_))
            p_value = 0.5 * (1.0 - np.vectorize(erf)(zsc / sqrt(2.0)))
        return np.clip(1.0 - p_value, 0.0, 1.0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        win = batch["window"].detach().cpu().numpy().astype(np.float64)
        score = np.clip(self._score(win), 1e-7, 1.0 - 1e-7)
        return torch.from_numpy(np.log(score / (1.0 - score))).float()

    def predict_proba(self, loader, device=None) -> np.ndarray:
        win, _ = self._collect(loader)
        return self._score(win).astype(np.float32)

    # -- diagnostics -------------------------------------------------------

    def extra_metrics(self, y_true, y_score, attack_type=None) -> Dict[str, float]:
        out: Dict[str, float] = {"temporal_threshold": float(self.threshold_)}
        if attack_type is None:
            return out
        detected = y_score >= float(1.0 - self.alpha)
        normal_mask = attack_type == ATTACK_NORMAL
        if normal_mask.any():
            out["temporal_fpr_at_threshold"] = float(detected[normal_mask].mean())
        for code, short in {1: "random", 2: "stealthy", 3: "load_redist"}.items():
            mask = attack_type == code
            if mask.any():
                out[f"temporal_detect_rate_{short}"] = float(detected[mask].mean())
        return out
