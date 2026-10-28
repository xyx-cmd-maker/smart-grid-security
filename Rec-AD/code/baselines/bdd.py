"""Bad Data Detection (BDD) — the classical physics baseline.

Why this baseline matters for the revision
------------------------------------------
The whole point of a stealthy FDIA is the constraint ``a = Hc``: the attack
vector is built inside the column space of the measurement Jacobian, so the
measurement residual barely moves and the traditional chi-square detector
sees nothing. Reviewer 3 (comment 1) asked for a defined threat model;
Reviewer 3 (comment 3) asked for evidence that prior methods are insufficient
for FDIA detection. This baseline produces exactly that evidence: it shows
BDD catching random attacks and missing stealthy ones.

Two variants
------------
``subspace``  (always available)
    Learn the normal-measurement subspace from clean training data by PCA.
    Residual energy outside that subspace is the test statistic. FDIA with
    ``a = Hc`` lies inside the subspace, so the statistic is unchanged.

``model``     (requires ``h_matrix`` in the contract)
    Classical weighted-least-squares residual using the linearised
    power-flow Jacobian ``H``. Closer to the textbook BDD, and the variant a
    power-systems reviewer will expect to see.

Both use the same statistic form ``J = ||r||^2 / sigma^2`` with
``dof = n_measurements - n_states``, so the reported numbers are comparable.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from data.contract import ATTACK_NORMAL
from recad.detector import DataSpec, DetectorBase


class BDDDetector(DetectorBase):
    """Chi-square / largest-normalised-residual bad data detector."""

    needs_node_feat = False
    needs_adj = False

    def __init__(self,
                 spec: DataSpec,
                 cfg: Dict[str, Any],
                 name: str = "BDD",
                 variant: Optional[str] = None,
                 h_matrix: Optional[np.ndarray] = None) -> None:
        super().__init__(spec, cfg, name)

        bdd_cfg = (cfg.get("baselines", {}) or {}).get("bdd", {}) or {}
        self.alpha = float(bdd_cfg.get("alpha", 0.01))
        self.lnr_sigma = float(bdd_cfg.get("lnr_sigma", 3.0))

        if variant is None or variant == "auto":
            variant = str(bdd_cfg.get("variant", "auto"))
        if variant == "auto":
            variant = "model" if h_matrix is not None else "subspace"
        if variant == "model" and h_matrix is None:
            raise ValueError("variant='model' requires an h_matrix")
        self.variant = variant

        if variant == "model":
            # IMPORTANT: the Jacobian must be expressed in the SAME space as
            # the feature vectors this detector sees. Any per-dimension
            # rescaling of the measurements (min-max normalisation, z-score
            # standardisation) maps col(H) to col(D H) for a diagonal D, so a
            # Jacobian fitted on raw physical units no longer describes the
            # legitimate subspace of standardised features. In that case the
            # attack's a = Hc structure is destroyed and the BDD baseline
            # looks far stronger than it really is. Use `variant: subspace`
            # when features are rescaled, or disable rescaling.
            import logging
            logging.getLogger("recad").info(
                "BDD 'model' variant in use: ensure features are NOT rescaled, "
                "or the a=Hc stealth property will not hold."
            )

        self.h_matrix = None if h_matrix is None else np.asarray(h_matrix, dtype=np.float64)
        self.mean_: Optional[np.ndarray] = None
        self.basis_: Optional[np.ndarray] = None      # (m, k) orthonormal columns
        self.sigma2_: float = 1.0
        self.dof_: int = 1
        self.threshold_: float = np.inf
        self._fitted = False

        # Registered so `.to(device)` is a no-op rather than an error.
        self.register_buffer("_dummy", torch.zeros(1))

    # -- fitting -----------------------------------------------------------

    def fit(self, train_loader, val_loader=None, **kwargs: Any) -> Dict[str, Any]:
        """Estimate the normal subspace, then set the threshold on val.

        Fitting on NORMAL samples only is the honest setup for an anomaly
        detector: a real operator does not have labelled attacks to train on.
        """
        z_train, normal_mask = self._collect(train_loader)
        z_clean = z_train[normal_mask]
        if len(z_clean) < 10:
            raise RuntimeError("BDD needs at least 10 clean training samples")

        self.mean_ = z_clean.mean(axis=0)
        centered = z_clean - self.mean_
        m = centered.shape[1]

        if self.variant == "model":
            # Column space of H is the legitimate measurement subspace.
            q, r = np.linalg.qr(self.h_matrix)
            keep = np.abs(np.diag(r)) > 1e-10
            self.basis_ = q[:, keep]
        else:
            # PCA basis; keep components explaining 99% of clean variance.
            _, s, vt = np.linalg.svd(centered, full_matrices=False)
            var = s ** 2
            cum = np.cumsum(var) / max(var.sum(), 1e-12)
            k = int(np.searchsorted(cum, 0.99) + 1)
            k = max(1, min(k, m - 1))
            self.basis_ = vt[:k].T

        k = self.basis_.shape[1]
        self.dof_ = max(1, m - k)

        resid = self._residual(z_clean)
        # sigma^2 must be the variance *per residual dimension*, so that
        # E[J] = dof and the chi-square calibration is correct. Averaging over
        # all m measurement dimensions instead would inflate J by m/(m-k),
        # making every sample look anomalous.
        self.sigma2_ = float(np.sum(resid ** 2) / (resid.size or 1)) * (m / self.dof_)
        if not np.isfinite(self.sigma2_) or self.sigma2_ <= 0:
            self.sigma2_ = 1e-12

        # Threshold from the requested false-positive level on the VAL split
        # when available; otherwise from the chi-square quantile.
        z_val, val_normal = (None, None)
        if val_loader is not None:
            z_val, val_normal = self._collect(val_loader)
        ref = (z_val[val_normal] if z_val is not None and val_normal.any() else z_clean)

        stat_ref = self._statistic(ref)
        self.threshold_ = float(np.quantile(stat_ref, 1.0 - self.alpha))

        self._fitted = True
        return {
            "variant": self.variant,
            "subspace_dim": int(k),
            "dof": int(self.dof_),
            "sigma2": self.sigma2_,
            "threshold": self.threshold_,
            "alpha": self.alpha,
            "n_clean_train": int(len(z_clean)),
        }

    @staticmethod
    def _collect(loader) -> tuple[np.ndarray, np.ndarray]:
        zs, is_normal = [], []
        for batch in loader:
            zs.append(batch["dense"].cpu().numpy().astype(np.float64))
            is_normal.append((batch["attack_type"].cpu().numpy() == ATTACK_NORMAL))
        return np.concatenate(zs), np.concatenate(is_normal)

    # -- scoring -----------------------------------------------------------

    def _residual(self, z: np.ndarray) -> np.ndarray:
        centered = z - self.mean_
        return centered - (centered @ self.basis_) @ self.basis_.T

    def _statistic(self, z: np.ndarray) -> np.ndarray:
        """J = ||r||^2 / sigma^2, one value per sample."""
        r = self._residual(z)
        return np.sum(r ** 2, axis=1) / self.sigma2_

    def _score(self, z: np.ndarray) -> np.ndarray:
        """Map the statistic to P(attacked) in [0, 1] via the chi-square CDF.

        Using the analytic survival function rather than a hard threshold
        keeps the output on the same probability scale as the learned
        detectors, so ROC/AUC comparisons are meaningful.
        """
        stat = self._statistic(z)
        try:
            from scipy.stats import chi2
            p_value = chi2.sf(stat, self.dof_)
        except ImportError:                      # pragma: no cover
            # Wilson-Hilferty normal approximation to the chi-square tail.
            from math import erf, sqrt
            zsc = ((stat / self.dof_) ** (1.0 / 3.0)
                   - (1.0 - 2.0 / (9.0 * self.dof_))) / np.sqrt(2.0 / (9.0 * self.dof_))
            p_value = 0.5 * (1.0 - np.vectorize(erf)(zsc / sqrt(2.0)))
        return np.clip(1.0 - p_value, 0.0, 1.0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = batch["dense"].detach().cpu().numpy().astype(np.float64)
        score = self._score(z)
        # Inverse sigmoid so the base class's sigmoid() round-trips exactly.
        eps = 1e-7
        score = np.clip(score, eps, 1.0 - eps)
        return torch.from_numpy(np.log(score / (1.0 - score))).float()

    def predict_proba(self, loader, device=None) -> np.ndarray:
        z, _ = self._collect(loader)
        return self._score(z).astype(np.float32)

    # -- diagnostics -------------------------------------------------------

    def extra_metrics(self, y_true, y_score, attack_type=None) -> Dict[str, float]:
        """Per-attack-type detection rate at the fitted threshold.

        This is the table that answers R3.3: BDD catches random attacks and
        misses ``a = Hc`` stealthy attacks, which is precisely why a learned
        detector is needed.
        """
        out: Dict[str, float] = {"bdd_threshold": float(self.threshold_)}
        if attack_type is None:
            return out

        z_all = None  # statistics already reflected in y_score
        detected = y_score >= float(1.0 - self.alpha)
        normal_mask = attack_type == ATTACK_NORMAL
        if normal_mask.any():
            out["bdd_fpr_at_threshold"] = float(detected[normal_mask].mean())

        names = {1: "random", 2: "stealthy", 3: "load_redist"}
        for code, short in names.items():
            mask = attack_type == code
            if mask.any():
                out[f"bdd_detect_rate_{short}"] = float(detected[mask].mean())
        return out
