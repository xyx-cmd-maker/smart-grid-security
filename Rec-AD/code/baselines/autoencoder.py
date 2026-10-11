"""Autoencoder anomaly detector.

Trained on normal measurements only — the realistic unsupervised setting for
a substation operator. The anomaly score is the reconstruction error, mapped
to a probability by a logistic calibration fitted on the validation split so
that the output is comparable with the supervised baselines' probabilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from baselines._common import FlatFeatureDetector, make_mlp
from data.contract import ATTACK_NORMAL
from recad.detector import DataSpec


class AutoencoderDetector(FlatFeatureDetector):

    def __init__(self, spec: DataSpec, cfg: Dict, name: str = "Autoencoder",
                 h_matrix=None) -> None:
        super().__init__(spec, cfg, name, h_matrix=h_matrix)  # dim + input mode from cfg
        bcfg = (cfg.get("baselines", {}) or {}).get("autoencoder", {}) or {}
        hidden: List[int] = list(bcfg.get("hidden", [256, 64, 256]))
        latent = int(bcfg.get("latent", 16))

        enc_dims = hidden[:len(hidden) // 2] or [128]
        dec_dims = hidden[len(hidden) // 2:] or [128]

        self.encoder = make_mlp(self.in_dim, enc_dims, out_dim=latent)
        self.decoder = make_mlp(latent, dec_dims, out_dim=self.in_dim)

        # Logistic calibration: P(attacked) = sigmoid(a * err + b), fitted on val.
        self.register_buffer("_cal_a", torch.tensor(1.0))
        self.register_buffer("_cal_b", torch.tensor(-1.0))
        self._err_scale = 1.0

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return logits from the calibrated reconstruction error."""
        x = self.features(batch)
        recon = self.decoder(self.encoder(x))
        err = torch.mean((recon - x) ** 2, dim=1) / self._err_scale
        return self._cal_a * err + self._cal_b

    # -- training ----------------------------------------------------------

    def fit(self, train_loader, val_loader=None, **kwargs: Any) -> Dict[str, Any]:
        """Train on normal samples, then calibrate on validation data."""
        import copy
        import torch.nn.functional as F

        cfg = self.cfg.get("train", {}) or {}
        epochs = int(cfg.get("epochs", 20))
        lr = float(cfg.get("lr", 1e-3))
        device = kwargs.get("device") or torch.device("cpu")
        self.to(device)

        opt = torch.optim.Adam(self.parameters(), lr=lr,
                               weight_decay=float(cfg.get("weight_decay", 0.0)))

        history: Dict[str, Any] = {"epochs": [], "train_loss": []}
        best_state = copy.deepcopy(self.state_dict())
        best_loss = float("inf")
        patience = int(cfg.get("early_stop_patience", 4))
        bad_epochs = 0

        for epoch in range(epochs):
            self.train()
            total, n_batches = 0.0, 0
            for batch in train_loader:
                mask = batch["attack_type"] == ATTACK_NORMAL
                if not bool(mask.any()):
                    continue
                batch = {k: (v[mask] if torch.is_tensor(v) and v.shape[:1] == mask.shape
                             else v) for k, v in batch.items()}
                x = self.features(batch).to(device)
                recon = self.decoder(self.encoder(x))
                loss = F.mse_loss(recon, x)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.get("grad_clip"):
                    nn.utils.clip_grad_norm_(self.parameters(), float(cfg["grad_clip"]))
                opt.step()
                total += float(loss.detach())
                n_batches += 1

            mean_loss = total / max(n_batches, 1)
            history["epochs"].append(epoch)
            history["train_loss"].append(mean_loss)

            if mean_loss < best_loss - 1e-6:
                best_loss = mean_loss
                best_state = copy.deepcopy(self.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        self.load_state_dict(best_state)

        # Scale errors so the calibration is numerically well behaved.
        err_train = self._reconstruction_error(train_loader, device, normal_only=True)
        self._err_scale = float(np.median(err_train)) or 1.0

        if val_loader is not None:
            self._calibrate(val_loader, device)
        history["err_scale"] = self._err_scale
        self._fitted = True
        return history

    @torch.no_grad()
    def _reconstruction_error(self,
                              loader,
                              device,
                              normal_only: bool = False) -> np.ndarray:
        self.eval()
        errs: List[np.ndarray] = []
        for batch in loader:
            if normal_only:
                mask = batch["attack_type"] == ATTACK_NORMAL
                if not bool(mask.any()):
                    continue
                batch = {k: (v[mask] if torch.is_tensor(v) and v.shape[:1] == mask.shape
                             else v) for k, v in batch.items()}
            x = self.features(batch).to(device)
            recon = self.decoder(self.encoder(x))
            errs.append(((recon - x) ** 2).mean(dim=1).cpu().numpy())
        if not errs:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(errs).astype(np.float64)

    @torch.no_grad()
    def _calibrate(self, val_loader, device) -> None:
        """Fit a 1-D logistic map from reconstruction error to P(attacked).

        Without this the raw MSE is unbounded and would produce ROC curves on
        an arbitrary scale; the calibration makes the AUC comparison honest.
        """
        errs, labels = [], []
        for batch in val_loader:
            mask = torch.ones_like(batch["label"], dtype=torch.bool)
            sub = {k: (v[mask] if torch.is_tensor(v) and v.shape[:1] == mask.shape
                       else v) for k, v in batch.items()}
            x = self.features(sub).to(device)
            recon = self.decoder(self.encoder(x))
            errs.append(((recon - x) ** 2).mean(dim=1).cpu().numpy())
            labels.append(batch["label"].numpy())
        err = np.concatenate(errs) / max(self._err_scale, 1e-12)
        y = np.concatenate(labels).astype(int)
        if len(np.unique(y)) < 2:
            self._cal_a.fill_(1.0)
            self._cal_b.fill_(-float(np.median(err)))
            return

        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression().fit(err.reshape(-1, 1), y)
        self._cal_a.fill_(float(lr.coef_[0, 0]))
        self._cal_b.fill_(float(lr.intercept_[0]))

    def predict_proba(self, loader, device=None) -> np.ndarray:
        device = device or next(self.parameters()).device
        err = self._reconstruction_error(loader, device) / max(self._err_scale, 1e-12)
        a = float(self._cal_a)
        b = float(self._cal_b)
        return (1.0 / (1.0 + np.exp(-(a * err + b)))).astype(np.float32)
