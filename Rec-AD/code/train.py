"""Shared supervised training loop for the gradient-based baselines."""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recad.detector import DetectorBase, _to_device


def run_training_loop(model: DetectorBase,
                      train_loader,
                      val_loader=None,
                      device: Optional[torch.device] = None,
                      epochs: Optional[int] = None,
                      lr: Optional[float] = None,
                      weight_decay: Optional[float] = None,
                      grad_clip: Optional[float] = None,
                      pos_weight: Optional[float] = None,
                      early_stop_patience: Optional[int] = None,
                      log_prefix: str = "") -> Dict[str, Any]:
    """Train `model` in place and return a history dict.

    Identical optimisation settings are used for every gradient-based
    baseline. That uniformity is the point: a reviewer asking "were the
    baselines tuned equally?" should be answerable by pointing at one
    function rather than at seven training scripts.
    """
    tcfg = model.cfg.get("train", {}) or {}
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = int(epochs if epochs is not None else tcfg.get("epochs", 20))
    lr = float(lr if lr is not None else tcfg.get("lr", 1e-3))
    weight_decay = float(
        weight_decay if weight_decay is not None else tcfg.get("weight_decay", 0.0)
    )
    grad_clip = grad_clip if grad_clip is not None else tcfg.get("grad_clip")
    patience = int(
        early_stop_patience if early_stop_patience is not None
        else tcfg.get("early_stop_patience", 4)
    )
    if pos_weight is None:
        pos_weight = tcfg.get("pos_weight")

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pw = None
    if pos_weight is not None:
        pw = torch.tensor([float(pos_weight)], device=device)

    history: Dict[str, Any] = {
        "epochs": [], "train_loss": [], "val_loss": [], "val_auc": [],
        "best_epoch": 0, "device": str(device),
    }

    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        total, n_batches = 0.0, 0

        for batch in train_loader:
            batch = _to_device(
                batch, device,
                needs_node_feat=model.needs_node_feat,
                needs_adj=model.needs_adj,
                needs_window=model.needs_window,
            )
            logits = model(batch)
            target = batch["label"].float()
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()

            total += float(loss.detach())
            n_batches += 1

        train_loss = total / max(n_batches, 1)
        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)

        val_loss, val_auc = float("nan"), float("nan")
        if val_loader is not None:
            val_loss, val_auc = _validate(model, val_loader, device, pw)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

        score = val_auc if not np.isnan(val_auc) else -train_loss
        if score > best_score + 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

        if log_prefix:
            print(f"  {log_prefix} epoch {epoch:>3} "
                  f"train_loss={train_loss:.4f} "
                  f"val_auc={val_auc:.4f}", flush=True)

    model.load_state_dict(best_state)
    history["train_seconds"] = time.perf_counter() - t0
    history["n_epochs_run"] = len(history["epochs"])
    return history


@torch.no_grad()
def _validate(model: DetectorBase, loader, device, pos_weight=None):
    from evaluate import compute_metrics

    model.eval()
    logits_all, labels_all = [], []
    total, n_batches = 0.0, 0
    for batch in loader:
        batch = _to_device(
            batch, device,
            needs_node_feat=model.needs_node_feat,
            needs_adj=model.needs_adj,
            needs_window=model.needs_window,
        )
        logits = model(batch)
        target = batch["label"].float()
        total += float(F.binary_cross_entropy_with_logits(logits, target,
                                                          pos_weight=pos_weight))
        n_batches += 1
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(target.detach().cpu().numpy())

    if not logits_all:
        return float("nan"), float("nan")

    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)
    scores = 1.0 / (1.0 + np.exp(-logits))
    metrics = compute_metrics(labels.astype(int), scores)
    return total / max(n_batches, 1), float(metrics.get("auc", float("nan")))
