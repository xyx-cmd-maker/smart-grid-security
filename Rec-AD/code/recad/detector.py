"""Detector interface shared by every baseline, plus the embedding backend protocol.

A fair comparison across seven method families requires that they all see the
same data, the same splits, and are scored by the same code. This module is
where that guarantee lives.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Batch contract
# ---------------------------------------------------------------------------
# DataLoader yields dicts with these keys. Not every model uses every key;
# `needs_graph` on the detector declares which ones it reads.
#
#   dense       (B, D_dense)      float32
#   sparse      (B, S)            int64
#   node_feat   (B, B_bus, F)     float32
#   adj         (B_bus, B_bus)    float32     (graph models only)
#   label       (B,)              float32
#   attack_type (B,)              int64
# ---------------------------------------------------------------------------


@dataclass
class DataSpec:
    """Shape metadata a detector needs at construction time."""

    num_dense: int
    num_sparse: int
    sparse_cardinalities: List[int]
    num_bus: int
    node_feat_dim: int
    # Temporal structure. 0 means the contract carries no trajectory and the
    # sequence models fall back to treating the bus axis as the sequence.
    window_len: int = 0
    num_measurements: int = 0

    @property
    def has_window(self) -> bool:
        return self.window_len > 0 and self.num_measurements > 0

    @property
    def sparse_total_rows(self) -> int:
        return int(sum(self.sparse_cardinalities))


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingBackend(Protocol):
    """Contract for a compressed embedding table.

    `recad.tt_table.TTEmbeddingTable` implements this in pure PyTorch so the
    package runs standalone. The authors' C++/CUDA Eff-TT implementation can
    implement the same four methods and be swapped in via config
    (`embedding_backend: "cuda_efftt"`).
    """

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: (B, S) int64 or (B, k) int64 -> (B, k, N) or (B, N)."""
        ...

    def footprint_bytes(self) -> int:
        """Bytes actually resident for the embedding table, cores included."""
        ...

    def compression_ratio(self) -> float:
        """Dense table bytes / compressed bytes. Paper reports this."""
        ...

    def parameters(self) -> List[nn.Parameter]:
        """Trainable TT cores."""
        ...


# ---------------------------------------------------------------------------
# Detector base
# ---------------------------------------------------------------------------

class DetectorBase(abc.ABC, nn.Module):
    """Base class for all seven baseline families.

    Subclasses implement `forward` (deep models) and may override `fit` when
    the training procedure is not gradient based (BDD, SVM, RF).

    Invariants every subclass must honour:
      * `predict_proba` returns P(attacked) in [0, 1], shape (N,).
      * Probability convention is fixed so ROC/AUC and threshold sweeps are
        comparable across methods.
      * `extra_metrics` may add method-specific diagnostics (e.g. BDD's
        detection rate per attack type).
    """

    #: Set True for models that read `node_feat` from the batch.
    needs_node_feat: bool = False

    #: Set True for models that read `adj` from the batch (graph models only).
    needs_adj: bool = False

    #: Set True for models that read the `window` trajectory from the batch.
    #: Sequence models set this when the contract carries a trajectory, and
    #: clear `needs_node_feat` in that case so the profiler does not charge
    #: them for an input they never touch.
    needs_window: bool = False

    def __init__(self, spec: DataSpec, cfg: Dict[str, Any], name: str) -> None:
        super().__init__()
        self.spec = spec
        self.cfg = cfg
        self.name = name
        self._fitted = False

    # -- training ----------------------------------------------------------

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            **kwargs: Any) -> Dict[str, Any]:
        """Train and return a history dict.

        The default implementation runs `train.py:run_training_loop`.
        Non-gradient detectors override this entirely.
        """
        from train import run_training_loop  # local import avoids a cycle

        history = run_training_loop(self, train_loader, val_loader, **kwargs)
        self._fitted = True
        return history

    # -- inference ---------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self,
                      loader: torch.utils.data.DataLoader,
                      device: Optional[torch.device] = None) -> np.ndarray:
        """Return P(attacked) for every sample, in loader order.

        Subclasses that are not nn.Module-shaped in the usual way still get a
        usable default by overriding `predict_proba_batch`.
        """
        self.eval()
        device = device or next(self.parameters(), torch.empty(0)).device
        chunks: List[np.ndarray] = []
        for batch in loader:
            batch = _to_device(
                batch, device,
                needs_node_feat=self.needs_node_feat,
                needs_adj=self.needs_adj,
                needs_window=self.needs_window,
            )
            logits = self.forward(batch)
            chunks.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)

    @abc.abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return logits of shape (B,)."""
        raise NotImplementedError

    # -- introspection -----------------------------------------------------

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def footprint_bytes(self) -> int:
        """Parameter bytes for the model (excluding activations)."""
        total = 0
        for p in self.parameters():
            total += p.numel() * p.element_size()
        for b in self.buffers():
            total += b.numel() * b.element_size()
        return int(total)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def extra_metrics(self,
                      y_true: np.ndarray,
                      y_score: np.ndarray,
                      attack_type: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Method-specific diagnostics merged into the result row."""
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(batch: Dict[str, Any],
               device: torch.device,
               needs_node_feat: bool = False,
               needs_adj: bool = False,
               needs_window: bool = False) -> Dict[str, Any]:
    """Move a batch to `device`, dropping tensors the model will not read.

    Dropping unused tensors keeps memory measurements honest: otherwise a CNN
    over flat features would appear to require the graph model's input tensor,
    or a single-snapshot model would appear to require the whole trajectory.
    """
    skip = set()
    if not needs_node_feat:
        skip.add("node_feat")
    if not needs_adj:
        skip.add("adj")
    if not needs_window:
        skip.add("window")
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if key in skip:
            continue
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


class MLP(nn.Module):
    """Small MLP with optional dropout. Shared by several baselines."""

    def __init__(self,
                 in_dim: int,
                 hidden: List[int],
                 out_dim: int = 1,
                 dropout: float = 0.0,
                 activation: str = "relu") -> None:
        super().__init__()
        act_cls = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), act_cls()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
