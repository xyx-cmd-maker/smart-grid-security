"""Torch Dataset / DataLoader over the data contract.

Standardisation statistics are computed on the TRAIN split only and applied to
val/test. Anything else leaks information and would be caught in review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.contract import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL


@dataclass
class NormStats:
    dense_mean: np.ndarray
    dense_std: np.ndarray
    node_mean: Optional[np.ndarray]
    node_std: Optional[np.ndarray]

    @staticmethod
    def identity(dense_dim: int, node_dim: Optional[int] = None) -> "NormStats":
        return NormStats(
            dense_mean=np.zeros(dense_dim, dtype=np.float32),
            dense_std=np.ones(dense_dim, dtype=np.float32),
            node_mean=None if node_dim is None else np.zeros(node_dim, dtype=np.float32),
            node_std=None if node_dim is None else np.ones(node_dim, dtype=np.float32),
        )


class FDIADataset(Dataset):
    """One row per sample: measurements + label + attack-type tag."""

    def __init__(self,
                 dense: np.ndarray,
                 sparse: np.ndarray,
                 label: np.ndarray,
                 attack_type: np.ndarray,
                 node_feat: Optional[np.ndarray] = None,
                 adj: Optional[np.ndarray] = None,
                 index: Optional[np.ndarray] = None,
                 window: Optional[np.ndarray] = None) -> None:
        self.dense = dense
        self.sparse = sparse
        self.label = label
        self.attack_type = attack_type
        self.node_feat = node_feat
        self.adj = adj
        self.index = np.arange(len(label)) if index is None else index
        # (n, W, m) measurement trajectory. `dense` is its final snapshot.
        self.window = window

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        j = int(self.index[i])
        item = {
            "dense": torch.from_numpy(self.dense[j]),
            "sparse": torch.from_numpy(self.sparse[j]),
            "label": torch.tensor(float(self.label[j]), dtype=torch.float32),
            "attack_type": torch.tensor(int(self.attack_type[j]), dtype=torch.long),
            "row_index": torch.tensor(j, dtype=torch.long),
        }
        if self.node_feat is not None:
            item["node_feat"] = torch.from_numpy(self.node_feat[j])
        if self.adj is not None:
            item["adj"] = torch.from_numpy(self.adj)
        if self.window is not None:
            item["window"] = torch.from_numpy(self.window[j])
        return item


def compute_norm_stats(data: Dict, standardize: bool = True) -> NormStats:
    dense = data["dense"]
    node = data.get("node_feat")
    if not standardize:
        return NormStats.identity(
            dense.shape[1], None if node is None else node.shape[2]
        )

    split = data.get("split")
    train_mask = (split == SPLIT_TRAIN) if split is not None else np.ones(len(dense), bool)
    if train_mask.sum() == 0:
        raise ValueError("train split is empty; cannot compute normalisation stats")

    d_train = dense[train_mask]
    eps = 1e-6
    stats = NormStats(
        dense_mean=d_train.mean(axis=0).astype(np.float32),
        dense_std=(d_train.std(axis=0) + eps).astype(np.float32),
        node_mean=None,
        node_std=None,
    )
    if node is not None:
        n_train = node[train_mask]
        # Reduce over samples and buses, keep a per-feature statistic.
        stats.node_mean = n_train.mean(axis=(0, 1)).astype(np.float32)
        stats.node_std = (n_train.std(axis=(0, 1)) + eps).astype(np.float32)
    return stats


def apply_norm(data: Dict, stats: NormStats) -> Dict:
    """Rescale features, and rescale `h_matrix` CONSISTENTLY with them.

    Why the Jacobian must be transformed too
    -----------------------------------------
    Standardising maps z -> (z - mu) / sigma, i.e. z -> D z + const with
    D = diag(1/sigma). The measurement model z = H x + e therefore becomes

        z~ = D H x + D e + const

    so the legitimate subspace is no longer col(H) but col(D H). Passing the
    RAW H to a residual detector while handing it standardised measurements
    makes the test inconsistent: `a = Hc` is no longer in the subspace the
    detector projects onto, the stealth property is destroyed, and BDD looks
    far stronger than it is. That is exactly the failure measured on
    2026-10-06 (stealthy detection 0.0058 -> 1.0000 when standardising).

    The fix is not to abandon standardisation -- neural baselines need inputs
    with a sane dynamic range; the raw measurements here span +/-1600. It is to
    transform H by the same diagonal, after which standardisation is exactly
    equivalent and the stealth property is preserved.
    """
    out = dict(data)
    out["dense"] = ((data["dense"] - stats.dense_mean) / stats.dense_std).astype(np.float32)
    if data.get("node_feat") is not None and stats.node_mean is not None:
        out["node_feat"] = (
            (data["node_feat"] - stats.node_mean) / stats.node_std
        ).astype(np.float32)
    if data.get("window") is not None:
        out["window"] = (
            (data["window"] - stats.dense_mean) / stats.dense_std
        ).astype(np.float32)
    if data.get("h_matrix") is not None:
        # Same diagonal as the measurements: H -> D H, with D = 1/sigma.
        # D acts on the MEASUREMENT axis, which is the ROW axis of H (m, n).
        d = data["h_matrix"].shape[0]
        scale = stats.dense_std[:d].reshape(-1, 1).astype(np.float64)
        out["h_matrix"] = (data["h_matrix"] / scale).astype(np.float32)
        out["h_matrix_rescaled"] = np.array(True)
    return out


def _split_indices(data: Dict, ratios: Dict[str, float]) -> Dict[str, np.ndarray]:
    split = data.get("split")
    if split is not None:
        return {
            "train": np.where(split == SPLIT_TRAIN)[0],
            "val": np.where(split == SPLIT_VAL)[0],
            "test": np.where(split == SPLIT_TEST)[0],
        }

    n = len(data["label"])
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_train = int(ratios["train"] * n)
    n_val = int(ratios["val"] * n)
    return {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
    }


def build_loaders(data: Dict,
                  batch_size: int = 4096,
                  num_workers: int = 0,
                  standardize: bool = True,
                  ratios: Optional[Dict[str, float]] = None,
                  attack_filter: Optional[Iterable[int]] = None,
                  shuffle_train: bool = True,
                  drop_last_train: bool = False,
                  seed: int = 0) -> Dict[str, DataLoader]:
    """Build train/val/test loaders from a contract dict.

    attack_filter restricts which attack types are kept (normal samples are
    always kept). Used by the per-attack-type analysis required by R3.1.
    """
    ratios = ratios or {"train": 0.7, "val": 0.15, "test": 0.15}
    stats = compute_norm_stats(data, standardize=standardize)
    data = apply_norm(data, stats)

    parts = _split_indices(data, ratios)
    loaders: Dict[str, DataLoader] = {}

    keep = None
    if attack_filter is not None:
        allowed = set(int(a) for a in attack_filter) | {0}
        keep = np.isin(data["attack_type"], list(allowed))

    for name, idx in parts.items():
        if keep is not None:
            idx = idx[keep[idx]]
        if len(idx) == 0:
            raise ValueError(f"split {name!r} is empty after filtering")

        ds = FDIADataset(
            dense=data["dense"],
            sparse=data["sparse"],
            label=data["label"],
            attack_type=data["attack_type"],
            node_feat=data.get("node_feat"),
            adj=data.get("adj"),
            index=idx,
            window=data.get("window"),
        )
        generator = torch.Generator().manual_seed(seed + {"train": 0, "val": 1, "test": 2}[name])
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train" and shuffle_train),
            num_workers=num_workers,
            drop_last=(name == "train" and drop_last_train),
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )
    loaders["_stats"] = stats  # type: ignore[assignment]
    return loaders


def build_spec(data: Dict) -> "DataSpec":
    """Derive the model-facing shape metadata from a contract dict."""
    from recad.detector import DataSpec

    sparse = data["sparse"]
    cardinalities = [
        int(sparse[:, c].max()) + 1 for c in range(sparse.shape[1])
    ]
    node_feat = data.get("node_feat")
    window = data.get("window")
    return DataSpec(
        num_dense=int(data["dense"].shape[1]),
        num_sparse=int(sparse.shape[1]),
        sparse_cardinalities=cardinalities,
        num_bus=int(node_feat.shape[1]) if node_feat is not None else 0,
        node_feat_dim=int(node_feat.shape[2]) if node_feat is not None else 0,
        window_len=int(window.shape[1]) if window is not None else 0,
        num_measurements=int(window.shape[2]) if window is not None else 0,
    )


def loader_arrays(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialise (X_dense, X_sparse, y, attack_type) for non-gradient baselines.

    SVM, RF and BDD need the whole split in memory. Callers should subsample
    the train split for SVM; see config.baselines.svm.max_train_samples.
    """
    dense, sparse, labels, atypes = [], [], [], []
    for batch in loader:
        dense.append(batch["dense"].numpy())
        sparse.append(batch["sparse"].numpy())
        labels.append(batch["label"].numpy())
        atypes.append(batch["attack_type"].numpy())
    return (
        np.concatenate(dense),
        np.concatenate(sparse),
        np.concatenate(labels),
        np.concatenate(atypes),
    )
