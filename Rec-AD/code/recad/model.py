"""Rec-AD detector — the paper's own model, with each claimed component switchable.

This is what makes the ablation of Reviewer 1 (comment 4) and Reviewer 2
(comment 3) possible. The original manuscript reports only whole-system
numbers, so a reader cannot tell which of the three components produces the
gain. Here each component is a constructor flag:

  use_eff_tt   -- TT-compressed embedding table vs a plain nn.Embedding
  use_reorder  -- topology-aware index bijection vs the raw id order
  use_pipeline -- prefetching training path vs a strictly sequential one

Everything else — feature frontend, head, optimiser — is held fixed, so the
measured differences are attributable to the flags rather than to architecture
drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from recad.detector import DataSpec, DetectorBase
from recad.reorder import build_permutation
from recad.tt_table import TTEmbeddingBag, TTEmbeddingTable

#: Initialisation scale shared by BOTH ablation arms.
#:
#: Must equal `init_std` in tt_table.py. The two arms of the component ablation
#: differ only in the embedding layer, so they must start from the same weight
#: scale; otherwise the comparison measures the initialisation, not the method.
TT_INIT_STD = 0.02


@dataclass
class RecADConfig:
    """Component switches and shapes for one ablation cell."""

    use_eff_tt: bool = True
    use_reorder: bool = True
    use_pipeline: bool = True

    tt_rank: int = 32
    tt_core_dims: Optional[List[int]] = None
    embedding_dim: int = 16
    hidden: int = 128
    dropout: float = 0.1

    @staticmethod
    def from_cfg(cfg: Dict[str, Any], **overrides: Any) -> "RecADConfig":
        tt = (cfg.get("tt", {}) or {})
        abl = (cfg.get("ablation", {}) or {})
        comps = (abl.get("components", {}) or {})
        parts = (tt.get("split", {}) or {})
        d = int(parts.get("d", 3))
        n_per_core = int(parts.get("n_per_core", 0) or 0)
        default_dim = (n_per_core ** d) if n_per_core else 16
        conf = RecADConfig(
            use_eff_tt=bool(comps.get("eff_tt", True)),
            use_reorder=bool(comps.get("reorder", True)),
            use_pipeline=bool(comps.get("pipeline", True)),
            tt_rank=int(tt.get("rank", 32)),
            embedding_dim=int(tt.get("embedding_dim", default_dim)),
        )
        for key, value in overrides.items():
            if hasattr(conf, key) and value is not None:
                setattr(conf, key, value)
        return conf

    def label(self) -> str:
        return ("+".join(
            name for name, on in (("eff_tt", self.use_eff_tt),
                                  ("reorder", self.use_reorder),
                                  ("pipeline", self.use_pipeline)) if on
        ) or "none")


class RecADDetector(DetectorBase):
    """Tensor-Train compressed DLRM-style detector for FDIA classification."""

    def __init__(self,
                 spec: DataSpec,
                 cfg: Dict[str, Any],
                 name: str = "Rec-AD",
                 recad: Optional[RecADConfig] = None,
                 permutation: Optional[np.ndarray] = None,
                 data: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(spec, cfg, name)
        self.recad = recad or RecADConfig.from_cfg(cfg)

        # ---- embedding layer: the ablated component ----------------------
        if self.recad.use_eff_tt:
            total_rows = spec.sparse_total_rows
            self.embedding = TTEmbeddingBag(
                num_rows=total_rows,
                dim=self.recad.embedding_dim,
                rank=self.recad.tt_rank,
                mode="mean",
                reuse=True,
                split=(self.recad.tt_core_dims, None) if self.recad.tt_core_dims else None,
            )
            self._num_rows = total_rows
        else:
            # Dense control: same feature width, uncompressed table.
            self.embedding = nn.EmbeddingBag(
                spec.sparse_total_rows, self.recad.embedding_dim, mode="mean"
            )
            # Match the TT table's initialisation scale.
            #
            # PyTorch's default is N(0, 1); TTEmbeddingBag builds its cores from
            # a dense draw with init_std = 0.02 (tt_table.py). Left alone, the
            # two ablation cells start 50x apart, and the dense arm trains
            # visibly worse -- measured AUC 0.596 with the default against
            # 0.635 when matched, on an otherwise identical model. That is a
            # confound in the ablation, not an effect of compression, so the
            # scales are aligned here.
            nn.init.normal_(self.embedding.weight,
                            mean=0.0, std=TT_INIT_STD)
            self._num_rows = spec.sparse_total_rows

        self.emb_dim = self.recad.embedding_dim

        # ---- index bijection: the reordering component -------------------
        self.register_buffer(
            "_perm",
            torch.from_numpy(
                permutation if permutation is not None
                else np.arange(spec.sparse_total_rows, dtype=np.int64)
            ),
            persistent=False,
        )
        # Row offsets so each sparse column addresses its own slice of a
        # shared table, matching the paper's single-table design.
        offsets = np.concatenate([[0], np.cumsum(spec.sparse_cardinalities)[:-1]])
        self.register_buffer("_col_offsets", torch.from_numpy(offsets.astype(np.int64)),
                             persistent=False)
        self.register_buffer("_col_sizes",
                             torch.tensor(spec.sparse_cardinalities, dtype=torch.int64),
                             persistent=False)

        # ---- head --------------------------------------------------------
        in_dim = spec.num_dense + self.emb_dim
        self.bottom = nn.Sequential(
            nn.Linear(spec.num_dense, self.recad.hidden),
            nn.ReLU(),
            nn.Dropout(self.recad.dropout) if self.recad.dropout > 0 else nn.Identity(),
        )
        self.head = nn.Sequential(
            nn.Linear(self.recad.hidden + self.emb_dim, self.recad.hidden),
            nn.ReLU(),
            nn.Dropout(self.recad.dropout) if self.recad.dropout > 0 else nn.Identity(),
            nn.Linear(self.recad.hidden, 1),
        )
        self._in_dim = in_dim
        self._last_indices: Optional[torch.Tensor] = None

    # -- index handling ----------------------------------------------------

    def map_indices(self, sparse: torch.Tensor) -> torch.Tensor:
        """Column-local ids -> global table ids -> reordered ids."""
        offsets = self._col_offsets.to(sparse.device)
        sizes = self._col_sizes.to(sparse.device)
        # Clamp defensively: a val/test id outside the train-derived range
        # would otherwise index out of bounds silently.
        local = torch.minimum(sparse, sizes.unsqueeze(0) - 1)
        global_ids = local + offsets.unsqueeze(0)
        if self.recad.use_reorder:
            return self._perm.to(sparse.device)[global_ids]
        return global_ids

    # -- forward -----------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        dense = batch["dense"]
        sparse = batch["sparse"]
        ids = self.map_indices(sparse)
        self._last_indices = ids.detach()

        emb = self.embedding(ids.reshape(ids.shape[0], -1))
        bottom = self.bottom(dense)
        return self.head(torch.cat([bottom, emb], dim=1)).squeeze(-1)

    # -- introspection -----------------------------------------------------

    def embedding_bytes(self) -> int:
        if self.recad.use_eff_tt:
            return int(self.embedding.footprint_bytes())
        return int(self.embedding.weight.numel() * self.embedding.weight.element_size())

    def compression_ratio(self) -> float:
        if self.recad.use_eff_tt:
            return float(self.embedding.compression_ratio())
        return 1.0

    def extra_metrics(self, y_true, y_score, attack_type=None) -> Dict[str, float]:
        return {
            "use_eff_tt": float(self.recad.use_eff_tt),
            "use_reorder": float(self.recad.use_reorder),
            "use_pipeline": float(self.recad.use_pipeline),
            "component_label": self.recad.label(),
            "embedding_bytes": float(self.embedding_bytes()),
            "compression_ratio": float(self.compression_ratio()),
            "tt_rank": float(self.recad.tt_rank),
        }


# ---------------------------------------------------------------------------
# Permutation construction helper shared by the reorder experiment
# ---------------------------------------------------------------------------

def make_permutation(kind: str,
                     spec: DataSpec,
                     data: Dict[str, Any],
                     seed: int = 987654321) -> np.ndarray:
    """Build the index bijection for one reordering strategy.

    `seed` defaults to a value deliberately different from the usual data
    seeds: sharing a seed with the data generator makes `random` reproduce the
    data's own block structure and silently voids the comparison. See
    `recad.reorder.random_permutation`.
    """
    total = spec.sparse_total_rows
    ids = data["sparse"].astype(np.int64)

    # Map each global row id back to a bus. The second sparse column is the
    # measurement-point index in the contract, which is what carries topology.
    col_offset = 0
    id_to_node = np.zeros(total, dtype=np.int64)
    point_col = 1 if ids.shape[1] > 1 else 0
    point_card = spec.sparse_cardinalities[point_col]
    node = (ids[:, point_col] % max(point_card, 1))
    if spec.num_bus > 0:
        node = node % spec.num_bus
    for c in range(ids.shape[1]):
        off = int(np.sum(spec.sparse_cardinalities[:c]))
        size = spec.sparse_cardinalities[c]
        if c == point_col:
            id_to_node[off:off + size] = (np.arange(size) % max(spec.num_bus, 1)
                                          if spec.num_bus else 0)
        else:
            id_to_node[off:off + size] = 0

    adj = data.get("adj")
    if adj is None:
        # No topology available: fall back to a chain so BFS is still defined.
        adj = np.zeros((max(spec.num_bus, 1), max(spec.num_bus, 1)), dtype=np.float32)
        for i in range(spec.num_bus - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1.0

    return build_permutation(kind, total, ids=ids, adj=adj,
                             id_to_node=id_to_node, seed=seed)
