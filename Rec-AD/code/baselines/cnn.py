"""1-D CNN baseline over the bus measurement sequence.

Reviewer 3 (comment 6) explicitly asked for CNNs to be included in the
comparison, since the paper cites them as computationally intensive without
ever measuring them.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from baselines._sequence import SequenceDetectorBase
from recad.detector import DataSpec


class CNNDetector(SequenceDetectorBase):

    def __init__(self, spec: DataSpec, cfg: Dict, name: str = "CNN", h_matrix=None) -> None:
        bcfg = (cfg.get("baselines", {}) or {}).get("cnn", {}) or {}
        channels: List[int] = list(bcfg.get("channels", [32, 64, 128]))
        kernel = int(bcfg.get("kernel", 3))
        dropout = float(bcfg.get("dropout", 0.1))
        hidden = channels[-1] if channels else 64
        super().__init__(spec, cfg, name, hidden=hidden, h_matrix=h_matrix)

        blocks: List[nn.Module] = []
        # Tokens arrive projected to `hidden` by the base class.
        in_ch = hidden
        for out_ch in channels:
            blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                          padding=kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2, ceil_mode=True),
            ]
            in_ch = out_ch
        self.encoder = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, L, hidden) -> conv along the token axis -> (B, L', hidden)."""
        x = self.encoder(tokens.transpose(1, 2))   # (B, hidden, L')
        x = x.transpose(1, 2)                       # (B, L', hidden)
        return self.dropout(x)
