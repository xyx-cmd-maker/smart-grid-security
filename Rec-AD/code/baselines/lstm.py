"""LSTM baseline over a token sequence (temporal when the contract has one)."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from baselines._sequence import SequenceDetectorBase
from recad.detector import DataSpec


class LSTMDetector(SequenceDetectorBase):

    def __init__(self, spec: DataSpec, cfg: Dict, name: str = "LSTM", h_matrix=None) -> None:
        bcfg = (cfg.get("baselines", {}) or {}).get("lstm", {}) or {}
        hidden = int(bcfg.get("hidden", 128))
        super().__init__(spec, cfg, name, hidden=hidden, h_matrix=h_matrix)

        layers = int(bcfg.get("layers", 2))
        self.bidirectional = bool(bcfg.get("bidirectional", False))
        # Tokens arrive already projected to `hidden` by the base class, so the
        # encoder does not need to know whether the order is temporal or
        # spatial -- nor what the raw feature dimension was.
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=0.1 if layers > 1 else 0.0,
        )
        if self.bidirectional:
            self.hidden = hidden * 2
            self.head = self._make_head(self.hidden, self.emb_dim)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the full output sequence (B, L, hidden) for pooling."""
        out, _ = self.lstm(tokens)
        return out
