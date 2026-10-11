"""Transformer baseline over the bus measurement sequence.

Reviewer 3 (comment 6): the paper cites Transformers as computationally
intensive but never measures them. This baseline closes that gap, and being
the heaviest model in the comparison it is also the one most likely to
exceed the edge memory budget — which is the deployability argument.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn

from baselines._sequence import SequenceDetectorBase
from recad.detector import DataSpec


class PositionalEncoding(nn.Module):
    """Standard sinusoidal encoding over the bus index."""

    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerDetector(SequenceDetectorBase):

    def __init__(self, spec: DataSpec, cfg: Dict, name: str = "Transformer", h_matrix=None) -> None:
        bcfg = (cfg.get("baselines", {}) or {}).get("transformer", {}) or {}
        d_model = int(bcfg.get("d_model", 128))
        nhead = int(bcfg.get("nhead", 4))
        layers = int(bcfg.get("layers", 3))
        ff = int(bcfg.get("ff", 256))
        dropout = float(bcfg.get("dropout", 0.1))

        super().__init__(spec, cfg, name, hidden=d_model, h_matrix=h_matrix)

        # The base class already projected tokens to `d_model`; only the
        # positional encoding and the encoder stack are ours.
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the encoder output sequence (B, L, d_model) for pooling."""
        x = self.pos(tokens)
        x = self.encoder(x)
        return self.norm(x)
