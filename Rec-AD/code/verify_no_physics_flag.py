"""Verify what --no-physics-features actually does, before relying on it.

The flag's name implies it disables the three physics channels (raw / residual /
innovation). I inferred that from the name alone, which is exactly the kind of
inference this project has been burned by. So this checks the behaviour rather than the
label: build a CNN with the flag on and off, and report `physics_channels`, the token
tensor's last dimension, and the field width.

It also computes what the channel count implies, so a silent no-op cannot pass: if the
flag does nothing, `physics_channels` stays 3 and the tensor stays 3*m wide.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                     # noqa: E402
from data.dataset import build_spec, build_loaders          # noqa: E402
from recad.common import load_config, set_seed              # noqa: E402
from baselines.registry import build_detector               # noqa: E402

d = load_contract(HERE / "data_cache/ieee118.npz")
spec = build_spec(d)
m = spec.num_measurements
dcfg = (load_config(None).get("data", {}) or {})

loaders = build_loaders(d, batch_size=32, num_workers=0,
                        standardize=bool(dcfg.get("standardize", False)),
                        ratios=dcfg.get("split_ratios"), seed=0)
batch = next(iter(loaders["test"]))

print("=" * 84)
print("  Does --no-physics-features change the token layout?")
print("=" * 84)
print()
print(f"  expected: physics off -> physics_channels 0, token width {m}")
print(f"            physics on  -> physics_channels 3, token width {3 * m}")
print()

for physics in (True, False):
    cfg = load_config(None)
    b = cfg.setdefault("baselines", {})
    seq = b.setdefault("sequence", {})
    seq["physics_features"] = physics
    set_seed(0)
    model = build_detector("CNN", spec, cfg, data=d)
    tok = model.token_sequence(batch)
    ch = getattr(model, "physics_channels", "?")
    ok = (ch == 3 and tok.shape[2] == 3 * m) if physics else (ch == 0 and tok.shape[2] == m)
    print(f"  physics_features={physics!s:<5}  physics_channels={ch}  "
          f"token {tuple(tok.shape)}   in_dim={getattr(model, 'in_dim', '?')}   "
          f"{'OK' if ok else 'UNEXPECTED'}")

print()
print("  Reading: if the off-arm shows physics_channels=0 and a token width of "
      f"{m}, the flag does what its name says and an ablation on it is meaningful.")
print("  If both arms match, the flag is a no-op and the ablation would be vacuous.")
