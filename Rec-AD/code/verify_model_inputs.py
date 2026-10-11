"""Behaviourally verify what the GNN detectors receive, and what the others receive.

I twice mis-classified result files by guessing which models go through the physics
channels. The lesson (AGENTS.md pattern B, static checks do not prove behaviour) is to
measure rather than read. So this asserts, by running the models, what each family
actually consumes -- in particular whether the GNNs ever see the temporal window.

Why it matters for the table: the stealthy attack's only usable signature in this
project is a step in the state trajectory, i.e. a TEMPORAL signature. A detector that
never receives `window` cannot see it for a structural reason, and presenting such a
detector next to an LSTM would imply the gap is architectural preference rather than
missing input.

Reports, per model: the input keys it touches, the shapes, and whether `window` is
among them.
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
cfg = load_config(None)
dcfg = cfg.get("data", {}) or {}

loaders = build_loaders(d, batch_size=64, num_workers=0,
                        standardize=bool(dcfg.get("standardize", False)),
                        ratios=dcfg.get("split_ratios"), seed=0)
batch = next(iter(loaders["test"]))

print("=" * 88)
print("  What each model actually consumes (measured, not read)")
print("=" * 88)
print()
print("  batch fields and shapes:")
for k, v in sorted(batch.items()):
    if torch.is_tensor(v):
        print(f"    {k:<14} {tuple(v.shape)}  {v.dtype}")
print()

# instrument the batch dict so every read is recorded
class Spy(dict):
    def __init__(self, inner):
        super().__init__(inner)
        self.touched = []
    def __getitem__(self, k):
        self.touched.append(k)
        return super().__getitem__(k)
    def get(self, k, default=None):
        self.touched.append(k)
        return super().get(k, default)

print(f"  {'model':<16}{'fields read':<44}{'sees window?':>14}")
print("  " + "-" * 74)
rows = []
for name in ("BDD", "TemporalResidual", "RF", "SVM", "Autoencoder",
             "CNN", "LSTM", "Transformer", "GCN", "GAT"):
    spy = Spy(batch)
    try:
        set_seed(0)
        m = build_detector(name, spec, cfg, data=d)
        m.eval()
        with torch.no_grad():
            m(spy)
        touched = [t for t in dict.fromkeys(spy.touched) if t != "spy"]
        sees = "YES" if "window" in touched else "no"
        rows.append((name, touched, sees))
        print(f"  {name:<16}{', '.join(touched):<44}{sees:>14}")
    except Exception as exc:                       # noqa: BLE001
        print(f"  {name:<16}!! {type(exc).__name__}: {exc}")

print()
print("=" * 88)
no_window = [r[0] for r in rows if r[2] == "no"]
print(f"  Models that never receive `window`: {', '.join(no_window) if no_window else '(none)'}")
print()
print("  The stealthy family's only usable signature here is a STEP in the state")
print("  trajectory -- a temporal signature. A model that never receives `window`")
print("  cannot see it for a structural reason, so those rows must be labelled as")
print("  temporal-blind in the table rather than compared as if architecture were the")
print("  only difference.")
