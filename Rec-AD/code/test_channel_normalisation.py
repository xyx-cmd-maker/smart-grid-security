"""Is the sequence models' failure caused by UNNORMALISED physics channels?

`token_sequence` concatenates three views of each timestep with no rescaling:

    z_t        raw measurement     ||.|| ~ 1150
    r_t        residual            ||.|| ~ 0.67
    nu_t       innovation          ||.|| small

That is roughly a 1700x scale difference inside a single first-layer weight matrix,
so the residual and innovation channels contribute almost nothing to the pre-activation
variance -- which would explain trained AUC ~0.50 on families whose signal lives
entirely in those channels.

This adds an opt-in per-channel standardisation (statistics estimated on the training
split, applied to all splits) and compares per-family AUC with and without it.

Outcome reading:
  normalised >> unnormalised  -> the failure was a scaling defect, the channels do
                                 carry the signal, and Table C4's CNN/LSTM rows must be
                                 re-run before any conclusion is drawn from them
  normalised ~ unnormalised   -> scaling is not the cause; keep looking
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

NAMES = {0: "normal", 1: "random", 2: "stealthy", 3: "load_redist"}
d = load_contract(HERE / "data_cache/ieee118.npz")
spec = build_spec(d)
cfg = load_config(None)
dcfg = cfg.get("data", {}) or {}


def auc_vs_benign(a, s, fam):
    sel = (a == fam) | (a == 0)
    y = (a[sel] == fam)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(s[sel])) + 1.0
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------- 1. scales
print("=" * 90)
print("  1. Channel scales (this is the suspected cause)")
print("=" * 90)
loaders = build_loaders(d, batch_size=512, num_workers=0,
                        standardize=bool(dcfg.get("standardize", False)),
                        ratios=dcfg.get("split_ratios"), seed=0)
probe = build_detector("CNN", spec, cfg, data=d)
m = spec.num_measurements
tr = build_loaders(d, batch_size=4096, num_workers=0,
                   standardize=bool(dcfg.get("standardize", False)),
                   ratios=dcfg.get("split_ratios"), seed=0)["train"]
tok_tr = torch.cat([probe.token_sequence(b) for b in tr], dim=0)
print(f"  train tokens {tuple(tok_tr.shape)}")
for ch, label in ((0, "raw z"), (1, "residual r"), (2, "innovation nu")):
    blk = tok_tr[:, :, ch * m:(ch + 1) * m]
    print(f"    {label:<16} mean|.| {blk.abs().mean():>12.4f}   "
          f"std {blk.std():>12.4f}   max|.| {blk.abs().max():>12.4f}")
print()
print("  -> a ~1700x spread across channels concatenated into one Linear layer will")
print("     let the largest channel dominate the pre-activation variance.")

# per-channel mean/std over the training split, shared across all three views
mu = tok_tr.mean(dim=(0, 1), keepdim=True)
sd = tok_tr.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)
print(f"  normalising with train stats: mu range "
      f"[{float(mu.min()):.3f}, {float(mu.max()):.3f}]  "
      f"sd range [{float(sd.min()):.4f}, {float(sd.max()):.4f}]")

# ---------------------------------------------------------------- 2. compare
print()
print("=" * 90)
print("  2. Per-family AUC, unnormalised vs normalised channels (CNN, seed 0)")
print("=" * 90)
print()

results = {}
for normalise in (False, True):
    set_seed(0)
    model = build_detector("CNN", spec, cfg, data=d)
    orig = model.token_sequence

    if normalise:
        def patched(batch, _orig=orig):
            t = _orig(batch)
            return (t - mu.to(t.dtype)) / sd.to(t.dtype)
        model.token_sequence = patched

    model.fit(loaders["train"], loaders["val"])
    s_all, a_all = [], []
    for bb in loaders["test"]:
        with torch.no_grad():
            out = model.score(bb) if hasattr(model, "score") else model(bb)
        s = out[0] if isinstance(out, (tuple, list)) else out
        s = s.detach().numpy().reshape(-1) if torch.is_tensor(s) else np.asarray(s).reshape(-1)
        s_all.append(s)
        a_all.append(bb["attack_type"].numpy().reshape(-1).astype(int))
    s = np.concatenate(s_all)
    a = np.concatenate(a_all)
    row = {}
    label = "normalised" if normalise else "unnormalised"
    print(f"  --- {label} ---")
    for k in (1, 2, 3):
        au = auc_vs_benign(a, s, k)
        thr = np.quantile(s[a == 0], 0.99)
        rate = float((s[a == k] >= thr).mean())
        row[NAMES[k]] = (au, rate)
        print(f"      {NAMES[k]:<14} AUC {au:.4f}   detect@1% {rate:.4f}")
    results[label] = row
    print()

print("=" * 90)
print("  3. Comparison")
print("=" * 90)
print(f"  {'family':<14}{'unnorm AUC':>13}{'norm AUC':>12}{'delta':>10}   verdict")
print("  " + "-" * 62)
for k in ("random", "stealthy", "load_redist"):
    u = results.get("unnormalised", {}).get(k, (np.nan, np.nan))[0]
    n = results.get("normalised", {}).get(k, (np.nan, np.nan))[0]
    v = ("SCALING WAS THE CAUSE" if n - u > 0.2
         else "no material change" if abs(n - u) <= 0.1 else "small change")
    print(f"  {k:<14}{u:>13.4f}{n:>12.4f}{n - u:>10.4f}   {v}")
print()
print("  Channel-wise ceiling for reference: residual channel separates random at")
print("  AUC 1.0000 with no training at all (measured in check_physics_channels.py).")
