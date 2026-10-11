"""Does FlatFeatureDetector have the same channel-scale defect as the sequence models?

`FlatFeatureDetector.features()` does:

    return torch.cat([dense, pooled], dim=1)

with `dense` at norm ~1150 and `pooled` a mean over embedding rows initialised at
std 0.02. If that spread is large, the flat models have the same defect that just cost
the sequence models their entire result set, and the Autoencoder residual experiment
(AUC 1.0000) needs re-measuring before it can be trusted.

Two things are reported:
  1. the actual scale of every field the flat models consume, in both input modes;
  2. per-family AUC with and without standardising the concatenated feature vector,
     which is the same controlled comparison that exposed the sequence defect.

This is deliberately run BEFORE trusting anything downstream of it: a second scaling
defect would invalidate the residual-input finding, which is currently the cleanest
evidence in the project.
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


# ------------------------------------------------------- 1. field scales
print("=" * 92)
print("  1. Scales of the fields FlatFeatureDetector concatenates")
print("=" * 92)
for mode in ("dense", "residual"):
    cfg_m = load_config(None)
    cfg_m.setdefault("baselines", {})["input_mode"] = mode
    loaders = build_loaders(d, batch_size=512, num_workers=0,
                            standardize=bool(dcfg.get("standardize", False)),
                            ratios=dcfg.get("split_ratios"), seed=0)
    set_seed(0)
    model = build_detector("Autoencoder", spec, cfg_m, data=d)
    b = next(iter(loaders["train"]))
    with torch.no_grad():
        dense_in = b["dense"]
        if getattr(model, "input_mode", "dense") == "residual":
            dense_in = model._to_residual(dense_in)
        pooled = model.embedding(b["sparse"])
        feat = model.features(b)
    print(f"  input_mode={mode}")
    print(f"    dense/残差  norm mean {float(dense_in.norm(dim=1).mean()):>12.4f}"
          f"   max|.| {float(dense_in.abs().max()):>10.4f}"
          f"   dim {dense_in.shape[1]}")
    print(f"    pooled emb  norm mean {float(pooled.norm(dim=1).mean()):>12.4f}"
          f"   max|.| {float(pooled.abs().max()):>10.4f}"
          f"   dim {pooled.shape[1]}")
    r = float(dense_in.norm(dim=1).mean() / max(float(pooled.norm(dim=1).mean()), 1e-12))
    print(f"    -> scale ratio (dense / pooled) = {r:,.1f}x")
    print(f"    feature vector dim {feat.shape[1]}   "
          f"per-dim sd spread [{float(feat.std(0).min()):.4g}, {float(feat.std(0).max()):.4g}]")
    print()

# ------------------------------------------- 2. standardised vs not, residual input
print("=" * 92)
print("  2. Autoencoder with RESIDUAL input: standardised vs not (seed 0)")
print("=" * 92)
print()

results = {}
for normalise in (False, True):
    cfg_m = load_config(None)
    cfg_m.setdefault("baselines", {})["input_mode"] = "residual"
    loaders = build_loaders(d, batch_size=512, num_workers=0,
                            standardize=bool(dcfg.get("standardize", False)),
                            ratios=dcfg.get("split_ratios"), seed=0)
    set_seed(0)
    model = build_detector("Autoencoder", spec, cfg_m, data=d)
    assert getattr(model, "input_mode", None) == "residual", "residual mode not active"

    if normalise:
        fb = next(iter(loaders["train"]))
        with torch.no_grad():
            F = model.features(fb)
        mu, sd = F.mean(0, keepdim=True), F.std(0, keepdim=True).clamp_min(1e-8)
        orig = model.features

        def patched(batch, _o=orig, _mu=mu, _sd=sd):
            f = _o(batch)
            return (f - _mu.to(f.dtype)) / _sd.to(f.dtype)
        model.features = patched

    model.fit(loaders["train"], loaders["val"])
    s_all, a_all = [], []
    for bb in loaders["test"]:
        with torch.no_grad():
            out = model.score(bb) if hasattr(model, "score") else model(bb)
        s = out[0] if isinstance(out, (tuple, list)) else out
        s = s.detach().numpy().reshape(-1) if torch.is_tensor(s) else np.asarray(s).reshape(-1)
        s_all.append(s)
        a_all.append(bb["attack_type"].numpy().reshape(-1).astype(int))
    s, a = np.concatenate(s_all), np.concatenate(a_all)

    label = "standardised" if normalise else "raw concat"
    print(f"  --- {label} ---")
    row = {}
    for k in (1, 2, 3):
        au = auc_vs_benign(a, s, k)
        thr = np.quantile(s[a == 0], 0.99)
        row[NAMES[k]] = (au, float((s[a == k] >= thr).mean()))
        print(f"      {NAMES[k]:<14} AUC {au:.4f}   detect@1% {row[NAMES[k]][1]:.4f}")
    results[label] = row
    print()

print("=" * 92)
print("  3. Comparison")
print("=" * 92)
print(f"  {'family':<14}{'raw AUC':>10}{'std AUC':>10}{'delta':>10}   verdict")
print("  " + "-" * 58)
for k in ("random", "stealthy", "load_redist"):
    u = results.get("raw concat", {}).get(k, (np.nan,))[0]
    n = results.get("standardised", {}).get(k, (np.nan,))[0]
    v = ("FLAT MODELS ALSO AFFECTED" if n - u > 0.1
         else "no material change" if abs(n - u) <= 0.05 else "small change")
    print(f"  {k:<14}{u:>10.4f}{n:>10.4f}{n - u:>10.4f}   {v}")
print()
print("  Note: the residual input is already a whitened quantity, so its scale is close")
print("  to the pooled embedding's -- unlike the raw-measurement mode, where dense is")
print("  ~1150. If the delta here is small while the sequence delta was large, the flat")
print("  models are not affected in the same way and the earlier result stands.")
