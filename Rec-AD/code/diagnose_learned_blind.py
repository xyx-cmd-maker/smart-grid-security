"""What do the learned detectors actually receive, and is the attack in it?

The previous check found something that contradicts the paper's explanation: BDD
detects the random family at 1.0000 from a single snapshot, while every learned
detector -- including a random forest, which needs no gradient descent at all --
stays at its false-alarm rate on that same family. A model that cannot separate a
30x anomaly is not "limited by representation"; something upstream is wrong.

This script inspects the feature path rather than the model:
  1. the raw contract arrays, per family, to confirm the attack is present in `dense`
  2. the pooled feature vector the detectors consume, per family
  3. the class balance of each split
  4. whether the label aligns with the family

Any of these can explain the failure, and they are cheap to check. The point is to
locate the fault before writing another word of interpretation.
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
from recad.common import load_config                        # noqa: E402

d = load_contract(HERE / "data_cache/ieee118.npz")
spec = build_spec(d)
cfg = load_config(None)

ATTACK_NAMES = {0: "normal", 1: "random", 2: "stealthy", 3: "load_redist",
                4: "?", 5: "?"}

print("=" * 90)
print("1. Raw contract arrays, per attack family")
print("=" * 90)
atk = np.asarray(d["attack_type"]).astype(int)
dense = np.asarray(d["dense"], dtype=np.float32)
print(f"  attack_type values present: {sorted(set(atk.tolist()))}")
print()
print(f"  {'family':<14}{'n':>7}{'||dense|| mean':>18}{'ratio vs normal':>18}")
base = None
for a in sorted(set(atk.tolist())):
    m = atk == a
    if not m.any():
        continue
    nrm = float(np.linalg.norm(dense[m], axis=1).mean())
    if base is None:
        base = nrm
    print(f"  {ATTACK_NAMES.get(a, str(a)):<14}{int(m.sum()):>7}{nrm:>18.4f}{nrm / base:>18.2f}")

print()
print("=" * 90)
print("2. What the detectors consume, and each split's composition")
print("=" * 90)
dcfg = cfg.get("data", {}) or {}
loaders = build_loaders(
    d, batch_size=4096, num_workers=0,
    standardize=bool(dcfg.get("standardize", False)),
    ratios=dcfg.get("split_ratios"), seed=0,
)

for split in ("train", "val", "test"):
    dl = loaders.get(split)
    if dl is None:
        continue
    feats, labels, atks = [], [], []
    for b in dl:
        feats.append(b["dense"].numpy())
        labels.append(b["label"].numpy())
        atks.append(b["attack_type"].numpy())
    F = np.concatenate(feats)
    y = np.concatenate(labels)
    a = np.concatenate(atks).astype(int)
    print(f"  {split}: n={len(y)}  feature dim={F.shape[1]}"
          f"  pos rate={y.mean():.3f}")
    fam = {ATTACK_NAMES.get(k, str(k)): int((a == k).sum()) for k in sorted(set(a.tolist()))}
    print(f"      families: {fam}")
    # separation of the RANDOM family inside this split, on the feature the model sees
    nrm = np.linalg.norm(F, axis=1)
    normal = nrm[a == 0]
    for k in sorted(set(a.tolist())):
        if k == 0:
            continue
        r = nrm[a == k].mean() / normal.mean() if normal.size else float("nan")
        print(f"      ||x|| {ATTACK_NAMES.get(k, k):<14} / normal = {r:.2f}")
    print()

print("=" * 90)
print("3. Would a trivial rule separate the random family?")
print("=" * 90)
tr = loaders.get("train")
if tr is not None:
    d0 = np.asarray(d["dense"], dtype=np.float32)
    nrm = np.linalg.norm(d0, axis=1)
    n_mask, r_mask = atk == 0, atk == 1
    if n_mask.any() and r_mask.any():
        print(f"  ||dense|| normal  : mean {nrm[n_mask].mean():.3f}  "
              f"p99 {np.percentile(nrm[n_mask], 99):.3f}")
        print(f"  ||dense|| random  : mean {nrm[r_mask].mean():.3f}  "
              f"p01 {np.percentile(nrm[r_mask], 1):.3f}")
        # a threshold at the 99th percentile of normal
        thr = np.percentile(nrm[n_mask], 99)
        tpr = float((nrm[r_mask] > thr).mean())
        print(f"  threshold = normal p99 = {thr:.3f}")
        print(f"  -> random-family detection with that ONE scalar rule: {tpr:.4f}")
        print()
        print("  If this number is near 1.0 while the learned detectors report ~0.01,")
        print("  then the attack IS separable from a single scalar of the same input,")
        print("  and the fault is in the learned pipeline, not in the representation.")
