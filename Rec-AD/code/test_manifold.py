"""Test whether the stealthy family actually lies on the benign manifold.

Why this is load-bearing: the project's strongest differentiation is that our stealthy
attack stays on the benign manifold spatially while PICL's does not, so that manifold
detectors cannot separate ours. The current evidence for that is two matching norms
(||dense|| 1.00x, residual 1.00x). Equal norms do not imply equal distributions -- a
perturbation can preserve the norm while moving the sample off-manifold -- so the claim
needs a distributional test, not a norm comparison.

Three tests, each answering a different version of the question:

  T1  Can a classifier trained to separate benign from stealthy do better than chance on
      held-out data? If AUC ~ 0.5, no single-snapshot statistic separates them.

  T2  Norm comparison for reference (the evidence currently in use), reported so the
      reader can see how much weaker it is than T1.

  T3  Residual comparison against the random family as a positive control. The random
      family IS separable (30x residual), so the same test must succeed on it. Without
      this control, a null result on stealthy would be uninterpretable -- it could just
      mean the test has no power.

T3 is the part that makes T1 meaningful: a negative result needs a working positive
control, otherwise it is indistinguishable from a broken measurement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                    # noqa: E402
from data.dataset import build_spec                        # noqa: E402

np.random.seed(0)

d = load_contract(HERE / "data_cache/ieee118.npz")
spec = build_spec(d)
names = ["benign", "random", "stealthy", "load_redist"]

# Locate the per-family label and the dense snapshot matrix.
at = np.asarray(d["attack_type"]).ravel()
dense = np.asarray(d["dense"], dtype=np.float64)

# The attack-family codes live as module constants, not as contract entries.
import data.contract as _C                                # noqa: E402
codes = {}
for label, cname in (("benign", "ATTACK_NORMAL"), ("random", "ATTACK_RANDOM"),
                     ("stealthy", "ATTACK_STEALTHY"),
                     ("load_redist", "ATTACK_LOAD_REDIST")):
    v = getattr(_C, cname, None)
    if v is not None:
        codes[label] = int(v)
if not codes:
    vals, cnt = np.unique(at, return_counts=True)
    print("  [info] no ATTACK_* constants; observed codes:", dict(zip(vals.tolist(), cnt.tolist())))
    order = vals[np.argsort(-cnt)]
    codes = {n: int(c) for n, c in zip(["benign", "random", "stealthy", "load_redist"], order)}

print("=" * 86)
print("  Does the stealthy family lie on the benign manifold?")
print("=" * 86)
print()
print(f"  file       : data_cache/ieee118.npz")
print(f"  dense      : {dense.shape}   (one snapshot per sample)")
print(f"  codes      : {codes}")
print()

ben = at == codes["benign"]
print(f"  benign samples: {int(ben.sum())}   family sizes: "
      + ", ".join(f"{k}={int((at == v).sum())}" for k, v in codes.items() if k != "benign"))
print()

def mask_of(label):
    m = at == codes[label]
    return m if m.any() else None

print("  " + "-" * 82)
print("  T2 (reference): norm comparison -- the evidence currently in use")
print("  " + "-" * 82)
nb = np.linalg.norm(dense[ben], axis=1)
print(f"  {'family':<14}{'mean ||dense||':>16}{'ratio vs benign':>18}")
print(f"  {'benign':<14}{nb.mean():>16.2f}{1.0:>17.2f}x")
fam_masks = {}
for label in ("random", "stealthy", "load_redist"):
    m = mask_of(label)
    if m is None:
        continue
    fam_masks[label] = m
    n = np.linalg.norm(dense[m], axis=1)
    print(f"  {label:<14}{n.mean():>16.2f}{n.mean()/nb.mean():>17.2f}x")

print()
print("  " + "-" * 82)
print("  T1: can a classifier separate benign from each family on a SINGLE snapshot?")
print("  " + "-" * 82)
print("  A held-out AUC near 0.5 means no single-snapshot statistic separates them.")
print()

from sklearn.linear_model import LogisticRegression      # noqa: E402
from sklearn.ensemble import RandomForestClassifier      # noqa: E402
from sklearn.preprocessing import StandardScaler         # noqa: E402
from sklearn.model_selection import train_test_split     # noqa: E402
from sklearn.metrics import roc_auc_score                # noqa: E402


def heldout_auc(pos_mask, seed=0):
    """AUC of a classifier trained to separate benign from the given family."""
    X = np.vstack([dense[ben], dense[pos_mask]])
    y = np.concatenate([np.zeros(ben.sum()), np.ones(pos_mask.sum())])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed,
                                          stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    out = {}
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    out["logreg"] = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
    rf = RandomForestClassifier(n_estimators=200, n_jobs=1,
                                random_state=seed).fit(Xtr, ytr)
    out["rf"] = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
    return out


print(f"  {'family':<14}{'logreg AUC':>14}{'RF AUC':>12}   verdict")
print("  " + "-" * 82)
for label in ("random", "stealthy", "load_redist"):
    m = fam_masks.get(label)
    if m is None:
        continue
    r = heldout_auc(m)
    best = max(r.values())
    verdict = ("SEPARABLE" if best > 0.65 else
               "on-manifold (not separable)" if best < 0.55 else "weakly separable")
    print(f"  {label:<14}{r['logreg']:>14.4f}{r['rf']:>12.4f}   {verdict}")

print()
print("  " + "-" * 82)
print("  Reading")
print("  " + "-" * 82)
print("  The random family is the POSITIVE CONTROL: it must come out SEPARABLE, because")
print("  it is 30x in the residual. If it did not, the test would have no power and the")
print("  stealthy result would be uninterpretable rather than informative.")
print()
