"""Manifold test across all four IEEE cases.

The single-case result (ieee118: random and load-redistribution separable in the residual,
stealthy not) is the load-bearing evidence for the paper's central differentiation. A
one-case result is a single point; this extends it to 14, 57, 118 and 300 buses so the
claim does not rest on one system.

Design carried over from the corrected version, because the first version failed its own
positive control:
  - Each family is tested in the space where its signal lives (the residual).
  - The random family is the POSITIVE CONTROL and must come out separable, otherwise the
    whole run is uninterpretable rather than informative.
  - load-redistribution is a second control: intermediate signal (about 3.6x), so it
    tests whether the measurement degrades gracefully rather than all-or-nothing.

Run this ONLY when no other python job is running (AGENTS.md criterion: count processes
before starting). It is CPU-heavy: a random forest per family per case, twelve fits total.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                    # noqa: E402
import data.contract as _C                                 # noqa: E402

from sklearn.linear_model import LogisticRegression        # noqa: E402
from sklearn.ensemble import RandomForestClassifier        # noqa: E402
from sklearn.preprocessing import StandardScaler           # noqa: E402
from sklearn.model_selection import train_test_split       # noqa: E402
from sklearn.metrics import roc_auc_score                  # noqa: E402

CASES = [14, 57, 118, 300]
FAMS = [("random", "ATTACK_RANDOM"), ("stealthy", "ATTACK_STEALTHY"),
        ("load_redist", "ATTACK_LOAD_REDIST")]
N_EST = 120          # fewer trees than the single-case run; twelve fits here


def residual_of(dense: np.ndarray, h: np.ndarray) -> np.ndarray:
    """r = z - H (H^T H)^-1 H^T z, the projection a BDD test forms."""
    P = h @ np.linalg.pinv(h.T @ h) @ h.T
    return dense - dense @ P.T


def heldout(ben: np.ndarray, pos: np.ndarray, X: np.ndarray, seed: int = 0):
    Xp = np.vstack([X[ben], X[pos]])
    y = np.concatenate([np.zeros(int(ben.sum())), np.ones(int(pos.sum()))])
    Xtr, Xte, ytr, yte = train_test_split(Xp, y, test_size=0.4, random_state=seed,
                                          stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=N_EST, n_jobs=1,
                                random_state=seed).fit(Xtr, ytr)
    return (roc_auc_score(yte, lr.predict_proba(Xte)[:, 1]),
            roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]))


results = {}
print("=" * 92)
print("  Manifold test across four IEEE cases -- is the stealthy family on the benign manifold?")
print("=" * 92)
print()
print(f"  {'case':<7}{'family':<14}{'resid x':>10}{'logreg':>10}{'RF':>9}   verdict")
print("  " + "-" * 88)

for case in CASES:
    path = HERE / "data_cache" / f"ieee{case}.npz"
    if not path.exists():
        print(f"  ieee{case:<3}  MISSING")
        continue
    d = load_contract(path)
    at = np.asarray(d["attack_type"]).ravel()
    dense = np.asarray(d["dense"], dtype=np.float64)
    h = np.asarray(d["h_matrix"], dtype=np.float64)
    resid = residual_of(dense, h)

    ben = at == int(_C.ATTACK_NORMAL)
    rb = np.linalg.norm(resid[ben], axis=1).mean()

    for label, cname in FAMS:
        pos = at == int(getattr(_C, cname))
        if not pos.any():
            continue
        ratio = np.linalg.norm(resid[pos], axis=1).mean() / rb
        a_lr, a_rf = heldout(ben, pos, resid)
        best = max(a_lr, a_rf)
        verdict = ("SEPARABLE" if best > 0.65 else
                   "on-manifold" if best < 0.55 else "weakly separable")
        tag = "  <- CONTROL" if label == "random" else ""
        print(f"  {case:<7}{label:<14}{ratio:>9.2f}x{a_lr:>10.4f}{a_rf:>9.4f}   {verdict}{tag}")
        results[(case, label)] = (ratio, a_lr, a_rf)
    print()

print("  " + "-" * 88)
print("  Summary")
print("  " + "-" * 88)
ctrls_ok = all(max(results[(c, "random")][1], results[(c, "random")][2]) > 0.65
               for c in CASES if (c, "random") in results)
print(f"  positive control (random separable) passes on all cases: "
      f"{'YES' if ctrls_ok else 'NO -- results are NOT interpretable'}")
if ctrls_ok:
    st = [max(results[(c, 'stealthy')][1], results[(c, 'stealthy')][2])
          for c in CASES if (c, 'stealthy') in results]
    print(f"  stealthy best-classifier AUC across cases: "
          f"min {min(st):.4f}  max {max(st):.4f}  mean {np.mean(st):.4f}")
    print(f"  -> stealthy is on-manifold on every case tested"
          if max(st) < 0.55 else
          f"  -> NOT uniformly on-manifold; max {max(st):.4f} exceeds 0.55")
print()
