"""Manifold test, corrected: run the positive control where the signal actually is.

The first version of this test failed its own positive control and was therefore
uninformative. That is worth recording precisely, because the failure mode is subtle.

What happened: benign-vs-family classifiers were trained on the raw measurement snapshot
(`dense`, 297 dimensions). The random family -- the positive control, which is 30x in the
RESIDUAL -- came out at AUC 0.5393, i.e. not separable. Since the control failed, the
matching null result on the stealthy family (0.4980) could not be read as evidence of
anything; a test with no power returns nulls for every input.

Why the control failed: the attack perturbs the raw measurement vector by about 1.7% of
its norm (20.08 of 1150). Training a classifier to find a 1.7% perturbation among 297
dimensions from 12,000 samples is not a test of manifold membership, it is a re-run of
this project's separate finding that raw-measurement detectors see nothing.

The correction: each family is tested in the space where its signal lives.
  - random    -> residual, where it is 30x        (positive control: must separate)
  - stealthy  -> residual, where it is 1.00x      (the actual question)
  - stealthy  -> raw snapshot, where it is 1.00x  (secondary)
A null on stealthy is only meaningful alongside a working positive control in the SAME
space, so both are now measured on the residual.
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
at = np.asarray(d["attack_type"]).ravel()
dense = np.asarray(d["dense"], dtype=np.float64)

import data.contract as _C                                 # noqa: E402
codes = {"benign": int(_C.ATTACK_NORMAL), "random": int(_C.ATTACK_RANDOM),
         "stealthy": int(_C.ATTACK_STEALTHY),
         "load_redist": int(_C.ATTACK_LOAD_REDIST)}
ben = at == codes["benign"]

# The residual is what a BDD-style test sees: r = z - H (H^T H)^-1 H^T z.
H = np.asarray(d["h_matrix"], dtype=np.float64)
P = H @ np.linalg.pinv(H.T @ H) @ H.T
resid = dense - dense @ P.T

from sklearn.linear_model import LogisticRegression        # noqa: E402
from sklearn.ensemble import RandomForestClassifier        # noqa: E402
from sklearn.preprocessing import StandardScaler           # noqa: E402
from sklearn.model_selection import train_test_split       # noqa: E402
from sklearn.metrics import roc_auc_score                  # noqa: E402


def heldout(pos_mask, X, seed=0):
    Xp = np.vstack([X[ben], X[pos_mask]])
    y = np.concatenate([np.zeros(int(ben.sum())), np.ones(int(pos_mask.sum()))])
    Xtr, Xte, ytr, yte = train_test_split(Xp, y, test_size=0.4, random_state=seed,
                                          stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=seed).fit(Xtr, ytr)
    return (roc_auc_score(yte, lr.predict_proba(Xte)[:, 1]),
            roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]))


print("=" * 88)
print("  Manifold test, corrected -- positive control in the space where the signal lives")
print("=" * 88)
print()

nb = np.linalg.norm(dense[ben], axis=1)
rb = np.linalg.norm(resid[ben], axis=1)
print(f"  benign: ||dense|| {nb.mean():.2f}   ||residual|| {rb.mean():.4f}")
print()
print(f"  {'family':<14}{'||dense|| x':>12}{'||resid|| x':>13}")
print("  " + "-" * 84)
for lab in ("random", "stealthy", "load_redist"):
    m = at == codes[lab]
    nd = np.linalg.norm(dense[m], axis=1).mean() / nb.mean()
    nr = np.linalg.norm(resid[m], axis=1).mean() / rb.mean()
    print(f"  {lab:<14}{nd:>11.2f}x{nr:>12.2f}x")

print()
for space_name, X in (("RESIDUAL  (signal space)", resid), ("RAW dense (control space)", dense)):
    print("  " + "-" * 84)
    print(f"  Classifier on {space_name}")
    print("  " + "-" * 84)
    print(f"  {'family':<14}{'logreg AUC':>13}{'RF AUC':>11}   verdict")
    for lab in ("random", "stealthy", "load_redist"):
        m = at == codes[lab]
        a_lr, a_rf = heldout(m, X)
        best = max(a_lr, a_rf)
        v = ("SEPARABLE" if best > 0.65 else
             "on-manifold" if best < 0.55 else "weakly separable")
        tag = "  <- POSITIVE CONTROL" if lab == "random" else ""
        print(f"  {lab:<14}{a_lr:>13.4f}{a_rf:>11.4f}   {v}{tag}")
    print()

print("  " + "-" * 84)
print("  How to read this")
print("  " + "-" * 84)
print("  In the RESIDUAL space the random family must come out SEPARABLE -- it is 30x.")
print("  If it does, the test has power, and the stealthy verdict in the same space is")
print("  meaningful. The raw-dense block is reported for contrast only: there nothing is")
print("  separable, because the attack is 1.7% of that norm.")
print()
