"""The decisive test: does supplying the residual projection fix the learned baselines?

Context. On the corrected data every learned baseline sits at AUC ~0.50, and the
positive control (benign vs random, whose residual is 30x benign) also fails at
train AUC 0.58. That failure has two candidate explanations with opposite
consequences for the paper:

  (a) Representation. The model is handed raw measurements, in which a random
      attack is a ~2.3% per-measurement perturbation. To exploit the 30x
      residual separation it must first learn the projector (I - P). A linear
      model cannot, so 0.58 is a representation problem that is fixable by
      handing the model a residual channel.

  (b) Something else entirely (labels, split, training budget).

If (a) holds, the residual input should be near-perfectly separable on the
TRAINING set. If it is not, (a) is dead and the search continues.

Same estimator, same split, same protocol -- only the input changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract      # noqa: E402

d = load_contract(HERE / "data_cache/ieee118.npz")
z = np.asarray(d["dense"], dtype=np.float64)
H = np.asarray(d["h_matrix"], dtype=np.float64)
at = np.asarray(d["attack_type"])
split = np.asarray(d["split"])
sparse = np.asarray(d["sparse"], dtype=np.float64)

P = H @ np.linalg.pinv(H)
res = z - z @ P.T                      # (I - P) z

REPS = {
    "A 原始量测 dense": z,
    "B 残差 residual=(I-P)z": res,
    "C dense + residual": np.hstack([z, res]),
    "D 残差范数 |res|": np.linalg.norm(res, axis=1, keepdims=True),
    "E sparse（8 维）": sparse,
}

tr, te = split == 0, split == 2
print("=" * 96)
print("决定性实验：给出投影，学习型基线能不能起来？")
print("=" * 96)
print()
print(f"  {'输入表示':<30}{'任务':<26}{'train AUC':>11}{'test AUC':>11}")
print("  " + "-" * 78)

for rname, X in REPS.items():
    if X.ndim != 2:
        X = X.reshape(len(X), -1)
    for tname, pos in (("vs random(正对照)", 1), ("vs stealthy", 2)):
        keep = np.isin(at, [0, pos])
        mtr, mte = tr & keep, te & keep
        y = (at == pos).astype(int)
        sc = StandardScaler().fit(X[mtr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(X[mtr]), y[mtr])
        a_tr = roc_auc_score(y[mtr], clf.predict_proba(sc.transform(X[mtr]))[:, 1])
        a_te = roc_auc_score(y[mte], clf.predict_proba(sc.transform(X[mte]))[:, 1])
        print(f"  {rname:<30}{tname:<26}{a_tr:>11.4f}{a_te:>11.4f}")
    print()

print("=" * 96)
print("判定")
print("=" * 96)
print()
print("  B 行的 train AUC 若 > 0.95  -> 问题是【缺少投影】，给残差通道即可修")
print("  B 行的 train AUC 若 ≈ 0.58  -> 【表示】解释不成立，另有原因")
print("  D 行（残差范数）是单快照幅度统计量 —— 它是 C2b 的直接检验：")
print("     若 D 对 stealthy 也是 0.5，则坐实『单快照幅度统计量也盲』")
