"""Where is the attack signal, and why does no learned model find it?

Every learned baseline sits at AUC ~0.50 on every attack family while two
deterministic detectors reach 0.81 and 0.98 on the same data. That pattern says
the problem is in the learned path, not in the task.

This isolates it. Plain logistic regression on each candidate feature block,
using scikit-learn directly -- no DataLoader, no torch, no custom training loop:

    dense            the raw measurement snapshot
    sparse           the categorical features
    residual         z - P z              (what BDD tests)
    innovation       z_t - P z_{t-1}      (what the temporal test uses)

If logistic regression on `residual` separates the families but on `dense` does
not, the signal is there and the neural pipeline is losing it. If nothing
separates anything, the labels or the splits are broken.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import (ATTACK_LOAD_REDIST, ATTACK_NORMAL,   # noqa: E402
                           ATTACK_RANDOM, ATTACK_STEALTHY)

P = HERE / "data_cache/temporal_smoke.npz"
with np.load(P, allow_pickle=False) as z:
    dense = z["dense"].astype(np.float64)
    sparse = z["sparse"].astype(np.float64)
    win = z["window"].astype(np.float64)
    h = z["h_matrix"].astype(np.float64)
    at = z["attack_type"]
    label = z["label"]
    split = z["split"]

tr, va, te = split == 0, split == 1, split == 2
proj = h @ np.linalg.pinv(h)

residual = dense - dense @ proj.T
pred = np.einsum("nwm,km->nwk", win[:, :-1, :], proj)
innovation = np.concatenate(
    [residual[:, None, :], win[:, 1:, :] - pred], axis=1)
# per-trajectory summary statistics, which is what a pooling encoder sees
innov_stat = np.concatenate([
    innovation.max(axis=1),
    innovation.std(axis=1),
    np.abs(innovation).mean(axis=1),
], axis=1)

BLOCKS = {
    "dense (297)": dense,
    "sparse (8)": sparse,
    "residual (297)": residual,
    "innovation stats (891)": innov_stat,
}

print("=" * 88)
print("逻辑回归：每个特征块能分开哪一族？（AUC vs benign，测试集）")
print("=" * 88)
print()
print(f"{'特征块':<26}{'随机':>10}{'隐蔽':>10}{'重分配':>10}{'全体攻击':>12}")
print("-" * 88)

benign_te = at[te] == ATTACK_NORMAL
for name, X in BLOCKS.items():
    Xtr, Xte = X[tr], X[te]
    sc = StandardScaler().fit(Xtr)          # fit on train only
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, label[tr])
    s = clf.predict_proba(Xte)[:, 1]

    row = []
    for code in (ATTACK_RANDOM, ATTACK_STEALTHY, ATTACK_LOAD_REDIST):
        m = at[te] == code
        a = roc_auc_score(np.r_[np.ones(m.sum()), np.zeros(benign_te.sum())],
                          np.r_[s[m], s[benign_te]])
        row.append(a)
    allatk = ~benign_te
    a_all = roc_auc_score(np.r_[np.ones(allatk.sum()), np.zeros(benign_te.sum())],
                          np.r_[s[allatk], s[benign_te]])
    print(f"{name:<26}{row[0]:>10.3f}{row[1]:>10.3f}{row[2]:>10.3f}{a_all:>12.3f}")

print()
print("=" * 88)
print("对照：二分类任务（只用 benign + random）")
print("=" * 88)
print()
keep = np.isin(at, [ATTACK_NORMAL, ATTACK_RANDOM])
for name in ("dense (297)", "residual (297)"):
    X = BLOCKS[name]
    sc = StandardScaler().fit(X[tr & keep])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(X[tr & keep]), label[tr & keep])
    t = te & keep
    s = clf.predict_proba(sc.transform(X[t]))[:, 1]
    print(f"  {name:<26} AUC = {roc_auc_score(label[t], s):.4f}")

print()
print("=" * 88)
print("数据完整性自检")
print("=" * 88)
print()
print(f"  样本 {len(label):,}   train/val/test = "
      f"{int(tr.sum()):,}/{int(va.sum()):,}/{int(te.sum()):,}")
print(f"  标签分布 (train): {np.bincount(label[tr])}")
print(f"  攻击族分布 (train): {np.bincount(at[tr])}")
print(f"  dense 范围 [{dense.min():.2f}, {dense.max():.2f}]  std={dense.std():.2f}")
print(f"  residual 中位范数: benign="
      f"{np.median(np.linalg.norm(residual[at == ATTACK_NORMAL], axis=1)):.3f}  "
      f"random={np.median(np.linalg.norm(residual[at == ATTACK_RANDOM], axis=1)):.3f}  "
      f"stealthy={np.median(np.linalg.norm(residual[at == ATTACK_STEALTHY], axis=1)):.3f}")
print()
print("  若 logistic 在 residual 上高、在 dense 上低 -> 信号在，神经管线丢了它")
print("  若所有块都 ~0.5                        -> 标签或划分有问题")
