"""Quantify the cost of the pure-noise channel in node_feat.

`encode_node_features` fills channel 3 with `rng.normal(0, 0.02)` -- 118 of the
472 dimensions the sequence and graph models receive are pure noise. It carries
no information about the grid, the state or the attack, so it can only dilute a
model's fit. The comment in the generator says it exists to "give the models
something to work with", which is a development convenience rather than a
measurement channel.

Before reporting that the learned baselines sit at AUC ~0.50, it matters whether
that is partly the noise channel's doing. Same estimator, same split, only the
channel set changes.

Channel layout (from the generator):
    0  raw bus injection measurements   -- carries the attack
    1  mean across buses                -- aggregate
    2  std across buses                 -- aggregate
    3  rng.normal(0, 0.02)              -- pure noise
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
nf = np.asarray(d["node_feat"], dtype=np.float64)      # (n, n_bus, 4)
at = np.asarray(d["attack_type"])
split = np.asarray(d["split"])
print(f"  node_feat {nf.shape}  -> 展平 {nf.shape[0]} x {nf.shape[1] * nf.shape[2]}")

# Sanity: is channel 3 actually noise (zero mutual information with the label)?
c3 = nf[:, :, 3]
print(f"  channel 3 与标签的 |相关| 最大 = "
      f"{np.abs(np.corrcoef(c3.mean(axis=1), (at > 0).astype(float))[0, 1]):.4f}"
      f"   （≈0 即确认纯噪声）")
print()

SETS = {
    "ch0 only (118, 含攻击)": [0],
    "ch0+1+2 (354, 无噪声)": [0, 1, 2],
    "ch0+1+2+3 (472, 全量含噪声)": [0, 1, 2, 3],
    "ch3 only (118, 纯噪声)": [3],
}

tr, te = split == 0, split == 2
print(f"  {'输入通道':<32}{'任务':<24}{'train AUC':>11}{'test AUC':>11}")
print("  " + "-" * 78)

for name, chans in SETS.items():
    X = nf[:, :, chans].reshape(len(nf), -1)
    for tname, pos in (("vs random(正对照)", 1), ("vs stealthy", 2)):
        keep = np.isin(at, [0, pos])
        mtr, mte = tr & keep, te & keep
        y = (at == pos).astype(int)
        sc = StandardScaler().fit(X[mtr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(X[mtr]), y[mtr])
        a_tr = roc_auc_score(y[mtr], clf.predict_proba(sc.transform(X[mtr]))[:, 1])
        a_te = roc_auc_score(y[mte], clf.predict_proba(sc.transform(X[mte]))[:, 1])
        print(f"  {name:<32}{tname:<24}{a_tr:>11.4f}{a_te:>11.4f}")
    print()

print("=" * 78)
print("判读")
print("=" * 78)
print()
print("  ch3-only 行若 ≈0.50          -> 确认它零信息（预期如此）")
print("  ch0 与 ch0+1+2+3 的差         -> 就是那 118 维噪声的代价")
print("  若 ch0+1+2 (354) 明显优于全量  -> 报告学习型基线时应剔除 channel 3")
