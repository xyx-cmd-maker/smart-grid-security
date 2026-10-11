"""Why do the learned detectors sit at AUC ~0.50 on the corrected data?

This is the blocking unknown. Two readings are possible and they have opposite
consequences for the paper:

  (a) It is the finding. Shallow/vector-input models cannot see a family whose
      attack changes neither the measurement norm nor the residual direction, so
      0.50 is the honest answer and C2a stands.
  (b) It is a bug. The input the models receive does not actually contain the
      attack, or the labels are misaligned, in which case C2a is an artefact and
      the per-family comparison is worthless.

The test that separates them is a POSITIVE CONTROL on the training set:

    benign vs random, same pipeline
      -> random attacks carry a 30x residual. If a model cannot fit even THIS on
         the training set, the pipeline is broken, not the task.

So this script reports, for the exact representation the learned baselines use:

    train AUC   (can it fit at all?)
    test  AUC
    the same, on the easy control task

plus a direct check that the attack is present in the input array.
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

from data.contract import load_contract      # noqa: E402

P = HERE / "data_cache/ieee118.npz"
d = load_contract(P)

print("=" * 90)
print("学习型模型为何停在 AUC 0.50 —— 正对照诊断")
print("=" * 90)
print()
print(f"  文件字段: {sorted(d.keys())}")

at = np.asarray(d["attack_type"])
split = np.asarray(d["split"])

# ---- what the learned models actually receive ---------------------------
if "node_feat" in d:
    nf_raw = np.asarray(d["node_feat"], dtype=np.float64)
    print(f"  node_feat {nf_raw.shape}   <- 学习型基线的输入（按 build_spec 的约定）")
    nf = nf_raw.reshape(len(nf_raw), -1)
    print(f"  展平后 {nf.shape}")
else:
    nf = np.asarray(d["dense"], dtype=np.float64)
    print(f"  dense {nf.shape}   <- 无 node_feat，退回 dense")

dense = np.asarray(d["dense"], dtype=np.float64)
H = np.asarray(d["h_matrix"], dtype=np.float64)
proj = H @ np.linalg.pinv(H)
res = dense - dense @ proj.T

# ---- is the attack actually IN the input? ------------------------------
print()
print("  攻击是否进入了输入数组（各族的偏离量）")
print(f"    {'族':<12}{'n':>7}{'||Δnode_feat||/||benign||':>28}{'||Δresidual||':>16}")
print("    " + "-" * 62)
base_nf = nf[at == 0].mean(axis=0)
base_res = np.linalg.norm(res[at == 0], axis=1).mean()
# Natural dispersion of the benign class, as the reference against which an
# attack's deviation must be judged -- an attack that moves the input less than
# the benign class moves on its own is not learnable from a snapshot.
benign_disp = np.linalg.norm(nf[at == 0] - base_nf, axis=1).mean() / (np.linalg.norm(base_nf) + 1e-12)
print(f"    良性自身离散度 = {benign_disp:.4f}（攻击的偏离必须显著大于它才可学）")
for code, nm in ((0, "benign"), (1, "random"), (2, "stealthy"), (3, "load_redist")):
    m = at == code
    if not m.any():
        continue
    dev = np.linalg.norm(nf[m] - base_nf, axis=1).mean() / (np.linalg.norm(base_nf) + 1e-12)
    rr = np.linalg.norm(res[m], axis=1).mean() / base_res
    print(f"    {nm:<12}{int(m.sum()):>7}{dev:>28.4f}{rr:>16.2f}")

# ---- train/test controls -----------------------------------------------
tr, te = split == 0, split == 2
print()
print("  正对照与任务对照（逻辑回归，与学习型基线同一输入表示）")
print(f"    {'任务':<34}{'train AUC':>12}{'test AUC':>11}{'n_tr':>8}{'n_te':>8}")
print("    " + "-" * 74)

for task, pos in (("benign vs RANDOM (正对照, 残差30x)", 1),
                  ("benign vs stealthy (隐蔽族)", 2),
                  ("benign vs load_redist (中间档)", 3)):
    keep = np.isin(at, [0, pos])
    mtr, mte = tr & keep, te & keep
    y = (at == pos).astype(int)
    sc = StandardScaler().fit(nf[mtr])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(sc.transform(nf[mtr]), y[mtr])
    a_tr = roc_auc_score(y[mtr], clf.predict_proba(sc.transform(nf[mtr]))[:, 1])
    a_te = roc_auc_score(y[mte], clf.predict_proba(sc.transform(nf[mte]))[:, 1])
    tag = ""
    if pos == 1:
        tag = "  <- 若这里也 ≈0.50，则管线坏了" if a_tr < 0.7 else "  <- 管线正常"
    print(f"    {task:<34}{a_tr:>12.4f}{a_te:>11.4f}{int(mtr.sum()):>8}{int(mte.sum()):>8}{tag}")

print()
print("  读法：")
print("    正对照 train AUC > 0.9  -> 管线正常，隐蔽族的 0.50 是真实结论（C2a 成立）")
print("    正对照 train AUC ≈ 0.50 -> 管线坏了，0.50 无意义，必须先修管线")
print()
