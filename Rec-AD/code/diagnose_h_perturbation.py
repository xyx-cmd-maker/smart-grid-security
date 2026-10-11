"""How much does a wrong Jacobian cost the temporal detector?

The temporal innovation test reaches AUC 0.983 against a stealthy injection --
but only because it is handed the true H. A reviewer's first objection is that
a real operator does not know H exactly: the network model carries line-
parameter error, the topology changes, the linearisation is only valid near the
operating point.

This script quantifies the sensitivity. The ATTACK is always built with the true
H (a = Hc); only the DETECTOR's H is perturbed. That is the realistic setting:
the attacker knows the system, the defender has a model.

Reported per perturbation level, relative to ||H||_F:

  * temporal innovation detector   (the new baseline)
  * BDD residual detector          (the spatial control)
  * per attack family, so a family that becomes easier under mismatch is
    visible rather than averaged away

A detector whose advantage collapses under a small model error is not a
deployable detector, and the gap it leaves is the honest justification for a
learned approach.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import (ATTACK_LOAD_REDIST, ATTACK_NORMAL,   # noqa: E402
                           ATTACK_RANDOM, ATTACK_STEALTHY)

P = HERE / "data_cache/temporal_smoke.npz"
ALPHA = 0.01

with np.load(P, allow_pickle=False) as z:
    win = z["window"].astype(np.float64)
    h_true = z["h_matrix"].astype(np.float64)
    at = z["attack_type"]
    split = z["split"]

TRAIN, VAL, TEST = 0, 1, 2
tr = split == TRAIN
va = split == VAL
te = split == TEST

m, n_state = h_true.shape
h_norm = np.linalg.norm(h_true)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    n_p, n_n = len(pos), len(neg)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def temporal_stat(win_: np.ndarray, proj: np.ndarray,
                  scale: np.ndarray) -> np.ndarray:
    pred = np.einsum("nwm,km->nwk", win_[:, :-1, :], proj)
    nu = (win_[:, 1:, :] - pred) / scale
    return np.sum(nu ** 2, axis=2).max(axis=1)


def bdd_stat(win_: np.ndarray, proj: np.ndarray,
             sigma2: float) -> np.ndarray:
    z = win_[:, -1, :]
    r = z - z @ proj.T
    return np.sum(r ** 2, axis=1) / sigma2


def run(h_det: np.ndarray) -> dict:
    """Fit both detectors on CLEAN training data using h_det, then score."""
    pinv = np.linalg.pinv(h_det)
    proj = h_det @ pinv

    clean = win[tr & (at == ATTACK_NORMAL)]
    if len(clean) < 10:
        return {}

    # temporal: per-dimension innovation scale from clean training
    pred = np.einsum("nwm,km->nwk", clean[:, :-1, :], proj)
    nu = clean[:, 1:, :] - pred
    scale = np.sqrt((nu ** 2).mean(axis=(0, 1))) + 1e-12
    t_ref = temporal_stat(clean, proj, scale)
    t_thr = float(np.quantile(t_ref, 1.0 - ALPHA))

    # BDD: residual variance from clean training
    r_clean = clean[:, -1, :] - clean[:, -1, :] @ proj.T
    dof = max(1, m - int(np.linalg.matrix_rank(h_det)))
    sigma2 = float(np.sum(r_clean ** 2) / max(r_clean.size, 1)) * (m / dof)
    b_ref = bdd_stat(clean, proj, sigma2)
    b_thr = float(np.quantile(b_ref, 1.0 - ALPHA))

    t_score = temporal_stat(win[te], proj, scale)
    b_score = bdd_stat(win[te], proj, sigma2)
    at_te = at[te]

    out = {}
    for label, score in (("temporal", t_score), ("bdd", b_score)):
        benign = score[at_te == ATTACK_NORMAL]
        out[f"{label}_fpr"] = float((benign >= (t_thr if label == "temporal"
                                                 else b_thr)).mean())
        for code, nm in ((ATTACK_RANDOM, "random"),
                         (ATTACK_STEALTHY, "stealthy"),
                         (ATTACK_LOAD_REDIST, "load_redist")):
            pos = score[at_te == code]
            out[f"{label}_auc_{nm}"] = auc(pos, benign)
    return out


print("=" * 92)
print("H 扰动实验：攻击始终用真实 H 构造，只有检测器的 H 被扰动")
print("=" * 92)
print()
print(f"  H: {m} x {n_state},  ||H||_F = {h_norm:,.1f}")
print(f"  测试集 {int(te.sum()):,} 样本")
print()

levels = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
rng = np.random.default_rng(12345)

print(f"{'‖ΔH‖/‖H‖':>11}{'随机':>9}{'隐蔽':>9}{'重分配':>9}   |"
      f"{'随机':>9}{'隐蔽':>9}{'重分配':>9}")
print(f"{'':>11}{'--- 时序新息 ---':>29}   |{'--- BDD 残差 ---':>29}")
print("-" * 92)

rows = []
for lvl in levels:
    if lvl == 0.0:
        h_det = h_true
    else:
        delta = rng.normal(size=h_true.shape)
        delta *= (lvl * h_norm) / np.linalg.norm(delta)
        h_det = h_true + delta
    r = run(h_det)
    if not r:
        continue
    r["level"] = lvl
    rows.append(r)
    print(f"{lvl:>11.3f}"
          f"{r['temporal_auc_random']:>9.3f}"
          f"{r['temporal_auc_stealthy']:>9.3f}"
          f"{r['temporal_auc_load_redist']:>9.3f}   |"
          f"{r['bdd_auc_random']:>9.3f}"
          f"{r['bdd_auc_stealthy']:>9.3f}"
          f"{r['bdd_auc_load_redist']:>9.3f}")

print()
print("=" * 92)
print("判定")
print("=" * 92)
print()
base_t = rows[0]["temporal_auc_stealthy"]
base_b = rows[0]["bdd_auc_stealthy"]
print(f"  时序基线对隐蔽攻击：无扰动 AUC {base_t:.3f}  ->  最差 {min(r['temporal_auc_stealthy'] for r in rows):.3f}")
print(f"  BDD     对隐蔽攻击：无扰动 AUC {base_b:.3f}  ->  最好 {max(r['bdd_auc_stealthy'] for r in rows):.3f}")
print()

# find the level at which the temporal advantage halves
half = None
for r in rows[1:]:
    if r["temporal_auc_stealthy"] < 0.5 * (base_t - 0.5) + 0.5:
        half = r["level"]
        break
if half is not None:
    print(f"  时序基线的隐蔽攻击 AUC 掉到「一半优势」时的扰动量：‖ΔH‖/‖H‖ ≈ {half}")
else:
    print("  在测试范围内（最大 1.0），时序基线的优势没有掉一半。")

# does a wrong H make the stealthy attack visible to BDD?
gain = max(r["bdd_auc_stealthy"] for r in rows) - base_b
print(f"  BDD 对隐蔽攻击的 AUC 因 H 失配最多变化 {gain:+.3f}")
if gain > 0.1:
    print("     -> H 失配让 BDD 反而能看见隐蔽攻击（泄漏效应）")
    print("        这削弱了「学习型方法抗模型误差」的论据。")
else:
    print("     -> H 失配没有让 BDD 获得对隐蔽攻击的可见性。")

json.dump(rows, open(HERE / "results" / "h_perturbation.json", "w",
                     encoding="utf-8"), ensure_ascii=False, indent=2)
print()
print(f"  明细已写入 results/h_perturbation.json")
