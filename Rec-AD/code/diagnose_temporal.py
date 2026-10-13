"""Does the temporal structure restore the difficulty gradient?

The claim under test: a stealthy injection (a = Hc) is invisible to any
single-snapshot residual test, but if it is applied as a STEP it makes the
apparent state jump by more than the state dynamics allow -- and that jump is
visible to a detector that looks along the trajectory.

Two statistics are compared per family:

  SPATIAL  residual norm ||z - H pinv(H) z||        -- single snapshot
  TEMPORAL max_t ||x_hat_t - x_hat_{t-1}||          -- state jump between
                                                       consecutive snapshots,
                                                       where x_hat = pinv(H) z

For the gradient to hold, SPATIAL must separate random from stealthy, and
TEMPORAL must separate stealthy from benign.
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

P = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data_cache/temporal_smoke.npz"
if not P.is_absolute():
    P = HERE / P
with np.load(P, allow_pickle=False) as z:
    win = z["window"].astype(np.float64)
    h = z["h_matrix"].astype(np.float64)
    at = z["attack_type"]
    meta = json.loads(str(z["meta"]))

names = {ATTACK_NORMAL: "benign", ATTACK_RANDOM: "random",
         ATTACK_STEALTHY: "stealthy", ATTACK_LOAD_REDIST: "load_redist"}

print(f"数据: {P.name}   window={win.shape[1]}   m={win.shape[2]}")
print(f"state_step_sigma = {meta['state_step_sigma']}")
print()

# ---- statistics -----------------------------------------------------------
pinv = np.linalg.pinv(h)                       # n_state x m
Psym = h @ pinv                                # projector onto col(H)

# SPATIAL: residual of the FINAL snapshot only (what a single-snapshot
# detector can see)
z_last = win[:, -1, :]
residual = np.linalg.norm(z_last - z_last @ Psym.T, axis=1)

# TEMPORAL: largest jump of the apparent state between consecutive snapshots
xhat = np.einsum("nwm,km->nwk", win, pinv)     # (n, window, n_state)
jumps = np.linalg.norm(np.diff(xhat, axis=1), axis=2)   # (n, window-1)
max_jump = jumps.max(axis=1)

print("=" * 84)
print("各族的两个统计量（中位数）")
print("=" * 84)
print()
print(f"  {'家族':<14}{'SPATIAL 残差范数':>20}{'TEMPORAL 最大状态跳变':>24}")
print("-" * 84)
tables = {}
for code, nm in names.items():
    mask = at == code
    r = np.median(residual[mask])
    j = np.median(max_jump[mask])
    tables[nm] = (residual[mask], max_jump[mask])
    print(f"  {nm:<14}{r:>20.4f}{j:>24.4f}")
print()

print("=" * 84)
print("分离度：各族中位数 / benign 中位数")
print("=" * 84)
print()
base_r = np.median(tables["benign"][0])
base_j = np.median(tables["benign"][1])
print(f"  {'家族':<14}{'SPATIAL 倍数':>16}{'TEMPORAL 倍数':>18}")
print("-" * 84)
for nm in ("random", "stealthy", "load_redist"):
    r = np.median(tables[nm][0]) / base_r
    j = np.median(tables[nm][1]) / base_j
    print(f"  {nm:<14}{r:>16.2f}{j:>18.2f}")
print()

print("=" * 84)
print("判定")
print("=" * 84)
print()
def auc(pos, neg):
    """Rank-based AUC: P(pos > neg)."""
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    n_pos, n_neg = len(pos), len(neg)
    return (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

print(f"  {'检测统计量':<26}{'random vs benign':>20}{'stealthy vs benign':>22}")
print("-" * 84)
for label, idx in (("SPATIAL  残差范数", 0), ("TEMPORAL 状态跳变", 1)):
    a_r = auc(tables["random"][idx], tables["benign"][idx])
    a_s = auc(tables["stealthy"][idx], tables["benign"][idx])
    print(f"  {label:<26}{a_r:>20.3f}{a_s:>22.3f}")
print()
print("  AUC = 0.5 表示完全分不开；1.0 表示完全可分。")
print()

a_res_s = auc(tables["stealthy"][0], tables["benign"][0])
a_jmp_s = auc(tables["stealthy"][1], tables["benign"][1])
a_res_r = auc(tables["random"][0], tables["benign"][0])

ok_spatial = a_res_r > 0.95 and a_res_s < 0.65
ok_temporal = a_jmp_s > 0.95
print(f"  空间统计量：抓住 random 且对 stealthy 失明   -> {ok_spatial}")
print(f"  时间统计量：抓住 stealthy                    -> {ok_temporal}")
print()
if ok_spatial and ok_temporal:
    print("  ✓ 难度梯度成立：")
    print("      random   -> 空间（残差）可检出")
    print("      stealthy -> 空间不可见，时间（状态跳变）可检出")
    print("    这正是论文需要的梯度，也是序列模型的用武之地。")
elif ok_temporal:
    print("  时间统计量有效，但空间统计量的行为不符合预期，需要查。")
else:
    print("  时间统计量未能分离 stealthy，需要调整 state_step_sigma 或攻击幅度。")
