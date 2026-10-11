"""Why does RandomForest miss random attacks but catch stealthy ones?

Measures the per-family attack magnitudes and checks whether the three families
are separable by a trivial statistic. If the families differ hugely in
magnitude, a tree model will key on magnitude rather than on the residual
signature, and the "difficulty gradient" the threat model assumes will not hold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import ATTACK_NORMAL, ATTACK_RANDOM, ATTACK_STEALTHY, ATTACK_LOAD_REDIST  # noqa: E402

P = HERE / "data_cache/stage1.npz"
with np.load(P, allow_pickle=False) as z:
    dense = z["dense"]
    at = z["attack_type"]
    h = z["h_matrix"]
    meta = json.loads(str(z["meta"]))

names = {ATTACK_NORMAL: "benign", ATTACK_RANDOM: "random",
         ATTACK_STEALTHY: "stealthy", ATTACK_LOAD_REDIST: "load_redist"}

print("=" * 80)
print("1. 各族的样本数与量测维度")
print("=" * 80)
print()
m = dense.shape[1]
n_state = h.shape[1]
print(f"  量测维度 m = {m},  状态维度 n = {n_state}")
print(f"  噪声 sigma = {meta['noise_sigma']},  攻击 scale = {meta['attack_scale']}")
print()
for code, nm in names.items():
    k = int((at == code).sum())
    print(f"    {nm:<14} {k:>8,}")
print()

print("=" * 80)
print("2. 各族的量测偏移幅度")
print("=" * 80)
print()
print("  注意：这里量的是『攻击向量本身』的幅度。")
print("  干净数据的量测分布有它自己的尺度，两者要一起看才有意义。")
print()
print(f"  {'家族':<14}{'|a| 均值':>12}{'|a| 中位':>12}{'|a| 最大':>12}{'信噪比':>10}")
print("-" * 80)

# Reconstruct the clean reference: benign samples are clean by construction
clean = dense[at == ATTACK_NORMAL]
scale_clean = float(np.abs(clean).mean())
noise = float(meta["noise_sigma"])

for code, nm in names.items():
    if code == ATTACK_NORMAL:
        continue
    sub = dense[at == code]
    # attack magnitude = distance from the clean mean, which the benign
    # samples estimate
    shift = np.abs(sub - clean.mean(axis=0))
    print(f"  {nm:<14}{shift.mean():>12.3f}{np.median(shift):>12.3f}"
          f"{shift.max():>12.3f}{shift.mean() / noise:>10.1f}x")
print()
print(f"  参考：干净数据的平均 |z| = {scale_clean:.3f}，噪声 sigma = {noise}")
print()

print("=" * 80)
print("3. 一个平凡统计量能不能分开三族")
print("=" * 80)
print()
print("  用『偏离干净均值的范数』这个单一统计量，看各族的分位数：")
print()
center = clean.mean(axis=0)
print(f"  {'家族':<14}{'10%':>10}{'50%':>10}{'90%':>10}")
print("-" * 80)
stats = {}
for code, nm in names.items():
    d = np.linalg.norm(dense[at == code] - center, axis=1)
    stats[nm] = d
    print(f"  {nm:<14}{np.quantile(d, .1):>10.2f}{np.quantile(d, .5):>10.2f}"
          f"{np.quantile(d, .9):>10.2f}")
print()

r, s = stats["random"], stats["stealthy"]
print(f"  random 中位数 / stealthy 中位数 = {np.median(r) / np.median(s):.2f}")
print()

print("=" * 80)
print("4. 结论")
print("=" * 80)
print()
if np.median(s) > 5 * np.median(r):
    print("  隐蔽攻击的偏移范数远大于随机攻击。")
    print("  这意味着：一个只看『偏离多远』的树模型会把 stealthy 判为攻击、")
    print("  把 random 判为正常 —— 正好与论文需要的难度梯度相反。")
    print()
    print("  => 三族的难度梯度没有建立起来。random 的幅度需要大幅提高，")
    print("     或者 stealthy 的幅度需要降低到与 random 可比。")
else:
    print("  两族幅度可比，RF 的反常不能只用幅度解释，需要继续查。")
