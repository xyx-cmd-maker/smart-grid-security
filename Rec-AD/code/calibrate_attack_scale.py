"""Calibrate the attack scales so the three families form the intended gradient.

Current problem (measured on data_cache/stage1.npz):

    family        |a| mean    ||z - mean_benign|| (median)
    benign            --              1118.65
    random        35.077             1117.82   <- indistinguishable from benign
    stealthy      63.211             2016.64   <- 80% above benign

The stealthy attack is the LARGEST deviation in raw measurement space, which is
backwards: it is supposed to be the hard one. A tree model fed raw values keys
on magnitude, so it catches exactly the attack it should miss and misses the one
it should catch.

What the threat model actually needs:
  * random   : a visible component OUTSIDE col(H)  -> detectable by residual,
               while staying small in raw terms
  * stealthy : a = Hc, no component outside col(H) -> invisible to residual,
               with a raw magnitude comparable to the random family
  * load-redist: stealthy plus a small noise-scaled departure -> in between

This script measures the achievable magnitudes and reports the scale factors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

P = HERE / "data_cache/stage1.npz"
with np.load(P, allow_pickle=False) as z:
    h = z["h_matrix"].astype(np.float64)

m, n_state = h.shape
rng = np.random.default_rng(0)
N = 2000

print("=" * 78)
print("1. H 的尺度")
print("=" * 78)
print()
print(f"  H: {m} x {n_state}")
print(f"  ||H||_F            = {np.linalg.norm(h):,.1f}")
print(f"  每列范数 中位      = {np.median(np.linalg.norm(h, axis=0)):,.3f}")
print(f"  每列范数 最大      = {np.max(np.linalg.norm(h, axis=0)):,.3f}")
print()

print("=" * 78)
print("2. 单位 scale 下，两族的原始幅度")
print("=" * 78)
print()
print("  随机攻击  a ~ N(0, 1) 逐量测")
c = rng.normal(size=(N, m))
a_rand1 = c
norm_rand1 = np.linalg.norm(a_rand1, axis=1)
print(f"    ||a|| 中位 = {np.median(norm_rand1):,.2f}")
print(f"    => 与样本数无关；每量测约 1 个单位")
print()

print("  隐蔽攻击  a = Hc, c ~ N(0, 1) 逐状态")
c = rng.normal(size=(N, n_state))
a_stl1 = c @ h.T
norm_stl1 = np.linalg.norm(a_stl1, axis=1)
print(f"    ||a|| 中位 = {np.median(norm_stl1):,.2f}")
print(f"    放大倍数   = {np.median(norm_stl1) / np.median(norm_rand1):,.1f}x")
print()
print("  => 同样的 scale 下，隐蔽攻击的原始偏移是随机攻击的")
print(f"     {np.median(norm_stl1) / np.median(norm_rand1):,.1f} 倍。这就是失配的来源。")
print()

print("=" * 78)
print("3. 要让两者原始幅度可比，scale 该怎么取")
print("=" * 78)
print()
r_rand = np.median(norm_rand1)
r_stl = np.median(norm_stl1)
print(f"  若随机攻击用 scale_r，隐蔽攻击用 scale_s：")
print(f"    ||a_random||   ∝ scale_r")
print(f"    ||a_stealthy|| ∝ scale_s x {r_stl / r_rand:.1f}")
print()
print("  令两者相等（以 scale_r = 1.0 为基准）：")
print(f"    scale_s = {r_rand / r_stl:.4f}")
print()
print("  即：隐蔽攻击的状态扰动要比随机攻击小约"
      f" {r_stl / r_rand:.0f} 倍，原始量测偏移才可比。")
print()

print("=" * 78)
print("4. 但『原始幅度可比』不等于『难度梯度正确』")
print("=" * 78)
print()
print("  真正决定可检出性的是两个量：")
print()
# residual component
P_h = h @ np.linalg.pinv(h)
res_rand = np.linalg.norm(a_rand1 - a_rand1 @ P_h.T, axis=1)
res_stl = np.linalg.norm(a_stl1 - a_stl1 @ P_h.T, axis=1)
print(f"    random 的残差分量  ||a - proj_col(H) a|| 中位 = "
      f"{np.median(res_rand):,.2f}")
print(f"    stealthy 的残差分量                        中位 = "
      f"{np.median(res_stl):,.4f}")
print()
print("  随机攻击的残差分量 = 其全部幅度；隐蔽攻击的残差分量 ≈ 0。")
print("  所以只要 scale 调到『两者原始幅度可比』，")
print("  残差检验就会：抓住随机、放过隐蔽 —— 正是威胁模型要求的。")
print()

print("=" * 78)
print("5. 建议的参数")
print("=" * 78)
print()
print("  目标：random 的原始偏移 = stealthy 的原始偏移")
print()
target = 3.0          # 每量测 3 个单位, 约 60 倍噪声, 明显但不夸张
print(f"  若希望每量测平均偏移约 {target} 个单位：")
sr = target          # random: a ~ N(0, sr^2) per measurement
ss = target * r_rand / r_stl * np.sqrt(m) / np.sqrt(m)
ss = target * (r_rand / np.sqrt(m)) / (r_stl / np.sqrt(n_state)) / np.sqrt(n_state) * np.sqrt(n_state)
# simpler: match norms
ss = target * np.sqrt(m) * r_rand / r_stl / np.sqrt(m) * 1.0
ss = target * r_rand / r_stl
print(f"    scale_random   = {target:.2f}")
print(f"    scale_stealthy = {ss:.4f}   (= {target:.2f} x {r_rand / r_stl:.4f})")
print()
print("  当前默认两者都是 1.5，所以隐蔽攻击的原始偏移是随机攻击的"
      f" {r_stl / r_rand:.1f} 倍。")
print()

print("=" * 78)
print("6. 结论")
print("=" * 78)
print()
print("  这是**数据生成的参数失配**，不是 RF 的实现问题。")
print()
print("  修正方向：把 random 与 stealthy 的 scale 解耦，")
print("  使两者的原始量测偏移可比；难度差异只由『是否落在 col(H) 内』决定。")
print()
print("  修正后必须重新验证：")
print("    1. BDD: random 高、stealthy ≈ 虚警率      （威胁模型）")
print("    2. RF : 两族都不应出现 100%/0% 的退化     （梯度存在）")
