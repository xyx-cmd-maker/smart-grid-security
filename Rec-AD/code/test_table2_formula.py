"""The advisor confirmed: '数据量' = number of training samples.

So the claimed derivation is:  embedding rows = sparse_features x dim x samples

This script tests that formula against all four rows of Table II. If it holds
only for the IEEE118 row, then that cell was computed by a method the other
three did not use -- which is the actual defect.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data_cache"

# Table II as printed (verified against Rec-AD.pdf page 7)
TABLE_II = [
    # name,            dense, sparse, rows,     dim, size_gb
    ("Avazu",              1,     20,   8.9e6,   16, 0.55),
    ("Criteo Terabyte",   13,     26, 242.5e6,   64, 59.2),
    ("Criteo Kaggle",     13,     26,  30.8e6,   16, 1.9),
    ("IEEE118-Bus",        6,      7,  19.53e6,  16, 1.22),
]

print("=" * 88)
print("A. 老师的公式：行数 = 稀疏特征 x 维度 x 样本数   —— 对四行分别反解样本数")
print("=" * 88)
print()
print(f"{'数据集':<18}{'稀疏':>5}{'维度':>5}{'Table II 行数':>16}{'反解样本数':>14}{'是整数?':>10}")
print("-" * 88)
implied = {}
for name, dense, sparse, rows, dim, size in TABLE_II:
    n = rows / (sparse * dim)
    implied[name] = n
    is_int = "是" if abs(n - round(n)) < 1e-6 else "否"
    print(f"{name:<18}{sparse:>5}{dim:>5}{rows:>16,.0f}{n:>14,.1f}{is_int:>10}")
print()
print("  反解出的『样本数』：")
for name, _, _, _, _, _ in TABLE_II:
    print(f"    {name:<18} {implied[name]:>12,.0f}")
print()
print("  真实规模参考（公开数据集样本数的量级）：")
print("    Avazu            ~4.0e7")
print("    Criteo Terabyte  ~4.4e9")
print("    Criteo Kaggle    ~4.6e7")
print()

print("=" * 88)
print("B. 结论：这个公式只对 IEEE118 那一行成立")
print("=" * 88)
print()
print("  若四行都用 `稀疏特征 x 维度 x 样本数`：")
print("    反解出的样本数是 "
      f"{', '.join(f'{implied[n]:,.0f}' for n, *_ in TABLE_II)}")
print("    四个数彼此无关，也都不等于各自数据集的真实样本数。")
print()
print("  而 IEEE118 反解出 174,375 —— 一个整数，量级也像真实训练集大小。")
print()
print("  => 前三行的『Rows』列另有来源（很可能是各特征基数之和，即真正的嵌入表行数）；")
print("     只有 IEEE118 那一行用了 `特征 x 维度 x 样本数`。")
print()

print("=" * 88)
print("C. `稀疏特征 x 维度 x 样本数` 到底算的是什么")
print("=" * 88)
print()
samples = implied["IEEE118-Bus"]
print(f"  代入 IEEE118：{7} x {16} x {samples:,.0f} = {7 * 16 * samples:,.0f}")
print()
print("  这个乘积 = 每个 epoch 里所有样本、所有稀疏特征的查表输出个数")
print("           = 前向传播产生多少个 16 维激活值")
print()
print("  它是【计算量】，随样本数线性增长。")
print("  它不是【存储量】——嵌入表的行数由『特征有多少种取值』决定，与样本数无关。")
print()

print("=" * 88)
print("D. IEEE118-Bus 那一行的正确数值")
print("=" * 88)
print()
with np.load(CACHE / "ieee118.npz", allow_pickle=False) as z:
    meta = json.loads(str(z["meta"]))
    sparse_arr = z["sparse"]
cols = meta["sparse_columns"]
cards = [int(sparse_arr[:, c].max()) + 1 for c in range(sparse_arr.shape[1])]
rows_true = sum(cards)
dim_true = 16
size_true_kb = rows_true * dim_true * 4 / 1024
n_samples = sparse_arr.shape[0]

print(f"  特征基数：")
for c, card in zip(cols, cards):
    print(f"      {c:<20} {card:>6,}")
print(f"      {'合计':<20} {rows_true:>6,}   <- 嵌入表行数")
print()
print(f"  {'':<24}{'Table II 现状':>18}{'正确值':>18}")
print("  " + "-" * 60)
print(f"  {'Emb. Table Rows':<24}{19.53e6:>18,.0f}{rows_true:>18,}")
print(f"  {'Emb. Table Size':<24}{'1.22 GB':>18}{f'{size_true_kb:.1f} KB':>18}")
print(f"  {'(本次样本数)':<24}{'—':>18}{n_samples:>18,}")
print()
print(f"  差 {19.53e6 / rows_true:,.0f} 倍。")
print()

print("=" * 88)
print("E. 对论文的影响")
print("=" * 88)
print()
print(f"  IEEE118 的嵌入表实际是 {rows_true:,} 行 / {size_true_kb:.0f} KB。")
print("  而论文第一条贡献是『让检测器装进变电站边缘节点的内存预算（8 GB）』。")
print()
print(f"  {size_true_kb:.0f} KB 的表，任何设备都装得下，不需要压缩。")
print()
print("  => 按当前特征设计，第一条贡献在 IEEE118 上不成立。")
print("     必须在『重新设计特征使表变大』和『把贡献改到别的轴上』之间选一条。")
