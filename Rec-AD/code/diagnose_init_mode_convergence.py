"""Do the two init modes converge as the table grows?

`init_mode="svd"` gives cores (1,18,18)/(18,32,32)/(32,96,1) = 21,828 params at
799 rows, while `init_mode="direct"` gives 4-D cores totalling 36,416. The
compression ratios therefore differ (2.34x vs 1.40x) at that size.

Hypothesis: the SVD truncates to min(rank, actual matricisation rank), and at
799 rows the actual rank is below 32, so the SVD path gets smaller cores for
free. At the scale the paper needs (16.7M rows) the first matricisation is
1024 x 1,048,576, whose rank far exceeds 32, so truncation binds and both paths
should agree.

If that is right, the direct path is not worse -- it is just not flattered by a
small table. If the ratios stay apart, switching modes would silently change the
paper's headline compression number and must not be done casually.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from recad.tt_table import TTEmbeddingTable          # noqa: E402

DIM, RANK = 64, 32

print("=" * 96)
print("两种初始化模式的核形状与压缩比随表规模的变化")
print("=" * 96)
print()
print(f"{'N':>12}{'模式':>9}{'参数':>12}{'压缩比':>10}{'中间核的秩':>13}   核形状")
print("-" * 96)

for N in (799, 10_000, 100_000, 1_000_000):
    for mode in ("svd", "direct"):
        torch.manual_seed(0)
        try:
            t = TTEmbeddingTable(num_rows=N, dim=DIM, rank=RANK, init_mode=mode)
        except RuntimeError as exc:
            print(f"{N:>12,}{mode:>9}   分配失败: {str(exc)[:44]}")
            continue
        shapes = [tuple(c.shape) for c in t.cores]
        mid = max(shapes, key=lambda s: 1)
        # The rank actually used at the middle bond, whatever the layout.
        mid_rank = max(shapes[1]) if len(shapes) > 1 else 0
        print(f"{N:>12,}{mode:>9}{t.n_parameters():>12,}"
              f"{t.compression_ratio():>10.3f}{mid_rank:>13}   {shapes}")
    print()

print("=" * 96)
print("判定")
print("=" * 96)
print()
print("  若两个模式的压缩比随 N 增大而收敛 -> 直接初始化不是变差，")
print("     只是小表上 SVD 的秩截断未绑定、白得了便宜。")
print("  若始终相差约 1.67x          -> 换模式会改变论文的压缩比数字，")
print("     必须显式说明并统一口径，不能默默切换。")
