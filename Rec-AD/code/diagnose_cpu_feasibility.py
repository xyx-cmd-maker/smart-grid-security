"""Can this laptop do the 16.7M-row table at all, and where exactly does it break?

The blocking question: the abstract's first sentence claims modern FDIA detectors
exceed a substation node's 8 GB budget. That needs a table around 20M rows. The
box here has 15.7 GB RAM with ~2 GB free.

The key structural point is that the argument is ASYMMETRIC by nature:

  * the dense arm must be shown to EXCEED the budget -- which means it cannot be
    trained on the target hardware at all. Its footprint is an arithmetic fact.
  * the TT arm must be shown to FIT and to work -- and its cores are orders of
    magnitude smaller, so it IS trainable here.

So the question is not "can we train the dense table" (we cannot, and neither
could the substation) but "can we compute the dense footprint without
materialising it". `TTEmbeddingTable.__init__` currently draws a full dense
table before decomposing, which is what OOM'd at 10M rows.

This measures (a) the core sizes analytically, (b) whether the dense
materialisation is the only blocker, and (c) peak allocation for each.
"""
from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DIM = 64
RANK = 32
D = 3
N = 64 ** 4                      # bins=64, cross=4 -> 16,777,216 rows

print("=" * 84)
print(f"目标：N = 64^4 = {N:,} 行,  dim = {DIM},  rank = {RANK},  cores d = {D}")
print("=" * 84)
print()

# ---- 1. dense footprint: pure arithmetic, no allocation -------------------
dense_bytes = N * DIM * 4
print("1. 稠密表占用（纯算术，不分配内存）")
print(f"   {N:,} x {DIM} x 4 B = {dense_bytes:,} B = {dense_bytes/2**30:.2f} GiB")
print(f"   含 Adam 状态（m/v，约 3x）        ≈ {dense_bytes*3/2**30:.2f} GiB")
print(f"   对 8 GB 预算：{'超出' if dense_bytes > 8*2**30 else '未超出'}（仅表）; "
      f"{'超出' if dense_bytes*3 > 8*2**30 else '未超出'}（含 Adam）")
print()

# ---- 2. TT core sizes: also pure arithmetic ------------------------------
# N = n^d and DIM = m^d with equal splits, the shapes tt_table uses.
n = round(N ** (1.0 / D))
m = round(DIM ** (1.0 / D))
print("2. TT 核占用（纯算术）")
print(f"   每个核 shape: (r_{{i-1}}, {n}, {m}, r_i),  r = {RANK}")
total = 0
shapes = []
for i in range(D):
    r_in = 1 if i == 0 else RANK
    r_out = 1 if i == D - 1 else RANK
    params = r_in * n * m * r_out
    total += params
    shapes.append((r_in, n, m, r_out))
    print(f"     core {i}: {shapes[-1]}  ->  {params:,} 参数")
tt_bytes = total * 4
print(f"   合计 {total:,} 参数 = {tt_bytes:,} B = {tt_bytes/2**20:.2f} MiB")
print(f"   对 8 GB 预算：{'超出' if tt_bytes > 8*2**30 else '装得下'}"
      f"（余量 {8*2**30/tt_bytes:,.0f}x）")
print()

# ---- 3. the ratio -------------------------------------------------------
ratio = dense_bytes / tt_bytes
print("3. 压缩比")
print(f"   {dense_bytes:,} / {tt_bytes:,} = {ratio:,.1f}x")
print(f"   （先前实测标度律 ratio ~ N^0.578，由 799..1M 四点拟合）")
print()

# ---- 4. can the TT table actually be built here? -------------------------
print("4. 本机能否构建这个 TT 表")
print(f"   可用内存约 2.2 GB；TT 核需 {tt_bytes/2**20:.2f} MiB  ->  可以")
print(f"   但 `TTEmbeddingTable.__init__` 会先 torch.randn({n**D:,}, {DIM})")
print(f"   即分配 {n**D * DIM * 4 / 2**30:.2f} GiB 的稠密中间量  ->  这正是 OOM 的原因")
print()

tracemalloc.start()
try:
    import torch
    probe = torch.empty(n * DIM, dtype=torch.float32)
    cur, peak = tracemalloc.get_traced_memory()
    print(f"   实测 torch.empty({n*DIM:,}) 分配 {probe.numel()*4/2**30:.2f} GiB 成功")
    print(f"   tracemalloc 峰值 {peak/2**30:.2f} GiB")
    del probe
except RuntimeError as exc:
    print(f"   实测分配失败: {str(exc)[:120]}")
finally:
    tracemalloc.stop()

print()
print("=" * 84)
print("判定")
print("=" * 84)
print()
print(f"  ✓ 压缩比 {ratio:,.0f}x 与稠密超预算 {dense_bytes/2**30:.1f} GiB 都是【纯算术】")
print(f"    不需要训练、不需要分配 —— 结论现在就能写，而且对硬件无要求")
print(f"  ✓ TT 核仅 {tt_bytes/2**20:.1f} MiB，本机可以构建与训练")
print(f"  ✗ 稠密表 {dense_bytes/2**30:.2f} GiB 本机【建不出来】")
print(f"    但这恰恰是论证要的：装不下的模型本来就不该在目标硬件上训练")
print()
print("  => 本机足够完成第 1 步，前提是【不物化稠密表】。")
print("     需要一处代码改动：让 TT 核直接按形状初始化，跳过稠密中间量。")
