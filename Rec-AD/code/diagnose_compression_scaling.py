"""How does the TT compression ratio scale with the embedding table size?

This decides which journal is even viable. The paper's efficiency claim rests on
compression mattering; at the table size this project currently uses (799 rows)
the measured ratio is only 2.34x, which is not an edge-deployment problem.

If the ratio grows steeply with table size, then scaling the table -- not
changing the method -- is what makes the contribution real, and that is a
CPU-side data decision. If it stays near 2x, the efficiency story has no
headroom at any scale and the paper must be reframed.

Uses the shipped TTEmbeddingTable so the numbers are the implementation's, not a
formula's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from recad.tt_table import TTEmbeddingTable          # noqa: E402

DIM = 64
RANK = 32
D = 3                      # number of TT cores, from config tt.split.d

print("=" * 92)
print("压缩比 vs 嵌入表规模")
print("=" * 92)
print()
print(f"  dim={DIM}  rank={RANK}  cores d={D}   （与 config.yaml 一致）")
print()
print(f"{'表行数 N':>14}{'稠密 GB':>12}{'TT GB':>12}{'压缩比':>10}"
      f"{'实际参数比':>12}{'每核 n':>9}")
print("-" * 92)

rows = []
for N in [799, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]:
    torch.manual_seed(0)
    try:
        t = TTEmbeddingTable(num_rows=N, dim=DIM, rank=RANK, reuse=True)
    except Exception as exc:                      # shape search may refuse
        print(f"{N:>14,}  构建失败: {type(exc).__name__}: {exc}")
        continue

    dense_bytes = t.dense_bytes()
    tt_bytes = t.footprint_bytes()
    n_par = t.n_parameters()
    ratio = dense_bytes / tt_bytes
    par_ratio = (N * DIM) / n_par
    n_core = round(N ** (1.0 / D))

    rows.append({"N": N, "dense_bytes": dense_bytes, "tt_bytes": tt_bytes,
                 "ratio": ratio, "param_ratio": par_ratio, "n_per_core": n_core})
    print(f"{N:>14,}{dense_bytes/2**30:>12.4f}{tt_bytes/2**30:>12.4f}"
          f"{ratio:>10.1f}{par_ratio:>12.1f}{n_core:>9}")

print()
print("=" * 92)
print("判定")
print("=" * 92)
print()
if len(rows) >= 2:
    first, last = rows[0], rows[-1]
    print(f"  从 N={first['N']:,} 到 N={last['N']:,}（放大 {last['N']/first['N']:.0f} 倍）：")
    print(f"    压缩比 {first['ratio']:.1f}x  ->  {last['ratio']:.1f}x"
          f"（放大 {last['ratio']/first['ratio']:.0f} 倍）")
    print()
    # Theory: ratio ~ n^(d-1) * dim / (d * rank^2), so ratio ~ N^((d-1)/d)
    import math
    exp = math.log(last["ratio"] / first["ratio"]) / math.log(last["N"] / first["N"])
    print(f"  实测标度指数  ratio ~ N^{exp:.3f}")
    print(f"  理论预期      ratio ~ N^{{(d-1)/d}} = N^{{{(D-1)/D:.3f}}}")
    print()
    if exp > 0.3:
        print("  => 压缩比随表规模【强烈增长】。")
        print("     结论：让贡献成立的办法是【把表做大】，不是改方法。")
        print("     而这是纯 CPU 的数据侧决定 —— 现在就能做。")
    else:
        print("  => 压缩比对表规模不敏感，效率叙事在任何规模都没有空间。")
