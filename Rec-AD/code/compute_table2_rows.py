"""Table II's "Rows" column, computed from how the embedding table is actually built.

Advisor's instruction (2026-10-06): "表上意思是转换为嵌入表后的行数，你要看嵌入表
你是怎么生成的，你根据 IEEE 118 总线数据集实际的行数转换为嵌入表后看看是多少就是多少."

So the number is not to be reverse-engineered from 19.53M. It is whatever this
implementation produces. This script derives it from first principles, prints the
full derivation rather than only the total, and does it for all four IEEE cases so
the table can be filled consistently.

The DLRM convention used here: each sparse feature owns a contiguous slice of one
shared table, so the table's row count is the SUM of the per-feature cardinalities
(not their product), and a feature's cardinality is one more than the largest id it
can take.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
CASES = ["ieee14", "ieee57", "ieee118", "ieee300"]

print("=" * 96)
print("嵌入表行数的推导（按 DLRM 约定：行数 = 各稀疏特征基数之和）")
print("=" * 96)
print()

rows = []
for case in CASES:
    p = CACHE / f"{case}.npz"
    if not p.exists():
        print(f"  {case}: 文件不存在，跳过")
        continue
    with np.load(p, allow_pickle=False) as z:
        sparse = z["sparse"]
        dense = z["dense"]
        meta = json.loads(str(z["meta"])) if "meta" in z else {}

    cols = meta.get("sparse_columns") or [f"col{i}" for i in range(sparse.shape[1])]
    cards = [int(sparse[:, c].max()) + 1 for c in range(sparse.shape[1])]
    total = int(sum(cards))
    table_bytes = total * 64 * 4          # dim=64 (tt.split -> 4**3), fp32

    print(f"  ── {case} ──")
    print(f"     样本数 {sparse.shape[0]:,}   量测数 {dense.shape[1]}")
    print(f"     {'特征':<22}{'基数':>12}{'观测到的最大 id':>18}")
    for name, c in zip(cols, cards):
        print(f"     {name:<22}{c:>12,}{c - 1:>18,}")
    print(f"     {'合计（= 表行数）':<22}{total:>12,}")
    print(f"     表大小（dim=64, fp32）  {table_bytes:,} B = "
          f"{table_bytes/2**20:.3f} MiB")
    print()
    rows.append({"case": case, "n_samples": int(sparse.shape[0]),
                 "n_measurements": int(dense.shape[1]),
                 "cards": cards, "cols": cols, "rows": total,
                 "table_bytes": table_bytes})

print("=" * 96)
print("汇总")
print("=" * 96)
print()
print(f"  {'算例':<12}{'样本':>10}{'量测':>8}{'表行数':>12}{'表大小(MiB)':>14}")
print("  " + "-" * 60)
for r in rows:
    print(f"  {r['case']:<12}{r['n_samples']:>10,}{r['n_measurements']:>8}"
          f"{r['rows']:>12,}{r['table_bytes']/2**20:>14.3f}")
print()

# ---- comparison with the claimed figure -----------------------------------
if any(r["case"] == "ieee118" for r in rows):
    n118 = next(r for r in rows if r["case"] == "ieee118")
    claimed = 19_530_000
    print("=" * 96)
    print("与稿件 Table II 声称值的对比")
    print("=" * 96)
    print()
    print(f"  Table II 声称（IEEE118-Bus）  {claimed:,} 行 / 1.22 GB")
    print(f"  本实现（DLRM 约定）           {n118['rows']:,} 行 / "
          f"{n118['table_bytes']/2**20:.3f} MiB")
    print(f"  相差                          {claimed / n118['rows']:,.0f} 倍")
    print()
    print(f"  而 7 x 16 x {n118['n_samples']:,} = {7*16*n118['n_samples']:,}"
          f"  <- 恰等于声称值（查表量口径）")
    print()
    print("  老师指示：按本实现算出来是多少就是多少 -> "
          f"采用 {n118['rows']:,} 行")
print()

out = ROOT / "results" / "table2_rows.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  明细 -> {out.relative_to(ROOT)}")
