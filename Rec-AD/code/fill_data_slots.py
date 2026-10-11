"""Fill the eight data.ieee*.rows / .size slots now that the column meaning is settled.

Advisor's instruction (2026-10-06): Table II's "Rows" column means the row count of
the embedding table as actually built -- "按本实现算出来是多少就是多少". These eight
slots were blocked on that question; they are unblocked now.

Values come from compute_table2_rows.py, which derives each cardinality from the
generated data file rather than from any assumption. The script reads that JSON
rather than hard-coding the numbers a second time, so the two cannot drift.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../Rec-AD/code
PROJ = HERE.parent                              # .../Rec-AD
rows = json.loads((HERE / "results" / "table2_rows.json").read_text(encoding="utf-8"))
by_case = {r["case"]: r for r in rows}

sp = PROJ / "slots.json"
d = json.loads(sp.read_text(encoding="utf-8"))

print("=" * 88)
print("填补 data.ieee*.rows / .size")
print("=" * 88)
print()

for case, r in by_case.items():
    short = case.replace("ieee", "ieee")
    rk, sk = f"data.{short}.rows", f"data.{short}.size"
    for key, val, unit, note in (
        (rk, r["rows"], "行",
         f"嵌入表行数 = 各稀疏特征基数之和（{'+'.join(str(c) for c in r['cards'])}）"
         f" = {r['rows']:,}。样本 {r['n_samples']:,}，量测 {r['n_measurements']}。"
         "按 DLRM 约定，行数是各特征基数之和，不是乘积。"),
        (sk, r["table_bytes"], "字节",
         f"表大小 = 行数 x dim(64) x 4B = {r['table_bytes']:,} B "
         f"= {r['table_bytes']/2**20:.3f} MiB。dim=64 由 tt.split{{d:3,n_per_core:4}} 推出。"),
    ):
        grp = d.setdefault("data", {})
        if key in grp:
            grp[key].update({"value": val, "unit": unit,
                             "status": "resolved", "note": note})
            print(f"  {key:<22} 更新为 {val:,} {unit}")
        else:
            grp[key] = {"meaning": f"{case} 的{'嵌入表行数' if 'rows' in key else '嵌入表大小'}",
                        "unit": unit, "reference": "—",
                        "source": "code/compute_table2_rows.py（从生成的 .npz 推导）",
                        "used_in": ["revision-sections.md"], "status": "resolved",
                        "value": val, "note": note}
            print(f"  {key:<22} 新增 {val:,} {unit}")

sp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print("  已写入 slots.json")
print()
print("=" * 88)
print("Table II 的 IEEE118-Bus 行应改为")
print("=" * 88)
print()
r = by_case["ieee118"]
print(f"  原：Dense 6 / Sparse 7 / Rows 19.53M / Dimension 16 / Size 1.22 GB")
print(f"  改：Dense 6 / Sparse 7 / Rows {r['rows']:,} / Dimension 64 / "
      f"Size {r['table_bytes']/2**20:.3f} MiB")
print()
print("  注：Dimension 也从 16 改为 64 —— Table II 原文写 16，而本实现的")
print("      dim 由 tt.split 推出为 4**3 = 64，且 §V.C 原文也写 '64 for T4'。")
