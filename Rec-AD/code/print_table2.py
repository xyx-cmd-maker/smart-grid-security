"""Print Table II verbatim, so the question about "how many rows" is unambiguous.

Everything so far about Table II came from a screenshot and a rendered image.
Before asking the advisor anything, quote the table exactly as it appears, and
show which column the question is about.
"""
from __future__ import annotations

from pathlib import Path

import fitz

PDF = Path(__file__).resolve().parent.parent / "Rec-AD.pdf"
doc = fitz.open(PDF)
pages = [doc[i].get_text() for i in range(doc.page_count)]

for pno, txt in enumerate(pages, 1):
    if "TABLE II" not in txt:
        continue
    lines = txt.splitlines()
    for k, ln in enumerate(lines):
        if "TABLE II" in ln:
            print("=" * 88)
            print(f"TABLE II 原文（PDF 第 {pno} 页）")
            print("=" * 88)
            for l2 in lines[k : k + 46]:
                if l2.strip():
                    print(f"  {l2.strip()}")
            break
    break

print()
print("=" * 88)
print("数字核算")
print("=" * 88)
print()
print("  Table II 的 IEEE118-Bus 行（据原文）：")
print("    Dense 6 / Sparse 7 / Dimension 16 / Rows 19.53M / Size 1.22 GB")
print()
print("  三种口径下 19.53M 的含义：")
print(f"    (a) 若 Rows 真是【表行数】      -> 表 = 19.53M x 16 x 4 B = {19.53e6*16*4/2**30:.2f} GB")
print(f"    (b) 若是【特征x维度x样本数】    -> 7 x 16 x 174375 = {7*16*174375:,}  (精确等于 19,530,000)")
print(f"    (c) 本参考实现按 DLRM 约定      -> 7 个特征的基数之和 = 799 行")
print()
print("  问老师的问题因此是：Table II 里那个 'Rows' 列，指的是 (a) 还是 (b)？")
print("  若是 (a)，真实表有多大？若是 (b)，表头写了 'Rows' 就是标错了列名。")
