"""Pull the full context around every speedup claim, and the whole of Table III.

The first pass found three different numbers for what reads like the same
quantity. Before drawing a conclusion, read each claim in its own paragraph --
they may be speedups of different things (training vs inference vs a specific
stage), in which case there is no conflict and the fix is purely editorial.
"""
from __future__ import annotations

from pathlib import Path

import fitz

PDF = Path(__file__).resolve().parent.parent / "Rec-AD.pdf"
doc = fitz.open(PDF)
pages = [doc[i].get_text() for i in range(doc.page_count)]


def block(page_no: int, needle: str, before: int = 6, after: int = 14) -> None:
    lines = pages[page_no - 1].splitlines()
    for k, ln in enumerate(lines):
        if needle in ln:
            print(f"  ---- p{page_no}, 命中「{needle}」 ----")
            for l2 in lines[max(0, k - before) : k + after]:
                if l2.strip():
                    print(f"    {l2.strip()}")
            print()


print("=" * 92)
print("A. 3× 的上下文（p8）")
print("=" * 92)
block(8, "average speedup of 3", before=10, after=12)

print("=" * 92)
print("B. 2.44× 的上下文（p8）")
print("=" * 92)
block(8, "2.44", before=6, after=14)

print("=" * 92)
print("C. Table III 全表（p7）")
print("=" * 92)
lines = pages[6].splitlines()
for k, ln in enumerate(lines):
    if "TABLE III" in ln:
        for l2 in lines[k : k + 40]:
            if l2.strip():
                print(f"    {l2.strip()}")
        break

print()
print("=" * 92)
print("D. 数值核算")
print("=" * 92)
print()
print(f"  Table III 4GPU 列 Rec-AD = 0.62（归一化时间）-> 加速比 {1/0.62:.2f}x")
print(f"  Table III 1GPU 列 Rec-AD = 0.74             -> 加速比 {1/0.74:.2f}x")
print(f"  Table III CPU  列 Rec-AD = 0.82             -> 加速比 {1/0.82:.2f}x")
print()
print("  文中另两个自称同一量的数字：3x（§V.C 写 average）、2.44x（§V.E 写 average）")
