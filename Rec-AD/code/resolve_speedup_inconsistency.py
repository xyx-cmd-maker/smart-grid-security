"""Resolve the 1.61x vs 2.44x inconsistency from the source, not from a screenshot.

Evidence so far came from a screenshot of Table III and a text-extracted copy of
section V.E. Before deciding whether the two numbers conflict, get both from the
PDF itself and find every other speedup claim in the paper, so the comparison is
against the full set rather than against one sentence.

A conflict here matters more than a normal numeric slip: it sits on the
boundary between "technical criticism" and "integrity criticism", and it does
not move when the paper's framing changes.
"""
from __future__ import annotations

import re
from pathlib import Path

import fitz

PDF = Path(__file__).resolve().parent.parent / "Rec-AD.pdf"

doc = fitz.open(PDF)
pages = [doc[i].get_text() for i in range(doc.page_count)]
full = "\n".join(pages)

print("=" * 92)
print(f"PDF: {PDF.name}   {doc.page_count} 页")
print("=" * 92)
print()

# ---- 1. every speedup / time-ratio claim ---------------------------------
print("1. 全文所有加速比与时延比声称")
print("-" * 92)
pat = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*[x×]\s*(?:speedup|faster|reduction)?", re.I)
seen = set()
for pno, txt in enumerate(pages, 1):
    for m in re.finditer(r"[^.\n]*?(?:speedup|faster|slowdown|reduce[sd]?)[^.\n]*?[0-9][^.\n]*", txt, re.I):
        s = " ".join(m.group(0).split())
        if len(s) < 25 or s in seen:
            continue
        seen.add(s)
        print(f"  [p{pno}] {s[:150]}")
print()

# ---- 2. the Table III region --------------------------------------------
print("2. Table III 所在的页与上下文")
print("-" * 92)
for pno, txt in enumerate(pages, 1):
    if re.search(r"TABLE III", txt, re.I):
        lines = txt.splitlines()
        for k, ln in enumerate(lines):
            if re.search(r"TABLE III", ln, re.I):
                print(f"  --- p{pno} ---")
                for l2 in lines[k : k + 22]:
                    if l2.strip():
                        print(f"    {l2.strip()}")
                print()
        break
print()

# ---- 3. the Fig. 10 / pipeline speedup region ---------------------------
print("3. §V.E 与 Fig. 10 的数字")
print("-" * 92)
for pno, txt in enumerate(pages, 1):
    if re.search(r"2\.44|1\.30", txt):
        print(f"  --- p{pno} ---")
        for ln in txt.splitlines():
            if re.search(r"2\.44|1\.30|speedup|Sequential|Pipeline", ln):
                print(f"    {ln.strip()}")
        print()

# ---- 4. any other table with a training-time column ---------------------
print("4. 其他含训练时间列的表格")
print("-" * 92)
for pno, txt in enumerate(pages, 1):
    if re.search(r"Training\s*time|training time", txt, re.I):
        head = [l.strip() for l in txt.splitlines()
                if re.search(r"TABLE [IVX]+", l, re.I)]
        print(f"  p{pno}: {head if head else '(该页无表号)'}")
