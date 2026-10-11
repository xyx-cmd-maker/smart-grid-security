"""Verify every link in the briefing resolves, and check the stray nested directory.

Two things this confirms:
  1. every markdown link in the briefing points at a file that exists, resolved from
     the briefing's OWN directory (the mistake that broke the first version)
  2. what is actually inside `Rec-AD/smart-grid-security/`, a directory whose name
     duplicates the parent topic -- it showed up while locating the slots files and
     is not something this conversation created on purpose
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "\u4e0b\u5348\u6c47\u62a5\u63d0\u7eb2.md"

text = DOC.read_text(encoding="utf-8")
links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)

print("=" * 78)
print(f"links in {DOC.name}")
print("=" * 78)
bad = []
for label, target in links:
    if target.startswith(("http://", "https://", "#")):
        print(f"  SKIP  {label}")
        continue
    resolved = (DOC.parent / target).resolve()
    if resolved.exists():
        kind = "dir " if resolved.is_dir() else f"{resolved.stat().st_size / 1024:6.1f}K"
        print(f"  OK    {kind}  {target}")
    else:
        print(f"  MISS         {target}")
        bad.append(target)

print()
print(f"  {len(links)} links, {len(bad)} broken")

# files the briefing names in prose but deliberately does not link
prose = ["_backup_baselines_20261008_105531", "data_cache/ieee118.npz"]
print()
print("  named in prose, intentionally not linked:")
for p in prose:
    full = (DOC.parent / p).resolve()
    print(f"    {'exists' if full.exists() else 'MISSING'}  {p}")

print()
print("=" * 78)
print("stray directory check")
print("=" * 78)
stray = ROOT / "smart-grid-security"
if stray.exists():
    items = sorted(stray.rglob("*"))
    print(f"  {stray}")
    print(f"  {len(items)} entries")
    for it in items[:30]:
        rel = it.relative_to(stray)
        depth = len(rel.parts) - 1
        print(f"    {'  ' * depth}{rel.parts[-1]}{'/' if it.is_dir() else ''}")
    if len(items) > 30:
        print(f"    ... and {len(items) - 30} more")
else:
    print("  not present")

raise SystemExit(1 if bad else 0)
