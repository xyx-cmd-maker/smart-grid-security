"""Compare the two 57-bus ablation runs, which is the dataset where the paper reports a disagreement.

The paper's reproducibility paragraph names 57 buses as the place where the same cell gave two different values, 0.4898 and 0.5079.
Two runs at that size, same seed, same shortened budget, both thread pools pinned, differ only in the output tag. If they agree, the
named disagreement does not reproduce at this budget; if they disagree, it does, and the paper's account gains a demonstration.

The comparison prints every cell, the absolute difference, and a verdict per cell, so a single matching cell cannot stand in for the
whole table.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "code" / "results"


def load(name: str) -> dict[str, float]:
    p = RES / name
    if not p.exists():
        raise SystemExit(f"  missing: {name}")
    d = json.loads(p.read_text(encoding="utf-8"))
    recs = d.get("records") or d
    tmp: dict[str, list[float]] = {}
    for r in recs:
        if isinstance(r, dict) and r.get("cell") and r.get("auc") is not None:
            tmp.setdefault(r["cell"], []).append(float(r["auc"]))
    return {k: statistics.fmean(v) for k, v in tmp.items()}


a = load("ablation_ieee57_local_s57a.json")
b = load("ablation_ieee57_local_s57b.json")

print(f"  run a cells: {len(a)}   run b cells: {len(b)}")
print()
print(f"  {'cell':<20}{'run a':>12}{'run b':>12}{'abs diff':>12}{'verdict':>10}")
same = diff = 0
for cell in sorted(set(a) | set(b)):
    va, vb = a.get(cell), b.get(cell)
    if va is None or vb is None:
        print(f"  {cell:<20}{'—':>12}{'—':>12}{'—':>12}{'missing':>10}")
        continue
    d = abs(va - vb)
    ok = d == 0.0
    same += ok
    diff += (not ok)
    print(f"  {cell:<20}{va:>12.6f}{vb:>12.6f}{d:>12.2e}{'EXACT' if ok else 'DIFFERS':>10}")

print()
print(f"  exact: {same}   differing: {diff}")
print()
print("  === context: what the paper names ===")
print("  paper reports 0.4898 and 0.5079 for the same cell at 57 buses")
for k, v in sorted(a.items()):
    if abs(v - 0.5079) < 5e-4:
        print(f"    run a {k} = {v:.6f}  (matches 0.5079)")
    if abs(v - 0.4898) < 5e-4:
        print(f"    run a {k} = {v:.6f}  (matches 0.4898)")
for k, v in sorted(b.items()):
    if abs(v - 0.4898) < 5e-4:
        print(f"    run b {k} = {v:.6f}  (matches 0.4898)")
