"""Compare the two ablation runs cell by cell and report whether each cell reproduces exactly.

This is the controlled repeat the paper says was not completed. Both runs used the same data, the same seed, two epochs, the same
tier, and the ablation runner with the inter-op pool pinned; the only difference between them is the tag in the output filename.

Two outcomes matter and they say different things. If every cell matches bit for bit, the divergence the paper reports does not
reproduce under a pinned pool and a shortened schedule, which supports treating the earlier disagreement as unexplained rather than
as a thread-order artefact. If some cells differ, the divergence is reproducible and the paper's claim gains a demonstration.

The comparison prints absolute and relative differences so neither is hidden behind rounding.
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
    out: dict[str, float] = {}
    for r in recs:
        if not isinstance(r, dict):
            continue
        c = r.get("cell")
        a = r.get("auc")
        if c and a is not None:
            out.setdefault(c, []).append(float(a))
    return {k: statistics.fmean(v) for k, v in out.items()}


a = load("ablation_ieee118_local_det1.json")
b = load("ablation_ieee118_local_det2.json")

print(f"  run1 cells: {len(a)}   run2 cells: {len(b)}")
print()
print(f"  {'cell':<20}{'run1':>12}{'run2':>12}{'abs diff':>12}{'same?':>8}")
identical = differing = 0
for cell in sorted(set(a) | set(b)):
    va, vb = a.get(cell), b.get(cell)
    if va is None or vb is None:
        print(f"  {cell:<20}{'—':>12}{'—':>12}{'—':>12}{'missing':>8}")
        continue
    diff = abs(va - vb)
    same = diff == 0.0
    if same:
        identical += 1
    else:
        differing += 1
    print(f"  {cell:<20}{va:>12.6f}{vb:>12.6f}{diff:>12.2e}{'EXACT' if same else 'DIFFERS':>8}")

print()
print(f"  identical cells: {identical} / {identical + differing}")
print(f"  differing cells: {differing}")
if differing == 0:
    print("  verdict: every cell reproduces exactly under the pinned pool and the shortened schedule")
else:
    print("  verdict: at least one cell does not reproduce; the divergence is reproducible")
