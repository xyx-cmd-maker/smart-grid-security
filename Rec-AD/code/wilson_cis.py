"""Compute Wilson score intervals for every detection rate that has a stored confusion matrix.

The review asks for interval estimates on the headline TPR/FPR numbers and allows "a suitable binomial interval" as one
acceptable route. The result records carry tp, fp, fn and tn plus the per-family sample counts, so a binomial interval can
be computed for every existing cell without retraining anything.

Wilson is used rather than the normal approximation because several rates are near 0 or 1, where the normal approximation
produces intervals that extend outside [0, 1] and shrink to zero width at the boundary. Wilson stays inside the range and
does not collapse.

What this cannot do: an interval on AUROC. That needs per-sample scores, and no prediction files were saved even though the
configuration asks for them. This script reports that limitation rather than papering over it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "code" / "results"
Z = 1.959963984540054  # two-sided 95%


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for k successes out of n trials."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


FILES = [
    "baselines_ieee118_t4_fam118_ae_on.json",
    "baselines_ieee14_t4_s14_ae_on.json",
    "baselines_ieee57_t4_s57_ae_on.json",
]

print("  === stored confusion matrices and Wilson 95% intervals ===")
print(f"  {'file':<44} {'model':<14} {'k/n':>12}  {'rate':>7}  {'95% CI':>18}")
found = 0
for name in FILES:
    p = RES / name
    if not p.exists():
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    for r in (d.get("records") or []):
        if not isinstance(r, dict):
            continue
        tp, fp, fn, tn = r.get("tp"), r.get("fp"), r.get("fn"), r.get("tn")
        if not all(isinstance(v, (int, float)) for v in (tp, fp, fn, tn)):
            continue
        n_pos = int(tp + fn)
        if n_pos <= 0:
            continue
        k = int(tp)
        lo, hi = wilson(k, n_pos)
        rate = k / n_pos
        print(f"  {name[:44]:<44} {str(r.get('model'))[:14]:<14} {k:>5}/{n_pos:<6}  {rate:>7.4f}  "
              f"[{lo:.4f}, {hi:.4f}]")
        found += 1

print()
print(f"  cells with a computable interval: {found}")

print()
print("  === why half-widths differ across cells ===")
print("    k/n and n both matter: at n near 780 a rate near 0.01 has a half-width near 0.007,")
print("    while at the same n a rate near 0.5 has a half-width near 0.035. The interval width is")
print("    therefore not a property of the detector but of where its rate sits and how many samples")
print("    that family contributes. That is the point the review makes about four-decimal reporting.")

print()
print("  === the limit that cannot be fixed without saved scores ===")
print("    AUROC intervals require per-sample scores. config.yaml sets save_predictions: true, but no")
print("    prediction file exists under code/results, so no AUROC interval can be computed from the")
print("    record. AUROC therefore keeps its seed standard deviation only, and the manuscript must say")
print("    so rather than implying an interval it does not have.")
