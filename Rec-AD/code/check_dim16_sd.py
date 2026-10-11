"""Does the dim=16 CNN run have a standard deviation, or only a single point?

This matters for how the C4 table may phrase the comparison. If the dim=16 side has
only a mean (or only one seed), then "the two dimensions agree" is unsupported --
what the data actually shows is that the dim=64 result sits at chance, consistent
with the earlier observation at dim=16. Those are different claims, and only the
second is defensible.

Prints the raw per-seed records for both sides so the answer comes from the files
rather than from a summary label.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIM16 = ["results/baselines_ieee118_local_iso2_*.json",
         "results/baselines_ieee118_local_iso_*.json"]


def show(pattern_glob: str, label: str) -> None:
    files = sorted(f for f in glob.glob(str(HERE / pattern_glob)) if "CNN" in f)
    if not files:
        print(f"  {label}: no CNN file matching {pattern_glob}")
        return
    for f in files:
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception as exc:                       # noqa: BLE001
            print(f"  {f}: unreadable ({exc})")
            continue
        name = Path(f).name
        if isinstance(d, dict) and "records" in d:
            recs = d["records"]
            print(f"  {name}  -> {len(recs)} per-seed record(s)")
            for r in recs:
                print(f"      seed={r.get('seed')}  auc={r.get('auc')}"
                      f"  stealthy={r.get('detect_rate_stealthy')}"
                      f"  fpr={r.get('detect_rate_normal_fp')}")
            if len(recs) > 1:
                a = np.array([r.get("auc", np.nan) for r in recs], dtype=float)
                s = np.array([r.get("detect_rate_stealthy", np.nan) for r in recs], dtype=float)
                print(f"      mean auc {a.mean():.4f}  sd {a.std(ddof=1):.4f}")
                print(f"      mean stealthy {s.mean():.4f}  sd {s.std(ddof=1):.4f}")
            else:
                print("      -> SINGLE SEED: no standard deviation available")
        else:
            # a summary file: show the aggregate keys only
            keys = [k for k in d if "auc" in k or "stealthy" in k][:8]
            print(f"  {name}  (aggregate) keys: {keys}")
            for k in keys:
                print(f"      {k} = {d[k]}")


print("=" * 78)
print("dim16 CNN")
print("=" * 78)
for p in DIM16:
    show(p, "dim16")

print()
print("=" * 78)
print("dim64 CNN (the rerun)")
print("=" * 78)
show("results/baselines_ieee118_local_dim64b_*.json", "dim64")
show("results/baselines_ieee118_local_dim64_*.json", "dim64a")
