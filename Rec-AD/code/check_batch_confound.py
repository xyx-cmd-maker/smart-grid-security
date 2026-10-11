"""Check which batch size each CNN run used before treating them as a clean comparison.

The dim=16 and dim=64 CNN runs differ in embedding dimension, which is the variable
under test. If they ALSO differ in batch size, then the comparison confounds two
variables, and any "the dimension did not matter" conclusion would be unsupported --
the difference could be batch noise masking or exaggerating a dimension effect.

AGENTS.md 4.8: before comparing two numbers, write down their frames of reference
{dataset, config, source file, tier, standardize, measurement scope}. Batch size is
part of `config`. This reads it from the stored records rather than from memory.

Also reports the LSTM rerun status, since that is the other half of the pending work.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

FIELDS = ("batch_size", "lr", "epochs", "tag", "tier", "standardize",
          "embedding_dim", "target_fpr", "attack_filter", "seed")


def inspect(pattern: str) -> None:
    for f in sorted(glob.glob(str(HERE / pattern))):
        if not f.endswith(".json") or f.endswith("_summary.json"):
            continue
        p = Path(f)
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:                       # noqa: BLE001
            print(f"  {p.name}: unreadable ({exc})")
            continue
        recs = d.get("records", d if isinstance(d, list) else [])
        if not recs:
            continue
        print(f"  {p.name}")
        # top-level meta first
        for k in FIELDS:
            if k in d:
                print(f"      meta.{k} = {d[k]}")
        r0 = recs[0]
        for k in FIELDS:
            if k in r0:
                vals = {r.get(k) for r in recs}
                shown = vals.pop() if len(vals) == 1 else sorted(map(str, vals))
                print(f"      rec.{k} = {shown}")
        auc = np.array([r.get("auc", np.nan) for r in recs], dtype=float)
        print(f"      -> {len(recs)} seeds, mean auc {auc.mean():.4f} "
              f"sd {auc.std(ddof=1) if len(recs) > 1 else float('nan'):.4f}")
        print()


print("=" * 78)
print("dim16 CNN (previous generation)")
print("=" * 78)
inspect("results/baselines_ieee118_local_iso_CNN*.json")

print("=" * 78)
print("dim64 CNN (the rerun)")
print("=" * 78)
inspect("results/baselines_ieee118_local_dim64b_CNN*.json")

print("=" * 78)
print("LSTM rerun status")
print("=" * 78)
inspect("results/baselines_ieee118_local_dim64b_LSTM*.json")
log = HERE / "d64b_LSTM.log"
if log.exists():
    lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
    print(f"  d64b_LSTM.log: {len(lines)} lines")
    for ln in lines[-5:]:
        print(f"      {ln.rstrip()}")
else:
    print("  no LSTM log yet")
