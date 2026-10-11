"""Compare the dim=64 reruns against the earlier dim=16 results.

The question this answers: did fixing the embedding-dimension mismatch change any
conclusion, or only the capacity of the models being compared? If every family's
detection rate is still at its own false-alarm rate, the "capacity is not the
binding constraint" claim survives -- now on a same-capacity comparison, which is
strictly stronger evidence than before.

Old results come from two tag generations (`iso_` and `iso2_`); both were dim=16 and
both are superseded, but they are the right baseline for seeing the effect.
"""
from __future__ import annotations

import glob
import json

import numpy as np


def load(pattern: str) -> dict:
    out = {}
    for f in glob.glob(pattern):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        recs = d if isinstance(d, list) else d.get("records", [])
        if not recs:
            continue
        # filename tail after the last underscore, minus the suffix
        name = f.replace("\\", "/").split("/")[-1]
        name = name.split("_")[-1].replace(".partial.json", "")
        g = lambda k: float(np.mean([r.get(k, np.nan) for r in recs]))  # noqa: E731
        out[name] = dict(fpr=g("detect_rate_normal_fp"), auc=g("auc"),
                         rnd=g("detect_rate_random"), st=g("detect_rate_stealthy"),
                         lr=g("detect_rate_load_redist"), n=len(recs))
    return out


old = load("results/baselines_ieee118_local_iso2_*.partial.json")
old.update(load("results/baselines_ieee118_local_iso_*.partial.json"))
new = load("results/baselines_ieee118_local_dim64_*.partial.json")

print(f"  {'model':<18}{'dim16  auc':>12}{'stealthy':>10}   {'dim64  auc':>12}{'stealthy':>10}")
print("  " + "-" * 66)
for k in sorted(set(old) | set(new)):
    a, b = old.get(k), new.get(k)
    sa = f"{a['auc']:.4f}" if a else "  --  "
    ss = f"{a['st']:.4f}" if a else "  --  "
    ba = f"{b['auc']:.4f}" if b else "  --  "
    bs = f"{b['st']:.4f}" if b else "  --  "
    print(f"  {k:<18}{sa:>12}{ss:>10}   {ba:>12}{bs:>10}")

print()
print("  Reading: if the dim64 columns are also ~0.50 and the stealthy rate tracks")
print("  the false-alarm rate, the fix changed capacity without changing the")
print("  conclusion -- which makes the 'capacity is not the constraint' claim")
print("  stronger, because it now rests on a same-dimension comparison.")
