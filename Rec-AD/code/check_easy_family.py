"""Do the learned detectors detect ANY family, including the easy one?

This is a check on my own reasoning. I claimed the learned detectors sit at chance
because of the INPUT REPRESENTATION. But the `dense` input is a single snapshot that
carries the random attack at ~30x the benign residual norm -- BDD detects that family
at 1.0000 from the same snapshot. If a random forest cannot reach even its
false-alarm rate on a 30x anomaly, then "representation" is the wrong explanation and
something in the learned pipeline is broken instead.

That distinction matters a lot: it decides whether the paper's central claim is
"capacity is not the constraint" or "our training pipeline is broken".

Prints, per detector, the detection rate per family next to its own false-alarm rate,
and flags any detector that fails to clear its false-alarm rate on the RANDOM family,
which is the one family that must be easy.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

FILES = sorted(
    f for f in glob.glob(str(HERE / "results/baselines_ieee118_local_*.json"))
    if "summary" not in f and "partial" not in f
)

print("=" * 100)
print("Per-family detection vs each detector's own false-alarm rate")
print("=" * 100)
print()
print(f"  {'detector':<20}{'tag':<12}{'fpr':>9}{'random':>9}{'stealthy':>10}"
      f"{'redist':>9}   {'random/fpr':>11}  verdict")
print("  " + "-" * 96)

suspect = []
for f in FILES:
    p = Path(f)
    d = json.loads(p.read_text(encoding="utf-8"))
    recs = d.get("records", d if isinstance(d, list) else [])
    if not recs:
        continue

    def m(key: str) -> float:
        v = np.array([r.get(key, np.nan) for r in recs], dtype=float)
        v = v[~np.isnan(v)]
        return float(v.mean()) if v.size else float("nan")

    name = p.name.replace("baselines_ieee118_local_", "").replace(".json", "")
    tag = name.rsplit("_", 1)[0] if "_" in name else name
    mdl = name.rsplit("_", 1)[-1]

    fpr, rnd, st, rd = (m("detect_rate_normal_fp"), m("detect_rate_random"),
                        m("detect_rate_stealthy"), m("detect_rate_load_redist"))
    ratio = rnd / fpr if fpr else float("nan")

    if mdl in ("BDD", "TemporalResidual"):
        verdict = "physics/temporal -- reference"
    elif ratio < 3.0:
        verdict = "!! FAILS on the EASY family"
        suspect.append(name)
    else:
        verdict = "detects random OK"

    print(f"  {mdl:<20}{tag:<12}{fpr:>9.4f}{rnd:>9.4f}{st:>10.4f}{rd:>9.4f}"
          f"   {ratio:>11.2f}  {verdict}")

print()
print("=" * 100)
if suspect:
    print("  SUSPECT -- these learned detectors do not clear their own false-alarm rate")
    print("  on the random family, which BDD detects at 1.0000 from the same snapshot:")
    for s in suspect:
        print(f"    {s}")
    print()
    print("  If the random family is genuinely ~30x in the dense input, a detector that")
    print("  cannot see it is not 'limited by representation' -- it is not learning.")
    print("  That would contradict my earlier explanation and needs its own investigation.")
else:
    print("  All learned detectors clear their false-alarm rate on the random family.")
    print("  The at-chance behaviour is then specific to the stealthy family, which is")
    print("  consistent with the structural-blindness explanation.")
