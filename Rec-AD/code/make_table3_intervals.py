"""Turn the per-seed standard deviations this project already has into the intervals A's footnote requires.

Her footnote says values need confidence intervals or run dispersion. The registry holds sd and
n_seeds for every baseline slot, all of them three, so the interval is a computation over existing
records rather than a reason to re-run anything. My previous audit said the opposite, and the audit
was reading its own assumption rather than the registry, which is the failure this session keeps
producing.

The interval is the t interval on three seeds: mean plus or minus t(0.975, 2) times sd over the
square root of three. With two degrees of freedom the multiplier is 4.303, which is wide, and
saying so is part of reporting honestly rather than a fault to hide. A three-seed interval is a
dispersion estimate, not a precise bound, and A's footnote allows either.

Each row also reports the interval half-width as a fraction of the point estimate, because that
ratio is what a reviewer will look at first.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
blob = json.loads((ROOT / "slots.json").read_text(encoding="utf-8"))
S: dict[str, dict] = {}
for ns, g in blob.items():
    if isinstance(g, dict):
        for k, v in g.items():
            if isinstance(v, dict) and "used_in" in v:
                S[k] = v

T2 = 4.302653  # t(0.975, df=2)

ROWS = ["BDD", "TemporalResidual", "RF", "SVM", "Autoencoder", "GCN", "GAT",
        "CNN", "LSTM", "Transformer"]
COLS = [("auc", "AUROC"), ("random", "Random"), ("stealthy", "Stealthy"),
        ("redist", "LRA"), ("fpr", "Benign FPR")]

out = {"multiplier": T2, "df": 2, "note": "t interval on 3 seeds", "rows": []}
print("  A 的 Table III 单元格 + 置信区间（三种子，t(0.975,2)=4.303）")
print()
hdr = f"  {'row':<17}" + "".join(f"{lab:>20}" for _, lab in COLS)
print(hdr)
print("  " + "-" * (17 + 20 * len(COLS)))

for det in ROWS:
    cells = {}
    line = f"  {det:<17}"
    for key, lab in COLS:
        e = S.get(f"baselines.{det}.{key}")
        if e is None or e.get("value") is None:
            line += f"{'--':>20}"
            cells[key] = None
            continue
        m = float(e["value"])
        sd = float(e.get("sd") or 0.0)
        n = int(e.get("n_seeds") or 1)
        if n > 1 and sd > 0:
            h = T2 * sd / math.sqrt(n)
            lo, hi = max(0.0, m - h), min(1.0, m + h)
            txt = f"{m:.3f}±{h:.3f}"
            cells[key] = {"mean": m, "sd": sd, "n": n, "lo": round(lo, 4),
                          "hi": round(hi, 4), "half_width": round(h, 4),
                          "relative": round(h / m, 3) if m else None,
                          "source": e.get("source")}
        else:
            txt = f"{m:.3f} (n={n})"
            cells[key] = {"mean": m, "sd": sd, "n": n, "lo": None, "hi": None}
        line += f"{txt:>20}"
    print(line)
    out["rows"].append({"detector": det, "cells": cells})

# ---- which cells still lack an interval -------------------------------------
print()
no_ci = [(r["detector"], k) for r in out["rows"] for k, v in r["cells"].items()
         if v is not None and v.get("lo") is None]
print(f"  ★ 仍无区间的单元格: {len(no_ci)}  {no_ci if no_ci else '（无）'}")

wide = [(r["detector"], k, v["relative"]) for r in out["rows"] for k, v in r["cells"].items()
        if v is not None and v.get("relative") is not None and v["relative"] > 0.5]
print(f"  ★ 相对半宽 > 50% 的单元格（点估计不可靠）: {len(wide)}")
for d, k, r_ in wide:
    print(f"      {d:<17} {k:<10} ±{r_*100:.0f}% of the estimate")

p = ROOT / "code" / "results" / "table3_intervals.json"
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"  wrote {p.name} ({os.stat(p).st_size/1024:.1f} KB)")
