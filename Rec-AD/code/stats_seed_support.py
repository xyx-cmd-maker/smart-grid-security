"""Report what three seeds support for the cross-scale claim, with the interval width made explicit.

The scaling records carry one row per (case, seed), with no model field: the runner sweeps one model across grid sizes. Grouping by case
alone gives the three seeds per point.

Three values support a mean and a standard deviation and very little else. Reporting a Student-t interval from n = 3 gives a width of
roughly four standard errors, which for these standard deviations will be wide enough to overlap between endpoints; the paper's claim that
the two scales differ should therefore be stated with the triple and not with an interval that implies more precision than exists.

This prints the per-case triples, the mean, the seed SD, the half-width of a 95 percent t interval, and whether the intervals for the two
extremes overlap.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "code" / "results"

# t(0.975, df=2) for a two-sided 95 percent interval
T975_DF2 = 4.302653

for fname in ("scaling_local_v3.json", "scaling_local_v2.json", "scaling_local_sc1.json"):
    p = RES / fname
    if not p.exists():
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    recs = d.get("records") or []
    by = {}
    for r in recs:
        case = r.get("case") or r.get("n_bus")
        auc = r.get("auc")
        seed = r.get("seed")
        if case is None or auc is None:
            continue
        by.setdefault(int(case), []).append((seed, float(auc)))

    print(f"  === {fname}: {len(recs)} records, {len(by)} cases ===")
    summary = {}
    for case in sorted(by):
        vals = [v for _, v in sorted(by[case])]
        if len(vals) < 2:
            print(f"    {case:>4} buses  only {len(vals)} seed(s) — no spread available")
            continue
        mu = statistics.fmean(vals)
        sd = statistics.stdev(vals)
        hw = T975_DF2 * sd / (len(vals) ** 0.5)
        summary[case] = (mu, sd, hw, min(vals), max(vals), len(vals))
        print(f"    {case:>4} buses  n={len(vals)}  mean={mu:.4f}  sd={sd:.4f}  "
              f"95%t-half-width={hw:.4f}  range=[{min(vals):.4f}, {max(vals):.4f}]")

    if len(summary) >= 2:
        lo_case, hi_case = min(summary), max(summary)
        m1, s1, h1, *_ = summary[lo_case]
        m2, s2, h2, *_ = summary[hi_case]
        overlap = abs(m1 - m2) <= (h1 + h2)
        print(f"    -> {lo_case} vs {hi_case}: difference={abs(m1 - m2):.4f}, "
              f"sum of half-widths={h1 + h2:.4f}, intervals overlap: {'YES' if overlap else 'no'}")
    print()
