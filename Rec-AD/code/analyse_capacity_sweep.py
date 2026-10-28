"""Analyse the capacity-limited 57-bus sweep and compare it with the 118-bus result.

The purpose is to test whether the operating-point effect follows the detector's AUROC rather than the
system it runs on. At 118 buses the learned models happen to have middling AUROC and show large gaps; at 57
buses they sit near ceiling and show small ones. Lowering the training budget moves the 57-bus model down
the ranking scale without touching its system, so if the gap appears there too, the effect tracks AUROC.

Reads the four tagged result files, reports per-budget rates, and computes the fraction of ten-percent
performance realised at the one-percent deployment point, which is the quantity the manuscript reports.
"""
import json
import statistics
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "code" / "results"
TAGS = {"0.01": "cap1ep57_fpr001", "0.02": "cap1ep57_fpr002",
        "0.05": "cap1ep57_fpr005", "0.10": "cap1ep57_fpr010"}

def load(tag):
    p = RES / f"baselines_ieee57_local_{tag}.json"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    return [r for r in d["records"] if isinstance(r, dict)]

print("  === 57-bus CNN, one epoch, three seeds ===")
print(f"  {'budget':<8}{'run rate':<10}{'stealthy rate':<15}{'stealthy AUROC':<16}{'realised FPR':<13}")
rates, aucs = {}, {}
for tgt, tag in TAGS.items():
    recs = load(tag)
    r = statistics.mean(x["detect_rate_stealthy"] for x in recs)
    a = statistics.mean(x["auc_stealthy"] for x in recs)
    f = statistics.mean(x["fpr"] for x in recs)
    rates[tgt], aucs[tgt] = r, a
    print(f"  {tgt:<8}{tgt:<10}{r:<15.4f}{a:<16.4f}{f:<13.4f}")

frac = 100 * rates["0.01"] / rates["0.10"]
gap1 = aucs["0.01"] - rates["0.01"]
gap10 = aucs["0.10"] - rates["0.10"]
print()
print(f"  realised fraction at 1% of the 10% rate: {frac:.1f} percent")
print(f"  gap between AUROC and rate: {gap1:.4f} at 1%, {gap10:.4f} at 10%")
print(f"  AUROC is budget-independent within a seed set: spread "
      f"{max(aucs.values()) - min(aucs.values()):.4f}")

print()
print("  === comparison with the 118-bus sweep in the manuscript ===")
print(f"  {'detector':<22}{'AUROC':<10}{'rate@1%':<10}{'rate@10%':<10}{'realised':<10}")
for name, a, r1, r10 in (
    ("118 TemporalResidual", 0.9837, 0.9581, 0.9797),
    ("118 Transformer", 0.8703, 0.3621, 0.6688),
    ("118 LSTM", 0.6844, 0.0805, 0.3168),
    ("118 CNN", 0.7189, 0.0737, 0.3477),
    ("57 CNN one epoch", aucs["0.01"], rates["0.01"], rates["0.10"]),
):
    print(f"  {name:<22}{a:<10.4f}{r1:<10.4f}{r10:<10.4f}{100*r1/r10:<10.1f}")
