"""Compute the closed-form detector's realised fraction at a lower attack amplitude.

The 118-bus closed-form test sits at AUROC 0.9837 at the default amplitude, near the top of the scale, so it
supplies only one point and that point is above the fitted relationship. Lowering the injection amplitude
moves the same detector down the scale without changing its class or its system, which is the closed-form
counterpart of the training-budget cut used on the learned network.

At amplitude 0.9 the threshold sweep recorded a temporal AUROC of 0.7569, inside the informative band. This
reports what that detector realises at the deployment point, so the class comparison can be made at matched
AUROC rather than across the whole scale.
"""
import json
import statistics
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "code" / "results"

def load(name):
    d = json.loads((RES / name).read_text(encoding="utf-8"))
    return [r for r in d["records"] if isinstance(r, dict)]

per = {}
for tgt, tag in (("0.01", "fpr001"), ("0.02", "fpr002"), ("0.05", "fpr005"), ("0.10", "fpr010")):
    recs = load(f"baselines_ieee118_s09_local_s09tr_{tag}.json")
    per[tgt] = (statistics.mean(r["auc_stealthy"] for r in recs),
                statistics.mean(r["detect_rate_stealthy"] for r in recs),
                statistics.mean(r["fpr"] for r in recs))

print("  === 118-bus closed-form test at attack scale 0.9 ===")
print(f"  {'budget':<9}{'stealthy AUROC':<17}{'stealthy rate':<15}{'realised FPR':<13}")
for tgt in ("0.01", "0.02", "0.05", "0.10"):
    a, r, f = per[tgt]
    print(f"  {tgt:<9}{a:<17.4f}{r:<15.4f}{f:<13.4f}")

auc = per["0.01"][0]
realised = 100 * per["0.01"][1] / per["0.10"][1]
print()
print(f"  stealthy AUROC (budget-independent): {auc:.4f}")
print(f"  realised fraction at 1% of the 10% rate: {realised:.1f} percent")

print()
print("  === class comparison at matched AUROC ===")
print(f"  {'detector':<30}{'AUROC':<10}{'realised %':<11}")
print(f"  {'118 closed-form, amplitude 0.9':<30}{auc:<10.4f}{realised:<11.1f}")
for name, a_, f_ in (("57 learned CNN, 1 epoch", 0.7805, 30.6), ("57 learned CNN, 2 epochs", 0.8732, 51.9)):
    print(f"  {name:<30}{a_:<10.4f}{f_:<11.1f}")

near = min((abs(auc - 0.7805), "1 epoch", 30.6), (abs(auc - 0.8732), "2 epochs", 51.9))
print()
print(f"  closest learned point: {near[1]}, AUROC gap {near[0]:.4f}, "
      f"realised gap {realised - near[2]:+.1f} points")
