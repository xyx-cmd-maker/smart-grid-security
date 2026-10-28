"""Assemble the full two-class, two-system table and test whether class adds anything to AUROC.

The closed-form detector now has four points: the default amplitude and three lowered amplitudes, each
giving an AUROC and a realised fraction. The learned detector has four points from the 57-bus capacity
ladder and four from the 118-bus sweep.

The question this answers is whether AUROC alone orders the realised fraction, or whether class does too. If
closed-form and learned points interleave along AUROC there is one relationship; if the closed-form points
sit systematically above at matched AUROC there are two.
"""
import json
import math
import statistics
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "code" / "results"

def load(name):
    d = json.loads((RES / name).read_text(encoding="utf-8"))
    return [r for r in d["records"] if isinstance(r, dict)]

def frac(f1, f10):
    a = load(f1)
    b = load(f10)
    auc = statistics.mean(r["auc_stealthy"] for r in a)
    r1 = statistics.mean(r["detect_rate_stealthy"] for r in a)
    r10 = statistics.mean(r["detect_rate_stealthy"] for r in b)
    return auc, r1, r10, 100 * r1 / r10

points = []

# closed-form, 118 bus, four amplitudes
for sfx, label in (("", "default 1.5"), ("s12", "amplitude 1.2"),
                   ("s09", "amplitude 0.9"), ("s07", "amplitude 0.7")):
    if sfx == "":
        p = frac("baselines_ieee118_local_fpr01_ieee118.json", "baselines_ieee118_local_fpr10_ieee118.json")
        # restrict to TemporalResidual
        a = [r for r in load("baselines_ieee118_local_fpr01_ieee118.json") if str(r.get("model")) == "TemporalResidual"]
        b = [r for r in load("baselines_ieee118_local_fpr10_ieee118.json") if str(r.get("model")) == "TemporalResidual"]
        auc = statistics.mean(r["auc_stealthy"] for r in a)
        r1 = statistics.mean(r["detect_rate_stealthy"] for r in a)
        r10 = statistics.mean(r["detect_rate_stealthy"] for r in b)
        points.append(("118 closed-form", "closed-form", label, auc, r1, r10, 100 * r1 / r10))
    else:
        auc, r1, r10, f = frac(f"baselines_ieee118_{sfx}_local_{sfx}tr_fpr001.json",
                               f"baselines_ieee118_{sfx}_local_{sfx}tr_fpr010.json")
        points.append(("118 closed-form", "closed-form", label, auc, r1, r10, f))

# learned, 57-bus ladder
for ep in ("1", "2", "3"):
    auc, r1, r10, f = frac(f"baselines_ieee57_local_cap{ep}ep57_fpr001.json",
                           f"baselines_ieee57_local_cap{ep}ep57_fpr010.json")
    points.append(("57 CNN", "learned", f"{ep} epoch", auc, r1, r10, f))
fam57 = [r for r in load("baselines_ieee57_local_fam57.json") if str(r.get("model")) == "CNN"]
b20 = load("baselines_ieee57_local_cap20ep57_fpr010.json")
points.append(("57 CNN", "learned", "20 epochs",
               statistics.mean(r["auc_stealthy"] for r in fam57),
               statistics.mean(r["detect_rate_stealthy"] for r in fam57),
               statistics.mean(r["detect_rate_stealthy"] for r in b20),
               100 * statistics.mean(r["detect_rate_stealthy"] for r in fam57)
               / statistics.mean(r["detect_rate_stealthy"] for r in b20)))

# learned, 118-bus sweep
for model in ("Transformer", "LSTM", "CNN"):
    a = [r for r in load("baselines_ieee118_local_fpr01_ieee118.json") if str(r.get("model")) == model]
    b = [r for r in load("baselines_ieee118_local_fpr10_ieee118.json") if str(r.get("model")) == model]
    points.append(("118 " + model, "learned", "default 1.5",
                   statistics.mean(r["auc_stealthy"] for r in a),
                   statistics.mean(r["detect_rate_stealthy"] for r in a),
                   statistics.mean(r["detect_rate_stealthy"] for r in b),
                   100 * statistics.mean(r["detect_rate_stealthy"] for r in a)
                   / statistics.mean(r["detect_rate_stealthy"] for r in b)))

points.sort(key=lambda p: -p[3])
print(f"  {'detector':<18}{'class':<13}{'setting':<15}{'AUROC':<9}{'rate@1%':<10}{'realised %':<11}")
for name, cls, st, auc, r1, r10, f in points:
    print(f"  {name:<18}{cls:<13}{st:<15}{auc:<9.4f}{r1:<10.4f}{f:<11.1f}")

def spearman(xs, ys):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0] * len(v)
        for pos, i in enumerate(s):
            rk[i] = pos + 1
        return rk
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n)))
    return num / den

xs = [p[3] for p in points]
ys = [p[6] for p in points]
print()
print(f"  points: {len(points)}  Spearman(all) = {spearman(xs, ys):.3f}")
learned = [p for p in points if p[1] == "learned"]
closed = [p for p in points if p[1] == "closed-form"]
print(f"  Spearman(learned only, n={len(learned)}) = "
      f"{spearman([p[3] for p in learned], [p[6] for p in learned]):.3f}")
print()
print("  === overlap region, AUROC 0.75 to 0.90 ===")
for name, cls, st, auc, r1, r10, f in points:
    if 0.75 <= auc <= 0.90:
        print(f"    {name:<18}{cls:<13}{st:<15}AUROC {auc:.4f}  realised {f:.1f}")
