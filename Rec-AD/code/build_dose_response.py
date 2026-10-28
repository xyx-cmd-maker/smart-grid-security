"""Build the dose-response table: realised fraction against AUROC, across two systems.

The 57-bus case is moved down the ranking scale by cutting the training budget, which changes the detector
without changing the system. If the realised fraction tracks AUROC rather than the system, the four
capacity points should fall on the same curve as the 118-bus detectors rather than on a separate one.

Reports the table the manuscript would carry, plus a rank correlation across all points.
"""
import json
import math
import statistics
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "code" / "results"

def load(name):
    p = RES / name
    d = json.loads(p.read_text(encoding="utf-8"))
    return [r for r in d["records"] if isinstance(r, dict)]

def point(f1, f10):
    a = load(f1)
    b = load(f10)
    auc = statistics.mean(x["auc_stealthy"] for x in a)
    r1 = statistics.mean(x["detect_rate_stealthy"] for x in a)
    r10 = statistics.mean(x["detect_rate_stealthy"] for x in b)
    return auc, r1, r10, 100 * r1 / r10

rows = []
for ep in ("1", "2", "3"):
    auc, r1, r10, frac = point(f"baselines_ieee57_local_cap{ep}ep57_fpr001.json",
                               f"baselines_ieee57_local_cap{ep}ep57_fpr010.json")
    rows.append((f"57-bus CNN, {ep} epoch", auc, r1, r10, frac))

# the full-budget 57-bus CNN from the main family run
fam = load("baselines_ieee57_local_fam57.json")
cnn = [r for r in fam if str(r.get("model")) == "CNN"]
auc = statistics.mean(r["auc_stealthy"] for r in cnn)
r1 = statistics.mean(r["detect_rate_stealthy"] for r in cnn)
rows.append(("57-bus CNN, 20 epochs", auc, r1, float("nan"), float("nan")))

print("  === 57-bus capacity ladder ===")
print(f"  {'setting':<26}{'AUROC_st':<11}{'rate@1%':<10}{'rate@10%':<10}{'realised %':<11}")
for name, a, x1, x10, f in rows:
    x10s = "n/a" if math.isnan(x10) else f"{x10:.4f}"
    fs = "n/a" if math.isnan(f) else f"{f:.1f}"
    print(f"  {name:<26}{a:<11.4f}{x1:<10.4f}{x10s:<10}{fs:<11}")

print()
print("  === combined dose-response, two systems ===")
COMBINED = [
    ("118 TemporalResidual", 0.9837, 97.8),
    ("118 Transformer", 0.8703, 54.1),
    ("118 LSTM", 0.6844, 25.4),
    ("118 CNN", 0.7189, 21.2),
] + [(n, a, f) for n, a, x1, x10, f in rows if not math.isnan(f)]
COMBINED.sort(key=lambda r: -r[1])
print(f"  {'detector':<26}{'AUROC_st':<11}{'realised %':<11}")
for n, a, f in COMBINED:
    print(f"  {n:<26}{a:<11.4f}{f:<11.1f}")

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
    return num / den if den else float("nan")

xs = [r[1] for r in COMBINED]
ys = [r[2] for r in COMBINED]
print()
print(f"  points: {len(COMBINED)} across two systems")
print(f"  Spearman(AUROC, realised fraction) = {spearman(xs, ys):.4f}")
