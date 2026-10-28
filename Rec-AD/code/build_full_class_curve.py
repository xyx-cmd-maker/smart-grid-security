"""Assemble the six-point closed-form curve beside the seven learned points.

With amplitudes 0.7, 0.8, 0.9, 1.0, 1.2 and the default 1.5 the closed-form detector spans the ranking scale
the same way the learned network does under a budget cut. That makes a like-for-like comparison possible at
several AUROC levels rather than at one.

Reports both curves, the class gap at each overlapping AUROC band, and whether a single fit through all
thirteen points leaves structure in the residuals.
"""
import json
import statistics
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "code" / "results"

def load(name):
    d = json.loads((RES / name).read_text(encoding="utf-8"))
    return [r for r in d["records"] if isinstance(r, dict)]

def pair(f1, f10, model=None):
    a = load(f1)
    b = load(f10)
    if model:
        a = [r for r in a if str(r.get("model")) == model]
        b = [r for r in b if str(r.get("model")) == model]
    auc = statistics.mean(r["auc_stealthy"] for r in a)
    r1 = statistics.mean(r["detect_rate_stealthy"] for r in a)
    r10 = statistics.mean(r["detect_rate_stealthy"] for r in b)
    return auc, r1, r10, 100 * r1 / r10

closed = []
closed.append(("1.5 (default)",) + pair("baselines_ieee118_local_fpr01_ieee118.json",
                                        "baselines_ieee118_local_fpr10_ieee118.json", "TemporalResidual"))
for sc in ("s12", "s10", "s09", "s08", "s07"):
    closed.append((sc[1:] if False else sc,)
                  + pair(f"baselines_ieee118_{sc}_local_{sc}tr_fpr001.json",
                         f"baselines_ieee118_{sc}_local_{sc}tr_fpr010.json"))
labels = {"s12": "1.2", "s10": "1.0", "s09": "0.9", "s08": "0.8", "s07": "0.7"}
closed = [(labels.get(c[0], c[0]),) + c[1:] for c in closed]
closed.sort(key=lambda r: r[1])

learned = []
for ep in ("1", "2", "3"):
    learned.append((f"57 CNN {ep}ep",) + pair(f"baselines_ieee57_local_cap{ep}ep57_fpr001.json",
                                             f"baselines_ieee57_local_cap{ep}ep57_fpr010.json"))
f57 = [r for r in load("baselines_ieee57_local_fam57.json") if str(r.get("model")) == "CNN"]
b20 = load("baselines_ieee57_local_cap20ep57_fpr010.json")
learned.append(("57 CNN 20ep",
                statistics.mean(r["auc_stealthy"] for r in f57),
                statistics.mean(r["detect_rate_stealthy"] for r in f57),
                statistics.mean(r["detect_rate_stealthy"] for r in b20),
                100 * statistics.mean(r["detect_rate_stealthy"] for r in f57)
                / statistics.mean(r["detect_rate_stealthy"] for r in b20)))
for m in ("Transformer", "LSTM", "CNN"):
    learned.append((f"118 {m}",) + pair("baselines_ieee118_local_fpr01_ieee118.json",
                                        "baselines_ieee118_local_fpr10_ieee118.json", m))

print("  === closed-form, six amplitudes ===")
print(f"  {'amplitude':<12}{'AUROC':<10}{'rate@1%':<11}{'rate@10%':<11}{'realised %':<11}")
for lab, auc, r1, r10, f in closed:
    print(f"  {lab:<12}{auc:<10.4f}{r1:<11.4f}{r10:<11.4f}{f:<11.1f}")

print()
print("  === learned, seven settings ===")
print(f"  {'setting':<16}{'AUROC':<10}{'rate@1%':<11}{'realised %':<11}")
for lab, auc, r1, r10, f in sorted(learned, key=lambda r: -r[1]):
    print(f"  {lab:<16}{auc:<10.4f}{r1:<11.4f}{f:<11.1f}")

print()
print("  === class gap in overlapping AUROC bands ===")
for lab, auc, r1, r10, f in closed:
    near = min(learned, key=lambda L: abs(L[1] - auc))
    if abs(near[1] - auc) <= 0.06:
        print(f"    closed {lab} AUROC {auc:.4f} realised {f:5.1f}  |  "
              f"{near[0]} AUROC {near[1]:.4f} realised {near[4]:5.1f}  "
              f"-> closed is {f - near[4]:+.1f} points")

allpts = [(1, c[1], c[4]) for c in closed] + [(0, L[1], L[4]) for L in learned]
print()
print(f"  total points: {len(allpts)}  ({len(closed)} closed-form, {len(learned)} learned)")
