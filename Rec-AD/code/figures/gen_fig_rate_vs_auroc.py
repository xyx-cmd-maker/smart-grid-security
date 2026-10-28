"""Figure 2: what survives to the deployment point, by detector class.

Two panels, because the previous single panel could not show the pairing rule and left one learned point
visually unconnected. Panel A carries both series with every matched pair annotated by its two AUROC values,
and marks the learned point that is in no pair. Panel B carries the six gaps as bars with their own value
labels, so the comparison is stated rather than left to the reader to reconstruct from nearest-neighbour
geometry.

Style follows the academic-plotting skill's data-chart rules: serif font for IEEEtran, Okabe-Ito colours,
distinct marker shapes for grayscale, panel titles, an external legend, a limited tick count, value labels,
vector PDF plus 300 dpi PNG, and the generating script kept beside the output.
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "code" / "results"
OUT = ROOT / "code" / "figures" / "pub2"
OUT.mkdir(parents=True, exist_ok=True)

def load(name):
    d = json.loads((RES / name).read_text(encoding="utf-8"))
    return [r for r in d["records"] if isinstance(r, dict)]

def pair(f1, f10, model=None, label=""):
    a, b = load(f1), load(f10)
    if model:
        a = [r for r in a if str(r.get("model")) == model]
        b = [r for r in b if str(r.get("model")) == model]
    auc = statistics.mean(r["auc_stealthy"] for r in a)
    r1 = statistics.mean(r["detect_rate_stealthy"] for r in a)
    r10 = statistics.mean(r["detect_rate_stealthy"] for r in b)
    return {"label": label, "auc": auc, "frac": 100 * r1 / r10}

closed = [pair("baselines_ieee118_local_fpr01_ieee118.json",
               "baselines_ieee118_local_fpr10_ieee118.json", "TemporalResidual", "1.5")]
for sc, lab in (("s12", "1.2"), ("s10", "1.0"), ("s09", "0.9"), ("s08", "0.8"), ("s07", "0.7")):
    closed.append(pair(f"baselines_ieee118_{sc}_local_{sc}tr_fpr001.json",
                       f"baselines_ieee118_{sc}_local_{sc}tr_fpr010.json", None, lab))
closed.sort(key=lambda d: d["auc"])

learned = []
for ep in ("1", "2", "3"):
    learned.append(pair(f"baselines_ieee57_local_cap{ep}ep57_fpr001.json",
                        f"baselines_ieee57_local_cap{ep}ep57_fpr010.json", None, f"57 CNN {ep}ep"))
f57 = [r for r in load("baselines_ieee57_local_fam57.json") if str(r.get("model")) == "CNN"]
b20 = load("baselines_ieee57_local_cap20ep57_fpr010.json")
learned.append({"label": "57 CNN 20ep",
                "auc": statistics.mean(r["auc_stealthy"] for r in f57),
                "frac": 100 * statistics.mean(r["detect_rate_stealthy"] for r in f57)
                / statistics.mean(r["detect_rate_stealthy"] for r in b20)})
for m in ("Transformer", "LSTM", "CNN"):
    learned.append(pair("baselines_ieee118_local_fpr01_ieee118.json",
                        "baselines_ieee118_local_fpr10_ieee118.json", m, "118 " + m))
learned.sort(key=lambda d: d["auc"])

TOL = 0.06
matched, unpaired = [], []
for c in closed:
    near = min(learned, key=lambda d: abs(d["auc"] - c["auc"]))
    if abs(near["auc"] - c["auc"]) <= TOL:
        matched.append((c, near))
for L in learned:
    if not any(id(L) == id(m[1]) for m in matched):
        unpaired.append(L)
print(f"  paired: {len(matched)}  unpaired: {[u['label'] for u in unpaired]}")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "legend.fontsize": 8, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "-",
    "lines.linewidth": 1.8, "lines.markersize": 5,
})
BLUE, VERM, INK = "#0072B2", "#D55E00", "#4D4D4D"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})

for c, L in matched:
    axA.plot([c["auc"], L["auc"]], [c["frac"], L["frac"]], color=INK, lw=0.7, ls=(0, (2, 2)),
             alpha=0.55, zorder=1)

cx = [c["auc"] for c in closed]
cy = [c["frac"] for c in closed]
lx = [L["auc"] for L in learned]
ly = [L["frac"] for L in learned]
axA.plot(cx, cy, color=BLUE, lw=1.8, zorder=3)
axA.plot(cx, cy, "s", color=BLUE, ms=5.5, mec="white", mew=0.7, zorder=4)
axA.plot(lx, ly, "o", color=VERM, ms=5.5, mec="white", mew=0.7, ls="none", zorder=4)

for L in unpaired:
    axA.annotate("no counterpart\nwithin 0.06 AUROC", xy=(L["auc"], L["frac"]),
                 xytext=(L["auc"] + 0.022, L["frac"] - 12), fontsize=6.8, color=INK,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, ls=":"))

axA.set_title("(a) Realised fraction against ranking quality")
axA.set_xlabel("Stealthy-family AUROC")
axA.set_ylabel("Ten-percent rate kept\nat a one-percent budget (%)")
axA.set_xlim(0.655, 1.015)
axA.set_ylim(0, 105)
axA.xaxis.set_major_locator(MaxNLocator(5))
axA.yaxis.set_major_locator(MaxNLocator(5))
handles = [
    plt.Line2D([], [], color=BLUE, marker="s", mec="white", mew=0.7, ls="-", lw=1.8,
               label="closed-form test, six amplitudes"),
    plt.Line2D([], [], color=VERM, marker="o", mec="white", mew=0.7, ls="none",
               label="learned detectors, seven settings"),
    plt.Line2D([], [], color=INK, lw=0.7, ls=(0, (2, 2)), label="matched pair"),
]
axA.legend(handles=handles, loc="upper left", handlelength=1.9, borderaxespad=0.3)

gaps = [(c, L, c["frac"] - L["frac"]) for c, L in matched]
gaps.sort(key=lambda m: m[0]["auc"])
ypos = range(len(gaps))
bars = axB.barh(list(ypos), [g[2] for g in gaps], height=0.55, color="#6E7B8B",
                edgecolor="white", linewidth=0.5, zorder=3)
for bar, (c, L, g) in zip(bars, gaps):
    axB.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2, f"+{g:.1f}",
             va="center", ha="left", fontsize=7.5, color="#333")
axB.set_yticks(list(ypos))
axB.set_yticklabels([f"closed {c['auc']:.3f}\n{L['label']} {L['auc']:.3f}" for c, L, _ in gaps], fontsize=6.8)
axB.set_title("(b) Gap at each matched level")
axB.set_xlabel("Closed-form minus learned (percentage points)")
axB.set_xlim(0, 40)
axB.xaxis.set_major_locator(MaxNLocator(5))
axB.grid(axis="y", visible=False)

fig.tight_layout(w_pad=1.4)
fig.savefig(OUT / "fig_rate_vs_auroc.pdf")
fig.savefig(OUT / "fig_rate_vs_auroc.png", dpi=300)
print(f"  wrote {OUT / 'fig_rate_vs_auroc.pdf'}")
print(f"  gaps: {[round(g[2], 1) for g in gaps]}")
