"""Figure 4: detection at a fixed one-percent false-alarm target, across four systems and three families.

The data has a structure the prose states but no figure shows: four of the nine detectors sit at the
false-alarm floor on every system and every family, three learned detectors and BDD separate on some families
and collapse on others, and the closed-form innovation test is the only one that keeps most of its rate at
300 buses. Drawing all nine as separate lines would bury that, so the four that never leave the floor are
drawn as a single grey band labelled as such, with their individual values reported in the supplement table.

The floor line is drawn at the target false-alarm rate, because a detection rate at the floor means the
detector has no separability rather than weak separability. That is the distinction the panel is meant to
carry.
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

FILES = {14: "baselines_ieee14_local_fam14.json", 57: "baselines_ieee57_local_fam57.json",
         118: "baselines_ieee118_local_fam118.json", 300: "baselines_ieee300_local_fam300.json"}
FAMS = ["random", "stealthy", "load_redist"]
TITLES = {"random": "(a) Random family", "stealthy": "(b) Stealthy family",
          "load_redist": "(c) Redistribution family"}

data = {}
for n, f in FILES.items():
    rows = [r for r in json.loads((RES / f).read_text(encoding="utf-8"))["records"]
            if isinstance(r, dict)]
    agg = {}
    for r in rows:
        m = str(r.get("model"))
        agg.setdefault(m, {k: [] for k in FAMS})
        for fam in FAMS:
            v = r.get(f"detect_rate_{fam}")
            if v is not None:
                agg[m][fam].append(float(v))
    data[n] = {m: {k: (statistics.mean(v) if v else float("nan")) for k, v in d.items()}
               for m, d in agg.items()}

models = sorted(set().union(*[set(d) for d in data.values()]))
sizes = sorted(FILES)

FLAT = ["Autoencoder", "GCN", "RF", "SVM"]
HILITE = {"TemporalResidual": ("#0072B2", "-", "s", "closed-form innovation test"),
          "Transformer": ("#D55E00", "-", "o", "Transformer"),
          "LSTM": ("#E69F00", (0, (4, 2)), "^", "LSTM"),
          "CNN": ("#009E73", (0, (4, 2)), "v", "CNN"),
          "BDD": ("#CC79A7", (0, (1, 1.5)), "D", "residual test (BDD)")}

def series(m, fam):
    return [data[n][m][fam] if m in data[n] else float("nan") for n in sizes]

print("  flat detectors stay below:", end=" ")
worst = max(data[n][m][f] for n in sizes for m in FLAT for f in FAMS if m in data[n])
print(f"{worst:.4f}")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "legend.fontsize": 7, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "-",
    "lines.linewidth": 1.5, "lines.markersize": 3.6,
})

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15), sharey=True)
for ax, fam in zip(axes, FAMS):
    flat_lo = min(min(series(m, fam)) for m in FLAT)
    flat_hi = max(max(series(m, fam)) for m in FLAT)
    ax.fill_between(sizes, flat_lo, flat_hi, color="#B0B0B0", alpha=0.45, lw=0, zorder=1)
    if fam == FAMS[0]:
        band_handle = matplotlib.patches.Patch(facecolor="#B0B0B0", alpha=0.45,
                                               label="four detectors, at or near the floor")
    for m, (col, ls, mk, lab) in HILITE.items():
        if m not in models:
            continue
        ax.plot(sizes, series(m, fam), color=col, ls=ls, marker=mk, zorder=3,
                label=lab, clip_on=False)
    ax.axhline(0.01, color="#333", lw=0.9, ls=":", zorder=2)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlim(11, 400)
    ax.set_ylim(-0.04, 1.06)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_title(TITLES[fam])
    ax.set_xlabel("Buses")
axes[0].set_ylabel("Detection rate at a 1% false-alarm target")
axes[0].annotate("false-alarm floor", xy=(14, 0.01), xytext=(15, 0.16), fontsize=6.4,
                 color="#333", arrowprops=dict(arrowstyle="-", lw=0.6, color="#333"))
axes[0].text(30, 0.55, "five detectors at 1.00", fontsize=6.6, color="#333")
axes[0].text(30, 0.40, "four at the floor", fontsize=6.6, color="#555")

handles, labels = axes[0].get_legend_handles_labels()
handles.append(band_handle)
labels.append("four detectors, at or near the floor")
fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06),
           handlelength=2.2, columnspacing=1.4)
fig.tight_layout(w_pad=0.8, rect=(0, 0.06, 1, 1))
fig.savefig(OUT / "fig_cross_scale.pdf")
fig.savefig(OUT / "fig_cross_scale.png", dpi=300)
print(f"  wrote {OUT / 'fig_cross_scale.pdf'}")
for m in ("TemporalResidual", "Transformer", "LSTM", "CNN"):
    print(f"    {m:<18} stealthy: " + "  ".join(f"{v:.3f}" for v in series(m, "stealthy")))
