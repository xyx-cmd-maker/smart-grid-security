"""Figure: how much headroom each detector leaves against the two assumed budgets.

Plotting latency as a share of the budget puts every bar at one to three percent of the axis, which reads as
zero and hides the differences that matter. The headroom factor, budget divided by the measured value, is
what the text already reports ("no measured detector comes within a factor of 28 of either budget"), so it is
what the figure should show. A log axis keeps both the smallest and the largest margin on one scale, and the
reference line at 1x marks the point at which a budget would actually bind.

Reads the result files directly so the figure cannot drift from the numbers in the text.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "code" / "results"
OUT = ROOT / "code" / "figures" / "pub2"
OUT.mkdir(parents=True, exist_ok=True)

LAT_BUDGET_MS = 500.0
MEM_BUDGET_MIB = 8192.0

def load(name):
    d = json.loads((RES / name).read_text(encoding="utf-8"))
    return d.get("rows") or d.get("records") or []

rows = []
for r in load("power_latency.json"):
    if r.get("verdict") == "fits" and r.get("p99_ms") and r.get("peak_mib"):
        rows.append((r["model"], float(r["p99_ms"]), float(r["peak_mib"])))
for r in load("recad_inference.json"):
    cell = str(r.get("cell", ""))
    if cell in ("Rec-AD full (TT+reorder+pipeline)", "Rec-AD no-TT (reorder+pipeline)",
                "Rec-AD TT only", "Rec-AD none"):
        short = {"Rec-AD full (TT+reorder+pipeline)": "Rec-AD full",
                 "Rec-AD no-TT (reorder+pipeline)": "Rec-AD no TT",
                 "Rec-AD TT only": "Rec-AD TT only",
                 "Rec-AD none": "Rec-AD none"}[cell]
        rows.append((short, float(r["p99_ms"]), float(r["peak_mib"])))

print(f"  detectors: {len(rows)}")
lat_h = [LAT_BUDGET_MS / p for _, p, _ in rows]
mem_h = [MEM_BUDGET_MIB / m for _, _, m in rows]
names = [n for n, _, _ in rows]
print(f"  latency headroom {min(lat_h):.1f}x to {max(lat_h):.1f}x")
print(f"  memory headroom  {min(mem_h):.1f}x to {max(mem_h):.1f}x")
print(f"  tightest of either: {min(min(lat_h), min(mem_h)):.1f}x")

order = sorted(range(len(rows)), key=lambda i: max(lat_h[i], mem_h[i]))
names = [names[i] for i in order]
lat_h = [lat_h[i] for i in order]
mem_h = [mem_h[i] for i in order]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "legend.fontsize": 8, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "-",
    "lines.linewidth": 1.6,
})
BLUE, VERM, INK = "#0072B2", "#D55E00", "#4D4D4D"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
ypos = list(range(len(names)))

axA.barh(ypos, lat_h, height=0.55, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
axB.barh(ypos, mem_h, height=0.55, color=VERM, edgecolor="white", linewidth=0.5, zorder=3)

for ax, vals, title, lab in (
        (axA, lat_h, "(a) Latency headroom", "500 ms budget / measured p99"),
        (axB, mem_h, "(b) Memory headroom", "8 GiB budget / measured peak")):
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300, 1000]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.axvline(1.0, color=INK, lw=1.0, ls="--", zorder=2)
    ax.set_title(title)
    ax.set_xlabel(lab + "  (1\u00d7 = budget exhausted)")
    for y, v in zip(ypos, vals):
        ax.text(v * 1.10, y, f"{v:.0f}\u00d7", va="center", ha="left", fontsize=7, color="#333")

axA.set_yticks(ypos)
axA.set_yticklabels(names, fontsize=7.5)
axA.set_ylim(-0.7, len(names) - 0.15)
axA.grid(axis="y", visible=False)
axB.grid(axis="y", visible=False)

fig.tight_layout(w_pad=1.2)
fig.savefig(OUT / "fig_headroom.pdf")
fig.savefig(OUT / "fig_headroom.png", dpi=300)
print(f"  wrote {OUT / 'fig_headroom.pdf'}")
