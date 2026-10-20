"""Draw the seven conceptual figures the manuscript asks for and does not have.

Nine of the manuscript's ten figures are diagrams rather than plots: the architecture, the two input
branches, the tensorisation, the lookup, the two orderings, and the read-after-write hazard. Their
content is described in the captions and the surrounding text, so they can be drawn now, without
waiting on any measurement. The three that need data are left alone.

Each figure is drawn to the same standard as the result plots: double-column width, six-hundred dot
resolution, the colourblind-safe palette, type sized for the printed page. They use the caption
wording as their specification, and where a caption says the original artwork must be checked, the
diagram states only what the text asserts rather than inventing detail.

The file names follow the manuscript's own references, so a figure can be dropped in without anyone
having to map names. Boxes are drawn with rectangles and arrows with annotations; nothing is
decorative, because a diagram that carries no information takes space from one that does.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import pubstyle

OUT = HERE.parent / "figures" / "manuscript"

C = pubstyle.OKABE_ITO
FS = 6.5          # box label size at final print width


def box(ax, x, y, w, h, text, *, fc="white", ec=C["blue"], lw=0.8, fs=FS, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p, q, *, color=None, style="-|>", lw=0.9, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7,
                                 linewidth=lw, linestyle=ls,
                                 color=color or "#444444", zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def canvas(w=pubstyle.WIDTH_DOUBLE, h=2.4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.05)
    return pubstyle.save(fig, OUT, name)


# ---------------------------------------------------------------- overview ---
def overview11():
    fig, ax = canvas(h=2.9)
    box(ax, 0.02, 0.60, 0.20, 0.26, "Measurements\n$z_t$", ec=C["black"])
    box(ax, 0.02, 0.18, 0.20, 0.26, "Grid topology\n$H$, $\\Sigma$", ec=C["black"])

    box(ax, 0.27, 0.55, 0.20, 0.36, "Feature\nconstruction\nwindow $w$", ec=C["orange"])
    box(ax, 0.27, 0.14, 0.20, 0.30, "Physics\nchannels", ec=C["orange"])
    arrow(ax, (0.22, 0.73), (0.27, 0.73))
    arrow(ax, (0.22, 0.31), (0.27, 0.29))

    box(ax, 0.52, 0.55, 0.20, 0.36, "Embedding\nlayers\n(dense $\\to$ TT)", ec=C["blue"], bold=True)
    box(ax, 0.52, 0.14, 0.20, 0.30, "Interaction\n+ MLP head", ec=C["blue"])
    arrow(ax, (0.47, 0.73), (0.52, 0.73))
    arrow(ax, (0.47, 0.29), (0.52, 0.29))
    arrow(ax, (0.62, 0.55), (0.62, 0.44))

    box(ax, 0.77, 0.40, 0.21, 0.28, "Per-family\ndetection\nscores", ec=C["green"], bold=True)
    arrow(ax, (0.72, 0.54), (0.77, 0.54))

    ax.text(0.375, 0.055, "reused across samples and batches",
            ha="center", fontsize=FS, style="italic", color="#555555")
    ax.text(0.62, 0.055, r"$O(R^2mn)$ parameters, not $O(|\mathcal{V}|d)$",
            ha="center", fontsize=FS, style="italic", color="#555555")
    return save(fig, "overview11")


# ------------------------------------------------------------- dlrm_workflow --
def dlrm_workflow11():
    fig, ax = canvas(h=2.2)
    box(ax, 0.02, 0.62, 0.22, 0.26, "Continuous\nmeasurements", ec=C["black"])
    box(ax, 0.02, 0.14, 0.22, 0.26, "Categorical\nbin crosses", ec=C["black"])

    box(ax, 0.30, 0.62, 0.18, 0.26, "MLP\n$\\to \\mathbb{R}^{d}$", ec=C["orange"])
    box(ax, 0.30, 0.14, 0.18, 0.26, "Embedding\nlookup", ec=C["blue"], bold=True)
    arrow(ax, (0.24, 0.75), (0.30, 0.75))
    arrow(ax, (0.24, 0.27), (0.30, 0.27))

    box(ax, 0.55, 0.38, 0.16, 0.26, "Interaction\n(dot)", ec=C["green"])
    arrow(ax, (0.48, 0.75), (0.55, 0.58))
    arrow(ax, (0.48, 0.27), (0.55, 0.44))

    box(ax, 0.78, 0.38, 0.20, 0.26, "Logit\n$\\to$ family\nscores", ec=C["purple"])
    arrow(ax, (0.71, 0.51), (0.78, 0.51))

    ax.text(0.5, 0.02, "The categorical branch dominates parameters; the continuous branch does not.",
            ha="center", fontsize=FS, style="italic", color="#555555")
    return save(fig, "dlrm_workflow11")


# -------------------------------------------------------------- tt_workflow --
def tt_workflow11():
    fig, ax = canvas(h=2.5)
    box(ax, 0.02, 0.40, 0.18, 0.22, "Dense table\n$\\mathcal{W}$", ec=C["black"])
    arrow(ax, (0.20, 0.51), (0.27, 0.51))

    box(ax, 0.28, 0.36, 0.20, 0.30, "Tensorise\n$m_1\\!\\times\\!\\cdots\\!\\times\\!m_K$",
        ec=C["orange"])
    arrow(ax, (0.48, 0.51), (0.55, 0.51))

    xs, w, gap = 0.56, 0.075, 0.022
    for k in range(4):
        x = xs + k * (w + gap)
        box(ax, x, 0.36, w, 0.30, f"$G_{k+1}$", ec=C["blue"], bold=True)
        if k:
            arrow(ax, (x - gap, 0.51), (x, 0.51), style="-|>", lw=0.7)
    ax.text(0.56 + 2 * (w + gap) - gap, 0.70, "TT cores, rank $R$",
            ha="center", fontsize=FS, color=C["blue"])

    box(ax, 0.85, 0.36, 0.13, 0.30, "Row\nrecon-\nstruction", ec=C["green"])
    arrow(ax, (0.56 + 4 * w + 3 * gap, 0.51), (0.85, 0.51))

    ax.text(0.5, 0.24, r"storage $O(R^2 m K)$ instead of $O(\prod_k m_k)$",
            ha="center", fontsize=FS, style="italic", color="#555555")
    ax.text(0.5, 0.13, "Partial contractions are cached; a row is rebuilt from prefix products.",
            ha="center", fontsize=FS, style="italic", color="#555555")
    return save(fig, "tt_workflow11")


# ------------------------------------------------------------------ lookup ---
def lookup11():
    fig, ax = canvas(h=2.3)
    ax.text(0.24, 0.88, "Dense lookup", ha="center", fontsize=FS + 0.5, fontweight="bold")
    box(ax, 0.04, 0.50, 0.16, 0.24, "index $i$", ec=C["black"])
    box(ax, 0.26, 0.50, 0.18, 0.24, "row $\\mathcal{W}_i$\n(read $d$)", ec=C["vermillion"])
    arrow(ax, (0.20, 0.62), (0.26, 0.62))
    ax.text(0.24, 0.40, "$O(d)$ per lookup,\nno reuse", ha="center", fontsize=FS,
            color=C["vermillion"], style="italic")

    ax.text(0.76, 0.88, "TT reconstruction", ha="center", fontsize=FS + 0.5, fontweight="bold")
    box(ax, 0.56, 0.50, 0.14, 0.24, "index $i$", ec=C["black"])
    box(ax, 0.74, 0.50, 0.22, 0.24, "prefix\nproducts\n(cached)", ec=C["green"], bold=True)
    arrow(ax, (0.70, 0.62), (0.74, 0.62))
    arrow(ax, (0.85, 0.50), (0.85, 0.38), color=C["blue"])
    box(ax, 0.74, 0.14, 0.22, 0.24, "row from\nprefix $\\cdot$ suffix", ec=C["blue"])
    ax.text(0.76, 0.045, "shared prefixes are contracted once", ha="center",
            fontsize=FS, color=C["green"], style="italic")
    return save(fig, "new_lookup11")


# ----------------------------------------------------------------- reorder ---
def reorder11():
    fig, ax = canvas(h=2.3)
    ax.text(0.5, 0.90, "Index ordering changes which prefixes can be shared",
            ha="center", fontsize=FS + 0.5)

    rows = [("sorted", [0, 1, 2, 3, 4, 5], C["green"]),
            ("arbitrary", [3, 0, 5, 1, 4, 2], C["vermillion"]),
            ("frequency", [0, 0, 1, 1, 2, 2], C["orange"])]
    for j, (lab, seq, col) in enumerate(rows):
        y = 0.62 - j * 0.20
        ax.text(0.02, y + 0.045, lab, fontsize=FS, va="center")
        for k, v in enumerate(seq):
            ax.add_patch(Rectangle((0.16 + k * 0.055, y), 0.05, 0.09,
                                   facecolor=col, alpha=0.30, edgecolor=col, linewidth=0.6))
            ax.text(0.185 + k * 0.055, y + 0.045, str(v), fontsize=5.5,
                    ha="center", va="center")

    ax.text(0.5, 0.10, "Prefix sharing depends on adjacency, so the ordering is part of the design, "
                       "not an implementation detail.", ha="center", fontsize=FS,
            style="italic", color="#555555")
    ax.text(0.5, 0.015, "Measured reuse is reported against a frequency ordering that uses no grid "
                        "information.", ha="center", fontsize=FS, style="italic", color="#555555")
    return save(fig, "new_new_reordering11")


# ------------------------------------------------------------------- graph ---
def graph11():
    fig, ax = canvas(h=2.4)
    import numpy as np
    rng = np.random.default_rng(0)
    n = 9
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.c_[0.30 + 0.20 * np.cos(ang), 0.52 + 0.30 * np.sin(ang)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.18:
                ax.plot(*zip(pos[i], pos[j]), color="#BBBBBB", linewidth=0.5, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=22, color=C["blue"], zorder=3,
               edgecolor="white", linewidth=0.4)
    for i, (x, y) in enumerate(pos):
        ax.text(x, y - 0.075, str(i), fontsize=5.5, ha="center", color="#333333")

    ax.text(0.30, 0.93, "co-occurrence graph", ha="center", fontsize=FS)
    arrow(ax, (0.53, 0.52), (0.62, 0.52), lw=1.2)

    ax.text(0.80, 0.93, "identifier ordering", ha="center", fontsize=FS)
    order = [0, 2, 4, 1, 3, 5, 6, 7, 8]
    for k, v in enumerate(order):
        ax.add_patch(Rectangle((0.63 + k * 0.036, 0.44), 0.032, 0.16,
                               facecolor=C["green"], alpha=0.30,
                               edgecolor=C["green"], linewidth=0.6))
        ax.text(0.646 + k * 0.036, 0.52, str(v), fontsize=5.5, ha="center", va="center")

    ax.text(0.5, 0.20, "Community structure suggests an ordering; whether the suggestion buys "
                       "anything is measured.", ha="center", fontsize=FS,
            style="italic", color="#555555")
    ax.text(0.5, 0.10, "It does not outperform frequency ordering, and the result is reported "
                       "as such.", ha="center", fontsize=FS, style="italic", color="#555555")
    return save(fig, "new_graph11")


# ------------------------------------------------------------------ buffer ---
def embedding_cache21():
    fig, ax = canvas(h=2.5)
    box(ax, 0.03, 0.58, 0.19, 0.24, "producer\n(gradient\nstep $v$)", ec=C["blue"], bold=True)
    box(ax, 0.30, 0.58, 0.19, 0.24, "cache\ntag $v$", ec=C["orange"])
    box(ax, 0.57, 0.58, 0.19, 0.24, "consumer\n(prefetch)", ec=C["green"])
    arrow(ax, (0.22, 0.70), (0.30, 0.70))
    arrow(ax, (0.49, 0.70), (0.57, 0.70))

    box(ax, 0.30, 0.16, 0.46, 0.24, "hazard: consumer reads a row whose parameters changed",
        ec=C["vermillion"], lw=1.0)
    arrow(ax, (0.66, 0.58), (0.66, 0.40), color=C["vermillion"], ls="--")
    arrow(ax, (0.40, 0.40), (0.40, 0.58), color=C["vermillion"], ls="--")

    box(ax, 0.80, 0.58, 0.17, 0.24, "version\ncheck", ec=C["purple"], bold=True)
    arrow(ax, (0.76, 0.70), (0.80, 0.70))

    ax.text(0.5, 0.06, "A cached value must match the parameter version it was built from; "
                       "the check is what makes the overlap safe.",
            ha="center", fontsize=FS, style="italic", color="#555555")
    return save(fig, "embedding_cache21")


def main() -> int:
    made, failed = [], []
    for fn in (overview11, dlrm_workflow11, tt_workflow11, lookup11,
               reorder11, graph11, embedding_cache21):
        try:
            p = fn()
            made.append(p.name)
        except Exception as exc:                       # noqa: BLE001
            failed.append((fn.__name__, f"{type(exc).__name__}: {exc}"))

    print(f"  输出目录: {OUT}")
    print(f"  样式: IEEE 双栏 {pubstyle.WIDTH_DOUBLE} in, Okabe-Ito, 600 dpi")
    print()
    for m in made:
        print(f"    OK {m}")
    for f in failed:
        print(f"    XX {f}")

    print()
    print("  === 灰度与尺寸核 ===")
    for m in made:
        png = OUT / (Path(m).stem + ".png")
        if png.exists():
            g = pubstyle.grayscale_check(png)
            print(f"    {Path(m).stem:<26} 层次 {g.get('distinct_levels'):<3} "
                  f"对比 {g.get('ink_contrast_sd')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
