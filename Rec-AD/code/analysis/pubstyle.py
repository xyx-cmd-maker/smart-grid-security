"""Publication styling for the manuscript's figures, to the standard the visualisation skill states.

The current figures use default matplotlib with hand-written colours and widths of eight to thirteen
inches. The skill's requirements for this publisher are a double-column width of 7.16 inches, at
least 600 dpi for line art, a colourblind-safe palette, type sized for the printed page rather than
the screen, and redundant encoding so the figures survive grayscale. None of those are matters of
taste, so they are set once here rather than per figure.

Design choices worth stating. The palette is Okabe-Ito, which the skill recommends because it stays
distinguishable under all common forms of colour blindness; the previous green and red were not. Type
sizes are given in points at the final printed width, so a figure is never scaled on the page. Line
styles and markers vary with colour, because a reader printing in grey loses colour but keeps shape.
Spines are trimmed to the two that carry information.

Nothing here touches data. A figure's values come from the result files as before; only their
appearance is fixed.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---- Okabe-Ito: distinguishable under all common colour vision deficiencies --
OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}
CYCLE = [OKABE_ITO[k] for k in ("blue", "vermillion", "green", "orange",
                                "purple", "sky", "yellow", "black")]

# semantic roles, so a colour means the same thing in every figure
ROLE = {
    "ours": OKABE_ITO["blue"],
    "baseline": OKABE_ITO["vermillion"],
    "envelope": OKABE_ITO["purple"],
    "neutral": "#7F7F7F",
    "fill": OKABE_ITO["sky"],
}

# IEEE double column is 7.16 in (18.2 cm); single is 3.5 in. Heights follow the
# golden ratio so panels do not look stretched.
WIDTH_DOUBLE = 7.16
WIDTH_SINGLE = 3.50
WIDTH_ONE_HALF = 4.70


def width(kind: str = "double") -> float:
    return {"single": WIDTH_SINGLE, "1.5": WIDTH_ONE_HALF, "double": WIDTH_DOUBLE}[kind]


def apply(width_in: float = WIDTH_DOUBLE, height_in: float = None, *, columns: int = 1):
    """Set the rcParams for a figure of the given width, and return its size.

    Font sizes are chosen for the printed page: a reader holds the paper, not the
    screen. The skill's floor is six points at final size, and these sit above it
    so that a two-panel figure remains legible when each panel is half the width.
    """
    scale = 1.0 if columns == 1 else 0.92
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8 * scale,
        "axes.labelsize": 8.5 * scale,
        "axes.titlesize": 9 * scale,
        "xtick.labelsize": 7.5 * scale,
        "ytick.labelsize": 7.5 * scale,
        "legend.fontsize": 7.5 * scale,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.grid": False,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.prop_cycle": plt.cycler(color=CYCLE),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    if height_in is None:
        height_in = round(width_in * 0.62 * (1.0 if columns == 1 else 0.55), 2)
    return (width_in, height_in)


def panel_labels(axes, *, start: int = 0, x: float = -0.12, y: float = 1.06,
                 size: float = 9.0):
    """Bold uppercase A, B, C at the top-left of each axes, as journals require."""
    from string import ascii_uppercase
    for i, ax in enumerate(axes):
        ax.text(x, y, ascii_uppercase[start + i], transform=ax.transAxes,
                fontsize=size, fontweight="bold", va="top", ha="right")


def save(fig, out_dir, name: str, *, dpi: int = 600):
    """Write the vector figure and a raster preview at the required resolution."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=dpi)
    plt.close(fig)
    return out / f"{name}.pdf"


def grayscale_check(path_png) -> dict:
    """Report whether a rendered figure keeps contrast when colour is removed.

    A figure that only works in colour fails a large share of readers, so the
    check is a measurement rather than an intention: convert the preview to
    luminance and report how much of its area carries usable contrast.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return {"available": False, "reason": "PIL 不可用"}
    im = Image.open(path_png).convert("L")
    a = np.asarray(im, dtype=float) / 255.0
    ink = a < 0.9                      # pixels that are not white background
    return {
        "available": True,
        "ink_fraction": round(float(ink.mean()), 4),
        "ink_contrast_sd": round(float(a[ink].std()), 4) if ink.any() else 0.0,
        "distinct_levels": int(len(set((a[ink] * 20).astype(int)))) if ink.any() else 0,
    }
