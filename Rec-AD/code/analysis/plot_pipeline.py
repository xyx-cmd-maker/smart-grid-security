"""Generate Fig. 11: the data generation pipeline.

The revised manuscript lists a figure for the data pipeline and no script produced it. This is
that script.

The figure is a schematic, which is the stated exception to this project's rule that figures be
regenerated from result files: a pipeline diagram has no results to be derived from. So it is
made verifiable instead of decorative. Every number on it is read from the dataset contract at
build time, and the script fails on a missing field rather than drawing a blank box.

What it has to convey, in the order the reviewer asked (R1.5, R1.7):

  1. the grid and its state
  2. the measurement map, which is what the attack exploits
  3. the three attack families and what distinguishes them
  4. the observation contract each detector family consumes

The load-bearing panel is the second one. `h_matrix` is stored in the dataset, so the figure can
state the structural fact the paper rests on with the dataset's own numbers: H is (m x n) with
full column rank, so the stealthy injection `a = Hc` occupies an n-dimensional subspace of the
m-dimensional measurement space, and the remaining m - n directions are the only ones a residual
test can see. For IEEE 118 that is 117 directions invisible out of 297.

Reading the contract needed correcting twice. `meta` is a JSON string, not a mapping, and
`attack_type` holds integers, not names; the first version assumed both and failed on the first
field it touched.

Usage
-----
    python analysis/plot_pipeline.py --data data_cache/ieee118.npz [--out figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

from data.contract import ATTACK_NAMES  # noqa: E402

# Order and colour chosen so the stealthy row is visually distinct: it is the paper's problem.
FAMILY_ROWS = [
    (0, "benign", "no injection", "#3a6ea5"),
    (1, "random", "drawn outside $\\mathrm{col}(H)$; large residual", "#c46a1f"),
    (3, "load_redist", "$Hc$ plus a sparse noise-scaled bias", "#7a5ea8"),
    (2, "stealthy", "$a = Hc$: inside the column space", "#b03030"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_cache/ieee118.npz")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = CODE / data_path
    if not data_path.exists():
        raise SystemExit(f"  ABORT: {data_path} does not exist")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    d = np.load(data_path, allow_pickle=True)
    if "meta" not in d:
        raise SystemExit("  ABORT: no 'meta' in the dataset")
    meta = json.loads(str(d["meta"]))          # stored as a JSON string, not a mapping
    for key in ("topology_source", "n_bus", "n_branch", "n_measurements", "n_state",
                "state_step_sigma", "noise_sigma", "window"):
        if key not in meta:
            raise SystemExit(f"  ABORT: meta lacks {key!r}")
    if "h_matrix" not in d:
        raise SystemExit("  ABORT: no 'h_matrix'; the structural panel needs it")

    n_bus = meta["n_bus"]
    n_br = meta["n_branch"]
    m = meta["n_measurements"]
    n = meta["n_state"]
    window = meta["window"]
    src = meta["topology_source"]
    step_sigma = meta["state_step_sigma"]
    noise_sigma = meta["noise_sigma"]

    H = np.asarray(d["h_matrix"], dtype=np.float64)
    if H.shape != (m, n):
        raise SystemExit(f"  ABORT: h_matrix is {H.shape}, meta says ({m}, {n})")
    rank = int(np.linalg.matrix_rank(H))
    invisible = rank
    visible = m - rank

    dense = d["dense"]
    n_samples = int(dense.shape[0])
    at = d["attack_type"]
    counts = {int(v): int((at == v).sum()) for v in np.unique(at)}
    missing = [c for c, _, _, _ in FAMILY_ROWS if c not in counts]
    if missing:
        raise SystemExit(f"  ABORT: attack_type lacks {missing}")

    print(f"  data       {data_path.name}")
    print(f"  topology   {src}, {n_bus} buses, {n_br} branches")
    print(f"  contract   {n_samples} samples, m={m}, n={n}, window={window}")
    print(f"  H          {H.shape}, rank {rank} -> {visible} visible / {invisible} invisible")
    print(f"  families   { {ATTACK_NAMES[c]: counts[c] for c, _, _, _ in FAMILY_ROWS} }")

    fig, ax = plt.subplots(figsize=(12.2, 7.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 68)
    ax.axis("off")

    def box(x, y, w, h, title, lines, edge, face="#ffffff", tsize=10, lsize=8.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.6,rounding_size=1.1",
                                    linewidth=1.5, edgecolor=edge, facecolor=face, zorder=2))
        ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top",
                fontsize=tsize, fontweight="bold", color=edge, zorder=3)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 6.2 - i * 3.0, ln, ha="center", va="top",
                    fontsize=lsize, color="#222222", zorder=3)

    def arrow(x1, y1, x2, y2, color="#555555", lw=1.5, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=lw, color=color,
                                     linestyle=ls, zorder=1, shrinkA=1, shrinkB=1))

    # -- row 1: grid, state, measurements -------------------------------------
    box(2, 50, 27, 13, f"Power system ({n_bus}-bus)",
        [f"{src}", f"{n_br} branches", f"$m$ = {m} measurements"], "#3a6ea5")
    box(36, 50, 27, 13, "State trajectory",
        [f"$x_t$: random walk, $\\sigma$ = {step_sigma}",
         "bounded step, so a jump is visible",
         f"$n$ = {n} state variables"], "#3a6ea5")
    box(70, 50, 28, 13, "Measurements",
        [f"$z_t = h(x_t) + e_t$,  $\\sigma_e$ = {noise_sigma}",
         f"window of {window} snapshots",
         f"{n_samples} samples total"], "#3a6ea5")
    arrow(29.6, 56.5, 35.4, 56.5)
    arrow(63.6, 56.5, 69.4, 56.5)

    # -- row 2: the structural panel (the load-bearing one) -------------------
    ax.add_patch(FancyBboxPatch((2, 33.5), 96, 12.5,
                                boxstyle="round,pad=0.6,rounding_size=1.1",
                                linewidth=2.0, edgecolor="#b03030", facecolor="#fff6f6", zorder=2))
    ax.text(50, 44.6, "Why the stealthy family is structurally invisible",
            ha="center", va="top", fontsize=10.5, fontweight="bold", color="#b03030", zorder=3)
    ax.text(50, 41.2,
            f"$H$ is ({m} x {n}) with rank {rank}.  An injection $a = Hc$ is a legal measurement "
            f"under some state $x + c$,",
            ha="center", va="top", fontsize=8.6, color="#222222", zorder=3)
    ax.text(50, 38.4,
            f"so it lies in a {rank}-dimensional subspace of the {m}-dimensional measurement space.  "
            f"Only the other {visible} directions leave a residual.",
            ha="center", va="top", fontsize=8.6, color="#222222", zorder=3)
    # A bar showing the split, which is the number a reader should remember.
    bx, bw, bh = 8, 84, 1.7
    ax.add_patch(plt.Rectangle((bx, 35.2), bw * visible / m, bh,
                               facecolor="#2a7f4f", edgecolor="none", zorder=3))
    ax.add_patch(plt.Rectangle((bx + bw * visible / m, 35.2), bw * invisible / m, bh,
                               facecolor="#b03030", edgecolor="none", zorder=3))
    ax.text(bx + bw * visible / m / 2, 34.4, f"{visible} directions visible to the residual test",
            ha="center", va="top", fontsize=7.8, color="#2a7f4f", zorder=4)
    ax.text(bx + bw * (visible + invisible / 2) / m, 34.4, f"{invisible} invisible",
            ha="center", va="top", fontsize=7.8, color="#b03030", zorder=4)
    arrow(50, 49.4, 50, 46.6, color="#b03030", lw=2.0)

    # -- row 3: the families --------------------------------------------------
    ax.text(2, 31.4, "Attack injection   $z'_t = z_t + a$", ha="left", va="top",
            fontsize=10, fontweight="bold", color="#333333", zorder=3)
    ax.text(98, 31.4, "families differ only in where $a$ is drawn from", ha="right", va="top",
            fontsize=8.2, style="italic", color="#666666", zorder=3)
    for i, (code, name, desc, col) in enumerate(FAMILY_ROWS):
        x = 2 + i * 24.5
        ax.add_patch(FancyBboxPatch((x, 20.0), 22.5, 9.0,
                                    boxstyle="round,pad=0.4,rounding_size=0.9",
                                    linewidth=1.3, edgecolor=col, facecolor="#ffffff", zorder=3))
        ax.text(x + 11.25, 27.2, f"{name}    n = {counts[code]}", ha="center", va="top",
                fontsize=9.0, fontweight="bold", color=col, zorder=4)
        ax.text(x + 11.25, 24.3, desc, ha="center", va="top",
                fontsize=7.2, color="#333333", zorder=4)
        ax.text(x + 11.25, 21.6, f"({ATTACK_NAMES[code]})", ha="center", va="top",
                fontsize=6.4, color="#888888", style="italic", zorder=4)
    arrow(50, 33.0, 50, 29.6, color="#b03030", lw=2.0)

    # -- row 4: the observation contract --------------------------------------
    ax.text(2, 19.4, "Observation contract: what each detector family is allowed to see",
            ha="left", va="top", fontsize=10, fontweight="bold", color="#2a7f4f", zorder=3)
    box(2, 1.8, 30, 14.0, "Temporal",
        [f"window: {n_samples} x {window} x {m}",
         "CNN / LSTM / Transformer",
         "the step at $t_0$ is the signature"], "#2a7f4f")
    box(35, 1.8, 30, 14.0, "Spatial / residual",
        [f"dense: last snapshot, {m} dims",
         "$\\| (I - P) z \\|$, rank $m - n$",
         "blind when $a = Hc$"], "#2a7f4f")
    box(68, 1.8, 30, 14.0, "Graph",
        [f"node_feat: {n_samples} x {n_bus} x 4",
         "adjacency: {0} x {0}".format(n_bus),
         "GCN / GAT baselines"], "#2a7f4f")
    for x in (17, 50, 83):
        arrow(x, 18.4, x, 16.2, color="#2a7f4f")

    ax.set_title("Fig. 11  Data generation pipeline: where the attack enters, and what each "
                 "detector family is allowed to see",
                 fontsize=11.8, fontweight="bold", pad=12)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = CODE / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig11_pipeline.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'fig11_pipeline.pdf'} and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
