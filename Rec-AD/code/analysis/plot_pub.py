"""Redraw the manuscript figures to the publisher's figure requirements.

The existing figures were drawn with matplotlib defaults at widths of eight to thirteen inches in
hand-picked colours, two of which form a red-green pair. This produces the same figures at the
double-column width for this publisher, in a palette that survives colour blindness, with the type
sized for the printed page, uncertainty drawn where the result files carry it, and redundant encoding
so the figures remain readable in grayscale.

It reads the same result files as the existing script and writes to a separate directory, so nothing
that currently works is replaced. That separation is deliberate: an earlier change to a shared reader
altered what three unrelated figures plotted, and the cost of a duplicate output directory is lower
than the cost of that failure recurring.

Where a result file carries only a mean, no error bar is invented. Where it carries a standard
deviation, the bar is drawn and the caption is expected to say so.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np

import pubstyle
from seedguard import assert_absolute, harmonise_blobs


def load(results_dir: Path) -> dict:
    out = {}
    for p in sorted(results_dir.glob("*.json")):
        if any(s in p.name for s in ("selftest", "SUPERSEDED", "smoke")):
            continue
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return out


def rows(blob):
    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict):
        for k in ("records", "summary", "rows"):
            if isinstance(blob.get(k), list):
                return blob[k]
    return []


def pick(results, prefix):
    return {k: v for k, v in results.items() if k.startswith(prefix)}


def fig_ablation(results, out):
    """Component ablation: what each component contributes, with seed spread."""
    # The ablation records carry `cell` as their identifier; there is no `model`
    # field, and filtering on one produced an empty figure before this was read.
    recs = []
    for name, blob in pick(results, "ablation_ieee118_local_abl3_topo").items():
        if "summary" in name or "partial" in name:
            continue
        recs = [r for r in rows(blob) if isinstance(r, dict) and r.get("cell")]
    if not recs:
        return None

    cells = {}
    for r in recs:
        cells.setdefault(str(r.get("cell") or r.get("variant") or "?"), []).append(r)
    order, means, sds = [], [], []
    for cell, rs in sorted(cells.items()):
        aucs = [r.get("auc") for r in rs if isinstance(r.get("auc"), (int, float))]
        if not aucs:
            continue
        order.append(cell.replace("cum:", "+").replace("only:", ""))
        means.append(float(np.mean(aucs)))
        sds.append(float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0)
    if not order:
        return None

    size = pubstyle.apply(pubstyle.WIDTH_DOUBLE, height_in=2.5, columns=1)
    fig, ax = plt.subplots(figsize=size)
    x = np.arange(len(order))
    ax.bar(x, means, yerr=sds, capsize=2.5, width=0.62,
           color=pubstyle.ROLE["ours"], edgecolor="white", linewidth=0.4,
           error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=28, ha="right")
    ax.set_ylabel("AUROC (pooled)")
    ax.set_xlabel("Component configuration")
    # The caption names a chance reference, and the caption is right to: every bar here lies within
    # about a hundredth of 0.5, so without the rule the reader cannot tell the scale. Drawn before the
    # bars so that the bars occlude it, matching how the scaling panel draws its own.
    ax.axhline(0.5, color=pubstyle.ROLE["neutral"], linestyle=":", linewidth=0.8, zorder=0)
    lo = min(m - s for m, s in zip(means, sds))
    ax.set_ylim(max(0.0, lo - 0.05), min(1.0, max(m + s for m, s in zip(means, sds)) + 0.03))
    ax.set_title(f"Component ablation, IEEE 118, n = 3 seeds")
    fig.tight_layout(pad=0.3)
    return pubstyle.save(fig, out, "figA_ablation")


def fig_scaling(results, out):
    """Compression and detection against grid size; two panels, shared x.

    The identifier field is `case` and the measures are `compression_ratio` and `auc`. The per-family
    detection rates in these files are deliberately not plotted: this runner selects its threshold by
    maximising F1, which here lands at a false-alarm rate near 0.99, so those rates describe the
    threshold rather than the detector. Compression and pooled AUROC are the quantities the operating
    point does not distort.
    """
    # Only blobs whose seeds agree may be pooled: averaging a one-seed run with a three-seed one
    # makes the error bar describe different things at different points.
    pool = {n: b for n, b in pick(results, "scaling_local").items()
            if "summary" not in n and "partial" not in n}
    kept_names, seeds, problems = assert_absolute(pool)
    if problems:
        print(f"    fig_scaling: refusing to pool -- {problems}")
        return None
    recs = []
    for name in kept_names:
        recs += [r for r in rows(pool[name])
                 if isinstance(r, dict) and (r.get("case") or r.get("n_bus"))]
    if not recs:
        return None

    by = {}
    for r in recs:
        b = int(r.get("case") or r.get("n_bus"))
        by.setdefault(b, []).append(r)
    buses = sorted(by)


    def series(key):
        m, s = [], []
        for b in buses:
            vals = [float(r[key]) for r in by[b] if isinstance(r.get(key), (int, float))]
            m.append(float(np.mean(vals)) if vals else float("nan"))
            s.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        return m, s


    comp, comp_sd = series("compression_ratio")
    auc, auc_sd = series("auc")

    size = pubstyle.apply(pubstyle.WIDTH_DOUBLE, height_in=2.5, columns=2)
    fig, axes = plt.subplots(1, 2, figsize=size)
    x = np.arange(len(buses))

    axes[0].errorbar(x, comp, yerr=comp_sd, fmt="-o", capsize=2.5, linewidth=1.2,
                     markersize=4, color=pubstyle.ROLE["ours"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(b) for b in buses])
    axes[0].set_xlabel("Bus count")
    axes[0].set_ylabel("Embedding compression ratio\n(nominal sizing included)")
    axes[0].set_ylim(0, max(comp) * 1.15 if comp else 1.0)
    axes[0].set_title("Compression is flat in grid size")

    axes[1].errorbar(x, auc, yerr=auc_sd, fmt="--s", capsize=2.5, linewidth=1.2,
                     markersize=4, color=pubstyle.ROLE["baseline"])
    axes[1].axhline(0.5, color=pubstyle.ROLE["neutral"], linestyle=":", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(b) for b in buses])
    axes[1].set_xlabel("Bus count")
    axes[1].set_ylabel("AUROC (pooled)")
    axes[1].set_title("Detection does not track grid size")

    pubstyle.panel_labels(axes)
    fig.tight_layout(pad=0.4)
    return pubstyle.save(fig, out, "figB_scaling")


def fig_deployability(results, out):
    """Detection against peak memory, with the envelope drawn, at batch = 1.

    The measurements live under names that describe what was measured rather than
    under a folder theme, so the prefix is `inference_vram`. Only batch-one rows
    are used: the larger batches in the same files are training-scale runs whose
    memory is not comparable with an inference envelope.
    """
    pts = []
    for name, blob in pick(results, "inference").items():
        for r in rows(blob):
            if not isinstance(r, dict):
                continue
            if r.get("batch") not in (1, 1.0):
                continue
            m = r.get("peak_mib", r.get("peak_vram_mb", r.get("peak_vram_mib")))
            v = r.get("detect_rate_stealthy", r.get("stealthy"))
            if isinstance(m, (int, float)) and isinstance(v, (int, float)):
                pts.append((float(m), float(v), str(r.get("name") or r.get("model") or "?")))
    if not pts:
        return None

    size = pubstyle.apply(pubstyle.WIDTH_DOUBLE, height_in=2.6, columns=1)
    fig, ax = plt.subplots(figsize=size)
    for m, v, lab in pts:
        ax.scatter(m, v, s=26, color=pubstyle.ROLE["ours"], edgecolor="black",
                   linewidth=0.4, zorder=3)
        ax.annotate(lab, (m, v), textcoords="offset points", xytext=(4, 3), fontsize=6.5)
    env = 8192.0
    ax.axvline(env, color=pubstyle.ROLE["envelope"], linestyle="--", linewidth=1.0)
    ax.text(env, ax.get_ylim()[1], " envelope", rotation=90, va="top", ha="left",
            fontsize=6.5, color=pubstyle.ROLE["envelope"])
    ax.set_xlabel("Peak inference memory (MiB), batch = 1")
    ax.set_ylabel("Detection rate, stealthy family")
    ax.set_title(f"Deployability envelope, {len(pts)} measured configurations")
    fig.tight_layout(pad=0.3)
    return pubstyle.save(fig, out, "figC_deployability")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE.parent / "results"))
    ap.add_argument("--out", default=str(HERE.parent / "figures" / "pub"))
    a = ap.parse_args()

    results = load(Path(a.results))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"  载入 {len(results)} 个结果文件 -> 输出 {out}")
    print(f"  样式: IEEE 双栏 {pubstyle.WIDTH_DOUBLE} 英寸, Okabe-Ito, 600 dpi")

    made, skipped = [], []
    for fn in (fig_ablation, fig_scaling, fig_deployability):
        try:
            p = fn(results, out)
        except Exception as exc:                      # noqa: BLE001
            skipped.append((fn.__name__, f"{type(exc).__name__}: {exc}"))
            continue
        (made if p else skipped).append(p.name if p else (fn.__name__, "无可用数据"))

    print()
    for m in made:
        print(f"    OK {m}")
    for s in skipped:
        print(f"    -- {s}")

    # grayscale report for each new figure
    print()
    print("  === 灰度检查 ===")
    for m in made:
        png = out / (Path(m).stem + ".png")
        if png.exists():
            g = pubstyle.grayscale_check(png)
            print(f"    {Path(m).stem:<22} 层次 {g.get('distinct_levels')}  "
                  f"对比 {g.get('ink_contrast_sd')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
