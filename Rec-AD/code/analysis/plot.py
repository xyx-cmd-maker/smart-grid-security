"""Figure generation for the revision.

Consumes the result JSON files written by the experiment scripts and produces
the figures the revised manuscript needs. Every figure is derived from a
result file rather than hand-assembled, so a re-run updates the figures
without anyone having to remember what changed.

Figures produced (numbers match the revised manuscript):
  fig12_deployability   accuracy versus peak memory, with the envelope drawn
  fig13_rank_sweep      compression versus detection quality
  fig14_scaling         benefit versus grid size
  fig15_reorder         reordering strategies: reuse rate and F1
  fig16_ablation        component ablation bars

Usage
-----
    python analysis/plot.py --results results --out figures
    python analysis/plot.py --results results --out figures --only rank_sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load every result JSON, keyed by the experiment name."""
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                out[path.stem] = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipped {path.name}: {exc}")
    return out


def _records(blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(blob, dict):
        return []
    if "summary" in blob and isinstance(blob["summary"], list):
        return blob["summary"]
    if "records" in blob and isinstance(blob["records"], list):
        return blob["records"]
    return []


def _all_records(blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Like `_records` but also accepts the `rows` key.

    The experiment scripts write `records` or `summary`; the measurement scripts
    write `rows`. Widening the shared reader to accept all three changed what
    three other figures plotted, because they began to pick up measurement rows
    they were never meant to see. So the wider reader is separate and only the
    deployability figure, which needs resource measurements, uses it.
    """
    if not isinstance(blob, dict):
        return []
    for key in ("summary", "records", "rows"):
        if key in blob and isinstance(blob[key], list):
            return blob[key]
    return []


def _load_registry() -> Dict[str, str]:
    """Map each detector to the result file the registry names as its source.

    `slots.json` records, for every published number, which file it came from.
    The deployability figure needs one number per detector from the detection
    side, and several files carry that field with values at the false-alarm rate,
    so picking by field name would draw a meaningless figure that looks like data.
    The registry is the authority the manuscript uses, so the figure uses it too.

    Returns an empty mapping when the registry is unavailable, which makes the
    caller skip the figure rather than draw it from the wrong source.
    """
    path = Path(__file__).resolve().parents[2] / "slots.json"
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    base = blob.get("baselines", {})
    out: Dict[str, str] = {}
    for key, spec in base.items():
        if not key.endswith(".stealthy") or not isinstance(spec, dict):
            continue
        det = key.split(".")[1]
        src = str(spec.get("source") or "").strip()
        if src:
            out[det] = src
    return out


def _get(rec: Dict[str, Any], key: str) -> Optional[float]:
    """Read `key` as either a raw value or an aggregated `key_mean`."""
    if key in rec and isinstance(rec[key], (int, float)):
        return float(rec[key])
    agg = f"{key}_mean"
    if agg in rec and isinstance(rec[agg], (int, float)):
        return float(rec[agg])
    return None


def _save(fig, out_dir: Path, name: str) -> Optional[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_deployability(results: Dict[str, Dict[str, Any]],
                      out_dir: Path,
                      budget: Optional[Dict[str, Any]] = None) -> None:
    """Accuracy versus peak memory — the paper's lead figure.

    The envelope is drawn as a region, not a line: a method outside it cannot
    be deployed at a substation regardless of its accuracy.
    """
    # Two quantities per detector, from two different places, joined on the
    # detector name.
    #
    # Memory comes from the measurement files, which name what they measured.
    # Detection comes from the file the registry records as the source of that
    # value. Picking a detection file by field name alone is wrong here: several
    # files carry the field and the ones that do hold values at the false-alarm
    # rate, because they are smoke tests and superseded runs. A figure drawn from
    # those would look like data and mean nothing, so the registry decides.
    mem: Dict[str, float] = {}
    for blob in results.values():
        for r in _all_records(blob):
            n = r.get("detector") or r.get("model")
            x = _get(r, "peak_mib")
            if not n or x is None:
                continue
            # Only the inference protocol. The envelope file also holds
            # training-scale rows at batch 256 and 4096 whose memory is roughly
            # five times larger, and taking the first match let one of those
            # stand in for an inference measurement.
            batch = r.get("batch")
            if isinstance(batch, (int, float)) and float(batch) != 1.0:
                continue
            mem.setdefault(str(n), float(x))

    det: Dict[str, float] = {}
    registry = _load_registry()
    for det_name in registry:
        src = registry[det_name]
        blob = results.get(src)
        if not isinstance(blob, dict):
            continue
        for r in _all_records(blob):
            y = _get(r, "detect_rate_stealthy")
            if y is not None:
                det.setdefault(det_name, float(y))

    if not mem or not det:
        print(f"  deployability: memory for {len(mem)} methods, detection for {len(det)}")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    budget = budget or {}
    vram = float(budget.get("vram_gb", 8)) * 1024

    plotted = 0
    xs: List[float] = []
    for r in [{"detector": k, "peak_mib": v} for k, v in mem.items()]:
        name = r["detector"]
        x = r["peak_mib"]
        y = det.get(name)
        if x is None or y is None:
            continue
        plotted += 1
        xs.append(float(x))
        label = name
        fits = x <= vram
        ax.scatter(x, y, s=110, marker="o" if fits else "X",
                   color="#2a7f4f" if fits else "#b03030",
                   edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(7, 5),
                    fontsize=8)

    # The measured values sit far inside the budget, so an axis that spans the
    # budget flattens every point against the left edge and makes the envelope
    # the subject rather than the comparison. Draw the measured range, and mark
    # where the budget lies relative to it.
    ax.set_xlim(0, max(vram, max(xs) * 1.15))
    ax.axvspan(vram, ax.get_xlim()[1], color="#b03030", alpha=0.07)
    ax.axvline(vram, color="#b03030", linestyle="--", linewidth=1.2)
    ax.text(vram, ax.get_ylim()[1], "  substation envelope",
            color="#b03030", fontsize=9, va="top")
    ax.set_xlabel("Peak inference memory (MB)")
    ax.set_ylabel("Stealthy-attack detection rate")
    ax.set_title("Deployability frontier: detection quality against the memory a node can hold")
    ax.grid(alpha=0.25)
    _save(fig, out_dir, "fig12_deployability")


def fig_rank_sweep(results: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    recs: List[Dict[str, Any]] = []
    for name, blob in results.items():
        if name.startswith("rank_sweep"):
            recs.extend(_records(blob))
    if not recs:
        print("  rank_sweep: no results")
        return

    pts = []
    for r in recs:
        c = _get(r, "compression_ratio")
        f = _get(r, "f1")
        if c is None or f is None:
            continue
        pts.append((c, f, r.get("config", "?"), _get(r, "tt_rank")))
    if not pts:
        return
    pts.sort(key=lambda t: t[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, "-o", color="#2b6cb0", linewidth=1.8, markersize=7, zorder=3)

    dense = next((p for p in pts if p[2] == "dense"), None)
    if dense:
        ax.axhline(dense[1], color="#888", linestyle=":", linewidth=1.2)
        ax.annotate("uncompressed ceiling", (xs[-1], dense[1]),
                    textcoords="offset points", xytext=(-6, 8),
                    fontsize=9, color="#666", ha="right")

    for x, y, label, _ in pts:
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, -14),
                    fontsize=8, ha="center")

    ax.set_xscale("log")
    ax.set_xlabel("Embedding-table compression ratio (log scale)")
    ax.set_ylabel("Detection F1")
    ax.set_title("Compression versus detection quality: where loss begins")
    ax.grid(alpha=0.25, which="both")
    _save(fig, out_dir, "fig13_rank_sweep")


def fig_scaling(results: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    recs: List[Dict[str, Any]] = []
    for name, blob in results.items():
        if name.startswith("scaling"):
            recs.extend(_records(blob))
    recs = [r for r in recs if _get(r, "n_bus") is not None]
    if not recs:
        print("  scaling: no results")
        return
    recs.sort(key=lambda r: _get(r, "n_bus") or 0)

    buses = [_get(r, "n_bus") for r in recs]
    comp = [_get(r, "compression_ratio") for r in recs]
    f1 = [_get(r, "f1") for r in recs]
    stealth = [_get(r, "detect_rate_stealthy") for r in recs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    if all(c is not None for c in comp):
        axes[0].plot(buses, comp, "-o", color="#2a7f4f", linewidth=1.8, markersize=7)
        axes[0].set_xlabel("Grid size (buses)")
        axes[0].set_ylabel("Compression ratio")
        axes[0].set_title("Compression grows with grid size")
        axes[0].grid(alpha=0.25)

    if all(s is not None for s in stealth):
        axes[1].plot(buses, stealth, "-s", color="#b06030", linewidth=1.8,
                     markersize=7, label="stealthy")
        if all(f is not None for f in f1):
            axes[1].plot(buses, f1, "-^", color="#2b6cb0", linewidth=1.8,
                         markersize=7, label="F1")
        axes[1].set_xlabel("Grid size (buses)")
        axes[1].set_ylabel("Detection rate")
        axes[1].set_title("Detection performance across topologies")
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.25)

    fig.suptitle("Scaling study: a growing benefit is the prediction of the "
                 "memory-bound account", fontsize=11)
    _save(fig, out_dir, "fig14_scaling")


def fig_reorder(results: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    blob = next((b for n, b in results.items() if n.startswith("reorder")), None)
    if not blob:
        print("  reorder: no results")
        return

    detection = _records(blob)
    reuse = blob.get("reuse_diagnostic") or []

    order = ["none", "random", "frequency", "topology"]
    by_order = {r.get("ordering"): r for r in detection if r.get("ordering")}
    reuse_by = {r.get("ordering"): r for r in reuse if "error" not in r}

    present = [k for k in order if k in by_order or k in reuse_by]
    if not present:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(present))

    rates = [reuse_by.get(k, {}).get("reuse_rate", np.nan) for k in present]
    axes[0].bar(x, rates, color="#2b6cb0", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(present, rotation=15)
    axes[0].set_ylabel("TT prefix reuse rate")
    axes[0].set_title("Mechanism: reuse induced by each ordering")
    axes[0].grid(alpha=0.25, axis="y")

    f1s = [_get(by_order.get(k, {}), "f1") for k in present]
    axes[1].bar(x, f1s, color="#2a7f4f", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(present, rotation=15)
    axes[1].set_ylabel("Detection F1")
    axes[1].set_title("Outcome: detection under each ordering")
    axes[1].grid(alpha=0.25, axis="y")

    fig.suptitle("Reordering study: report both panels — a reuse gain with no "
                 "detection effect is a systems result, not the structural claim",
                 fontsize=10)
    _save(fig, out_dir, "fig15_reorder")


def fig_ablation(results: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    recs: List[Dict[str, Any]] = []
    for name, blob in results.items():
        if name.startswith("ablation"):
            recs.extend(_records(blob))
    if not recs:
        print("  ablation: no results")
        return

    labels = [r.get("cell", "?") for r in recs]
    f1 = [_get(r, "f1") for r in recs]
    comp = [_get(r, "compression_ratio") for r in recs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(labels))

    axes[0].bar(x, f1, color="#2b6cb0", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("Detection F1")
    axes[0].set_title("Ablation: detection performance")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(x, comp, color="#2a7f4f", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("Embedding compression ratio")
    axes[1].set_title("Ablation: compression")
    axes[1].grid(alpha=0.25, axis="y")

    fig.suptitle("Component ablation: isolate-at-a-time and cumulative",
                 fontsize=11)
    _save(fig, out_dir, "fig16_ablation")


def fig_fedavg(results: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    recs: List[Dict[str, Any]] = []
    for name, blob in results.items():
        if name.startswith("fedavg"):
            recs.extend(_records(blob))
    if not recs:
        print("  fedavg: no results")
        return

    labels = [r.get("variant", "?") for r in recs]
    emb = [_get(r, "comm_embedding_bytes") for r in recs]
    other = [_get(r, "comm_other_bytes") for r in recs]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(labels))
    emb_kb = [v / 1024 if v else 0.0 for v in emb]
    oth_kb = [v / 1024 if v else 0.0 for v in other]

    ax.bar(x, emb_kb, color="#2b6cb0", alpha=0.9, label="embedding table")
    ax.bar(x, oth_kb, bottom=emb_kb, color="#cccccc", alpha=0.9,
           label="non-embedding (exchanged either way)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Communication per round (KB)")
    ax.set_title("Federated exchange: only the embedding differs between variants")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    _save(fig, out_dir, "fig17_fedavg")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

FIGURES = {
    "deployability": fig_deployability,
    "rank_sweep": fig_rank_sweep,
    "scaling": fig_scaling,
    "reorder": fig_reorder,
    "ablation": fig_ablation,
    "fedavg": fig_fedavg,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    p.add_argument("--config", default=None)
    p.add_argument("--only", default=None,
                   help="comma-separated subset: " + ", ".join(FIGURES))
    p.add_argument("--list", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for k in FIGURES:
            print(k)
        return 0

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run the experiment scripts first.")
        return 1

    results = load_results(results_dir)
    if not results:
        print(f"No result JSON files in {results_dir}")
        return 1

    print(f"Loaded {len(results)} result file(s) from {results_dir}")
    for name in sorted(results):
        n = len(_records(results[name]))
        print(f"  {name:<44} {n:>5} record(s)")
    print()

    budget = None
    if args.config:
        from recad.common import load_config
        budget = (load_config(args.config) or {}).get("budget")
    else:
        default_cfg = CODE_ROOT / "config.yaml"
        if default_cfg.exists():
            from recad.common import load_config
            budget = (load_config(default_cfg) or {}).get("budget")

    wanted = ([s.strip() for s in args.only.split(",")] if args.only
              else list(FIGURES))
    unknown = [w for w in wanted if w not in FIGURES]
    if unknown:
        print(f"Unknown figure(s): {unknown}. Available: {list(FIGURES)}")
        return 1

    print(f"Generating {len(wanted)} figure(s) into {out_dir}")
    for key in wanted:
        fn = FIGURES[key]
        if key == "deployability":
            fn(results, out_dir, budget)
        else:
            fn(results, out_dir)

    print()
    print("Done. Every figure is derived from a result file, so re-running an")
    print("experiment and re-running this script updates the figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
