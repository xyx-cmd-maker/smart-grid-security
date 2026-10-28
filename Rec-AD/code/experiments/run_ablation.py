"""P0-2 — component ablation for Rec-AD.

Answers Reviewer 1 (comment 4) and Reviewer 2 (comment 3): "no ablation
studies are provided to isolate the contribution of each component."

Two views are produced, because they answer different questions:

  one-at-a-time  -- turn each component on by itself. Shows whether a
                    component does anything at all on its own.
  cumulative     -- none -> +eff_tt -> +reorder -> +pipeline. Shows how the
                    gains stack, which is what the contribution list implies.

Report both. A component can be worthless alone yet essential in combination
(suppressing one would misrepresent it), and the reverse is equally possible.

Usage
-----
    python experiments/run_ablation.py --data data_cache/ieee118.npz
    python experiments/run_ablation.py --data ... --quick
    python experiments/run_ablation.py --data ... --check-backend
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# The thread count changes the low-order bits of every reduction in this pipeline, so it is set
# here rather than left to the caller's environment. Without these lines a repeat comparison
# cannot be attributed to anything, because the pool size would differ between runs.
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch as _torch
_torch.set_num_threads(1)
try:
    _torch.set_num_interop_threads(1)
except Exception:
    pass  # already initialised, or this build has no interop pool

# Allow `python experiments/run_ablation.py` as well as `-m experiments.run_ablation`.
# Running a script directly puts its own directory on sys.path, not the root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._harness import ExperimentContext
from recad.model import RecADConfig

COMPONENTS = ("eff_tt", "reorder", "pipeline")

METRICS = [
    "accuracy", "precision", "recall", "f1", "auc", "fpr",
    "tpr_at_fpr_1pct",
    "detect_rate_random", "detect_rate_stealthy", "detect_rate_load_redist",
    "compression_ratio", "embedding_bytes", "n_parameters",
    "latency_ms_p50", "latency_ms_p99", "peak_vram_mb",
    "energy_j_per_inference", "train_seconds", "fit_seconds",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--tier", default=None)
    p.add_argument("--seeds", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--measure-iters", type=int, default=None)
    p.add_argument("--no-energy", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--orderings", default="topology",
                   help="index ordering used throughout (default: topology)")
    p.add_argument("--target-fpr", type=float, default=0.01,
                   help="calibrate every cell's threshold to this benign "
                        "false-alarm rate on validation before reporting the "
                        "per-attack-family rates. Required for the family "
                        "breakdown to mean anything: with max_f1 a chi-square "
                        "detector sits at ~50%% false alarms and appears to "
                        "'detect' every family, including the ones it is blind "
                        "to. Negative falls back to max_f1.")
    p.add_argument("--check-backend", action="store_true",
                   help="verify the TT reference against a dense table and exit")
    return p.parse_args()


def build_cells(cumulative: bool) -> List[Tuple[str, Dict[str, bool]]]:
    """Configuration cells for the ablation table."""
    cells: List[Tuple[str, Dict[str, bool]]] = []

    # 1. All off (dense embedding, raw order, sequential training).
    cells.append(("none", {c: False for c in COMPONENTS}))

    # 2. One at a time.
    for comp in COMPONENTS:
        name = f"only:{comp}"
        flags = {c: (c == comp) for c in COMPONENTS}
        cells.append((name, flags))

    # 3. Cumulative build-up in the order the contribution list presents them.
    if cumulative:
        running = {c: False for c in COMPONENTS}
        for comp in COMPONENTS:
            running = dict(running)
            running[comp] = True
            cells.append((f"cum:{comp}", dict(running)))

    # 4. Full system.
    cells.append(("full", {c: True for c in COMPONENTS}))
    return cells


def check_backend(ctx: ExperimentContext) -> int:
    """Verify the TT table's lookup against its own independent contraction."""
    from recad.tt_table import TTEmbeddingTable

    rows = ctx.spec.sparse_total_rows
    rank = int((ctx.cfg.get("tt", {}) or {}).get("rank", 32))
    table = TTEmbeddingTable(rows, 16, rank=rank)
    stats = table.verify_against_dense(check_rows=min(rows, 512))
    print("TT backend consistency check")
    print("=" * 60)
    print(f"table              : {rows} x 16")
    print(f"ranks              : {[int(r) for r in table.ranks()]}")
    print(f"tt parameters      : {int(stats['tt_parameters']):,}")
    print(f"dense parameters   : {int(stats['dense_parameters']):,}")
    print(f"compression ratio  : {stats['compression_ratio']:.2f}x")
    print(f"max abs error      : {stats['max_abs_error']:.3e}")
    print(f"mean abs error     : {stats['mean_abs_error']:.3e}")
    print()
    if stats["max_abs_error"] > 1e-4:
        print("FAILED: the lookup path disagrees with the reference contraction.")
        print("Do not report any compression result until this is resolved.")
        return 1
    print("PASSED: lookup and reference contraction agree to float32 precision.")
    print("Quote max_abs_error in the paper if asked whether the TT path is exact.")

    if stats["compression_ratio"] < 1.0:
        print()
        print("NOTE: this table is SMALLER compressed than dense "
              f"({stats['compression_ratio']:.2f}x).")
        print("TT cost is O(R^2 * m * n) while dense is O(M * N), so compression")
        print("only pays off above a vocabulary-size threshold. For a table this")
        print("small, report the dense control in the rank sweep and do not claim")
        print("a memory win. The production tables (10^7-10^9 rows) are far above")
        print("the crossover, but the paper must not quote a compression ratio")
        print("from a table that is below it.")
    return 0


def main() -> int:
    args = parse_args()
    ctx = ExperimentContext(args)

    if args.check_backend:
        return check_backend(ctx)

    cfg_ab = ctx.cfg.get("ablation", {}) or {}
    cells = build_cells(cumulative=bool(cfg_ab.get("cumulative", True)))
    measure = not getattr(args, "no_resource_profile", False)

    ctx.describe_once()
    ctx.header("P0-2  Rec-AD component ablation",
               {"orderings": args.orderings,
                "cells": len(cells),
                "measure": measure})

    writer = ctx.writer(f"ablation_{Path(args.data).stem}"
                        f"_{ctx.tier_name}" + (f"_{args.tag}" if args.tag else ""))

    print(f"{'cell':<28}{'eff_tt':>7}{'reorder':>9}{'pipeline':>10}")
    print("-" * 56)
    for name, flags in cells:
        print(f"{name:<28}{str(flags['eff_tt']):>7}{str(flags['reorder']):>9}"
              f"{str(flags['pipeline']):>10}")
    print()

    # ---- per-arm learning rate -------------------------------------------
    # The two embedding parameterisations do not have the same effective
    # learning rate: on identical indices the dense table's backward gradient
    # is 2.6x the TT table's, so one nominal lr handicaps whichever arm it does
    # not suit. Measured optima on ieee118 are TT 1e-3 and dense 1e-2; used
    # naively, a single lr produces an apparent 0.13 AUC "TT advantage" that is
    # entirely a tuning artefact.
    #
    # So each cell is trained at the lr appropriate to its embedding, and the
    # value is recorded per row.
    lr_by_emb = (ctx.cfg.get("ablation", {}) or {}).get("lr_by_embedding", {}) or {}
    base_lr = float((ctx.cfg.get("train", {}) or {}).get("lr", 1e-3))
    if lr_by_emb:
        ctx.log.info("per-arm learning rates in use: %s (fallback %.0e)",
                     lr_by_emb, base_lr)

    for name, flags in cells:
        recad = RecADConfig.from_cfg(
            ctx.cfg,
            use_eff_tt=flags["eff_tt"],
            use_reorder=flags["reorder"],
            use_pipeline=flags["pipeline"],
        )
        cell_lr = float(lr_by_emb.get("eff_tt" if flags["eff_tt"] else "dense",
                                      base_lr))
        ctx.cfg.setdefault("train", {})["lr"] = cell_lr
        for seed in ctx.seeds:
            ctx.log.info("[ablation] cell=%s seed=%d lr=%.0e", name, seed, cell_lr)
            # Pass the seed so the weights are drawn from a seeded RNG. The
            # harness otherwise seeds only when training starts, which leaves
            # the initialisation depending on how many models were built
            # earlier in the run -- measured consequence: two computationally
            # identical cells scored 0.7927 and 0.5007.
            model = ctx.build_model(recad, ordering=args.orderings, seed=seed)
            row: Dict[str, Any] = {
                "cell": name,
                "variant": "cumulative" if name.startswith("cum:") else "oat",
                "lr": cell_lr,
            }
            row.update({f"flag_{c}": float(flags[c]) for c in COMPONENTS})
            row = ctx.run_one(model, seed, row, measure=measure)
            if row is not None:
                row.setdefault("cell", name)
                writer.add(**row)
                print(f"  {name:<28} seed={seed}  f1={row.get('f1', float('nan')):.4f}  "
                      f"auc={row.get('auc', float('nan')):.4f}  "
                      f"comp={row.get('compression_ratio', float('nan')):.1f}x  "
                      f"fit={row.get('fit_seconds', float('nan')):.1f}s")

    summary = ctx.finish(writer,
                         group_keys=["cell"],
                         metric_keys=METRICS,
                         extra={"cells": [{"cell": n, **f} for n, f in cells],
                                "orderings": args.orderings})

    print()
    print("=" * 100)
    print("Ablation summary (mean over seeds)")
    print("=" * 100)
    from evaluate import format_results_table
    print(format_results_table(summary, columns=[
        "cell", "compression_ratio", "f1", "auc", "detect_rate_stealthy",
        "latency_ms_p50", "peak_vram_mb", "train_seconds",
    ]))
    print()
    print("Reading this table: the isolate-at-a-time rows show whether a")
    print("component does anything alone; the cumulative rows show how the")
    print("gains stack, which is what the contribution list claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
