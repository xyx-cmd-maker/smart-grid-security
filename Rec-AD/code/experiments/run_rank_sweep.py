"""P0-3 — TT rank sweep: the compression-versus-accuracy trade-off.

This is the direct evidence for two claims the original manuscript makes
without support:

  * "provides extreme parameter compression **without sacrificing detection
    accuracy**" (Abstract)
  * "TT-compressed embeddings effectively preserve the structural information
    of power flow equations" (Section V.C, flagged by Reviewer 1 comment 6)

Neither claim can be assessed from a single operating point. Sweeping the TT
rank produces the curve that shows where accuracy starts to fall, and that
curve is the only honest way to say "compression is free up to rank R".

The rank is also the deployment knob: an operator with a tighter memory budget
can lower it. Report the curve, not one point.

Usage
-----
    python experiments/run_rank_sweep.py --data data_cache/ieee118.npz
    python experiments/run_rank_sweep.py --data ... --ranks 4,8,16,32 --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._harness import ExperimentContext
from recad.model import RecADConfig

METRICS = [
    "accuracy", "precision", "recall", "f1", "auc", "fpr",
    "detect_rate_stealthy", "detect_rate_random", "detect_rate_load_redist",
    "compression_ratio", "embedding_bytes", "n_parameters",
    "latency_ms_p50", "latency_ms_p95", "peak_vram_mb",
    "energy_j_per_inference", "train_seconds",
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
    p.add_argument("--orderings", default="topology")
    p.add_argument("--ranks", default=None,
                   help="comma-separated TT ranks; default from config tt.rank_sweep")
    p.add_argument("--include-dense", action="store_true", default=True,
                   help="also measure the uncompressed dense table as the ceiling")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ctx = ExperimentContext(args)

    tt_cfg = ctx.cfg.get("tt", {}) or {}
    ranks: List[int] = (
        [int(r) for r in args.ranks.split(",")] if args.ranks
        else [int(r) for r in tt_cfg.get("rank_sweep", [4, 8, 16, 24, 32, 48, 64])]
    )

    ctx.describe_once()
    ctx.header("P0-3  TT rank sweep (compression vs detection accuracy)",
               {"ranks": ranks,
                "dense ceiling": bool(args.include_dense),
                "orderings": args.orderings})

    stem = Path(args.data).stem
    writer = ctx.writer(f"rank_sweep_{stem}_{ctx.tier_name}"
                        + (f"_{args.tag}" if args.tag else ""))

    # Dense ceiling first: the reference point the compressed curve is read against.
    cells = ([("dense", None)] if args.include_dense else []) + \
            [(f"rank_{r}", r) for r in ranks]

    for label, rank in cells:
        recad = RecADConfig.from_cfg(
            ctx.cfg,
            use_eff_tt=(rank is not None),
            tt_rank=(rank if rank is not None else 0),
            use_reorder=True,
            use_pipeline=True,
        )
        for seed in ctx.seeds:
            ctx.log.info("[rank_sweep] %s seed=%d", label, seed)
            # Seed BEFORE building: the weights are drawn in the constructor,
            # so seeding only at training time leaves the initialisation to
            # depend on how many models were built earlier in the run.
            model = ctx.build_model(recad, ordering=args.orderings, seed=seed)
            row: Dict[str, Any] = {"config": label,
                                   "tt_rank": float(rank if rank is not None else 0),
                                   "compressed": float(rank is not None)}
            row = ctx.run_one(model, seed, row, measure=not args.quick or True)
            if row is None:
                continue
            row.setdefault("config", label)
            writer.add(**row)
            print(f"  {label:<10} seed={seed}  "
                  f"f1={row.get('f1', float('nan')):.4f}  "
                  f"auc={row.get('auc', float('nan')):.4f}  "
                  f"stealthy={row.get('detect_rate_stealthy', float('nan')):.4f}  "
                  f"comp={row.get('compression_ratio', float('nan')):6.1f}x  "
                  f"emb={row.get('embedding_bytes', 0) / 1024:8.1f} KB")

    summary = ctx.finish(writer,
                         group_keys=["config"],
                         metric_keys=METRICS,
                         extra={"ranks": ranks, "orderings": args.orderings})

    print()
    print("=" * 100)
    print("Rank sweep (mean over seeds)")
    print("=" * 100)
    from evaluate import format_results_table
    ordered = sorted(summary, key=lambda r: r.get("tt_rank", 0.0))
    print(format_results_table(ordered, columns=[
        "config", "tt_rank", "compression_ratio", "embedding_bytes",
        "f1", "auc", "detect_rate_stealthy", "latency_ms_p50", "peak_vram_mb",
    ]))

    dense = next((r for r in ordered if r.get("config") == "dense"), None)
    if dense is not None:
        print()
        print("Accuracy retained relative to the uncompressed ceiling:")
        base_f1 = dense.get("f1_mean", float("nan"))
        for r in ordered:
            if r.get("config") == "dense":
                continue
            f1 = r.get("f1_mean", float("nan"))
            delta = f1 - base_f1
            print(f"  {r.get('config'):<10} comp={r.get('compression_ratio_mean', float('nan')):6.1f}x  "
                  f"F1 {f1:.4f}  ({delta:+.4f} vs dense)  "
                  f"AUC {r.get('auc_mean', float('nan')):.4f}")
        print()
        print("Report the LOWEST rank whose F1 sits within noise of dense as the")
        print("recommended operating point, and state the noise band explicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
