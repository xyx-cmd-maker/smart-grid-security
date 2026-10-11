"""P1-5 — reordering strategy comparison: is topology actually used?

This is the quantitative answer to Reviewer 1 (comment 6). The manuscript
claims that TT-compressed embeddings "effectively preserve the structural
information of power flow equations" and offers no measurement. That claim is
also unfalsifiable as written.

The reframing here is deliberate. "Preserved" is a representation-level
statement a reviewer can wave away ("preserved is not the same as useful").
"Used" is functional and testable: hold the model, data and optimiser fixed
and change only the index ordering. If topology-aware ordering beats orderings
that ignore topology, the structural information is demonstrably being used.

Four orderings are compared:

  none       raw id order (hashed cross features, topology-free by construction)
  random     uniform permutation; the null hypothesis
  frequency  most-accessed-first; the standard systems heuristic
  topology   BFS over the grid graph

Detection performance and TT prefix reuse are reported together, because a
reuse gain that does not move detection metrics would be a systems result, not
the structural claim the paper makes.

Usage
-----
    python experiments/run_reorder.py --data data_cache/ieee118.npz
    python experiments/run_reorder.py --data ... --orderings none,random,frequency,topology
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._harness import ExperimentContext
from recad.model import RecADConfig, make_permutation
from recad.reorder import (build_permutation, evaluate_orderings,
                           multi_index, prefix_reuse_rate)

METRICS = [
    "accuracy", "precision", "recall", "f1", "auc", "fpr",
    "tpr_at_fpr_1pct",
    "detect_rate_random", "detect_rate_stealthy", "detect_rate_load_redist",
    "compression_ratio", "latency_ms_p50", "latency_ms_p99", "peak_vram_mb",
    "train_seconds",
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
    p.add_argument("--orderings", default="none,random,frequency,topology")
    p.add_argument("--skip-reuse-diagnostic", action="store_true")
    return p.parse_args()


def reuse_diagnostic(ctx: ExperimentContext, orderings: List[str]) -> List[Dict[str, Any]]:
    """Measure TT prefix reuse induced by each ordering, no training involved.

    Cheap, deterministic and hardware-independent, so it can be reported even
    if GPU time runs out. It is the mechanism-level view of the same effect.

    The permutation is built with `make_permutation`, which resolves the
    id -> bus mapping and the adjacency; `evaluate_orderings` would need both
    passed explicitly and silently errors for the topology strategy otherwise.
    """
    from recad.reorder import contraction_counts_for_batch, prefix_reuse_rate
    from recad.tt_table import suggest_split

    sparse = ctx.data["sparse"].astype(np.int64)
    flat = sparse.reshape(-1)
    total = ctx.spec.sparse_total_rows
    # TT reshape of the shared table, matching what the model actually uses.
    m, _ = suggest_split(total, 16, d=3)

    rows: List[Dict[str, Any]] = []
    for kind in orderings:
        perm = make_permutation(kind, ctx.spec, ctx.data)
        mapped = perm[flat]

        rates, plain_total, reuse_total = [], 0, 0
        for b in range(0, len(mapped), ctx.batch_size):
            chunk = mapped[b:b + ctx.batch_size]
            if len(chunk) == 0:
                continue
            rates.append(prefix_reuse_rate(chunk, m))
            plain_b, reuse_b = contraction_counts_for_batch(chunk, m)
            plain_total += plain_b
            reuse_total += reuse_b

        rows.append({
            "ordering": kind,
            "reuse_rate": float(np.mean(rates)) if rates else 0.0,
            "contractions_plain": float(plain_total),
            "contractions_reuse": float(reuse_total),
            "contraction_ratio": (float(plain_total / reuse_total)
                                  if reuse_total else float("nan")),
        })
    return rows


def main() -> int:
    args = parse_args()
    ctx = ExperimentContext(args)
    orderings = [o.strip() for o in args.orderings.split(",") if o.strip()]

    ctx.describe_once()
    ctx.header("P1-5  Index reordering strategies (is topology used?)",
               {"orderings": orderings})

    stem = Path(args.data).stem
    writer = ctx.writer(f"reorder_{stem}_{ctx.tier_name}"
                        + (f"_{args.tag}" if args.tag else ""))

    # ---- mechanism view: TT prefix reuse --------------------------------
    diag: List[Dict[str, Any]] = []
    if not args.skip_reuse_diagnostic:
        print("Mechanism check: TT prefix reuse induced by each ordering")
        print("-" * 74)
        diag = reuse_diagnostic(ctx, orderings)
        print(f"  {'ordering':<12}{'reuse rate':>13}{'contractions':>15}{'ratio':>10}")
        for r in diag:
            if "error" in r:
                print(f"  {r['ordering']:<12}  ERROR: {r['error']}")
                continue
            print(f"  {r['ordering']:<12}{r['reuse_rate']:>13.4f}"
                  f"{r['contractions_reuse']:>15.0f}{r['contraction_ratio']:>9.2f}x")
            writer.add(kind="reuse_diagnostic", ordering=r["ordering"],
                       reuse_rate=r["reuse_rate"],
                       contraction_ratio=r["contraction_ratio"])
        print()

    # ---- end-to-end view: detection performance -------------------------
    print("End-to-end: detection performance under each ordering")
    print("-" * 74)
    for kind in orderings:
        recad = RecADConfig.from_cfg(ctx.cfg, use_eff_tt=True,
                                     use_reorder=(kind != "none"),
                                     use_pipeline=True)
        for seed in ctx.seeds:
            ctx.log.info("[reorder] ordering=%s seed=%d", kind, seed)
            # Seed BEFORE building: the weights are drawn in the constructor,
            # so seeding only at training time leaves the initialisation to
            # depend on how many models were built earlier in the run.
            model = ctx.build_model(recad, ordering=kind, seed=seed)
            row: Dict[str, Any] = {"kind": "detection", "ordering": kind}
            row = ctx.run_one(model, seed, row, measure=not args.quick)
            if row is None:
                continue
            row.setdefault("kind", "detection")
            row.setdefault("ordering", kind)
            writer.add(**row)
            print(f"  {kind:<12} seed={seed}  "
                  f"f1={row.get('f1', float('nan')):.4f}  "
                  f"auc={row.get('auc', float('nan')):.4f}  "
                  f"stealthy={row.get('detect_rate_stealthy', float('nan')):.4f}")

    detection_rows = [r for r in writer.rows if r.get("kind") == "detection"]
    from recad.common import aggregate_seeds
    summary = aggregate_seeds(detection_rows, ["ordering"], METRICS)

    writer.write(extra={
        "config": ctx.cfg,
        "data_file": str(args.data),
        "seeds": ctx.seeds,
        "tier": ctx.tier_name,
        "reuse_diagnostic": diag,
        "summary": summary,
    })
    import json
    (ctx.results_dir / f"{writer.name}_summary.json").write_text(
        json.dumps({"detection": summary, "reuse": diag}, indent=2, default=float),
        encoding="utf-8",
    )

    print()
    print("=" * 96)
    print("Reorder summary")
    print("=" * 96)
    from evaluate import format_results_table
    print(format_results_table(summary, columns=[
        "ordering", "f1", "auc", "detect_rate_stealthy", "detect_rate_random",
        "train_seconds",
    ]))

    by = {r["ordering"]: r for r in summary}
    reuse = {r["ordering"]: r.get("reuse_rate") for r in diag if "error" not in r}
    topo, rand = by.get("topology"), by.get("random")
    if topo and rand:
        print()
        print("Topology vs random:")
        print(f"  F1            {topo.get('f1_mean', float('nan')):.4f} vs "
              f"{rand.get('f1_mean', float('nan')):.4f}  "
              f"({topo.get('f1_mean', 0) - rand.get('f1_mean', 0):+.4f})")
        print(f"  AUC           {topo.get('auc_mean', float('nan')):.4f} vs "
              f"{rand.get('auc_mean', float('nan')):.4f}  "
              f"({topo.get('auc_mean', 0) - rand.get('auc_mean', 0):+.4f})")
        print(f"  TT reuse rate {reuse.get('topology', float('nan')):.4f} vs "
              f"{reuse.get('random', float('nan')):.4f}")
        print()
        print("IMPORTANT for the paper: report BOTH the reuse-rate effect and the")
        print("detection effect. A reuse gain with no detection difference is a")
        print("systems result; the structural claim needs the detection number.")
        print("Also state the access regime - see recad/reorder.py, where topology")
        print("ordering pays off most when a batch concentrates on few buses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
