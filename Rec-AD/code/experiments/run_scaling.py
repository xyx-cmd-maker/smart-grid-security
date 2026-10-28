"""P1-6 — multi-topology scaling study.

Answers Reviewer 2 (comment 4): "The experiments rely on a limited set of
datasets, patterns, and sizes... This raises concerns about the
generalizability of the results."

The reviewer names three dimensions. Attack patterns are handled by the
threat model (three families, reported separately). Dataset roles are handled
by the FDIA-validity versus efficiency-benchmark split. This script addresses
the third: **size**.

The point is not to report four unrelated numbers but to test a prediction.
Rec-AD's mechanism is memory-bound: a larger grid means a larger embedding
table, heavier reliance on host memory, and a more severe CPU-GPU bottleneck.
If that account is right, the benefit should GROW with grid size. Plotting the
trend lets a reader check the mechanism rather than take four numbers on
trust — and a flat trend would falsify our explanation of why Rec-AD works.

Usage
-----
    python experiments/run_scaling.py --data-dir data_cache --cases 14,57,118,300
    python experiments/run_scaling.py --data-dir data_cache --cases 118 --quick
    python experiments/run_scaling.py --data-dir data_cache --generate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch

# The inter-op pool defaults to the core count and spins while one thread does the arithmetic. With 57 buses
# and two epochs this alone made a run take over twenty minutes. Pin it before any parallel work begins.
try:
    torch.set_num_interop_threads(1)
except Exception:  # noqa: BLE001
    pass
torch.set_num_threads(1)

from data.contract import describe_contract, load_contract
from data.dataset import build_loaders, build_spec
from evaluate import evaluate_detector, format_results_table
from recad.common import (
    ResultsWriter,
    aggregate_seeds,
    describe_environment,
    get_device,
    get_tier,
    load_config,
    set_seed,
    setup_logging,
    warn_if_unpublishable,
)
from recad.model import RecADConfig, RecADDetector, make_permutation

METRICS = [
    "accuracy", "precision", "recall", "f1", "auc", "fpr",
    "tpr_at_fpr_1pct",
    "detect_rate_random", "detect_rate_stealthy", "detect_rate_load_redist",
    "compression_ratio", "embedding_bytes", "n_parameters",
    "latency_ms_p50", "latency_ms_p99", "peak_vram_mb",
    "train_seconds", "fit_seconds",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--cases", default="14,57,118,300")
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
    p.add_argument("--generate", action="store_true",
                   help="generate missing case files before running")
    p.add_argument("--samples", type=int, default=20000,
                   help="samples per case when generating")
    p.add_argument("--no-pandapower", action="store_true")
    return p.parse_args()


def ensure_case(args, case: str, data_dir: Path) -> Optional[Path]:
    path = data_dir / f"ieee{case}.npz"
    if path.exists():
        return path
    if not args.generate:
        return None
    from data.generate_fdia import generate

    print(f"  generating {path} ...")
    generate(case=case, out=path, n_samples=args.samples,
             allow_pandapower=not args.no_pandapower)
    return path if path.exists() else None


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging((cfg.get("output", {}) or {}).get("log_level", "INFO"))

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    data_dir = Path(args.data_dir)
    tier_name = args.tier or (cfg.get("measure", {}) or {}).get("tier", "t4")
    warn_if_unpublishable(tier_name)
    device = get_device()
    budget = cfg.get("budget", {}) or {}

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(cfg.get("seeds", [0])))
    if args.quick:
        seeds = seeds[:1]
        cfg.setdefault("train", {})["epochs"] = 3
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)

    batch_size = int(args.batch_size
                     or (cfg.get("train", {}) or {}).get("batch_size", 4096))
    if args.quick:
        batch_size = min(batch_size, 512)
    num_workers = int(args.num_workers if args.num_workers is not None
                      else (cfg.get("data", {}) or {}).get("num_workers", 0))

    results_dir = Path(args.results_dir
                       or (cfg.get("output", {}) or {}).get("results_dir", "results"))
    writer = ResultsWriter(
        name=f"scaling_{tier_name}" + (f"_{args.tag}" if args.tag else ""),
        results_dir=results_dir,
    )

    print("=" * 96)
    print("P1-6  Multi-topology scaling study")
    print("=" * 96)
    print(f"cases       : {cases}")
    print(f"device      : {device}")
    print(f"tier        : {tier_name}  (publishable={get_tier(tier_name).publishable})")
    print(f"seeds       : {seeds}")
    print(f"batch size  : {batch_size}")
    print()

    available: List[tuple] = []
    for case in cases:
        path = ensure_case(args, case, data_dir)
        if path is None:
            print(f"  case {case:<4} SKIPPED - no data file and --generate not set")
            continue
        data = load_contract(path)
        available.append((case, path, data))
        meta = data.get("meta_parsed", {})
        print(f"  case {case:<4} buses={meta.get('n_bus', '?'):>4}  "
              f"measurements={meta.get('n_measurements', '?'):>4}  "
              f"samples={len(data['label']):>7,}  source={meta.get('topology_source','?')}")

    if not available:
        print()
        print("No cases available. Re-run with --generate, or point --data-dir at")
        print("a directory containing ieee14.npz, ieee57.npz, ieee118.npz, ...")
        return 1

    print()
    if not args.quiet:
        print(describe_contract(available[-1][1]))
        print()

    for case, path, data in available:
        spec = build_spec(data)
        recad = RecADConfig.from_cfg(cfg)
        perm = make_permutation(args.orderings, spec, data)
        meta = data.get("meta_parsed", {})
        n_bus = int(meta.get("n_bus", spec.num_bus))
        n_meas = int(meta.get("n_measurements", spec.num_dense))

        for seed in seeds:
            log.info("[scaling] case=%s seed=%d", case, seed)
            set_seed(seed)
            loaders = build_loaders(
                data, batch_size=batch_size, num_workers=num_workers,
                standardize=bool((cfg.get("data", {}) or {}).get("standardize", True)),
                ratios=(cfg.get("data", {}) or {}).get("split_ratios"),
                seed=seed,
            )
            model = RecADDetector(spec, cfg, name=f"Rec-AD[{case}]",
                                  recad=recad, permutation=perm, data=data)

            t0 = time.perf_counter()
            try:
                hist = model.fit(loaders["train"], loaders["val"], device=device)
            except torch.cuda.OutOfMemoryError:
                log.warning("[scaling] case=%s OOM at batch=%d", case, batch_size)
                writer.add(case=case, n_bus=n_bus, n_measurements=n_meas,
                           seed=seed, tier=tier_name, train_oom=1)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            row: Dict[str, Any] = {
                "case": case,
                "n_bus": n_bus,
                "n_measurements": n_meas,
                "seed": seed,
                "tier": tier_name,
                "topology_source": meta.get("topology_source", "unknown"),
                "ordering": args.orderings,
                "fit_seconds": time.perf_counter() - t0,
                "train_seconds": hist.get("train_seconds", float("nan")),
            }
            metrics_row, *_ = evaluate_detector(
                model, loaders, attack_type_test=data["attack_type"],
                threshold_strategy="max_f1", device=device,
            )
            row.update(metrics_row)
            row.setdefault("case", case)
            row.setdefault("n_bus", n_bus)
            writer.add(**row)

            print(f"  case {case:<4} seed={seed}  "
                  f"f1={row.get('f1', float('nan')):.4f}  "
                  f"auc={row.get('auc', float('nan')):.4f}  "
                  f"stealthy={row.get('detect_rate_stealthy', float('nan')):.4f}  "
                  f"comp={row.get('compression_ratio', float('nan')):7.1f}x  "
                  f"fit={row.get('fit_seconds', float('nan')):.1f}s")

    summary = aggregate_seeds(writer.rows, ["case", "n_bus"], METRICS)
    summary = sorted(summary, key=lambda r: r.get("n_bus", 0))
    writer.write(extra={
        "config": cfg,
        "environment": describe_environment(tier_name),
        "budget": budget,
        "cases": cases,
        "seeds": seeds,
        "orderings": args.orderings,
        "summary": summary,
    })
    (results_dir / f"{writer.name}_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")

    print()
    print("=" * 96)
    print("Scaling trend (mean over seeds)")
    print("=" * 96)
    print(format_results_table(summary, columns=[
        "case", "n_bus", "compression_ratio", "embedding_bytes",
        "f1", "auc", "detect_rate_stealthy", "fit_seconds",
    ]))

    if len(summary) >= 2:
        first, last = summary[0], summary[-1]
        print()
        print("Mechanism check: does the benefit grow with grid size?")
        for key, label in (("compression_ratio", "compression ratio"),
                           ("embedding_bytes", "embedding bytes"),
                           ("fit_seconds", "training time")):
            a = first.get(f"{key}_mean", float("nan"))
            b = last.get(f"{key}_mean", float("nan"))
            if np.isnan(a) or np.isnan(b) or a == 0:
                continue
            ratio = b / a
            trend = "grows" if ratio > 1.05 else ("shrinks" if ratio < 0.95 else "flat")
            print(f"  {label:<20} {a:>14.2f} -> {b:>14.2f}   ({trend}, {ratio:.2f}x)")
        print()
        print("  Compression ratio growing with bus count is the prediction of the")
        print("  memory-bound account. A FLAT trend would falsify our explanation of")
        print("  why Rec-AD works, and must be reported if observed rather than")
        print("  presented as a scaling win.")
        print()
        print("  Report detections per attack family alongside this trend, so a")
        print("  reader can confirm the comparison is not confounded by a different")
        print("  attack mix at different sizes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
