"""P0-1 — FDIA detection baselines on the edge deployment budget.

Answers Reviewer 1 (comment 8: no domain baselines), Reviewer 2 (comment 1:
baselines too old), Reviewer 3 (comment 2: no quantification of prior
overhead), Reviewer 3 (comment 3: no demonstrated gap) and Reviewer 3
(comment 6: CNNs and Transformers missing).

Every model is trained and measured on the same data, splits and hardware, and
is judged against the substation-node resource budget. The output is the
deployability table that the revision's first contribution rests on.

Usage
-----
    python experiments/run_baselines.py --data data_cache/ieee118.npz --tier t4
    python experiments/run_baselines.py --data ... --models BDD,GCN,Transformer
    python experiments/run_baselines.py --data ... --quick     # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# The thread count changes the bits for every operation in this pipeline, so it is set here
# rather than left to the caller's environment. The counts are recorded by
# describe_environment, which every entry point calls.
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch as _torch
_torch.set_num_threads(1)
try:
    _torch.set_num_interop_threads(1)
except Exception:
    pass  # already initialised, or this build has no interop pool

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch

from baselines.registry import BASELINE_FAMILIES, build_detector, enabled_baselines
from data.contract import describe_contract, load_contract
from data.dataset import build_loaders, build_spec
from evaluate import evaluate_detector, format_results_table
from measure.resources import budget_table, profile_resources
from recad.common import (
    ResultsWriter,
    describe_environment,
    get_device,
    get_tier,
    load_config,
    set_seed,
    setup_logging,
    warn_if_unpublishable,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="path to a .npz contract file")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--tier", default=None,
                   help="device tier for resource measurement (t4/rtx2060/jetson/...)")
    p.add_argument("--models", default=None,
                   help="comma-separated subset, e.g. BDD,GCN,Transformer")
    p.add_argument("--seeds", default=None, help="comma-separated seeds")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--tag", default="", help="suffix for the output filenames")
    p.add_argument("--epochs", type=int, default=None, help="override epoch count")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader worker processes (default from config; "
                        "use 0 where worker spawn is restricted)")
    p.add_argument("--no-resource-profile", action="store_true",
                   help="skip latency/memory/energy measurement (fast mode)")
    p.add_argument("--no-energy", action="store_true",
                   help="skip power sampling (useful when nvidia-smi is absent)")
    p.add_argument("--measure-iters", type=int, default=None)
    p.add_argument("--target-fpr", type=float, default=0.01,
                   help="calibrate every detector's threshold to this benign "
                        "false-alarm rate on the VALIDATION split, then report "
                        "per-attack-family detection rates at that operating "
                        "point. This is required for the paper's central claim: "
                        "residual BDD is nearly blind to a = Hc attacks. With "
                        "max-F1 the threshold collapses (BDD lands at ~50%% "
                        "false alarms) and the per-family numbers stop meaning "
                        "anything. Set to a negative value to fall back to "
                        "max_f1.")
    p.add_argument("--quick", action="store_true",
                   help="tiny run for smoke testing the toolchain")
    p.add_argument("--no-physics-features", action="store_true",
                   help="feed sequence models the RAW measurement trajectory "
                        "only, without the H-derived residual/innovation "
                        "channels. This is the decisive control: if a model "
                        "still learns, it does not depend on knowing H, and "
                        "that is the only honest argument for a learned "
                        "detector over the deterministic temporal test")
    p.add_argument("--patience", type=int, default=None,
                   help="early-stopping patience (overrides train config)")
    p.add_argument("--describe-data", action="store_true",
                   help="print the contract diagnostic report and exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging((cfg.get("output", {}) or {}).get("log_level", "INFO"))

    if args.describe_data:
        print(describe_contract(args.data))
        return 0

    data = load_contract(args.data)
    print(describe_contract(args.data))
    print()

    spec = build_spec(data)
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(cfg.get("seeds", [0])))
    tier_name = args.tier or (cfg.get("measure", {}) or {}).get("tier", "t4")
    warn_if_unpublishable(tier_name)

    batch_size = args.batch_size or int((cfg.get("train", {}) or {}).get("batch_size", 4096))
    mcfg = cfg.get("measure", {}) or {}
    lcfg = mcfg.get("latency", {}) or {}
    ecfg = mcfg.get("energy", {}) or {}
    budget = cfg.get("budget", {}) or {}

    if args.quick:
        seeds = seeds[:1]
        batch_size = min(batch_size, 512)
        cfg.setdefault("train", {})["epochs"] = 2
        cfg["baselines"]["svm"]["max_train_samples"] = 2000
        cfg["baselines"]["rf"]["n_estimators"] = 20
        cfg["baselines"]["gnn"]["hidden"] = 16
        cfg["baselines"]["transformer"]["d_model"] = 32
        cfg["baselines"]["transformer"]["layers"] = 1

    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.patience is not None:
        cfg.setdefault("train", {})["early_stop_patience"] = int(args.patience)
    if args.no_physics_features:
        seq_cfg = cfg.setdefault("baselines", {}).setdefault("sequence", {})
        seq_cfg["physics_features"] = False
        log.info("sequence models will see the RAW trajectory only "
                 "(physics_features=false)")

    measure_iters = args.measure_iters or int(lcfg.get("measure_iters", 500))
    warmup_iters = int(lcfg.get("warmup_iters", 50))
    if args.quick:
        measure_iters, warmup_iters = 20, 3

    num_workers = (args.num_workers if args.num_workers is not None
                   else int((cfg.get("data", {}) or {}).get("num_workers", 0)))
    if num_workers > 0:
        log.info("DataLoader num_workers=%d", num_workers)

    target_fpr = float(args.target_fpr)
    if target_fpr and target_fpr > 0:
        log.info("threshold: calibrated to a %g false-alarm rate on validation "
                 "(per-attack-family numbers are comparable only at a matched "
                 "operating point)", target_fpr)
    else:
        log.warning("threshold: max_f1 selected; per-attack-family rates are "
                    "NOT comparable across models and the a=Hc blindness of "
                    "residual BDD will not be visible")

    include = [m.strip() for m in args.models.split(",")] if args.models else None
    models = enabled_baselines(cfg, data=data, include=include)

    out_cfg = cfg.get("output", {}) or {}
    results_dir = Path(args.results_dir or out_cfg.get("results_dir", "results"))
    name = f"baselines_{Path(args.data).stem}_{tier_name}"
    if args.tag:
        name += f"_{args.tag}"
    writer = ResultsWriter(name=name, results_dir=results_dir)

    device = get_device()
    log.info("device=%s  tier=%s  models=%s  seeds=%s", device, tier_name, models, seeds)

    print("=" * 96)
    print("P0-1  FDIA detection baselines under the edge deployment budget")
    print("=" * 96)
    print(f"data        : {args.data}")
    print(f"device      : {device}")
    print(f"tier        : {tier_name}  (publishable={get_tier(tier_name).publishable})")
    print(f"budget      : {budget.get('vram_gb')} GB VRAM / "
          f"{budget.get('power_w')} W / {budget.get('latency_ms')} ms")
    print(f"models      : {', '.join(models)}")
    print(f"seeds       : {seeds}")
    print()

    profiles: Dict[str, Any] = {}
    skipped: List[str] = []

    for model_name in models:
        for seed in seeds:
            set_seed(seed)
            loaders = build_loaders(
                data, batch_size=batch_size,
                num_workers=num_workers,
                standardize=bool((cfg.get("data", {}) or {}).get("standardize", True)),
                ratios=(cfg.get("data", {}) or {}).get("split_ratios"),
                seed=seed,
            )

            family = BASELINE_FAMILIES[model_name][0]
            log.info("[%s] seed=%d  family=%s", model_name, seed, family)
            t0 = time.perf_counter()

            try:
                model = build_detector(model_name, spec, cfg, data=data)
                hist = model.fit(loaders["train"], loaders["val"], device=device)
            except torch.cuda.OutOfMemoryError:
                log.warning("[%s] OOM during training at batch=%d; recording as "
                            "over-budget and continuing", model_name, batch_size)
                writer.add(model=model_name, seed=seed, tier=tier_name,
                           train_oom=1, fits_budget=0.0,
                           note="OOM during training")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except Exception as exc:                      # noqa: BLE001
                log.error("[%s] failed: %s", model_name, exc)
                skipped.append(f"{model_name}: {exc}")
                continue

            fit_seconds = time.perf_counter() - t0

            # Calibrate to a common false-alarm rate so the per-attack-family
            # numbers are comparable across models. With max_f1 each model
            # picks its own operating point, and BDD ends up at ~50% false
            # alarms -- at which point "detects 49% of stealthy attacks" says
            # nothing about detection ability. The paper's claim that residual
            # BDD is blind to a = Hc only shows up at a matched low FPR.
            if target_fpr and target_fpr > 0:
                thr_strategy, thr_fpr = "target_fpr", target_fpr
            else:
                thr_strategy, thr_fpr = "max_f1", None

            row, *_ = evaluate_detector(
                model, loaders, attack_type_test=data["attack_type"],
                threshold_strategy=thr_strategy, target_fpr=thr_fpr,
                device=device,
            )
            row.update({
                "model": model_name,
                "family": family,
                "seed": seed,
                "tier": tier_name,
                "fit_seconds": fit_seconds,
                "train_seconds": hist.get("train_seconds", float("nan")),
                "n_epochs_run": hist.get("n_epochs_run", 0),
                "batch_size": batch_size,
                "standardize": bool((cfg.get("data", {}) or {}).get("standardize", False)),
                "target_fpr": thr_fpr,
            })

            if not args.no_resource_profile:
                sample_batch = next(iter(loaders["test"]))
                one = {k: (v[:1] if torch.is_tensor(v) and v.dim() > 0 else v)
                       for k, v in sample_batch.items()}
                prof = profile_resources(
                    model, one, tier_name=tier_name, budget=budget,
                    warmup_iters=warmup_iters, measure_iters=measure_iters,
                    percentiles=tuple(lcfg.get("percentiles", [50, 95, 99])),
                    measure_energy=not args.no_energy,
                    energy_interval_ms=int(ecfg.get("sample_interval_ms", 100)),
                    device=device,
                )
                row.update(prof.as_dict())
                row["profile_notes"] = "; ".join(prof.notes)
                profiles.setdefault(model_name, prof)
                if prof.oom:
                    log.warning("[%s] OOM at batch size 1 on tier %s", model_name, tier_name)

            writer.add(**row)
            print(f"  {model_name:<12} seed={seed}  "
                  f"acc={row['accuracy']:.4f}  f1={row['f1']:.4f}  "
                  f"auc={row['auc']:.4f}  "
                  f"stealthy={row.get('detect_rate_stealthy', float('nan')):.4f}  "
                  f"fit={fit_seconds:.1f}s")

    # ---- summary --------------------------------------------------------
    from recad.common import aggregate_seeds

    metric_keys = ["accuracy", "precision", "recall", "f1", "auc", "fpr",
                   "tpr_at_fpr_1pct", "detect_rate_random", "detect_rate_stealthy",
                   "detect_rate_load_redist", "detect_rate_normal_fp",
                   "latency_ms_p50", "latency_ms_p99", "peak_vram_mb",
                   "energy_j_per_inference", "train_seconds"]
    summary = aggregate_seeds(writer.rows, ["model", "family"], metric_keys)

    print()
    print("=" * 96)
    print("Detection performance (mean over seeds, threshold chosen on val)")
    print("=" * 96)
    print(format_results_table(summary))

    if profiles:
        print()
        print(budget_table(profiles, tier_name=tier_name, budget=budget))
        print()
        print("Report this table as the paper's lead result: it is the evidence")
        print("that existing FDIA detectors do not fit a substation-node budget.")

    if skipped:
        print()
        print("Skipped:")
        for s in skipped:
            print(f"  - {s}")

    writer.write(extra={
        "config": cfg,
        "environment": describe_environment(tier_name),
        "budget": budget,
        "data_file": str(args.data),
        "models": models,
        "seeds": seeds,
        "summary": summary,
        "skipped": skipped,
    })

    (results_dir / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    print()
    print(f"Results written to {results_dir}/{name}.json / .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
