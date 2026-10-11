"""Shared setup for the Rec-AD component experiments (ablation, sweep, reorder).

Keeps data loading, model construction, training and evaluation in one place so
that every experiment differs only in what it varies. If the runners each did
their own setup, a reviewer could reasonably ask whether the comparison is
between components or between harnesses.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch

from data.contract import describe_contract, load_contract
from data.dataset import build_loaders, build_spec
from evaluate import evaluate_detector
from measure.resources import profile_resources
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
from recad.model import RecADConfig, RecADDetector, make_permutation

#: Permutation seed kept deliberately distinct from the data seed. Reusing the
#: data seed makes the `random` control reproduce the data's own block
#: structure; see recad/reorder.py.
PERM_SEED = 987654321


class ExperimentContext:
    """Everything a component experiment needs, assembled once."""

    def __init__(self, args) -> None:
        self.args = args
        self.cfg = load_config(args.config)
        self.log = setup_logging(
            (self.cfg.get("output", {}) or {}).get("log_level", "INFO")
        )

        self.data = load_contract(args.data)
        self.spec = build_spec(self.data)
        self.device = get_device()

        self.tier_name = args.tier or (self.cfg.get("measure", {}) or {}).get("tier", "t4")
        warn_if_unpublishable(self.tier_name)
        self.budget = self.cfg.get("budget", {}) or {}

        if getattr(args, "quick", False):
            self.cfg.setdefault("train", {})["epochs"] = 3
            self.cfg["train"]["batch_size"] = min(
                int(self.cfg["train"].get("batch_size", 4096)), 512
            )
            for key in ("svm", "rf"):
                self.cfg.setdefault("baselines", {}).setdefault(key, {})
            self.cfg.setdefault("tt", {})["rank_sweep"] = [2, 4, 8]
        if getattr(args, "epochs", None) is not None:
            self.cfg.setdefault("train", {})["epochs"] = int(args.epochs)

        self.batch_size = int(
            getattr(args, "batch_size", None)
            or (self.cfg.get("train", {}) or {}).get("batch_size", 4096)
        )

        # Operating point for the per-attack-family numbers. Every detector
        # must be judged at the SAME false-alarm rate, otherwise the family
        # breakdown is not comparable: with max_f1 a chi-square detector lands
        # at ~50% false alarms and then "detects" nearly every family, which
        # says nothing. A negative value falls back to max_f1 for backwards
        # compatibility, and warns.
        self.target_fpr = float(getattr(args, "target_fpr", 0.01) or 0.01)
        if self.target_fpr > 0:
            self.log.info("threshold calibrated to a %g false-alarm rate on "
                          "validation", self.target_fpr)
        else:
            self.log.warning("target_fpr<=0: falling back to max_f1; the "
                             "per-attack-family rates will NOT be comparable "
                             "across cells")
            self.target_fpr = None
        self.num_workers = int(
            getattr(args, "num_workers", None)
            if getattr(args, "num_workers", None) is not None
            else (self.cfg.get("data", {}) or {}).get("num_workers", 0)
        )
        self.seeds = (
            [int(s) for s in args.seeds.split(",")] if getattr(args, "seeds", None)
            else list(self.cfg.get("seeds", [0]))
        )
        if getattr(args, "quick", False):
            self.seeds = self.seeds[:1]

        self.results_dir = Path(
            getattr(args, "results_dir", None)
            or (self.cfg.get("output", {}) or {}).get("results_dir", "results")
        )

        self._describe_printed = False

    # -- helpers -----------------------------------------------------------

    def loaders(self, seed: int) -> Dict[str, Any]:
        return build_loaders(
            self.data, batch_size=self.batch_size, num_workers=self.num_workers,
            standardize=bool((self.cfg.get("data", {}) or {}).get("standardize", True)),
            ratios=(self.cfg.get("data", {}) or {}).get("split_ratios"),
            seed=seed,
        )

    def build_model(self,
                    recad: RecADConfig,
                    ordering: str = "topology",
                    data: Optional[Dict[str, Any]] = None,
                    seed: Optional[int] = None) -> RecADDetector:
        """Build the model, seeding FIRST when a seed is given.

        `run_one` seeds at the start of training, which is too late: the
        weights are drawn in the constructor, so without seeding here every
        cell in a run is initialised from whatever RNG state the previous cell
        happened to leave behind. Measured consequence -- in one ablation,
        `none` and `only:pipeline` are computationally identical (the pipeline
        flag is a no-op) yet scored AUC 0.7927 and 0.5007, because they were
        built at different points in the RNG stream. Run in isolation with the
        seed applied before construction, both give bit-identical numbers.

        Pass `seed` from any caller that reports per-seed results.
        """
        if seed is not None:
            set_seed(int(seed))
        data = self.data if data is None else data
        perm = make_permutation(ordering, self.spec, data) if ordering else None
        return RecADDetector(self.spec, self.cfg, name=f"Rec-AD[{recad.label()}]",
                             recad=recad, permutation=perm, data=data)

    def describe_once(self) -> None:
        if not self._describe_printed and not getattr(self.args, "quiet", False):
            print(describe_contract(self.args.data))
            print()
            self._describe_printed = True

    def header(self, title: str, extra: Optional[Dict[str, Any]] = None) -> None:
        print("=" * 92)
        print(title)
        print("=" * 92)
        print(f"data        : {self.args.data}")
        print(f"device      : {self.device}")
        print(f"tier        : {self.tier_name}  "
              f"(publishable={get_tier(self.tier_name).publishable})")
        print(f"seeds       : {self.seeds}")
        if extra:
            for key, value in extra.items():
                print(f"{key:<12}: {value}")
        print()

    def run_one(self,
                model: RecADDetector,
                seed: int,
                row: Dict[str, Any],
                measure: bool = False) -> Optional[Dict[str, Any]]:
        """Train, evaluate (and optionally profile) one model configuration."""
        set_seed(seed)
        loaders = self.loaders(seed)

        t0 = time.perf_counter()
        try:
            hist = model.fit(loaders["train"], loaders["val"], device=self.device)
        except torch.cuda.OutOfMemoryError:
            self.log.warning("%s: OOM during training", model.name)
            row.update({"seed": seed, "train_oom": 1, "fits_budget": 0.0})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return row

        fit_seconds = time.perf_counter() - t0
        if self.target_fpr is not None:
            thr_strategy, thr_fpr = "target_fpr", self.target_fpr
        else:
            thr_strategy, thr_fpr = "max_f1", None
        metrics, *_ = evaluate_detector(
            model, loaders, attack_type_test=self.data["attack_type"],
            threshold_strategy=thr_strategy, target_fpr=thr_fpr,
            device=self.device,
        )
        row["target_fpr"] = thr_fpr
        row.update(metrics)
        row.update({
            "seed": seed,
            "tier": self.tier_name,
            "fit_seconds": fit_seconds,
            "train_seconds": hist.get("train_seconds", float("nan")),
            "n_epochs_run": hist.get("n_epochs_run", 0),
            "n_parameters": model.n_parameters(),
        })

        if measure:
            mcfg = self.cfg.get("measure", {}) or {}
            lcfg = mcfg.get("latency", {}) or {}
            sample = next(iter(loaders["test"]))
            one = {k: (v[:1] if torch.is_tensor(v) and v.dim() > 0 else v)
                   for k, v in sample.items()}
            iters = int(self.args.measure_iters or lcfg.get("measure_iters", 500))
            warmup = int(lcfg.get("warmup_iters", 50))
            if getattr(self.args, "quick", False):
                iters, warmup = 15, 3
            prof = profile_resources(
                model, one, tier_name=self.tier_name, budget=self.budget,
                warmup_iters=warmup, measure_iters=iters,
                percentiles=tuple(lcfg.get("percentiles", [50, 95, 99])),
                measure_energy=not getattr(self.args, "no_energy", False),
                energy_interval_ms=int((mcfg.get("energy", {}) or {})
                                       .get("sample_interval_ms", 100)),
                device=self.device,
            )
            row.update(prof.as_dict())
            row["profile_notes"] = "; ".join(prof.notes)
        return row

    def writer(self, name: str) -> ResultsWriter:
        return ResultsWriter(name=name, results_dir=self.results_dir)

    def finish(self, writer: ResultsWriter,
               group_keys: List[str],
               metric_keys: List[str],
               extra: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        from recad.common import aggregate_seeds

        summary = aggregate_seeds(writer.rows, group_keys, metric_keys)
        payload = {
            "config": self.cfg,
            "environment": describe_environment(self.tier_name),
            "budget": self.budget,
            "data_file": str(self.args.data),
            "seeds": self.seeds,
            "summary": summary,
        }
        payload.update(extra or {})
        writer.write(extra=payload)

        import json
        (self.results_dir / f"{writer.name}_summary.json").write_text(
            json.dumps(summary, indent=2, default=float), encoding="utf-8"
        )
        print()
        print(f"Results written to {self.results_dir}/{writer.name}.json / .csv")
        self._warn_reference_backend()
        return summary

    def _warn_reference_backend(self) -> None:
        """Refuse to let reference-implementation timings pass as results.

        The PyTorch TT table exists to define Eff-TT's semantics and to make
        the ablation runnable without the authors' CUDA kernel. Its per-group
        Python loop costs far more than the tensor work it saves, so ITS
        TIMINGS ARE NOT THE PAPER'S PERFORMANCE CLAIMS. Reporting them would
        understate Rec-AD by an order of magnitude and hand a reviewer an easy
        refutation.
        """
        backend = str(self.cfg.get("embedding_backend", "reference"))
        if backend != "reference":
            return
        print()
        print("!" * 84)
        print("WARNING: results above used the PYTHON REFERENCE TT backend.")
        print("  * Correctness numbers (error, compression ratio, convergence)")
        print("    are valid and reportable.")
        print("  * TIMING numbers are NOT. The reference implementation loops")
        print("    over index groups in Python; its wall clock is dominated by")
        print("    interpreter overhead, not by the algorithm.")
        print("  * Re-run on the CUDA Eff-TT backend before quoting any")
        print("    speedup, latency or throughput figure in the paper.")
        print("    Set embedding_backend and cuda_efftt_module in config.yaml.")
        print("!" * 84)
