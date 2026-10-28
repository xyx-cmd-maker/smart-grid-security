"""Shared utilities: seeding, device selection, config loading, result I/O."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Device tiers
# ---------------------------------------------------------------------------
# Resource numbers that appear in the paper may ONLY come from the
# "publishable" tiers. A100/H100/5090 have enough HBM that no baseline
# OOMs, which would silently invalidate the deployability argument.
# ---------------------------------------------------------------------------

PUBLISHABLE_TIERS = {"t4", "rtx2060", "jetson"}


@dataclass(frozen=True)
class DeviceTier:
    name: str
    vram_gb: float
    power_w: float
    publishable: bool
    description: str


DEVICE_TIERS: Dict[str, DeviceTier] = {
    "jetson": DeviceTier(
        "jetson", 8.0, 25.0, True,
        "Embedded edge device (Jetson Orin Nano/NX class). Strongest claim.",
    ),
    "rtx2060": DeviceTier(
        "rtx2060", 6.0, 160.0, True,
        "Industrial-grade GPU used in the paper's Table V.",
    ),
    "t4": DeviceTier(
        "t4", 16.0, 70.0, True,
        "Cloud edge GPU (AWS g4dn). Matches the paper's secondary platform.",
    ),
    "h100": DeviceTier(
        "h100", 80.0, 700.0, False,
        "DEBUG ONLY. 80GB HBM3 -> no baseline OOMs. Never report these numbers.",
    ),
    "a100": DeviceTier(
        "a100", 80.0, 400.0, False,
        "DEBUG ONLY. Same reason as h100.",
    ),
    "local": DeviceTier(
        "local", 0.0, 0.0, False,
        "DEBUG ONLY. Unknown hardware; do not report.",
    ),
}


def get_tier(name: str) -> DeviceTier:
    key = name.strip().lower()
    if key not in DEVICE_TIERS:
        raise KeyError(
            f"Unknown device tier {name!r}. "
            f"Valid: {sorted(DEVICE_TIERS)}"
        )
    return DEVICE_TIERS[key]


def warn_if_unpublishable(tier_name: str) -> None:
    tier = get_tier(tier_name)
    if not tier.publishable:
        logging.getLogger("recad").warning(
            "Device tier %r is NOT publishable (%s). Resource numbers produced "
            "here must not be reported in the paper.",
            tier.name, tier.description,
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that can affect a result.

    Called at the top of every experiment entry point. The seed is written
    into the result JSON so reported numbers are traceable to a run.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:  # torch is required in practice, but stay importable
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cudnn autotuning picks different kernels per run, which breaks
        # bit-exact reproducibility of latency measurements.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "cuda") -> "Any":
    import torch

    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def describe_environment(tier_name: str = "local") -> Dict[str, Any]:
    """Capture the run environment. Written into every result JSON.

    Reviewers asking "on what hardware?" must be answerable from the
    artifacts themselves, not from memory.
    """
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device_tier": tier_name,
        "tier_publishable": get_tier(tier_name).publishable,
    }
    try:
        import torch
        env["torch"] = torch.__version__
        # The thread count changes the bits for every operation in this pipeline, so it is
        # recorded with the results rather than left to whichever shell ran the job.
        try:
            env["torch_num_threads"] = int(torch.get_num_threads())
            env["torch_num_interop_threads"] = int(torch.get_num_interop_threads())
        except Exception:
            env["torch_num_threads"] = None
            env["torch_num_interop_threads"] = None
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env[var.lower()] = os.environ.get(var)

        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            env["gpu_total_mem_gb"] = [
                round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        env["torch"] = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            env["nvidia_driver"] = out.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.SubprocessError, IndexError):
        pass
    return env


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("recad")


# ---------------------------------------------------------------------------
# Result writing
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)!r}")


@dataclass
class ResultsWriter:
    """Accumulates rows and writes <name>.json + <name>.csv.

    Both files carry the same records; the CSV is for eyeballing and the
    JSON is for the plotting script and for archiving.
    """

    name: str
    results_dir: Path
    rows: list = field(default_factory=list)
    _env_cache: dict | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def add(self, **row: Any) -> None:
        self.rows.append(row)
        self.checkpoint()

    def extend(self, rows: Iterable[Dict[str, Any]]) -> None:
        for r in rows:
            self.rows.append(dict(r))
        self.checkpoint()

    def _env(self) -> dict:
        """Environment block, computed once and reused across checkpoints.

        A crashed run used to leave records with no record of the machine that produced them, because the
        block was written only by the final write. Recording it here means any salvageable row is attributable.
        """
        if self._env_cache is None:
            self._env_cache = describe_environment()
        return self._env_cache

    def checkpoint(self) -> None:
        """Persist the rows accumulated so far, after every measurement.

        Why this exists: results used to be written only by `write()`, called
        once when a run finished. A 90-minute ablation was then interrupted at
        cell 2 of 8 and **every completed cell was lost** -- the numbers had
        been computed and were simply never flushed. Long runs on a
        memory-constrained box will keep getting interrupted, so partial
        results must survive.

        Writes `<name>.partial.json`, which is not read by the plotting script;
        `write()` still produces the authoritative `<name>.json` / `.csv`.
        Failures are swallowed: bookkeeping must never kill a running
        experiment.
        """
        import json

        try:
            (self.results_dir / f"{self.name}.partial.json").write_text(
                json.dumps(
                    {
                        "experiment": self.name,
                        "partial": True,
                        "n_records": len(self.rows),
                        "environment": self._env(),
                        "records": self.rows,
                    },
                    indent=2,
                    default=float,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def write(self, extra: Optional[Dict[str, Any]] = None) -> tuple[Path, Path]:
        payload = {
            "experiment": self.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_records": len(self.rows),
            "records": self.rows,
        }
        if extra:
            payload.update(extra)

        json_path = self.results_dir / f"{self.name}.json"
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)

        csv_path = self.results_dir / f"{self.name}.csv"
        if self.rows:
            import pandas as pd
            pd.DataFrame(self.rows).to_csv(csv_path, index=False)
        else:
            csv_path.write_text("", encoding="utf-8")

        logging.getLogger("recad").info("Wrote %s and %s", json_path, csv_path)
        return json_path, csv_path


def aggregate_seeds(rows: list[Dict[str, Any]],
                    group_keys: Iterable[str],
                    metric_keys: Iterable[str]) -> list[Dict[str, Any]]:
    """Collapse per-seed rows into mean +/- std per configuration.

    Reporting mean +/- std over 3 seeds is the minimum bar for a major
    revision; single-run numbers invite "is this stable?" in round two.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    group_keys = list(group_keys)
    metric_keys = [m for m in metric_keys if m in df.columns]
    if df.empty or not metric_keys:
        return []

    agg = df.groupby(group_keys, dropna=False)[metric_keys].agg(["mean", "std"])
    out: list[Dict[str, Any]] = []
    for idx, sub in agg.iterrows():
        rec: Dict[str, Any] = {}
        if isinstance(idx, tuple):
            rec.update(dict(zip(group_keys, idx)))
        else:
            rec[group_keys[0]] = idx
        for metric in metric_keys:
            mean = sub[(metric, "mean")]
            std = sub[(metric, "std")]
            rec[f"{metric}_mean"] = None if pd.isna(mean) else float(mean)
            rec[f"{metric}_std"] = 0.0 if pd.isna(std) else float(std)
        out.append(rec)
    return out
