"""Resource measurement: latency distribution, peak VRAM, energy.

This module exists because of Reviewer 3 (comment 5): the paper reported a
single inference latency number (25 ms -> 21.5 ms) with no description of how
it was measured, and Reviewer 2 (comment 5) asked for energy. Both are fixed
here by measuring the same quantity the same way for every model, and by
reporting a distribution rather than a point estimate.

Measurement boundaries, stated explicitly (report these in the paper):
  * latency   : forward pass only, batch size 1, steady state after warmup
  * VRAM      : peak allocated during the measurement loop, weights included
  * energy    : GPU board power sampled during inference, idle power
                subtracted. NOT whole-system power.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from recad.common import DeviceTier, get_tier
from recad.detector import DetectorBase, _to_device

# ---------------------------------------------------------------------------
# Power sampling
# ---------------------------------------------------------------------------

class PowerSampler:
    """Samples GPU board power in a background thread.

    Uses pynvml when available and falls back to parsing `nvidia-smi`. The
    fallback matters on locked-down machines where only the CLI is allowed.
    """

    def __init__(self, interval_ms: int = 100, device_index: int = 0) -> None:
        self.interval = max(interval_ms, 10) / 1000.0
        self.device_index = device_index
        self.samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml = None
        self._handle = None
        self._cli = shutil.which("nvidia-smi")
        self.available = False
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.available = True
            return
        except Exception:
            self._nvml = None

        if self._cli:
            try:
                out = subprocess.run(
                    [self._cli, "-i", str(self.device_index),
                     "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                self.available = out.returncode == 0
            except (subprocess.SubprocessError, OSError):
                self.available = False

    def _read(self) -> Optional[float]:
        if self._nvml is not None and self._handle is not None:
            try:
                return float(self._nvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
            except Exception:
                return None
        if self._cli:
            try:
                out = subprocess.run(
                    [self._cli, "-i", str(self.device_index),
                     "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if out.returncode == 0:
                    return float(out.stdout.strip().splitlines()[0])
            except (subprocess.SubprocessError, ValueError, OSError, IndexError):
                return None
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            w = self._read()
            if w is not None:
                self.samples.append(w)
            self._stop.wait(self.interval)

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @property
    def mean_watts(self) -> float:
        return float(np.mean(self.samples)) if self.samples else float("nan")

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


def measure_idle_power(seconds: float = 5.0, interval_ms: int = 100) -> float:
    """Idle GPU power baseline, subtracted from measured inference power."""
    sampler = PowerSampler(interval_ms=interval_ms)
    if not sampler.available:
        sampler.close()
        return float("nan")
    sampler.start()
    time.sleep(seconds)
    sampler.stop()
    watts = sampler.mean_watts
    sampler.close()
    return watts


# ---------------------------------------------------------------------------
# Profile result
# ---------------------------------------------------------------------------

@dataclass
class ResourceProfile:
    latency_ms_mean: float = float("nan")
    latency_ms_p50: float = float("nan")
    latency_ms_p95: float = float("nan")
    latency_ms_p99: float = float("nan")
    throughput_tps: float = float("nan")
    peak_vram_mb: float = float("nan")
    model_bytes: float = float("nan")
    energy_j_per_inference: float = float("nan")
    gpu_power_w: float = float("nan")
    oom: bool = False
    fits_vram_budget: bool = True
    fits_power_budget: bool = True
    fits_latency_budget: bool = True
    notes: List[str] = field(default_factory=list)

    @property
    def fits_budget(self) -> bool:
        return (self.fits_vram_budget and self.fits_power_budget
                and self.fits_latency_budget and not self.oom)

    def as_dict(self, prefix: str = "") -> Dict[str, Any]:
        p = f"{prefix}_" if prefix else ""
        return {
            f"{p}latency_ms_mean": self.latency_ms_mean,
            f"{p}latency_ms_p50": self.latency_ms_p50,
            f"{p}latency_ms_p95": self.latency_ms_p95,
            f"{p}latency_ms_p99": self.latency_ms_p99,
            f"{p}throughput_tps": self.throughput_tps,
            f"{p}peak_vram_mb": self.peak_vram_mb,
            f"{p}model_bytes": self.model_bytes,
            f"{p}energy_j_per_inference": self.energy_j_per_inference,
            f"{p}gpu_power_w": self.gpu_power_w,
            f"{p}oom": float(self.oom),
            f"{p}fits_budget": float(self.fits_budget),
        }


# ---------------------------------------------------------------------------
# Main profiling routine
# ---------------------------------------------------------------------------

def profile_resources(model: DetectorBase,
                      sample_batch: Dict[str, torch.Tensor],
                      tier_name: str = "t4",
                      budget: Optional[Dict[str, Any]] = None,
                      warmup_iters: int = 50,
                      measure_iters: int = 500,
                      percentiles: tuple = (50, 95, 99),
                      measure_energy: bool = True,
                      energy_interval_ms: int = 100,
                      idle_watts: float = float("nan"),
                      device: Optional[torch.device] = None) -> ResourceProfile:
    """Measure one model under the deployment budget.

    A batch size of 1 is the default because the deployment scenario is
    streaming inference on a substation node, which is also what the paper's
    Table V claims. Reporting batch-1 latency makes the comparison to that
    table valid.
    """
    tier: DeviceTier = get_tier(tier_name)
    budget = budget or {}
    profile = ResourceProfile(model_bytes=float(model.footprint_bytes()))

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    vram_limit_gb = budget.get("vram_gb", tier.vram_gb)
    power_limit_w = budget.get("power_w", tier.power_w)
    latency_limit_ms = budget.get("latency_ms", float("inf"))

    prepared = _to_device(sample_batch, device,
                          needs_node_feat=model.needs_node_feat,
                          needs_adj=model.needs_adj,
                          needs_window=model.needs_window)

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # ---- warmup ---------------------------------------------------------
    try:
        with torch.no_grad():
            for _ in range(max(1, warmup_iters)):
                model(prepared)
                if use_cuda:
                    torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        profile.oom = True
        profile.notes.append("OOM during warmup at batch size 1")
        profile.fits_vram_budget = False
        if use_cuda:
            torch.cuda.empty_cache()
        return profile

    # ---- idle power -----------------------------------------------------
    if measure_energy and use_cuda and np.isnan(idle_watts):
        idle_watts = measure_idle_power(seconds=3.0, interval_ms=energy_interval_ms)

    sampler = None
    if measure_energy and use_cuda:
        sampler = PowerSampler(interval_ms=energy_interval_ms)
        if sampler.available:
            sampler.start()

    # ---- timed loop -----------------------------------------------------
    starts = torch.cuda.Event(enable_timing=True) if use_cuda else None
    ends = torch.cuda.Event(enable_timing=True) if use_cuda else None
    times_ms: List[float] = []

    t_wall = time.perf_counter()
    try:
        with torch.no_grad():
            for _ in range(max(1, measure_iters)):
                if use_cuda:
                    starts.record()
                    model(prepared)
                    ends.record()
                    torch.cuda.synchronize()
                    times_ms.append(starts.elapsed_time(ends))
                else:
                    t0 = time.perf_counter()
                    model(prepared)
                    times_ms.append((time.perf_counter() - t0) * 1000.0)
    except torch.cuda.OutOfMemoryError:
        profile.oom = True
        profile.notes.append("OOM during the latency loop at batch size 1")
        profile.fits_vram_budget = False
        if sampler is not None:
            sampler.stop()
            sampler.close()
        if use_cuda:
            torch.cuda.empty_cache()
        return profile

    wall = time.perf_counter() - t_wall

    if sampler is not None:
        sampler.stop()
        profile.gpu_power_w = sampler.mean_watts
        sampler.close()

    arr = np.asarray(times_ms, dtype=np.float64)
    profile.latency_ms_mean = float(arr.mean())
    for p in percentiles:
        setattr(profile, f"latency_ms_p{int(p)}", float(np.percentile(arr, p)))
    profile.throughput_tps = float(measure_iters / wall) if wall > 0 else float("nan")

    if use_cuda:
        profile.peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / 1024**2)

    # ---- energy ---------------------------------------------------------
    if not np.isnan(profile.gpu_power_w):
        active_w = profile.gpu_power_w
        if not np.isnan(idle_watts):
            active_w = max(active_w - idle_watts, 0.0)
            profile.notes.append(f"idle baseline {idle_watts:.1f} W subtracted")
        profile.energy_j_per_inference = float(active_w * (arr.mean() / 1000.0))
    elif measure_energy:
        profile.notes.append("power sampling unavailable; energy not measured")

    # ---- budget verdict -------------------------------------------------
    profile.fits_vram_budget = (
        np.isnan(profile.peak_vram_mb)
        or profile.peak_vram_mb <= vram_limit_gb * 1024
    )
    profile.fits_power_budget = (
        np.isnan(profile.gpu_power_w) or profile.gpu_power_w <= power_limit_w
    )
    profile.fits_latency_budget = profile.latency_ms_p99 <= latency_limit_ms

    if not profile.fits_vram_budget:
        profile.notes.append(
            f"peak VRAM {profile.peak_vram_mb:.0f} MB exceeds "
            f"{vram_limit_gb * 1024:.0f} MB budget"
        )
    if not profile.fits_power_budget:
        profile.notes.append(
            f"power {profile.gpu_power_w:.1f} W exceeds {power_limit_w:.0f} W budget"
        )
    if not profile.fits_latency_budget:
        profile.notes.append(
            f"p99 latency {profile.latency_ms_p99:.1f} ms exceeds "
            f"{latency_limit_ms:.0f} ms budget"
        )

    if not tier.publishable:
        profile.notes.append(
            f"tier {tier.name!r} is NOT publishable - do not report these numbers"
        )
    return profile


def budget_table(profiles: Dict[str, ResourceProfile],
                 tier_name: str = "t4",
                 budget: Optional[Dict[str, Any]] = None) -> str:
    """Render the deployability verdict table.

    This is the paper's new Table: which FDIA detectors can actually run
    inside a substation node's resource envelope, and which cannot.
    """
    tier = get_tier(tier_name)
    budget = budget or {}
    vram = budget.get("vram_gb", tier.vram_gb)
    power = budget.get("power_w", tier.power_w)
    lat = budget.get("latency_ms", float("inf"))

    lines = [
        f"Deployability against the substation-node budget "
        f"({vram:g} GB VRAM / {power:g} W / {lat:g} ms)",
        "=" * 104,
        f"{'model':<14}{'p50 ms':>9}{'p99 ms':>9}{'VRAM MB':>10}"
        f"{'power W':>9}{'mJ/inf':>10}{'verdict':>14}",
        "-" * 104,
    ]
    for name, p in profiles.items():
        if p.oom:
            verdict = "OOM"
        elif not p.fits_budget:
            verdict = "OVER BUDGET"
        else:
            verdict = "FITS"
        mj = (p.energy_j_per_inference * 1000.0
              if not np.isnan(p.energy_j_per_inference) else float("nan"))
        lines.append(
            f"{name:<14}{p.latency_ms_p50:>9.2f}{p.latency_ms_p99:>9.2f}"
            f"{p.peak_vram_mb:>10.0f}{p.gpu_power_w:>9.1f}{mj:>10.2f}"
            f"{verdict:>14}"
        )
    lines.append("-" * 104)
    n_fit = sum(1 for p in profiles.values() if p.fits_budget)
    lines.append(f"{n_fit}/{len(profiles)} models fit the budget.")
    return "\n".join(lines)
