"""Rec-AD's own inference memory and latency: the quantity `deployability.recad_vram` asks for.

The earlier probes built only the baselines, so the slot that names Rec-AD itself had no
measurement behind it. The first attempt to include it failed with `No module named '_harness'`
because `experiments/` was not on `sys.path`; that is fixed here.

The same protocol as the baseline measurement is used so the numbers are comparable: batch=1,
warmup=50, measure=500, p50/p95/p99, forward only under torch.no_grad. Comparing a Rec-AD
number taken under one protocol against baseline numbers taken under another would be the
cross-frame comparison this project has already recorded as a recurring error.

The four ablation cells are measured, not just `full`, because the compression claim is about
the difference between the TT and dense tables, so the pair is the informative one.

Usage
-----
    python measure_recad_inference.py [--warmup 50] [--measure 500]
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments"))          # for _harness; its absence broke run 1

CELLS = [
    ("Rec-AD full (TT+reorder+pipeline)", dict(use_eff_tt=True, use_reorder=True,
                                               use_pipeline=True)),
    ("Rec-AD no-TT (reorder+pipeline)", dict(use_eff_tt=False, use_reorder=True,
                                             use_pipeline=True)),
    ("Rec-AD TT only", dict(use_eff_tt=True, use_reorder=False, use_pipeline=False)),
    ("Rec-AD none", dict(use_eff_tt=False, use_reorder=False, use_pipeline=False)),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--measure", type=int, default=500)
    ap.add_argument("--data", default="data_cache/ieee118.npz")
    ap.add_argument("--out", default="results/recad_inference.json")
    args = ap.parse_args()

    import torch
    import argparse as _ap
    from _harness import ExperimentContext
    from recad.model import RecADConfig
    from data.dataset import build_loaders
    from data.contract import load_contract

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = HERE / data_path
    d = load_contract(data_path)

    ns = _ap.Namespace(config="config.yaml", data=str(data_path), tier="local",
                       num_workers=0, no_energy=True, target_fpr=0.01, measure_tier=None,
                       epochs=1, batch_size=1, orderings="topology", seeds="0", tag="probe")
    ctx = ExperimentContext(ns)
    budget = ctx.cfg.get("budget") or {}
    vram_mb = float(budget.get("vram_gb", 8)) * 1024
    lat_ms = float(budget.get("latency_ms", 500))

    dev = torch.device("cuda")
    loaders = build_loaders(d, batch_size=1, num_workers=0, seed=0)
    tl = loaders["test"] if isinstance(loaders, dict) and "test" in loaders else loaders["train"]
    first = next(iter(tl))

    print(f"  device   {torch.cuda.get_device_name(0)}")
    print(f"  envelope {vram_mb:.0f} MiB / {lat_ms:.0f} ms")
    print(f"  protocol batch=1, warmup={args.warmup}, measure={args.measure}, forward only")
    print()
    print("  %-38s %9s %9s %9s %10s  %s" %
          ("cell", "p50 ms", "p95 ms", "p99 ms", "peak MiB", "verdict"))
    print("  " + "-" * 92)

    rows = []
    for name, flags in CELLS:
        try:
            recad = RecADConfig.from_cfg(ctx.cfg, **flags)
            model = ctx.build_model(recad, ordering="topology", seed=0)
        except Exception as exc:                            # noqa: BLE001
            print("  %-38s %9s %9s %9s %10s  build_failed: %s" %
                  (name, "-", "-", "-", "-", str(exc)[:28]))
            rows.append({"cell": name, "verdict": "build_failed", "note": str(exc)[:200]})
            continue
        try:
            model = model.to(dev).eval()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in first.items()}
            with torch.no_grad():
                for _ in range(args.warmup):
                    model(b)
                torch.cuda.synchronize()
                times = []
                for _ in range(args.measure):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model(b)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000.0)
            times.sort()
            p50 = statistics.median(times)
            p95 = times[int(0.95 * (len(times) - 1))]
            p99 = times[int(0.99 * (len(times) - 1))]
            peak = torch.cuda.max_memory_allocated() / 1024**2
            over = []
            if peak > vram_mb:
                over.append("vram")
            if p99 > lat_ms:
                over.append("p99")
            print("  %-38s %9.2f %9.2f %9.2f %10.0f  %s" %
                  (name, p50, p95, p99, peak, "OVER:" + ",".join(over) if over else "FITS"))
            rows.append({
                "cell": name, "flags": flags,
                "p50_ms": round(p50, 3), "p95_ms": round(p95, 3), "p99_ms": round(p99, 3),
                "peak_mib": round(peak, 1),
                "compression_ratio": (float(model.compression_ratio())
                                      if hasattr(model, "compression_ratio") else None),
                "embedding_bytes": (float(model.embedding_bytes())
                                    if hasattr(model, "embedding_bytes") else None),
                "verdict": "over_envelope" if over else "fits",
            })
            del model, b
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:                            # noqa: BLE001
            print("  %-38s %9s %9s %9s %10s  %s: %s" %
                  (name, "-", "-", "-", "-", type(exc).__name__, str(exc)[:30]))
            rows.append({"cell": name, "verdict": "error",
                         "note": f"{type(exc).__name__}: {str(exc)[:160]}"})
            gc.collect()
            torch.cuda.empty_cache()

    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "envelope": {"vram_mib": vram_mb, "latency_ms": lat_ms},
        "protocol": f"batch=1, warmup={args.warmup}, measure={args.measure}, forward only",
        "tier": "local",
        "publishable": False,
        "fills": "deployability.recad_vram (magnitude only; tier=local is not publishable)",
        "note": ("Same protocol as measure_power_latency.py so the numbers are comparable to "
                 "the baseline rows. AGENTS.md 4.6 restricts reported resource numbers to "
                 "t4/rtx2060/jetson, so these cannot go into the manuscript as they stand."),
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
