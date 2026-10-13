"""The other two envelope constraints, plus Rec-AD's own inference memory.

`deployability.n_over_budget` counts detectors exceeding ANY of the three envelope constraints,
and only memory has been measured so far. Inference memory came back at 0.03x the envelope for
every baseline, so whether the envelope binds at all now depends on power and latency. This
measures them.

The latency protocol is fixed by the project rather than chosen here: `config.yaml` sets
`batch_size: 1  # streaming inference` and AGENTS.md 4.6 requires batch=1, warmup=50,
measure=500, reported as p50/p95/p99. The power budget is 60 W and the latency budget 500 ms.

It also builds Rec-AD itself, which `deployability.recad_vram` asks for and which the earlier
scripts did not cover -- they only built the baselines, so the slot's own quantity was missing
from the evidence.

Power is sampled externally by an nvidia-smi loop while this holds an inference load; the
script cannot read board power itself.

Usage
-----
    python measure_power_latency.py [--warmup 50] [--measure 500] [--data data_cache/ieee118.npz]
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

BASE_DETECTORS = ["Autoencoder", "LSTM", "CNN", "GCN", "GAT", "Transformer"]
NON_TORCH = ["BDD", "TemporalResidual", "SVM", "RF"]
# Rec-AD's four cells, as the ablation names them.
RECAD_CELLS = [
    ("Rec-AD (full)", dict(use_eff_tt=True, use_reorder=True, use_pipeline=True)),
    ("Rec-AD (no TT)", dict(use_eff_tt=False, use_reorder=True, use_pipeline=True)),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--measure", type=int, default=500)
    ap.add_argument("--data", default="data_cache/ieee118.npz")
    ap.add_argument("--out", default="results/power_latency.json")
    args = ap.parse_args()

    import torch
    from data.contract import load_contract
    from data.dataset import build_spec, build_loaders
    from recad.common import load_config, set_seed
    from baselines.registry import build_detector

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = HERE / data_path
    d = load_contract(data_path)
    spec = build_spec(d)
    cfg = load_config(None)
    budget = cfg.get("budget") or {}
    vram_mb = float(budget.get("vram_gb", 8)) * 1024
    lat_budget = float(budget.get("latency_ms", 500))
    pwr_budget = float(budget.get("power_w", 60))

    if not torch.cuda.is_available():
        raise SystemExit("  ABORT: CUDA not available")
    dev = torch.device("cuda")
    loaders = build_loaders(d, batch_size=1, num_workers=0, seed=0)
    tl = loaders["test"] if isinstance(loaders, dict) and "test" in loaders else loaders["train"]

    print(f"  device   {torch.cuda.get_device_name(0)}")
    print(f"  envelope {vram_mb:.0f} MiB / {pwr_budget:.0f} W / {lat_budget:.0f} ms")
    print(f"  protocol batch=1, warmup={args.warmup}, measure={args.measure}")
    print()
    print("  %-16s %10s %10s %10s %10s  %s" %
          ("model", "p50 ms", "p95 ms", "p99 ms", "peak MiB", "verdict"))
    print("  " + "-" * 76)

    rows = []
    models = [(n, "baseline", None) for n in BASE_DETECTORS]
    models += [(n, "recad", f) for n, f in RECAD_CELLS]

    for name, kind, flags in models:
        set_seed(0)
        try:
            if kind == "baseline":
                model = build_detector(name, spec, cfg, d)
            else:
                from recad.model import RecADConfig
                from _harness import ExperimentContext       # noqa: F401
                import argparse as _ap
                from experiments._harness import ExperimentContext as EC
                ns = _ap.Namespace(config="config.yaml", data=str(data_path), tier="local",
                                   num_workers=0, no_energy=True, target_fpr=0.01,
                                   measure_tier=None, epochs=1, batch_size=1,
                                   orderings="topology", seeds="0", tag="probe")
                ctx = EC(ns)
                recad = RecADConfig.from_cfg(ctx.cfg, **flags)
                model = ctx.build_model(recad, ordering="topology", seed=0)
        except Exception as exc:                            # noqa: BLE001
            print("  %-16s %10s %10s %10s %10s  build_failed: %s" %
                  (name, "-", "-", "-", "-", str(exc)[:26]))
            rows.append({"model": name, "verdict": "build_failed", "note": str(exc)[:180]})
            continue

        if not isinstance(model, torch.nn.Module):
            rows.append({"model": name, "verdict": "not_applicable",
                         "note": "not a torch module"})
            continue

        try:
            model = model.to(dev).eval()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # One fixed sample, reused so the timing excludes data loading.
            first = next(iter(tl))
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
            if p99 > lat_budget:
                over.append("p99")
            print("  %-16s %10.2f %10.2f %10.2f %10.0f  %s" %
                  (name, p50, p95, p99, peak, "OVER: " + ",".join(over) if over else "FITS"))
            rows.append({"model": name, "kind": kind, "p50_ms": round(p50, 3),
                         "p95_ms": round(p95, 3), "p99_ms": round(p99, 3),
                         "peak_mib": round(peak, 1),
                         "over": over, "verdict": "over_envelope" if over else "fits"})
            del model, b
            gc.collect()
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print("  %-16s %10s %10s %10s %10s  OOM" % (name, "-", "-", "-", "-"))
            rows.append({"model": name, "verdict": "oom"})
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:                            # noqa: BLE001
            print("  %-16s %10s %10s %10s %10s  %s: %s" %
                  (name, "-", "-", "-", "-", type(exc).__name__, str(exc)[:26]))
            rows.append({"model": name, "verdict": "error",
                         "note": f"{type(exc).__name__}: {str(exc)[:150]}"})
            gc.collect()
            torch.cuda.empty_cache()

    for n in NON_TORCH:
        rows.append({"model": n, "verdict": "not_applicable", "note": "not a torch model"})

    over_any = [r["model"] for r in rows if r.get("verdict") == "over_envelope"]
    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "envelope": {"vram_mib": vram_mb, "power_w": pwr_budget, "latency_ms": lat_budget},
        "protocol": f"batch=1, warmup={args.warmup}, measure={args.measure}, p50/p95/p99",
        "tier": "local",
        "publishable": False,
        "covers": ["latency_ms", "vram_mib"],
        "does_not_cover": ["power_w (sampled externally)"],
        "over_envelope": over_any,
        "note": ("tier=local is not publishable per config.yaml and AGENTS.md 4.6. "
                 "Power must be read from the external sampler."),
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"  over envelope (vram or p99 latency): {over_any or 'none'}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
