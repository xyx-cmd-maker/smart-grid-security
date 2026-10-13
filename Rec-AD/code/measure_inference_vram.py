"""Inference-side VRAM, which is the quantity the deployability slots actually ask for.

`slots.json` defines `deployability.recad_vram` and `deployability.transformer_vram` as the peak
INFERENCE memory, and `config.yaml` sets `batch_size: 1  # streaming inference, matches Table V
of the paper`. The earlier envelope experiment measured training memory at batch 1024 and 4096
-- forward plus backward plus an optimiser step. Those are different quantities. Training memory
is larger, because the backward pass retains activations, so "training fits" implies "inference
fits", but the number itself cannot go into a slot that asks for inference.

This measures the slot's quantity: forward only, no gradients, at the streaming batch size.

It also reports the denominator question. `n_over_budget` is "the number of detectors exceeding
at least one envelope constraint", and the envelope has three constraints (memory, power,
latency). Only memory is measured here, so this script can bound one of the three terms and
nothing more; the docstring says so rather than implying the slot is now fillable.

Usage
-----
    python measure_inference_vram.py [--batch 1,32] [--data data_cache/ieee118.npz]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TORCH_DETECTORS = ["Autoencoder", "LSTM", "CNN", "GCN", "GAT", "Transformer"]
NON_TORCH = ["BDD", "TemporalResidual", "SVM", "RF"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default="1,32")
    ap.add_argument("--data", default="data_cache/ieee118.npz")
    ap.add_argument("--out", default="results/inference_vram.json")
    args = ap.parse_args()

    import torch
    from data.contract import load_contract
    from data.dataset import build_spec, build_loaders
    from recad.common import load_config, set_seed, get_tier
    from baselines.registry import build_detector

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = HERE / data_path
    d = load_contract(data_path)
    spec = build_spec(d)
    cfg = load_config(None)
    budget = cfg.get("budget") or {}
    vram_mb = float(budget.get("vram_gb", 8)) * 1024
    batches = [int(b) for b in args.batch.split(",")]

    if not torch.cuda.is_available():
        raise SystemExit("  ABORT: CUDA not available")
    dev = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"  device    {torch.cuda.get_device_name(0)}")
    print(f"  total     {total:.0f} MiB   envelope {vram_mb:.0f} MiB")
    print(f"  protocol  forward only (torch.no_grad), batches {batches}")
    print()
    print("  %-14s %6s %12s %9s  %s" % ("detector", "batch", "peak MiB", "ratio", "verdict"))
    print("  " + "-" * 62)

    rows = []
    for name in TORCH_DETECTORS:
        for bs in batches:
            set_seed(0)
            try:
                model = build_detector(name, spec, cfg, d)
            except Exception as exc:                        # noqa: BLE001
                print("  %-14s %6d %12s %9s  build_failed: %s" %
                      (name, bs, "-", "-", str(exc)[:28]))
                rows.append({"detector": name, "batch": bs, "peak_mib": None,
                             "verdict": "build_failed", "note": str(exc)[:160]})
                break
            if not isinstance(model, torch.nn.Module):
                rows.append({"detector": name, "batch": bs, "peak_mib": None,
                             "verdict": "not_applicable", "note": "not a torch module"})
                break
            try:
                loaders = build_loaders(d, batch_size=bs, num_workers=0, seed=0)
                tl = loaders["test"] if isinstance(loaders, dict) and "test" in loaders \
                    else (loaders["train"] if isinstance(loaders, dict) else loaders[0])
                batch = next(iter(tl))
                model = model.to(dev).eval()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                with torch.no_grad():
                    out = model(b)
                peak = torch.cuda.max_memory_allocated() / 1024**2
                fits = peak <= vram_mb
                print("  %-14s %6d %12.0f %8.3fx  %s" %
                      (name, bs, peak, peak / vram_mb, "FITS" if fits else "OVER"))
                rows.append({"detector": name, "batch": bs, "peak_mib": round(peak, 1),
                             "ratio_to_envelope": round(peak / vram_mb, 3),
                             "verdict": "fits" if fits else "over_envelope"})
                del model, b, out
                gc.collect()
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print("  %-14s %6d %12s %9s  OOM" % (name, bs, "-", "-"))
                rows.append({"detector": name, "batch": bs, "peak_mib": None, "verdict": "oom"})
                gc.collect()
                torch.cuda.empty_cache()
                break
            except Exception as exc:                        # noqa: BLE001
                print("  %-14s %6d %12s %9s  %s: %s" %
                      (name, bs, "-", "-", type(exc).__name__, str(exc)[:28]))
                rows.append({"detector": name, "batch": bs, "peak_mib": None,
                             "verdict": "error", "note": f"{type(exc).__name__}: {str(exc)[:120]}"})
                gc.collect()
                torch.cuda.empty_cache()
                break

    for name in NON_TORCH:
        rows.append({"detector": name, "batch": None, "peak_mib": None,
                     "verdict": "not_applicable", "note": "not a torch model"})

    over = [r for r in rows if r.get("verdict") == "over_envelope"]
    out_blob = {
        "device": torch.cuda.get_device_name(0),
        "total_mib": round(total, 1),
        "envelope_mib": vram_mb,
        "protocol": "forward only under torch.no_grad, eval mode",
        "tier": "local",
        "publishable": False,
        "covers": "MEMORY constraint only",
        "does_not_cover": ["power_w", "latency_ms"],
        "note": ("Covers one of the three envelope constraints. `n_over_budget` counts detectors "
                 "exceeding ANY constraint, so this cannot fill that slot on its own. tier=local "
                 "is not publishable per config.yaml measure.tier and AGENTS.md 4.6."),
        "memory_over_envelope": [r["detector"] for r in over],
        "rows": rows,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_blob, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"  memory-over-envelope: {out_blob['memory_over_envelope'] or 'none'}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
