"""Envelope binding: which baselines actually fit in the deployment envelope?

The paper's central claim is that the envelope binds -- that most baselines do NOT fit an 8 GB
node, and that the compressed model is what makes deployment possible. That claim is only
observable on a device at or below the envelope. A 16 GB T4 cannot show it, because nothing
runs out of memory there; the measurement would come back "everything fits" and say nothing
about the claim. The local GPU reports 8151 MiB, which is 7.96 GiB and therefore does touch the
constraint.

This runs each detector that can use a GPU, at a ladder of batch sizes, and records the peak
allocation and whether it raised out-of-memory. Non-torch baselines (BDD, TemporalResidual,
SVM, RF) are recorded as not applicable rather than silently omitted, and a detector that
cannot be built at all is recorded as such -- a build failure is a finding too, because it means
the row in the comparison table does not exist.

This measures TRAINING memory, which includes activations and is the binding quantity. It does
not include the optimiser state that Adam adds, so the numbers are a lower bound on what a full
training run needs; the report says so.

Envelope: 8 GB (8192 MiB) from config.yaml, `budget.vram_gb`.

Usage
-----
    python measure_envelope_binding.py [--batch 256,1024,4096] [--data data_cache/ieee118.npz]
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

# Order: cheap classical detectors first, then the deep ones, so an early crash still leaves
# partial results worth reporting.
ORDER = ["BDD", "TemporalResidual", "SVM", "RF", "Autoencoder", "LSTM", "CNN", "GCN", "GAT",
         "Transformer"]
NON_TORCH = {"BDD", "TemporalResidual", "SVM", "RF"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default="256,1024,4096")
    ap.add_argument("--data", default="data_cache/ieee118.npz")
    ap.add_argument("--out", default="results/envelope_binding.json")
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
    vram_budget_mb = float(budget.get("vram_gb", 8)) * 1024
    batches = [int(b) for b in args.batch.split(",")]

    if not torch.cuda.is_available():
        raise SystemExit("  ABORT: CUDA not available; this measures a GPU")
    dev = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"  device   {torch.cuda.get_device_name(0)}")
    print(f"  total    {total:.0f} MiB   envelope {vram_budget_mb:.0f} MiB")
    print(f"  batches  {batches}")
    print()
    print("  %-18s %8s %12s %10s   %s" % ("detector", "batch", "peak MiB", "verdict", "note"))
    print("  " + "-" * 78)

    rows = []
    for name in ORDER:
        if name in NON_TORCH:
            print("  %-18s %8s %12s %10s   %s" % (name, "-", "-", "n/a", "not a torch model"))
            rows.append({"detector": name, "batch": None, "peak_mib": None,
                         "verdict": "not_applicable", "note": "not a torch model"})
            continue

        for bs in batches:
            set_seed(0)
            try:
                model = build_detector(name, spec, cfg, d)
            except Exception as exc:                            # noqa: BLE001
                print("  %-18s %8d %12s %10s   %s: %s" %
                      (name, bs, "-", "BUILD FAIL", type(exc).__name__, str(exc)[:40]))
                rows.append({"detector": name, "batch": bs, "peak_mib": None,
                             "verdict": "build_failed",
                             "note": f"{type(exc).__name__}: {str(exc)[:120]}"})
                break

            is_torch = isinstance(model, torch.nn.Module)
            if not is_torch:
                print("  %-18s %8s %12s %10s   %s" % (name, "-", "-", "n/a", "not a torch module"))
                rows.append({"detector": name, "batch": None, "peak_mib": None,
                             "verdict": "not_applicable", "note": "not a torch module"})
                break

            try:
                loaders = build_loaders(d, batch_size=bs, num_workers=0, seed=0)
                tl = loaders["train"] if isinstance(loaders, dict) else loaders[0]
                batch = next(iter(tl))
                model = model.to(dev).train()
                opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                       lr=1e-3)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                y = b["label"].float().reshape(-1)
                opt.zero_grad(set_to_none=True)
                out = model(b)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if isinstance(out, dict):
                    out = next(iter(out.values()))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(out.reshape(-1), y)
                loss.backward()
                opt.step()
                peak = torch.cuda.max_memory_allocated() / 1024**2
                fits = peak <= vram_budget_mb
                print("  %-18s %8d %12.0f %10s   %s" %
                      (name, bs, peak, "FITS" if fits else "OVER",
                       f"{peak/vram_budget_mb:.1f}x envelope"))
                rows.append({"detector": name, "batch": bs, "peak_mib": round(peak, 1),
                             "verdict": "fits" if fits else "over_envelope",
                             "note": f"{peak/vram_budget_mb:.2f}x envelope"})
                del model, opt, b, out, loss
                gc.collect()
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print("  %-18s %8d %12s %10s   %s" %
                      (name, bs, "-", "OOM", "exceeded device memory"))
                rows.append({"detector": name, "batch": bs, "peak_mib": None, "verdict": "oom",
                             "note": "torch.cuda.OutOfMemoryError"})
                gc.collect()
                torch.cuda.empty_cache()
                break
            except Exception as exc:                            # noqa: BLE001
                print("  %-18s %8d %12s %10s   %s: %s" %
                      (name, bs, "-", "ERROR", type(exc).__name__, str(exc)[:40]))
                rows.append({"detector": name, "batch": bs, "peak_mib": None,
                             "verdict": "error", "note": f"{type(exc).__name__}: {str(exc)[:120]}"})
                gc.collect()
                torch.cuda.empty_cache()
                break

    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "device": torch.cuda.get_device_name(0),
        "total_mib": round(total, 1),
        "envelope_mib": vram_budget_mb,
        "tier": "local",
        "publishable": False,
        "note": ("Measured on the local RTX 5060 (7.96 GiB). DEVICE_TIERS['local'] has "
                 "publishable=False and config.yaml restricts reported resource numbers to "
                 "t4/rtx2060/jetson, so these numbers cannot enter the paper as they stand. "
                 "They answer whether the envelope binds, which a 16 GB device cannot. "
                 "Training memory only; Adam state is not included."),
        "rows": rows,
    }
    out.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
