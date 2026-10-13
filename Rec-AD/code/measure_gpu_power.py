"""Measure what this GPU actually draws while training, so the envelope question is answered.

The paper's envelope is 8 GB / 60 W. The local card reports `Default Power Limit = 50 W` and
`Current Power Limit = 115 W`, which are two different claims about what it does, and
`power.draw` at idle (17 W) says nothing about either. The number that matters is what it draws
under sustained training load. This script produces that load and holds it while an external
sampler reads the power.

Why not just run the existing verification: `verify_gpu_train.py` does one forward and one
backward, which lasts well under a second. GPU power ramps over seconds, so a single step
reads near idle and would answer the question wrongly. This holds the load for a fixed
duration instead.

The script prints its own VRAM high-water mark and, if `pynvml` is importable, samples power
itself. It does not require pynvml: the intended use is with an external `nvidia-smi` sampler
running alongside, and its job is only to keep the GPU busy and report memory.

Usage
-----
    python measure_gpu_power.py --seconds 150 [--model CNN] [--batch 256]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=150.0,
                    help="how long to hold the load; power needs seconds to ramp")
    ap.add_argument("--model", default="CNN")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--data", default="data_cache/ieee118.npz")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("  ABORT: CUDA not available; this measures a GPU")

    from data.contract import load_contract
    from data.dataset import build_spec, build_loaders
    from recad.common import load_config
    from baselines.registry import build_detector

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = HERE / data_path
    if not data_path.exists():
        raise SystemExit(f"  ABORT: {data_path} does not exist")

    d = load_contract(data_path)
    spec = build_spec(d)
    cfg = load_config(None)
    loaders = build_loaders(d, batch_size=args.batch, num_workers=0, seed=0)
    train_loader = loaders["train"] if isinstance(loaders, dict) else loaders[0]

    dev = torch.device("cuda")
    model = build_detector(args.model, spec, cfg, d).to(dev)
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    lossf = torch.nn.functional.binary_cross_entropy_with_logits

    props = torch.cuda.get_device_properties(0)
    print(f"  device     {torch.cuda.get_device_name(0)}")
    print(f"  memory     {props.total_memory/1024**3:.2f} GiB")
    print(f"  model      {args.model}, {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  batch      {args.batch}")
    print(f"  duration   {args.seconds:.0f} s")
    print()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = 0
    # `warmup` steps are excluded from the timing so the reported rate is steady-state.
    warmup = 5
    timed = 0.0

    while True:
        for batch in train_loader:
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            y = b.get("label")
            if y is None:
                raise SystemExit("  ABORT: batch has no label")
            opt.zero_grad(set_to_none=True)
            out = model(b)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if isinstance(out, dict):
                out = next(iter(out.values()))
            out = out.reshape(-1)
            loss = lossf(out, y.float().reshape(-1))
            loss.backward()
            opt.step()
            steps += 1
            if steps == warmup:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            if time.perf_counter() - t0 >= args.seconds:
                torch.cuda.synchronize()
                timed = time.perf_counter() - t0
                break
        else:
            continue
        break

    peak = torch.cuda.max_memory_allocated() / 1024**2
    rate = (steps - warmup) / timed if timed > 0 else float("nan")

    print(f"  steps      {steps}  ({rate:.1f} steps/s after {warmup} warmup)")
    print(f"  elapsed    {timed:.1f} s of timed load")
    print(f"  peak VRAM  {peak:.0f} MiB")
    print()
    print("  Power was sampled externally; see the sampler output for the peak and mean.")

    try:
        import pynvml                                          # optional
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mw = pynvml.nvmlDeviceGetPowerManagementLimit(h)
        print(f"  NVML limit {mw/1000:.1f} W")
        pynvml.nvmlShutdown()
    except Exception:                                          # noqa: BLE001
        print("  (pynvml not available; external sampler is the source of truth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
