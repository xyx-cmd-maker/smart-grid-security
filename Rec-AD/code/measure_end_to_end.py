"""Measure input preparation, host-to-device transfer and forward pass as three separate segments.

The paper reports forward compute only and says so. The reviewer asks for the parts to be separated, because a sequence model reads
eight snapshots per decision while a single-snapshot model reads one, so a comparison at equal batch size is not a comparison at
equal input. This measures both models on the same data and the same device, with the three segments timed on their own.

Weights are initialised rather than trained. Timing depends on the shapes and the operations, not on the values, so trained weights
would change the numbers by nothing measurable; using untrained ones avoids a training run for a latency measurement. That is stated
in the output so nobody mistakes it for an accuracy run.

The protocol matches the one the paper reports: batch one, fifty warm-up iterations, five hundred timed, a synchronise before and
after each timed region, and the device named in the output since the charge-bearing numbers are the ones with a named platform.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

RESULTS = ROOT / "results"

WARMUP = 50
MEASURE = 500


def timed(fn, n):
    """Run fn n times after warm-up, returning p50/p95/p99 in milliseconds."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts)
    return {k: float(np.percentile(a, q)) for k, q in (("p50", 50), ("p95", 95), ("p99", 99))}


def main() -> int:
    from data.dataset import build_loaders, build_spec
    from baselines.registry import build_detector

    dev = torch.device("cuda")
    print(f"  device {torch.cuda.get_device_name(0)}")

    d = np.load(ROOT / "data_cache" / "ieee118.npz", allow_pickle=True)
    data = {k: d[k] for k in d.files}
    spec = build_spec(data)
    loaders = build_loaders(data, batch_size=1, num_workers=0, seed=0)

    rows = []
    for name in ("CNN", "Transformer"):
        model = build_detector(name, spec, {}).to(dev).eval()

        # segment 1: preparing one batch from the loader, on CPU
        it = iter(loaders["test"])
        first = next(it)

        def prep():
            try:
                next(it2)
            except StopIteration:
                pass

        it2 = iter(loaders["test"])
        s_prep = timed(prep, MEASURE)

        # segment 2: host-to-device transfer of that batch
        def xfer():
            for v in first.values():
                if torch.is_tensor(v):
                    v.to(dev)

        s_xfer = timed(xfer, MEASURE)

        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in first.items()}

        # segment 3: forward pass only
        def fwd():
            with torch.no_grad():
                model(b)

        s_fwd = timed(fwd, MEASURE)

        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        torch.cuda.reset_peak_memory_stats()

        row = {
            "model": name,
            "input_snapshots": 8 if name in ("Transformer", "LSTM", "CNN") and hasattr(spec, "window") else 1,
            "prep_ms": s_prep,
            "transfer_ms": s_xfer,
            "forward_ms": s_fwd,
            "end_to_end_p50_ms": s_prep["p50"] + s_xfer["p50"] + s_fwd["p50"],
            "peak_allocated_mib": peak,
        }
        rows.append(row)
        print(f"  {name:<14} prep {s_prep['p50']:.3f}  xfer {s_xfer['p50']:.3f}  "
              f"fwd {s_fwd['p50']:.3f}  sum {row['end_to_end_p50_ms']:.3f} ms")

    out = RESULTS / "end_to_end_segments.json"
    out.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        # This machine is not a T4. The tier is recorded as a claim, not an observation, and a
        # local-tier record is debug-only under the project rule; see the rule-violation note.
        "tier": "local",
        "protocol": f"batch=1, warmup={WARMUP}, measure={MEASURE}, three segments timed separately",
        "weights": "initialised, not trained: timing depends on shapes and operations, not values",
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
