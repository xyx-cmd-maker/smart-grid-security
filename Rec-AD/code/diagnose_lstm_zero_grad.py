"""Isolate the LSTM's zero gradient: is it the device, the physics path, or the model?

`verify_gpu_train.py` reported `grad_norm=0.0000` for the LSTM after it was corrected to pass
`data=`, which enables the residual-projection path in `baselines/_sequence.py`. The same
script previously reported a non-zero gradient for the LSTM when it was building models
WITHOUT `data=`, so the physics path had never been exercised and the zero was invisible.

Two things need separating before anything is written down:

  * is the zero caused by the physics path, or would it appear without it too?
  * is it caused by being on CUDA, or does it happen on CPU as well?

The second question decides whether existing results are affected. If the LSTM also produces
zero gradients on CPU with physics on, then every LSTM row already in the result files was
trained with a dead backward pass.

The grid is device x physics. Each cell reports the gradient norm, how many parameters have a
non-None gradient, and the loss, so a zero can be attributed to a specific part of the graph.

Usage
-----
    python diagnose_lstm_zero_grad.py [--model LSTM] [--batch 256] [--steps 2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="LSTM")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--data", default="data_cache/ieee118.npz")
    args = ap.parse_args()

    import torch
    from data.contract import load_contract
    from data.dataset import build_spec, build_loaders
    from recad.common import load_config
    from baselines.registry import build_detector

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = HERE / data_path
    d = load_contract(data_path)
    spec = build_spec(d)
    cfg = load_config(None)
    loaders = build_loaders(d, batch_size=args.batch, num_workers=0, seed=0)
    train_loader = loaders["train"] if isinstance(loaders, dict) else loaders[0]

    batches = []
    for i, b in enumerate(train_loader):
        batches.append(b)
        if i + 1 >= args.steps:
            break
    if not batches:
        raise SystemExit("  ABORT: no batches")
    print(f"  model {args.model}  batch {args.batch}  steps {len(batches)}")
    print()
    print("  %-6s %-8s %12s %10s %8s   %s" %
          ("device", "physics", "grad_norm", "n_grads", "loss", "verdict"))
    print("  " + "-" * 74)

    for dev_name in ("cpu", "cuda"):
        if dev_name == "cuda" and not torch.cuda.is_available():
            print(f"  {dev_name:<6} (skipped: no CUDA)")
            continue
        for use_physics in (False, True):
            dev = torch.device(dev_name)
            try:
                m = build_detector(args.model, spec, cfg, d if use_physics else None)
            except Exception as exc:                        # noqa: BLE001
                print(f"  {dev_name:<6} {str(use_physics):<8} build failed: "
                      f"{type(exc).__name__}: {str(exc)[:40]}")
                continue
            m = m.to(dev).train()
            opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
            lossf = torch.nn.functional.binary_cross_entropy_with_logits

            norms, n_grads, last_loss = [], 0, float("nan")
            for b in batches:
                bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                y = bb["label"].float().reshape(-1)
                opt.zero_grad(set_to_none=True)
                out = m(bb)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if isinstance(out, dict):
                    out = next(iter(out.values()))
                out = out.reshape(-1)
                loss = lossf(out, y)
                loss.backward()
                last_loss = float(loss.detach())
                g = [p.grad for p in m.parameters() if p.grad is not None]
                n_grads = len(g)
                norms.append(float(sum(float(x.norm()) ** 2 for x in g) ** 0.5) if g else 0.0)
                opt.step()

            worst = min(norms)
            if worst > 0:
                verdict = "OK"
            elif n_grads == 0:
                verdict = "NO GRAD AT ALL"
            else:
                verdict = "ZERO GRADIENT"
            print("  %-6s %-8s %12.6f %10d %8.4f   %s" %
                  (dev_name, use_physics, worst, n_grads, last_loss, verdict))

    print()
    print("  Reading: if the 'physics=True' rows are zero on BOTH devices, the physics path")
    print("  is the cause and any LSTM result trained through it is void. If only the CUDA")
    print("  rows are zero, the cause is device-specific and the CPU results stand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
