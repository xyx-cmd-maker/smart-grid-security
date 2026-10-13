"""Bisect the first epoch: is the divergence in the data order, the initialisation, or the step?

Training the same cell twice with the same seed gives different weight hashes after one epoch --
three runs, three hashes, no overlap. So the cause acts immediately rather than accumulating.
`torch.argsort` in the table's lookup was the most specific suspect and is now ruled out
(identical across four processes and both thread settings), as are hash-order dependence, the
table's construction, the thread count as an instability source, and a single backward pass.

This script prints three hashes from one process, in the order the quantities become available:

  INIT   the weights straight after construction, before any data is touched
  DATA   the first training batch, as the loader actually yields it
  STEP   the weights after one optimizer step on that batch

Run it twice and read which hash differs first. That names the stage:

  INIT differs   -> seeding-before-construction is not holding as assumed
  DATA differs   -> the loader or the seeding order around it
  STEP differs   -> the forward/backward/optimizer, given identical inputs

Hashing the batch matters more than it looks: if the loader hands out a different order, every
downstream number changes while every individual operation stays perfectly deterministic, which
is exactly the pattern the earlier single-operation probes kept failing to find.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "experiments"))


def tensor_hash(t) -> str:
    import torch
    h = hashlib.sha256()
    if isinstance(t, dict):
        for k in sorted(t):
            h.update(k.encode())
            h.update(tensor_hash(t[k]).encode())
    elif torch.is_tensor(t):
        h.update(t.detach().cpu().float().numpy().tobytes())
    elif isinstance(t, (list, tuple)):
        for x in t:
            h.update(tensor_hash(x).encode())
    else:
        h.update(repr(t).encode())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tt", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--data", default="data_cache/ieee118.npz")
    args = ap.parse_args()

    import argparse as _ap
    import torch
    from _harness import ExperimentContext
    from recad.model import RecADConfig
    from recad.common import set_seed

    ns = _ap.Namespace(
        config="config.yaml", data=args.data, tier="local", num_workers=0,
        no_energy=True, target_fpr=0.01, measure_tier=None, epochs=1,
        batch_size=args.batch_size, orderings="topology", seeds="0", tag="probe",
    )
    ctx = ExperimentContext(ns)
    recad = RecADConfig.from_cfg(ctx.cfg, use_eff_tt=bool(args.tt),
                                 use_reorder=False, use_pipeline=False)
    model = ctx.build_model(recad, ordering="topology", seed=args.seed)

    init_h = tensor_hash(model.state_dict())

    set_seed(args.seed)
    loaders = ctx.loaders(args.seed)

    train_loader = loaders["train"]
    it = iter(train_loader)
    first = next(it)
    data_h = tensor_hash(first)

    # One optimizer step on that batch, using the model's own training configuration.
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(args.steps):
        batch = {k: (v.to(ctx.device) if torch.is_tensor(v) else v) for k, v in first.items()
                 if k in ("dense", "sparse")}
        y = first.get("label")
        y = y.to(ctx.device).float() if torch.is_tensor(y) else None
        opt.zero_grad()
        out = model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        opt.step()
    step_h = tensor_hash(model.state_dict())

    print(f"  tt={args.tt} seed={args.seed} batch={args.batch_size} steps={args.steps} "
          f"threads={torch.get_num_threads()}")
    print(f"  INIT {init_h}")
    print(f"  DATA {data_h}")
    print(f"  STEP {step_h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
