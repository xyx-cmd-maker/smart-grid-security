"""Isolate one lookup + backward on the tensor-train table at the real configuration.

The bisection narrowed this as far as the training loop allows. With identical initial weights
and an identical first batch, one forward/backward/step gives identical weights for the dense
table in two processes and four different sets of weights for the tensor-train table in four
processes. So the difference is inside the table's own forward and backward, not in seeding,
not in the loader, not in accumulation over many steps.

This script drops the rest of the model and tests that one path directly, at the size the
ablation actually builds rather than a convenient one:

    rows = 4,193,420   (reverse-engineered from the dense footprint, 268,378,880 / (16 * 4))
    dim  = 16
    rank = 32          (config tt.rank; rank > dim, so the factorisation truncates hard)
    reuse = True       (the grouping path, which `lookup` only takes when reuse is enabled)

The earlier single-backward attempt used 4096 rows -- a thousand times smaller -- and came back
reproducible, which is the same mistake as the construction probe that never called `lookup`.
Both were measuring a different object from the one that varies.

Indices come from a seeded generator so that both processes test the same rows. Gradients are
hashed rather than weights, because gradients are where a many-to-one accumulation would show.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=4193420)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--batch", type=int, default=128, help="rows looked up per step")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-reuse", action="store_true",
                    help="disable the grouping path, to see whether it is the grouping that varies")
    args = ap.parse_args()

    import torch
    from recad.common import set_seed
    from recad.tt_table import TTEmbeddingTable

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    set_seed(args.seed)
    table = TTEmbeddingTable(args.rows, args.dim, rank=args.rank,
                             reuse=not args.no_reuse)
    # Fixed indices for both processes: draw once from the seeded RNG, then reuse.
    idx = torch.randint(0, args.rows, (args.batch,), generator=torch.Generator().manual_seed(7))

    init = hashlib.sha256()
    for name, p in sorted(table.state_dict().items()):
        init.update(name.encode())
        init.update(p.detach().cpu().float().numpy().tobytes())

    opt = torch.optim.SGD(table.parameters(), lr=0.0)   # lr 0: weights must not move
    loss_vals = []
    for _ in range(args.steps):
        opt.zero_grad()
        out = table.lookup(idx)
        loss = out.float().pow(2).mean()
        loss.backward()
        loss_vals.append(float(loss))
        opt.step()

    grad = hashlib.sha256()
    for name, p in sorted(table.named_parameters()):
        g = p.grad
        if g is None:
            grad.update(b"<none>")
            continue
        grad.update(name.encode())
        grad.update(g.detach().cpu().float().numpy().tobytes())

    print(f"  torch {torch.__version__} threads={torch.get_num_threads()} "
          f"rows={args.rows} dim={args.dim} rank={args.rank} reuse={not args.no_reuse}")
    print(f"  loss      {loss_vals[0]:.10e}")
    print(f"  INIT_HASH {init.hexdigest()[:16]}")
    print(f"  GRAD_HASH {grad.hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
