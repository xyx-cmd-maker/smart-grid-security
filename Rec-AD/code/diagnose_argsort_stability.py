"""Ask whether `torch.argsort` is stable across processes on a multi-threaded CPU.

Background: training the same cell twice with the same seed produced different weight hashes
after a single epoch -- three runs, three different hashes. That rules out slow amplification
over many iterations; the difference exists immediately. Four other candidates were already
eliminated: hash-order dependence, the table's construction (six processes, identical hashes),
the thread count as a source of instability, and a single backward through the table.

The remaining candidate is specific. `tt_table.py` groups rows with

    order = torch.argsort(prefix, stable=True)

and every row's output depends on that grouping. The construction test never exercised this
line -- `__init__` does not call `lookup` -- which is why it came back reproducible while
training does not. The dense table has no equivalent step.

This script tests the line directly, at the size and dtype the model actually uses:

  * is `argsort` output identical across processes?
  * does it depend on the thread count?

If the order varies across processes, the candidate is supported. If it is identical, the
search moves to the rest of the first epoch.
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
    ap.add_argument("--threads", type=int, default=0, help="0 leaves the default")
    ap.add_argument("--n", type=int, default=128, help="rows in a batch")
    ap.add_argument("--repeat", type=int, default=2000,
                    help="sorts per process; more chances to catch a rare reordering")
    ap.add_argument("--dup", type=int, default=4,
                    help="distinct prefix values; fewer values means more ties")
    args = ap.parse_args()

    import torch
    torch.manual_seed(0)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    # Prefix keys are integers in the model: they encode multi-indices, so ties are common
    # whenever several rows share a leading index. Few distinct values makes ties frequent,
    # which is the case stability is supposed to decide.
    acc = hashlib.sha256()
    changed = 0
    prev = None
    for i in range(args.repeat):
        prefix = torch.randint(0, args.dup, (args.n,), dtype=torch.long)
        order = torch.argsort(prefix, stable=True)
        acc.update(order.numpy().tobytes())
        if prev is not None and not torch.equal(order, prev):
            changed += 1
        prev = order

    doc = hashlib.sha256()
    prefix = torch.randint(0, args.dup, (args.n,), dtype=torch.long)
    order = torch.argsort(prefix, stable=True)
    doc.update(prefix.numpy().tobytes())
    doc.update(order.numpy().tobytes())

    print(f"  torch {torch.__version__}  threads={torch.get_num_threads()}  n={args.n} "
          f"dup={args.dup} repeat={args.repeat}")
    print(f"  ORDER_HASH {acc.hexdigest()[:16]}")
    print(f"  FIXED_HASH {doc.hexdigest()[:16]}   (one fixed input, for cross-process compare)")
    print(f"  consecutive-different: {changed}/{max(args.repeat - 1, 1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
