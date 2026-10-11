"""Test whether pinning the BLAS thread count makes the tensor-train table reproducible.

The observation this tests: cells that build a tensor-train table vary between runs while
cells that use a dense table are bit-identical (17 of 17 across two reruns). The hypothesis
is that the TT path's SVD goes through multi-threaded BLAS, whose float32 reduction order
depends on thread scheduling, and that nothing in the codebase pins that order --
`set_seed` sets four RNGs and two cuDNN flags, and no `set_num_threads` or `OMP_NUM_THREADS`
appears anywhere.

Design notes that matter for the answer:

  * It compares WEIGHTS, not AUC. AUC is a lossy projection of the weights: identical weights
    must give identical AUC, but identical AUC does not imply identical weights. So an
    AUC comparison is a lower bound on divergence, and cannot separate "the same model
    scored the same" from "two different models happened to score the same".

  * It compares across PROCESSES, because that is where the divergence was observed. Hashing
    twice inside one process would test nothing -- the thread pool is already warm and the
    reduction order is fixed for the life of the process.

  * `--threads 0` means "leave the default alone", i.e. the current unpinned behaviour, and
    serves as the control. If the default is also stable, the hypothesis is wrong and the
    cause lies elsewhere.

Usage (run each setting twice, in separate processes, and compare the printed hashes):

    python diagnose_tt_determinism.py --threads 0
    python diagnose_tt_determinism.py --threads 1
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=0,
                    help="BLAS thread count; 0 leaves the default (the control)")
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3,
                    help="builds within this process, to separate within- from across-process")
    args = ap.parse_args()

    import torch
    from recad.common import set_seed
    from recad.tt_table import TTEmbeddingTable

    print(f"  torch {torch.__version__}")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        print(f"  set_num_threads({args.threads})")
    else:
        print(f"  threads: default (unpinned) = {torch.get_num_threads()}")

    hashes = []
    for i in range(args.repeats):
        # Seed immediately before construction. The harness learned this the hard way: the
        # seed used to be applied at training time while weights were drawn in the
        # constructor, so a table's initialisation depended on how many tables had been
        # built before it.
        set_seed(args.seed)
        t = TTEmbeddingTable(args.rows, args.dim, rank=args.rank)
        h = hashlib.sha256()
        for name, p in sorted(t.state_dict().items()):
            h.update(name.encode())
            h.update(p.detach().cpu().numpy().tobytes())
        hashes.append(h.hexdigest()[:16])
        print(f"  build {i}: {hashes[-1]}")

    print()
    if len(set(hashes)) == 1:
        print(f"  within-process: STABLE  ({hashes[0]})")
    else:
        print(f"  within-process: VARIES  {hashes}")
    print(f"  HASH threads={args.threads} seed={args.seed} -> {hashes[0]}")
    print("  (run this script again and compare HASH lines across processes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
