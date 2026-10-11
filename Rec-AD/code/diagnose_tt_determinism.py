"""Test whether pinning the BLAS thread count makes the tensor-train table reproducible.

**Result as of 2026-10-11: the table construction is reproducible either way.**

    threads=0 (unpinned, 16 threads):  b0b8bf62a33b4400   three separate processes, all equal
    threads=1:                         625a0789d1d0fb69   three separate processes, all equal

So the thread count does change the weights -- the two settings disagree with each other --
but it does not make them *unstable*. This refutes the hypothesis that the ablation's
run-to-run divergence comes from multi-threaded reduction order in the table's SVD. The
search has to move downstream, and the standing unexplored candidate is the `PYTHONHASHSEED`
residue in `recad/common.py`: the variable is assigned inside `set_seed`, which runs after
interpreter start, so it has no effect; set and dict iteration order over strings can
therefore differ between processes. Recorded as an open item since 2026-10-06.

An earlier version of this script defaulted to `--rows 8192 --dim 64 --rank 8`, which is not
what the ablation builds. At that size the factorisation is small and reproducible no matter
what, so the script returned STABLE and looked like a refutation. It was measuring a different
object. The defaults now match the real configuration.

Design notes that matter for the answer:

  * It compares WEIGHTS, not AUC. AUC is a lossy projection of the weights: identical weights
    must give identical AUC, but identical AUC does not imply identical weights. So an
    AUC comparison is a lower bound on divergence, and cannot separate "the same model
    scored the same" from "two different models happened to score the same".

  * It compares across PROCESSES, because that is where the divergence was observed. Hashing
    twice inside one process would test nothing -- the thread pool is already warm and the
    reduction order is fixed for the life of the process.

  * `--threads 0` means "leave the default alone", i.e. the observed unpinned behaviour, and
    serves as the control.

Usage (run each setting several times, in separate processes, and compare the printed hashes):

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
    # Defaults match what the ablation actually builds, not round numbers. The first version
    # of this script defaulted to 8192 x 64 with rank 8 -- 512x fewer rows, a different
    # embedding dimension and a different rank -- and at that size the factorisation is small
    # enough to be reproducible regardless, so it returned STABLE and appeared to refute the
    # hypothesis. Measured at the real size the answer is different in one respect and the
    # same in another; see the module docstring. Defaults that do not match the experiment
    # under test are worse than no defaults.
    ap.add_argument("--rows", type=int, default=4193420,
                    help="sparse_total_rows in the IEEE 118 run; reverse-engineered from "
                         "dense embedding_bytes / (dim * 4)")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--rank", type=int, default=32,
                    help="config tt.rank; note rank > dim here, so the factorisation "
                         "truncates hard")
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
