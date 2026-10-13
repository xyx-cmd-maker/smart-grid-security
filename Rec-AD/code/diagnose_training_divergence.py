"""Find the epoch at which two runs of the same cell stop agreeing.

Four candidate causes have been eliminated: hash-order dependence (every set iteration in the
codebase is sorted, and dicts are insertion-ordered), the tensor-train table's construction
(six processes, two thread settings, identical hashes within each setting), the thread count as
a source of *instability* (it changes the weights but reproducibly), and a single backward pass
through the table (five processes, identical gradient norms).

What is left is a different kind of explanation: not one non-deterministic operation, but a
trajectory that amplifies a very small difference over many steps. That predicts something
testable and specific -- at one epoch the weights should agree, and disagreement should appear
only after enough steps.

This probe trains one configuration for a small number of epochs and prints a hash of the
trained weights. Run it twice in separate processes at increasing `--epochs` and find the first
setting where the two hashes differ. That epoch is the divergence point.

It uses the same construction path as the ablation (`ExperimentContext.build_model`, which seeds
before building) so that a negative result is about the training loop rather than about a
difference between this probe and the experiment it is meant to explain.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "experiments"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tt", type=int, default=1, help="1 = tensor-train table, 0 = dense")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--data", default="data_cache/ieee118.npz")
    args = ap.parse_args()

    ns = argparse.Namespace(
        config="config.yaml",
        data=args.data, tier="local", num_workers=0, no_energy=True,
        target_fpr=0.01, measure_tier=None, epochs=args.epochs,
        batch_size=args.batch_size, orderings="topology", seeds="0", tag="probe",
    )

    import torch
    from _harness import ExperimentContext
    from recad.model import RecADConfig

    ctx = ExperimentContext(ns)

    # Same construction the ablation uses: a RecADConfig with the component flags, handed to
    # build_model, which seeds before drawing weights.
    recad = RecADConfig.from_cfg(ctx.cfg,
                                 use_eff_tt=bool(args.tt),
                                 use_reorder=False,
                                 use_pipeline=False)
    model = ctx.build_model(recad, ordering="topology", seed=args.seed)

    from recad.common import set_seed
    set_seed(args.seed)                             # same order as run_one
    loaders = ctx.loaders(args.seed)

    hist = model.fit(loaders["train"], loaders["val"], device=ctx.device)

    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().float().numpy().tobytes())

    print(f"  epochs={args.epochs} tt={args.tt} seed={args.seed} batch={args.batch_size}")
    print(f"  n_epochs_run={hist.get('n_epochs_run', '?')}")
    print(f"  WEIGHT_HASH {h.hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
