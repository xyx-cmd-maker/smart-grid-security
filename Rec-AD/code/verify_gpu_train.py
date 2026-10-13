"""Verify that the project's torch models actually train on this GPU.

Why this is separate from the device check: `verify_gpu_runtime.py` proved the device
executes kernels. It said nothing about whether the PROJECT's model code runs there.
That gap matters because the next step is paying for a cloud T4, and a pipeline that
breaks on GPU would waste it.

The tier machinery deliberately routes `local` to CPU (`DEVICE_TIERS["local"]` has
vram_gb 0.0, publishable False, "Unknown hardware; do not report"). That is right for
REPORTING but it blocks verification, so this script places the model on CUDA
explicitly. It is a verification, not a result: nothing here is reportable and no
result files are written.

Checks, in the order a failure would appear:
  1. data loads at all
  2. the model builds under the GPU environment
  3. parameters and inputs move to the device
  4. forward produces finite output
  5. backward produces finite, non-zero gradients
  6. peak VRAM is readable
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                     # noqa: E402
from data.dataset import build_spec, build_loaders          # noqa: E402
from recad.common import load_config, set_seed                # noqa: E402
from baselines.registry import build_detector               # noqa: E402

DEV = "cuda"
d = load_contract(HERE / "data_cache/ieee118.npz")
spec = build_spec(d)
cfg = load_config(None)

# Seed before ANY model is constructed. Without this the script draws fresh initial weights
# every run and its gradient norms move by an order of magnitude between invocations -- CNN
# reported 476.2 and 45.6 on two consecutive runs, LSTM 256.2 and 27.3. One unseeded run
# landed on a saturated initialisation and printed grad_norm=0.0000, which looks exactly like
# a dead backward pass and is not one. A verification whose verdict changes with the draw
# cannot serve as a gate. This is the same rule the project already carries for experiments
# (AGENTS.md section 4.9): seed before building, here applied to the checker itself.
set_seed(0)

print("=" * 84)
print("GPU training path verification (bypasses the tier machinery on purpose)")
print("=" * 84)
print()
print(f"  device: {torch.cuda.get_device_name(0)}")
print("  note:   DEVICE_TIERS['local'] has vram_gb=0.0 / publishable=False, so")
print("          --tier local routes to CPU. This script bypasses that to verify only.")
print()

# ---- 1. data ---------------------------------------------------------------
try:
    dcfg = cfg.get("data", {}) or {}
    # standardize defaults to True in build_loaders, which is the AGENTS 5.4 trap:
    # it destroys the stealth property. Pass the config value explicitly so this
    # verification exercises the same regime the experiments use.
    loaders = build_loaders(
        d, batch_size=256, num_workers=0,
        standardize=bool(dcfg.get("standardize", False)),
        ratios=dcfg.get("split_ratios"), seed=0,
    )
    train_loader = loaders["train"] if isinstance(loaders, dict) else loaders[0]
    batch = next(iter(train_loader))
    print(f"  batch keys: {sorted(batch.keys())}")
    for k in ("dense", "sparse", "label"):
        if k in batch:
            print(f"    {k:8} {tuple(batch[k].shape)}  {batch[k].dtype}")
except Exception as exc:                            # noqa: BLE001
    print(f"  !! data load failed: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

# ---- 2..6 one torch model at a time ---------------------------------------
results = {}
for name in ("CNN", "LSTM", "Transformer"):
    print()
    print(f"  ---- {name} ----")
    try:
        # `data` must be passed: the sequence baselines build their fixed residual
        # projection from the contract's Jacobian, and without it they silently skip the
        # physics path. This check previously omitted it, so it exercised only the
        # branch that needs no Jacobian -- and reported success while the branch that
        # does need one was broken on GPU. See the note in baselines/_sequence.py.
        model = build_detector(name, spec, cfg, d)
    except Exception as exc:                        # noqa: BLE001
        print(f"    !! build failed: {type(exc).__name__}: {exc}")
        results[name] = False
        continue

    n_phys = getattr(model, "physics_channels", None)
    print(f"    parameters {sum(p.numel() for p in model.parameters()):,}"
          + (f"  physics_channels={n_phys}" if n_phys is not None else ""))
    if n_phys == 0:
        print("    !! physics_channels is 0: the Jacobian was not used, so this does NOT "
              "exercise the projection path")

    try:
        model = model.to(DEV)
        moved = all(p.device.type == "cuda" for p in model.parameters())
        print(f"    moved to GPU: {moved}")
    except Exception as exc:                        # noqa: BLE001
        print(f"    !! .to(cuda) failed: {type(exc).__name__}: {exc}")
        results[name] = False
        continue

    try:
        torch.cuda.reset_peak_memory_stats()
        model.train()
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

        b = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(b)
        logits = out
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if isinstance(logits, dict):
            logits = next(iter(logits.values()))
        finite = bool(torch.isfinite(logits).all())
        print(f"    forward  shape={tuple(logits.shape)}  finite={finite}")

        y = b.get("label")
        if y is None:
            print("    !! no label in batch; backward skipped")
            results[name] = finite
            continue

        # Output shape is (batch,) -- ONE score per sample, and the label tensor is
        # float32. That is a binary formulation, so cross_entropy (which wants a
        # (batch, n_class) input and a Long target) is the wrong loss and is what
        # produced "Expected floating point type for target with class probabilities".
        # Branch on the actual shape rather than assuming.
        if logits.dim() == 1 or logits.shape[-1] == 1:
            loss = torch.nn.functional.binary_cross_entropy(
                logits.reshape(-1).clamp(1e-6, 1 - 1e-6),
                y.reshape(-1).float())
            print(f"    loss = bce (output is one score per sample, shape {tuple(logits.shape)})")
        else:
            loss = torch.nn.functional.cross_entropy(logits, y.long())
            print(f"    loss = cross_entropy (output shape {tuple(logits.shape)})")
        opt.zero_grad()
        loss.backward()
        gnorm = float(sum(float(p.grad.norm()) ** 2
                          for p in model.parameters() if p.grad is not None) ** 0.5)
        opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"    backward loss={float(loss):.4f}  grad_norm={gnorm:.4f}"
              f"  {'(non-zero OK)' if gnorm > 0 else '(ZERO - suspicious)'}")
        print(f"    peak VRAM {peak:.1f} MiB")
        results[name] = finite and gnorm > 0
    except Exception as exc:                        # noqa: BLE001
        print(f"    !! training step failed: {type(exc).__name__}: {exc}")
        results[name] = False

print()
print("=" * 84)
for k, v in results.items():
    print(f"  {k:<8} {'PASS' if v else 'FAIL'}")
ok = bool(results) and all(results.values())
print(f"  overall: {'GPU training path usable' if ok else 'PROBLEM'}")
print("=" * 84)
print()
print("  This is a VERIFICATION, not a result. No number here may enter the paper.")
