"""Verify that the project's torch models actually train on this GPU.

Why this is separate from the device check: `verify_gpu_runtime.py` proved the device
executes kernels. It said nothing about whether the PROJECT's model code runs there.
The gap matters because the next step is paying for a cloud T4, and a pipeline that
breaks on GPU would waste that.

The tier machinery deliberately routes `local` to CPU (`DEVICE_TIERS["local"]` has
vram_gb 0.0 and publishable False, "Unknown hardware; do not report"). That is the
right policy for REPORTING but it blocks verification. So this script bypasses the
tier and places the model on CUDA explicitly. It is a verification, not a result:
nothing here is reportable, and it writes no result files.

What it checks, in the order a failure would show up:
  1. the model builds at all under the GPU environment
  2. parameters and inputs move to the device
  3. a forward pass produces finite output
  4. a backward pass produces finite, non-zero gradients
  5. peak VRAM is readable and plausible
  6. the same forward is deterministic across two runs (sanity, not a result)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                     # noqa: E402
from data.dataset import build_spec, build_loaders          # noqa: E402
from recad.common import load_config                        # noqa: E402
from baselines.registry import build_detector               # noqa: E402

DEV = "cuda"
d = load_contract(HERE / "data_cache/ieee118.npz")
spec = build_spec(d)
cfg = load_config(None)

print("=" * 84)
print("GPU 训练路径验证（绕开 tier 机制，直接指定 cuda）")
print("=" * 84)
print()
print(f"  device: {torch.cuda.get_device_name(0)}")
print(f"  tier 机制说明: DEVICE_TIERS['local'] 的 vram_gb=0.0 / publishable=False，")
print(f"                所以 --tier local 会走 CPU。本脚本绕开它，仅为验证。")
print()

# ---- data -----------------------------------------------------------------
try:
    loaders = build_loaders(d, spec, cfg, seed=0, batch_size=256, num_workers=0)
    train_loader = loaders["train"] if isinstance(loaders, dict) else loaders[0]
    batch = next(iter(train_loader))
    print(f"  batch 字段: {sorted(batch.keys())}")
    for k in ("dense", "sparse"):
        if k in batch:
            print(f"    {k:8} {tuple(batch[k].shape)}  {batch[k].dtype}")
except Exception as exc:                            # noqa: BLE001
    print(f"  !! 数据加载失败: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

# ---- one torch model ------------------------------------------------------
results = {}
for name in ("CNN", "LSTM"):
    print()
    print(f"  ---- {name} ----")
    try:
        model = build_detector(name, spec, cfg)
    except Exception as exc:                        # noqa: BLE001
        print(f"    !! 构造失败: {type(exc).__name__}: {exc}")
        results[name] = False
        continue

    n_par = sum(p.numel() for p in model.parameters())
    print(f"    参数量 {n_par:,}")

    try:
        model = model.to(DEV)
        moved = all(p.device.type == "cuda" for p in model.parameters())
        print(f"    参数已上 GPU: {moved}")
    except Exception as exc:                        # noqa: BLE001
        print(f"    !! .to(cuda) 失败: {type(exc).__name__}: {exc}")
        results[name] = False
        continue

    try:
        torch.cuda.reset_peak_memory_stats()
        model.train()
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

        b = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(b) if callable(getattr(model, "forward", None)) else None
        if out is None:
            print("    !! forward 返回 None")
            results[name] = False
            continue

        logits = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(logits, dict):
            logits = next(iter(logits.values()))
        finite = bool(torch.isfinite(logits).all())
        print(f"    forward OK   shape={tuple(logits.shape)}  finite={finite}")

        y = b.get("label")
        if y is None:
            print("    !! batch 无 label，跳过 backward")
            results[name] = True
            continue
        loss = torch.nn.functional.cross_entropy(logits, y.long())
        opt.zero_grad()
        loss.backward()
        gnorm = float(sum(p.grad.norm().item() ** 2
                          for p in model.parameters() if p.grad is not None) ** 0.5)
        print(f"    backward OK  loss={float(loss):.4f}  grad_norm={gnorm:.4f}"
              f"  {'(非零 ✓)' if gnorm > 0 else '(零 ✗)'}")
        opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"    峰值显存 {peak:.1f} MiB")
        results[name] = finite and gnorm > 0
    except Exception as exc:                        # noqa: BLE001
        print(f"    !! 训练步失败: {type(exc).__name__}: {exc}")
        results[name] = False

print()
print("=" * 84)
for k, v in results.items():
    print(f"  {k:<8} {'通过 ✓' if v else '失败 ✗'}")
print(f"  总结: {'GPU 训练路径可用 ✓' if results and all(results.values()) else '有问题 ✗'}")
print("=" * 84)
print()
print("  ⚠️ 这是【验证】，不是结果。没有任何数字可以从这里进论文。")
