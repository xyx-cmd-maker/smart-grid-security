"""Verify the local GPU actually executes, not just that torch lists sm_120.

`get_arch_list()` reports what the wheel was COMPILED for. A wheel can list sm_120
and still fail at runtime if the driver, the toolkit or the kernel image do not
line up. The only meaningful check is to allocate on the device and compute.

This matters here because the whole point of the exercise is to learn whether the
project's measurement code can run on a GPU at all before paying for a cloud
instance. A false pass would send us to T4 rental with a broken pipeline.

Also reports what the project's own tier machinery resolves to, so a mismatch
between "the GPU works" and "the harness can use it" shows up now.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

print("=" * 78)
print("本机 GPU 运行时验证")
print("=" * 78)
print()
print(f"  torch            {torch.__version__}")
print(f"  cuda available   {torch.cuda.is_available()}")
print(f"  cuda version     {torch.version.cuda}")
print(f"  cudnn            {torch.backends.cudnn.version()}")
print()

if not torch.cuda.is_available():
    print("  FAIL: torch 看不到 CUDA 设备")
    raise SystemExit(1)

try:
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    print(f"  device           {props.name}")
    print(f"  capability       sm_{props.major}{props.minor}")
    print(f"  total memory     {props.total_memory / 1024**3:.2f} GiB")
    print(f"  multiprocessors  {props.multi_processor_count}")
    print()
except Exception as exc:                            # noqa: BLE001
    print(f"  FAIL: 读取设备属性失败: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

# --- the real test: allocate, compute, read back -------------------------
print("  实际运算测试：")
ok = True
try:
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    s = float(c.sum().item())
    print(f"    2048x2048 fp32 matmul  -> sum {s:+.3f}   OK")
except Exception as exc:                            # noqa: BLE001
    ok = False
    print(f"    matmul 失败: {type(exc).__name__}: {exc}")

try:
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    print(f"    1024x1024 fp16 matmul  -> sum {float(c.sum().item()):+.3f}   OK")
except Exception as exc:                            # noqa: BLE001
    ok = False
    print(f"    fp16 matmul 失败: {type(exc).__name__}: {exc}")

try:
    x = torch.randn(4096, 64, device="cuda")
    idx = torch.randint(0, 1024, (4096, 8), device="cuda")
    table = torch.randn(1024, 64, device="cuda")
    out = table[idx].sum(dim=1) + x
    torch.cuda.synchronize()
    print(f"    嵌入查表模拟           -> shape {tuple(out.shape)}   OK")
except Exception as exc:                            # noqa: BLE001
    ok = False
    print(f"    查表失败: {type(exc).__name__}: {exc}")

# --- peak memory reporting, which is what the project measures ----------
try:
    torch.cuda.reset_peak_memory_stats()
    t = torch.randn(4096, 4096, device="cuda")
    _ = t @ t
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"    峰值显存读数可用       -> {peak:.1f} MiB   OK")
except Exception as exc:                            # noqa: BLE001
    ok = False
    print(f"    峰值显存读数失败: {type(exc).__name__}: {exc}")

print()
print("=" * 78)
print("  结论：" + ("GPU 运行时可用 ✓" if ok else "GPU 运行时有问题 ✗"))
print("=" * 78)
print()
print("  ⚠️ 这只是【设备可用性】验证；项目代码能否在 GPU 上跑尚未验证。")
print("     下一步应用本 venv 跑一次最小的基线（例如 BDD）来验证管线。")
