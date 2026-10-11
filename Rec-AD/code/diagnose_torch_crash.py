"""0.1 -- Is the c10.dll crash reproducible, and is it a memory leak?

Event log shows five APPCRASHes in three hours, all with
`Exception code 0xc0000005` in `c10.dll` while running the ablation. Two long
ablation runs died at the start of a TT cell and reported exit code 0, because
the crash happens inside the piped python process, not in PowerShell.

Two competing explanations, and this script separates them:

  (a) a binary incompatibility between torch 2.13.0+cpu and Python 3.14.4 --
      `torchvision` ships `Requires-Python: >=3.10,!=3.14.1`, so the ecosystem
      is demonstrably wary of 3.14. A crash would then land at an arbitrary
      point in a torch operation.
  (b) a slow memory leak that eventually takes the process down. The System log
      has `Resource-Exhaustion-Detector` id 2004 at the same timestamps, which
      is consistent with either reading -- a crashed process never releases its
      commit charge.

So: loop the exact workload that crashes (build a TT detector, train it), print
resident memory every iteration, and flush after each line. If RSS climbs
monotonically, it is (b). If it is flat and the process still dies, it is (a).

Run with output redirected at the shell (`python -u ... > log 2>&1`), never
through a pipe into a cmdlet -- a pipe buffers, and the tail is exactly what we
need to see when the process dies.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                             # noqa: E402
from data.dataset import build_loaders, build_spec                   # noqa: E402
from recad.common import load_config                                 # noqa: E402
from recad.model import RecADConfig, RecADDetector                   # noqa: E402
from train import run_training_loop                                  # noqa: E402

DATA = HERE / "data_cache/ieee118.npz"
ITERATIONS = int(os.environ.get("REPRO_ITERS", "6"))
EPOCHS = int(os.environ.get("REPRO_EPOCHS", "4"))
BATCH = 512


def rss_mb() -> float:
    """Resident set size of this process, in MB."""
    try:
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong),
                        ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        info = PMC()
        info.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(info),
            info.cb)
        return info.WorkingSetSize / 2**20
    except Exception:
        return float("nan")


def say(msg: str = "") -> None:
    print(msg, flush=True)


say("=" * 88)
say("0.1  最小复现：torch c10.dll 崩溃是否必现 / 是否内存泄漏")
say("=" * 88)
say()
say(f"  Python {sys.version.split()[0]}   torch {torch.__version__}")
say(f"  迭代 {ITERATIONS} 次 x {EPOCHS} epoch，batch={BATCH}")
say(f"  起始 RSS {rss_mb():.1f} MB")
say()

data = load_contract(DATA)
spec = build_spec(data)
base = load_config(None)
dim = RecADConfig.from_cfg(base).embedding_dim
say(f"  表 {spec.sparse_total_rows:,} 行  dim={dim}")
say()
say(f"  {'迭代':>5}{'建表后 MB':>12}{'训练后 MB':>12}{'Δ MB':>9}"
    f"{'耗时 s':>9}{'loss':>10}")
say("  " + "-" * 60)

baseline = None
rows = []
t_start = time.perf_counter()

for i in range(ITERATIONS):
    try:
        torch.manual_seed(i)
        np.random.seed(i)

        tt_cfg = dict(base.get("tt", {}) or {})
        tt_cfg["split"] = (base.get("tt", {}) or {}).get("split", {})
        cfg = {"tt": tt_cfg, "train": {},
               "ablation": {"components": {"eff_tt": True, "reorder": False,
                                           "pipeline": False}}}
        model = RecADDetector(spec, cfg, name="tt", data=data)
        after_build = rss_mb()

        loaders = build_loaders(data, batch_size=BATCH, num_workers=0,
                                standardize=False, seed=i)
        t0 = time.perf_counter()
        hist = run_training_loop(model, loaders["train"], loaders["val"],
                                 epochs=EPOCHS, lr=1e-3, weight_decay=0.0)
        dt = time.perf_counter() - t0

        after_train = rss_mb()
        if baseline is None:
            baseline = after_build
        losses = hist.get("train_loss") or [float("nan")]
        rows.append(after_train)

        say(f"  {i + 1:>5}{after_build:>12.1f}{after_train:>12.1f}"
            f"{after_train - baseline:>+9.1f}{dt:>9.1f}{losses[-1]:>10.4f}")

        del model, loaders
    except Exception as exc:                      # noqa: BLE001
        say(f"  {i + 1:>5}  异常 {type(exc).__name__}: {str(exc)[:70]}")
        break

say()
say("=" * 88)
say("判定")
say("=" * 88)
say()
say(f"  总耗时 {time.perf_counter() - t_start:.0f} s   完成 {len(rows)}/{ITERATIONS} 次迭代")
if rows:
    say(f"  RSS 起 {rows[0]:.1f} MB  终 {rows[-1]:.1f} MB  增长 {rows[-1] - rows[0]:+.1f} MB")
    say()
    growth = rows[-1] - rows[0]
    if growth > 200:
        say("  => RSS 单调增长 -> 指向【内存泄漏】，不是 3.14 兼容性问题。")
        say("     处置：分批跑 —— 每次全新进程，绕开累积。")
    else:
        say("  => RSS 平稳。若本脚本正常跑完而消融仍崩，")
        say("     则崩溃与累积内存无关，指向【torch/Python 3.14 兼容性】。")
        say("     处置：在 Python 3.13 上装 torch 重试。")
say()
say("  若本脚本中途无输出地死掉，说明崩溃可复现 —— 看它死在第几次迭代。")
say("  事件日志可用以下命令复核：")
say("    Get-WinEvent -FilterHashtable @{LogName='Application';StartTime=(Get-Date).AddMinutes(-30)} |")
say("      Where-Object { $_.Message -match 'c10.dll' } | Select-Object TimeCreated, Message")
