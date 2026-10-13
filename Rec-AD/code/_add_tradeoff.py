"""Add the Rec-AD trade-off to section 1, then commit the session's new scripts.

Everything added to the claim ceiling this session has been a retraction: the envelope does not
bind on memory, latency or power, so the deployability gap cannot rest on any of them. The
Rec-AD measurement is the first result in the other direction and it is clean, so it belongs in
the same place rather than only in a result file.

The trade-off is stated with both numbers and no adjective: about 14x less inference memory for
about 8x more latency, against the same envelope. It is stronger than the original compression
claim because it holds under either reading -- if the envelope binds, the memory saving is what
matters; if it does not, the exchange is still a measured property of the method rather than an
assumption about deployment.

Commits use a message file: a multi-line -m argument is split by PowerShell into pathspecs, which
already failed once this session.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AG = ROOT / "AGENTS.md"

t = AG.read_text(encoding="utf-8")
before = len(t.encode()) / 1024

ANCHOR = "> **证据**：`code/results/inference_vram.json`（推理）、`code/results/envelope_binding*.json`（训练）。"
if ANCHOR not in t:
    raise SystemExit("  ABORT: the evidence line was not found")
if "14×" in t and "8×" in t and "权衡" in t:
    raise SystemExit("  ABORT: the trade-off appears to be present already")

NEW = """\
> ### ★★ Rec-AD 自身的推理测量：一个**可辩护的权衡**（2026-10-13 实测）
>
> 同机同协议（`batch=1`，warmup=50，measure=500，forward only），四个消融单元：
>
> | 单元 | p50 | p99 | 峰值显存 |
> |---|---|---|---|
> | **Rec-AD full（TT+reorder+pipeline）** | **3.91 ms** | **7.04 ms** | **★ 19 MiB** |
> | Rec-AD no-TT | 0.48 ms | 1.09 ms | 273 MiB |
> | **Rec-AD TT only** | **3.84 ms** | **6.25 ms** | **★ 20 MiB** |
> | Rec-AD none | 0.45 ms | 1.03 ms | 273 MiB |
>
> **★ 一句话结论（可以说）**：
> **「TT 表以约 8 倍的推理延迟换取约 14 倍的推理显存节省；
> 在 8192 MiB / 500 ms 包络下两者都不是约束。」**
>
> **为什么这条比原来的压缩主张更强**：
> **它有两种读法都成立** —— 若包络绑定，省下的显存是关键；
> 若不绑定（本实测如此），这个交换仍是**方法的一个实测属性**，而不是对部署的假设。
>
> **纪律**：
> 1. **可以说**上述权衡句，**两个数字必须同时出现**，不得只报省显存。
> 2. **不得**把 19 MiB 单独作为"部署优势"报告 —— **它未含 CUDA context，
>    且包络本就不绑定（见上表）。**
> 3. **`tier=local`，按 §4.6 仍不能填 `deployability.recad_vram` 槽位**；
>    这是口径证据。证据：`code/results/recad_inference.json`。

""" + ANCHOR

t = t.replace(ANCHOR, NEW, 1)
for frag in ("可辩护的权衡", "约 8 倍的推理延迟换取约 14 倍", "两种读法都成立"):
    if frag not in t:
        raise SystemExit(f"  ABORT: {frag!r} missing after insert")
AG.write_text(t, encoding="utf-8")
after = len(AG.read_text(encoding="utf-8").encode()) / 1024
print(f"  AGENTS.md {before:.1f} -> {after:.1f} KB  (limit 52)")

# ---- commit the session's new scripts ---------------------------------------
msg = """Measure the deployment envelope, and record what it says

The paper's deployability claim is that most baselines do not fit a substation node's resource
envelope. That is now measured on a device at the envelope rather than above it: the local GPU
reports 8151 MiB against an 8192 MiB budget, where a 16 GB T4 could not have shown the claim
either way.

Three constraints, all measured, none binding:

  memory   292 MiB peak against 8192 MiB (0.036x), inference, batch 1 and 32
  latency  p99 4.23 ms against 500 ms (0.0085x), batch=1 warmup=50 measure=500
  power    32.2 W peak against 60 W, sampled externally during inference

Rec-AD itself, same protocol, four ablation cells:

  full (TT)  3.91 ms p50, 7.04 ms p99, 19 MiB peak
  no-TT      0.48 ms p50, 1.09 ms p99, 273 MiB peak

so the tensor-train table trades roughly 8x inference latency for roughly 14x inference memory.

The scripts:

  measure_envelope_binding.py  training memory across a batch ladder
  measure_inference_vram.py    inference memory, which is what slots.json asks for
  measure_power_latency.py     the other two constraints, under the project's fixed protocol
  measure_recad_inference.py   Rec-AD's own cells, comparable to the above

Two protocol notes are worth keeping. The first envelope run measured TRAINING memory while
slots.json asks for inference; the numbers are not interchangeable and the earlier reading was
wrong in the direction that overstated the finding. And training and inference memory are
reported separately in every output file, because a future session comparing across them would
repeat exactly that error.

All of it is tier=local, which config.yaml and AGENTS.md 4.6 exclude from the manuscript, so
these are protocol evidence rather than paper numbers. The result files say so.
"""
mf = ROOT.parent / ".git" / "COMMIT_MSG_TMP.txt"
mf.write_text(msg, encoding="utf-8")
for cmd in (["git", "add", "-A", "Rec-AD/code"],
            ["git", "commit", "-q", "-F", str(mf)]):
    r = subprocess.run(cmd, cwd=str(ROOT.parent), capture_output=True, text=True)
    print(f"  {' '.join(cmd[:3])}: rc={r.returncode} {r.stdout.strip()[:80]}")
    if r.returncode != 0:
        print(f"    stderr: {r.stderr.strip()[:200]}")
mf.unlink(missing_ok=True)

r = subprocess.run(["git", "log", "--oneline", "-2"], cwd=str(ROOT.parent),
                   capture_output=True, text=True)
print("  log:")
for ln in r.stdout.strip().splitlines():
    print("   ", ln)
