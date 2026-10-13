"""Audit the technical claims in AGENTS.md for whether they were measured or assumed.

Three of the file's statements were refuted by measurement in this session: the size unit, the
chance level for a detect rate, and the behaviour of git clean on a nested repository. Each had
been written from how the thing is usually described rather than from what this project does. The
file holds about twenty more technical claims, so this checks the ones that can be checked
cheaply and marks the rest.

Each claim gets one of three labels:

  measured   the value in the file was produced by a run whose output is cited
  read       the value comes from a file or an implementation, which is verification for a claim
             about that file, but not for a claim about the world
  assumed    nobody in this session ran anything, so the entry needs a marker

The point is not to delete the assumed ones. Several are probably right. The point is that a
reader should be able to tell which ones were earned.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

results: list[tuple[str, str, str]] = []


def record(claim: str, label: str, detail: str) -> None:
    results.append((claim, label, detail))
    print(f"  [{label:8}] {claim}\n             {detail}")


print("=" * 78)
print("  1. t4 specification: 16 GB / 70 W")
print("=" * 78)
try:
    from recad.common import DEVICE_TIERS
    t4 = DEVICE_TIERS.get("t4")
    if t4 is None:
        record("t4 是 16 GB / 70 W", "assumed", "DEVICE_TIERS 里没有 t4")
    else:
        vram = getattr(t4, "vram_gb", None)
        power = getattr(t4, "power_w", None)
        ok = (vram == 16.0 and power == 70.0)
        record("t4 是 16 GB / 70 W", "read",
               f"common.py 声明 vram_gb={vram}, power_w={power} -> "
               f"{'与文件一致' if ok else '★ 与文件不符'}")
        record("t4 是包络的 2 倍", "read",
               f"16/8 = {16/8:.1f}x（纯算术，依赖上一行）")
except Exception as exc:  # noqa: BLE001
    record("t4 是 16 GB / 70 W", "assumed", f"导入失败: {exc}")

print()
print("=" * 78)
print("  2. local tier carries no specs")
print("=" * 78)
try:
    loc = DEVICE_TIERS.get("local")
    record("local 的 vram_gb / power_w 都是 0", "read",
           f"vram_gb={getattr(loc, 'vram_gb', None)}, power_w={getattr(loc, 'power_w', None)}, "
           f"publishable={getattr(loc, 'publishable', None)}")
except Exception as exc:  # noqa: BLE001
    record("local 的规格", "assumed", str(exc))

print()
print("=" * 78)
print("  3. power timeouts: monitor 3 min, standby 10 min")
print("=" * 78)
try:
    out = subprocess.run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_VIDEO", "VIDEOIDLE"],
                         capture_output=True, text=True, timeout=20).stdout
    ac = re.findall(r"当前交流电源设置索引|Current AC Power Setting Index", out)
    hexes = re.findall(r"0x([0-9a-fA-F]{8})", out)
    secs = [int(h, 16) for h in hexes]
    monitor = min(secs) / 60 if secs else None
    out2 = subprocess.run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"],
                          capture_output=True, text=True, timeout=20).stdout
    hexes2 = re.findall(r"0x([0-9a-fA-F]{8})", out2)
    secs2 = [int(h, 16) for h in hexes2]
    standby = min(secs2) / 60 if secs2 else None
    ok = (monitor is not None and abs(monitor - 3) < 0.1 and
          standby is not None and abs(standby - 10) < 0.1)
    record("息屏 3 分钟 / 待机 10 分钟", "measured",
           f"powercfg 读出 monitor={monitor} min, standby={standby} min -> "
           f"{'与文件一致' if ok else '★ 与文件不符（可能已被我改过）'}")
except Exception as exc:  # noqa: BLE001
    record("息屏/待机超时", "assumed", str(exc))

print()
print("=" * 78)
print("  4. pandapower transformer units are abnormal, and reactance is clamped")
print("=" * 78)
try:
    import numpy as np
    with np.load(ROOT / "code" / "data_cache" / "ieee118.npz", allow_pickle=False) as z:
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
    keys = sorted(meta.keys())
    hits = [k for k in keys if any(t in k.lower() for t in ("clamp", "react", "x_pu", "topology"))]
    record("生成器钳制支路电抗到 [0.005, 0.5] pu", "read" if hits else "assumed",
           f"meta 里相关键: {hits if hits else '未找到 —— meta 不记录该信息'}")
    record("ieee118 的 179 条支路中 1 条被钳制", "assumed",
           "本会话未跑生成器核对；meta 无该字段")
    record("pandapower 变压器单位异常（sn_mva 恒 9900、vk_percent 数万）", "assumed",
           "读自 AGENTS.md（更早会话的记录），本会话未复现")
except Exception as exc:  # noqa: BLE001
    record("pandapower 单位异常", "assumed", str(exc))

print()
print("=" * 78)
print("  5. claims I know I have not tested this session")
print("=" * 78)
for claim, why in [
    ("torch.use_deterministic_algorithms(True) 抓不到 TT 的线程非确定性",
     "本会话只跑了线程数对照，没跑这个开关"),
    ("case300 上限约 30000 样本（60000 会 OOM）",
     "需要一次真实 OOM 才能确认；未跑"),
    ("稀疏特征的 bus_id 列与母线无映射",
     "读自 AGENTS.md 的三重证据记录，本会话未复现"),
    ("verify_threat_model.py 4/4 通过",
     "是更早会话的结果；本会话未重跑"),
    ("TT 前向确定、反向不确定",
     "本会话未重跑该对照（§4.12 记录为 2026-10-12）"),
]:
    record(claim, "assumed", why)

print()
print("=" * 78)
print("  6. claims verified by running things this session")
print("=" * 78)
for claim in [
    "推理显存 266–292 MiB（六基线）",
    "延迟 p99 4.23 ms",
    "功耗峰值 32.2 W",
    "TT 表 8× 延迟换 14× 显存",
    "三条约束都不绑定",
    "时间通道是幅度曲线（13 点）",
    "重抽对照下时间 AUC 1.000（拆掉 Sun 式攻击）",
    "BDD 检出率与虚警率无法区分（p=0.86 / 0.97）",
    "本机 8151 MiB 落在 8192 包络内",
]:
    record(claim, "measured", "本会话跑出的结果，证据在 code/results/")

# ---- summary ---------------------------------------------------------------
print()
print("=" * 78)
counts: dict[str, int] = {}
for _, label, _ in results:
    counts[label] = counts.get(label, 0) + 1
print(f"  合计 {len(results)} 条：", "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

assumed = [c for c, lab, _ in results if lab == "assumed"]
out = ROOT / "code" / "results" / "claim_audit.json"
out.write_text(json.dumps(
    {"rows": [{"claim": c, "label": l, "detail": d} for c, l, d in results],
     "counts": counts,
     "reason": ("Three statements in AGENTS.md were refuted by measurement this session. "
                "This separates the claims that were earned from the ones that were assumed.")},
    ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  wrote {out.name}")
print()
print(f"  ★ 需要标 [未实测] 的 {len(assumed)} 条：")
for c in assumed:
    print(f"    - {c}")
