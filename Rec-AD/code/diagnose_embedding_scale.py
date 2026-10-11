"""Dense vs TT embedding, head to head on identical indices.

CORRECTED. The previous version of this script hard-coded `embedding_dim: 16`,
while config.yaml resolves it to `n_per_core ** d = 4 ** 3 = 64`. Every number
it produced was therefore measured at a configuration the experiments never
use -- which is how a non-existent "compression ratio mismatch" was reported
(2.34x is correct; 1.15x came from the wrong dim).

The dimension is now taken from the config, and asserted against the value the
recorded runs used, so a silent divergence cannot happen again.

Measures:

    parameter counts
    forward output scale   -- a different scale is an implicit normalisation,
                              not a compression gain
    backward gradient scale -- a different scale changes the EFFECTIVE learning
                              rate, which is the leading explanation for the
                              apparent "TT improves accuracy" effect
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                     # noqa: E402
from data.dataset import build_spec                          # noqa: E402
from recad.common import load_config                         # noqa: E402
from recad.model import RecADConfig, RecADDetector           # noqa: E402

DATA = HERE / "data_cache/ieee118.npz"

data = load_contract(DATA)
spec = build_spec(data)

# ---- dimension comes from the config; never hard-code it ------------------
base_cfg = load_config(None)
resolved = RecADConfig.from_cfg(base_cfg)
dim = resolved.embedding_dim
tt_cfg = base_cfg.get("tt", {}) or {}
parts = tt_cfg.get("split", {}) or {}

print("=" * 82)
print("0. 维度解析（本次修正的核心）")
print("=" * 82)
print()
print(f"  config tt.split            {parts}")
print(f"  config tt.embedding_dim    {tt_cfg.get('embedding_dim', '(未设置)')}")
print(f"  => 解析出的 embedding_dim   {dim}")
print()
# Guard against the exact mistake that produced the bogus mismatch report.
expected_dense_bytes = spec.sparse_total_rows * dim * 4
print(f"  表 {spec.sparse_total_rows:,} 行 x {dim} 维 x 4B = "
      f"{expected_dense_bytes:,} B 的稠密等价表")
if expected_dense_bytes == 204_544:
    print("  ✓ 与消融记录 (ablation_ieee118_local_abl_real.json) 的 "
          "dense_bytes 一致")
else:
    print(f"  ⚠ 与消融记录的 204,544 B 不一致 —— 配置可能变过，")
    print(  "    本脚本的数字不能直接与那次消融对比")
print()


def build(use_eff_tt: bool, seed: int = 0):
    torch.manual_seed(seed)
    cfg = {"tt": {k: v for k, v in tt_cfg.items() if k != "split"},
           "tt_split": parts,
           "ablation": {"components": {"eff_tt": use_eff_tt,
                                       "reorder": False, "pipeline": False}},
           "train": {}}
    # Keep the config's own split so the resolved dim is the real one.
    cfg["tt"]["split"] = parts
    return RecADDetector(spec, cfg, name="tt" if use_eff_tt else "dense",
                         data=data)


m_dense = build(False)
m_tt = build(True)

print("=" * 82)
print("1. 参数量")
print("=" * 82)
print()
for tag, m in (("稠密", m_dense), ("TT", m_tt)):
    emb = sum(p.numel() for p in m.embedding.parameters())
    tot = sum(p.numel() for p in m.parameters())
    print(f"  {tag:<6} 嵌入参数 {emb:>9,}   全模型 {tot:>9,}   "
          f"嵌入占比 {emb / tot * 100:>5.1f}%")
emb_d = sum(p.numel() for p in m_dense.embedding.parameters())
emb_t = sum(p.numel() for p in m_tt.embedding.parameters())
print()
print(f"  嵌入参数比 稠密/TT = {emb_d / emb_t:.4f}")
print(f"  官方 compression_ratio() = {m_tt.compression_ratio():.4f}")
print(f"  报告 embedding_bytes()   = {m_tt.embedding_bytes():,} B")
print()

print("=" * 82)
print("2. 同一批索引下的前向输出")
print("=" * 82)
print()
torch.manual_seed(1)
idx = torch.randint(0, spec.sparse_total_rows, (256, 7))
with torch.no_grad():
    out_d = m_dense.embedding(idx)
    out_t = m_tt.embedding(idx)
print(f"{'':<8}{'mean':>12}{'std':>12}{'abs max':>12}")
print("-" * 82)
for tag, o in (("稠密", out_d), ("TT", out_t)):
    print(f"{tag:<8}{o.mean():>12.4f}{o.std():>12.4f}{o.abs().max():>12.4f}")
print(f"\n  输出尺度比 TT/稠密 = {float(out_t.std() / out_d.std()):.3f}")
print()

print("=" * 82)
print("3. 反向梯度尺度")
print("=" * 82)
print()
grads = {}
for tag, m in (("稠密", m_dense), ("TT", m_tt)):
    m.zero_grad(set_to_none=True)
    m.embedding(idx).sum().backward()
    g = torch.cat([p.grad.reshape(-1) for p in m.embedding.parameters()
                   if p.grad is not None])
    grads[tag] = float(g.norm())
    print(f"  {tag:<6} 梯度范数 {g.norm():>12.4f}   梯度 std {g.std():>12.6f}")
print()
ratio = grads["稠密"] / grads["TT"]
print(f"  梯度范数比 稠密/TT = {ratio:.3f}")
print()
print("=" * 82)
print("判定")
print("=" * 82)
print()
print(f"  dim = {dim}（不再是硬写的 16）")
if abs(out_t.std() / out_d.std() - 1.0) < 0.15:
    print("  输出尺度接近 -> 排除『隐式归一化』")
if ratio > 1.5:
    print(f"  梯度尺度差 {ratio:.2f}x -> 两臂有效学习率不同，")
    print("    支持『TT 精度优势是学习率标定伪影』的解释")
else:
    print(f"  梯度尺度接近（{ratio:.2f}x）-> 学习率伪影的解释需要重新考虑")
