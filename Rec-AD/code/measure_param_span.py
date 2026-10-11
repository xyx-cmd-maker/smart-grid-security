"""Quantify the parameter span across the detector families in Table V.

The revised abstract cites "four orders of magnitude in parameter count" as part of
the argument that capacity is not what separates the detectors that work from the
ones that do not. That phrase was written from an impression of the model types, not
from a count. This measures it, so the claim can either be supported or corrected.

Rec-AD and its dense counterpart are included: they share an architecture and differ
only in whether the embedding table is tensor-train compressed, which makes them the
cleanest single pair in the comparison.

The script prints what it resolved rather than assuming registry or config internals,
so a name mismatch fails loudly here instead of silently producing a wrong count.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                     # noqa: E402
from data.dataset import build_spec                         # noqa: E402
from recad.common import load_config                        # noqa: E402
from recad.model import RecADConfig, RecADDetector          # noqa: E402
from baselines.registry import build_detector               # noqa: E402

d = load_contract(HERE / "data_cache/ieee118.npz")
spec = build_spec(d)
cfg = load_config(None)

rconf = RecADConfig.from_cfg(cfg)
tt = cfg.get("tt", {}) or {}
split = tt.get("split", {}) or {}

print("=" * 84)
print("参数跨度实测（IEEE118，config.yaml 默认配置）")
print("=" * 84)
print()
print(f"  embedding_dim  = {rconf.embedding_dim}"
      f"   (config tt.split: n_per_core={split.get('n_per_core')}, d={split.get('d')})")
print(f"  tt_rank        = {rconf.tt_rank}")
print(f"  tt_core_dims   = {rconf.tt_core_dims}")
print(f"  sparse 表行数  = {spec.sparse_total_rows:,}")
print(f"  RecADConfig 字段: {[f.name for f in dataclasses.fields(RecADConfig)]}")
print()

rows = []

for label, use_tt in (("Rec-AD (TT table)", True), ("DLRM (dense table)", False)):
    r = RecADConfig.from_cfg(cfg)
    r.use_eff_tt = use_tt
    r.use_reorder = False
    r.use_pipeline = False
    try:
        rows.append((label, RecADDetector(spec, r).n_parameters()))
    except Exception as exc:                       # noqa: BLE001
        rows.append((label, None))
        print(f"  !! {label}: {type(exc).__name__}: {exc}")

for name in ("BDD", "TemporalResidual", "SVM", "RF", "Autoencoder",
             "LSTM", "CNN", "Transformer", "GCN", "GAT"):
    try:
        rows.append((name, build_detector(name, spec, cfg).n_parameters()))
    except Exception as exc:                       # noqa: BLE001
        rows.append((name, None))
        print(f"  !! {name}: {type(exc).__name__}: {exc}")

print(f"  {'模型':<24}{'参数量':>16}")
print("  " + "-" * 42)
vals = []
for name, n in rows:
    if n is None:
        print(f"  {name:<24}{'构造失败':>16}")
    else:
        vals.append((name, n))
        print(f"  {name:<24}{n:>16,}")

if vals:
    lo_name, lo = min(vals, key=lambda kv: kv[1])
    hi_name, hi = max(vals, key=lambda kv: kv[1])
    decades = float(np.log10(hi / max(lo, 1)))
    print()
    print(f"  最小 {lo_name} = {lo:,}")
    print(f"  最大 {hi_name} = {hi:,}")
    print(f"  比值 = {hi / max(lo, 1):,.0f}x   ~= {decades:.1f} 个数量级")
    print()
    print("  摘要第 4 句写的是 'four orders of magnitude'：")
    print("    -> 成立（>= 4 个数量级）" if decades >= 4
          else f"    -> 【不成立】，实际只有 {decades:.1f} 个数量级，必须改措辞")
    print()
    print("  ⚠️ BDD 与 TemporalResidual 无需训练，参数量近 0；")
    print("     若摘要措辞是『学习型检测器』，跨度应在学习型行内计算。")
    learned = [(n, v) for n, v in vals
               if n not in ("BDD", "TemporalResidual") and v and v > 1000]
    if learned:
        l_lo = min(v for _, v in learned)
        l_hi = max(v for _, v in learned)
        dl = float(np.log10(l_hi / l_lo))
        print(f"  仅学习型：最小 {l_lo:,}  最大 {l_hi:,}  ~= {dl:.1f} 个数量级")
        print("    -> 学习型跨度 >= 4" if dl >= 4
              else f"    -> 学习型跨度只有 {dl:.1f}，措辞需改")
