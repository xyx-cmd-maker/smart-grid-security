"""Which embedding dimension do the baselines and Rec-AD actually use?

The ablation's central row is "same architecture, dense table vs TT table". That
comparison is only valid if both arms use the same embedding dimension. Two numbers
are in play and they disagree:

  * the data contract's meta records embedding_dim = 16
  * RecADConfig.from_cfg derives embedding_dim = n_per_core ** d = 4**3 = 64

If the baselines read the spec (16) while Rec-AD derives from config (64), the two
arms differ in more than the table format and the ablation does not isolate what it
claims to isolate.

This prints what each side resolves, plus the parameter arithmetic that identifies
which dimension the built models actually contain.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                     # noqa: E402
from data.dataset import build_spec                         # noqa: E402
from recad.common import load_config                        # noqa: E402
from recad.model import RecADConfig, RecADDetector           # noqa: E402
from baselines.registry import build_detector               # noqa: E402

d = load_contract(HERE / "data_cache/ieee118.npz")
spec = build_spec(d)
cfg = load_config(None)

print("=" * 80)
print("嵌入维度核对")
print("=" * 80)
print()

# ---- what the contract says -------------------------------------------------
meta = d.get("meta")
if isinstance(meta, str):
    import json
    try:
        meta = json.loads(meta)
    except Exception:                                # noqa: BLE001
        meta = {}
if isinstance(meta, dict):
    print(f"  数据契约 meta:")
    for k in ("embedding_dim", "embedding_table_rows", "embedding_table_bytes_fp32",
              "n_bin_cross"):
        if k in meta:
            print(f"    {k:28} = {meta[k]}")

print()
print("  DataSpec 字段:")
names = [a for a in dir(spec) if not a.startswith("_")]
for k in sorted(names):
    v = getattr(spec, k)
    if isinstance(v, (int, float, str, bool)):
        print(f"    {k:28} = {v}")

# ---- what each side derives -------------------------------------------------
r = RecADConfig.from_cfg(cfg)
tt = cfg.get("tt", {}) or {}
sp = tt.get("split", {}) or {}
print()
print("  RecADConfig 推导:")
print(f"    tt.split = {sp}")
print(f"    embedding_dim = {r.embedding_dim}   (= {sp.get('n_per_core')}^{sp.get('d')})")
print(f"    tt_rank = {r.tt_rank}")

# ---- parameter arithmetic: which dimension do the BUILT models contain? -----
print()
print("  构建后的参数量（用来反推实际维度）:")
rows = spec.sparse_total_rows
print(f"    sparse_total_rows = {rows:,}")
for dim in (16, 32, 64):
    print(f"      若 dim={dim:3} -> 表参数 {rows * dim:>14,}")

try:
    m = RecADDetector(spec, r)
    print(f"    RecADDetector(默认配置) 参数量 = {m.n_parameters():,}")
except Exception as exc:                            # noqa: BLE001
    print(f"    !! RecADDetector 构造失败: {type(exc).__name__}: {exc}")

for name in ("CNN", "LSTM"):
    try:
        b = build_detector(name, spec, cfg)
        n = b.n_parameters()
        print(f"    {name:<10} 参数量 = {n:,}"
              f"   (n - 表参数16 = {n - rows*16:,};  n - 表参数64 = {n - rows*64:,})")
    except Exception as exc:                        # noqa: BLE001
        print(f"    !! {name} 构造失败: {type(exc).__name__}: {exc}")

print()
print("  判读：")
print("    某个 (参数量 - 表参数dim) 落在几千到几十万的小正数，说明该 dim 就是模型实际用的。")
print("    若两边的差值分布不同，则消融的『同架构』对照不成立。")
