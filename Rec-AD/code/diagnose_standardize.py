"""Is `data.standardize: true` the cause of the inverted attack-family results?

Runs the same residual BDD on the same dataset twice, changing only whether the
features are standardised. The threat model requires:
    random   -> high detection
    stealthy -> detection ~= false-alarm rate
    load-redist -> in between
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from baselines.bdd import BDDDetector                      # noqa: E402
from data.contract import load_contract                    # noqa: E402
from data.dataset import build_loaders, build_spec, loader_arrays  # noqa: E402
from evaluate import per_attack_type_metrics               # noqa: E402

DATA = HERE / "data_cache/ieee118.npz"
ALPHA = 0.01

print(f"数据: {DATA.name}")
print()


def run(standardize: bool):
    data = load_contract(DATA)
    spec = build_spec(data)
    loaders = build_loaders(data, batch_size=1024, num_workers=0,
                            standardize=standardize)
    bdd = BDDDetector(spec, {"baselines": {"bdd": {"alpha": ALPHA}}},
                      h_matrix=data["h_matrix"])
    bdd.fit(loaders["train"], loaders["val"])
    _, _, y, at = loader_arrays(loaders["test"])
    s = per_attack_type_metrics(y, bdd.predict_proba(loaders["test"]), at,
                                threshold=1.0 - ALPHA)
    return s


rows = {}
for flag in (True, False):
    try:
        rows[flag] = run(flag)
    except Exception as e:                                   # noqa: BLE001
        print(f"  standardize={flag}: 失败 {type(e).__name__}: {e}")
        rows[flag] = None

hdr = f"{'standardize':>13}{'normal_fp':>12}{'random':>11}{'stealthy':>11}{'load_redist':>13}"
print(hdr)
print("-" * len(hdr))
for flag in (True, False):
    s = rows[flag]
    if s is None:
        print(f"{str(flag):>13}   失败")
        continue
    print(f"{str(flag):>13}"
          f"{s['detect_rate_normal_fp']:>12.4f}"
          f"{s['detect_rate_random']:>11.4f}"
          f"{s['detect_rate_stealthy']:>11.4f}"
          f"{s['detect_rate_load_redist']:>13.4f}")

print()
print("威胁模型要求：random 高、stealthy ≈ normal_fp、load_redist 居中")
print()

s_false = rows[False]
s_true = rows[True]
if s_false and s_true:
    ok_false = (s_false["detect_rate_random"] > 0.8
                and s_false["detect_rate_stealthy"] < 0.10)
    ok_true = (s_true["detect_rate_random"] > 0.8
               and s_true["detect_rate_stealthy"] < 0.10)
    print(f"  standardize=False 符合威胁模型: {ok_false}")
    print(f"  standardize=True  符合威胁模型: {ok_true}")
    print()
    if ok_false and not ok_true:
        print("  => 确认：standardize=True 破坏了 a=Hc 的隐蔽性，")
        print("     所有用默认配置跑出的基线结果都不能用于论文。")
    elif ok_true:
        print("  => 标准化不是原因，需另找。")
