"""Why do the learned baselines invert the attack families?

The just-finished baselines run produces a pattern that AGENTS.md flags as a
trap:

    BDD          random 1.0000   stealthy 0.0058      <- correct
    SVM/RF/AE/CNN random ~0.01   stealthy ~1.00       <- inverted

But `standardize` was False in the run, so the documented cause does not apply,
and the false-alarm rates are ~1-2%, so the threshold is not the cause either.

Before theorising, measure the data: if the two families differ in residual
magnitude the way section 5.2 documents (random 30x benign, stealthy 1.0x), then
BDD's behaviour is expected and the learned models' behaviour is the anomaly.
If the magnitudes are the other way round in THIS file, then the file itself
differs from the documented one and everything downstream is suspect.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract      # noqa: E402

d = load_contract(HERE / "data_cache/ieee118.npz")
z = np.asarray(d["dense"], dtype=np.float64)
h = np.asarray(d["h_matrix"], dtype=np.float64)
at = np.asarray(d["attack_type"])

proj = h @ np.linalg.pinv(h)
res = z - z @ proj.T

print("ieee118.npz 各族统计")
print(f"  {'族':<12}{'n':>7}{'||dense||中位':>16}{'||residual||中位':>19}{'倍数':>10}")
print("  " + "-" * 64)

base = None
for code, nm in ((0, "benign"), (1, "random"), (2, "stealthy"), (3, "load_redist")):
    m = at == code
    if not m.any():
        continue
    rd = float(np.median(np.linalg.norm(res[m], axis=1)))
    dz = float(np.median(np.linalg.norm(z[m], axis=1)))
    if base is None:
        base = rd
    print(f"  {nm:<12}{int(m.sum()):>7}{dz:>16.3f}{rd:>19.3f}{rd / base:>9.2f}x")

print()
print("  对照：AGENTS.md §5.2 记录（temporal_smoke.npz）")
print("    benign 1.00x / random 30.0x / load_redist 3.0x / stealthy 1.00x")
print()
print("  若本文件与 §5.2 一致 -> BDD 的行为是预期的，学习型模型的反转是异常")
print("  若本文件相反        -> 文件本身与文档不符，下游全部可疑")
