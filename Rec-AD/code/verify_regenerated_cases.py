"""Verify the regenerated IEEE cases carry the correct attack construction.

The failure mode being guarded against: the previous ieee118.npz was produced
before attack magnitudes were normalised, so its stealthy and load-redistribution
families nearly doubled ||dense||. Reading `dense` then trivially revealed those
families, which inverted every learned detector's per-family result.

Pass condition for all four files:
  * every family has ||dense|| median within a few percent of benign
  * random keeps its large residual (it must stay easy for BDD)
  * stealthy keeps a benign-sized residual (it must stay invisible to BDD)

If any of those fail, the run must not be used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract      # noqa: E402

FILES = ["ieee14", "ieee57", "ieee118", "ieee300"]
ok_all = True

for tag in FILES:
    p = HERE / f"data_cache/{tag}.npz"
    if not p.exists():
        print(f"  {tag}: 缺失")
        ok_all = False
        continue

    d = load_contract(p)
    if "window" in d:
        w = np.asarray(d["window"])
        z = w[:, -1, :].astype(np.float64)
    else:
        z = np.asarray(d["dense"], dtype=np.float64)
    h = np.asarray(d["h_matrix"], dtype=np.float64)
    at = np.asarray(d["attack_type"])

    proj = h @ np.linalg.pinv(h)
    res = z - z @ proj.T

    print(f"  ===== {tag}  ({len(at):,} 样本, {z.shape[1]} 量测) =====")
    print(f"    {'族':<12}{'n':>8}{'||dense||':>12}{'倍数':>8}{'||res||':>11}{'倍数':>8}")
    print("    " + "-" * 58)

    b_z = b_r = None
    for code, nm in ((0, "benign"), (1, "random"), (2, "stealthy"), (3, "load_redist")):
        m = at == code
        if not m.any():
            continue
        dz = float(np.median(np.linalg.norm(z[m], axis=1)))
        dr = float(np.median(np.linalg.norm(res[m], axis=1)))
        if b_z is None:
            b_z, b_r = dz, dr
        print(f"    {nm:<12}{int(m.sum()):>8}{dz:>12.2f}{dz / b_z:>7.2f}x"
              f"{dr:>11.3f}{dr / b_r:>7.2f}x")

    # pass conditions
    m_s, m_r = at == 2, at == 1
    z_ratio = float(np.median(np.linalg.norm(z[m_s], axis=1))) / b_z
    r_ratio = float(np.median(np.linalg.norm(res[m_r], axis=1))) / b_r
    good = (0.9 <= z_ratio <= 1.1) and (r_ratio > 5.0)
    ok_all &= good
    print(f"    判定: stealthy 的 ||dense|| 倍数 {z_ratio:.2f} "
          f"({'✓' if 0.9 <= z_ratio <= 1.1 else '✗ 归一化未生效'})"
          f"    random 的残差倍数 {r_ratio:.1f} "
          f"({'✓' if r_ratio > 5.0 else '✗ 随机族太弱'})")
    print()

print("=" * 66)
print(f"  总计: {'✓ 四个算例全部通过' if ok_all else '✗ 有算例未通过，不得使用'}")
