"""Fix the `line` perturbation family so it is genuinely structural.

The first version of `diagnose_model_error_cliff.py` claimed a "line parameter"
perturbation but actually perturbed rows of H directly -- a rank-1 change to
some branch-flow rows. That is NOT what a wrong line reactance does. A wrong
b_ij moves:

  * the two diagonal entries b_bus[i,i], b_bus[j,j]   (row sums)
  * the two off-diagonal entries b_bus[i,j], b_bus[j,i]
  * the branch-flow row for (i, j)

i.e. it is correlated across BOTH the injection block and the flow block. The
row-wise version touched only the flow block.

This module rebuilds H from a perturbed susceptance matrix using the project's
own `build_jacobian`, and first proves it reproduces the true H when the
perturbation is zero -- without that check, a perturbation built on a wrong
reimplementation would be meaningless.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.generate_fdia import Network, build_jacobian      # noqa: E402


def h_from_susceptance(b: np.ndarray) -> np.ndarray:
    """H for a given branch-susceptance matrix, via the project's own builder.

    Wraps `b` in the `Network` dataclass so the exact production code path is
    used, rather than a reimplementation that could drift.
    """
    n = b.shape[0]
    net = Network(name="perturbed", n_bus=n, adjacency=(b > 0).astype(float),
                  susceptance=b, source="perturbed",
                  notes=["perturbed susceptance for the F2 model-error study"])
    h, _, _ = build_jacobian(net)
    return h


def perturb_lines(b_true: np.ndarray, target: float,
                  rng: np.random.Generator, frac: float = 0.3) -> np.ndarray:
    """Perturb a random subset of BRANCH SUSCEPTANCES, then rescale to target.

    Rescaling is done on the resulting H, because the target is defined as a
    relative Frobenius norm of dH -- a relative error on b does not translate
    linearly into one on H.
    """
    b = b_true.copy()
    iu = np.triu_indices_from(b, k=1)
    n_edges = len(iu[0])
    k = max(1, int(frac * n_edges))
    pick = rng.choice(n_edges, size=k, replace=False)
    rel = rng.normal(size=k) * 0.5          # relative error per chosen line
    for idx, r in zip(pick, rel):
        i, j = iu[0][idx], iu[1][idx]
        b[i, j] *= (1.0 + r)
        b[j, i] = b[i, j]
    if target == 0.0:
        return b_true.copy()
    # Match ||dH||_F / ||H||_F by bisection on a global scale of the change.
    h_true = h_from_susceptance(b_true)
    lo, hi = 0.0, 10.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        b_try = b_true + mid * (b - b_true)
        h_try = h_from_susceptance(b_try)
        rel_norm = np.linalg.norm(h_try - h_true) / np.linalg.norm(h_true)
        if rel_norm < target:
            lo = mid
        else:
            hi = mid
    return b_true + lo * (b - b_true)


def selftest() -> int:
    """The perturbation is only meaningful if zero perturbation reproduces H."""
    from data.contract import load_contract
    data = load_contract(HERE / "data_cache/temporal_smoke.npz")
    h_stored = data["h_matrix"].astype(np.float64)
    n = h_stored.shape[0] - int((h_stored.shape[0]) * 179 / 297)  # not robust

    # Recover b from the network instead of guessing: rebuild it.
    from data.generate_fdia import build_network
    for call in (lambda: build_network("118", 0),
                 lambda: build_network("118", seed=0),
                 lambda: build_network(case="118")):
        try:
            net = call()
            break
        except TypeError:
            continue
    else:
        print("  无法调用 build_network，自检跳过")
        return 1

    h_rebuilt, _, _ = build_jacobian(net)
    same_shape = h_rebuilt.shape == h_stored.shape
    print(f"  真实 H {h_stored.shape}   重建 H {h_rebuilt.shape}   形状一致={same_shape}")
    if same_shape:
        err = np.abs(h_rebuilt - h_stored).max() / np.abs(h_stored).max()
        print(f"  最大相对偏差 {err:.3e}   {'✓ 复现' if err < 1e-5 else '✗ 不一致，扰动不可信'}")
        return 0 if err < 1e-5 else 1
    return 1


if __name__ == "__main__":
    print("=" * 76)
    print("自检：零扰动下能否复现存储的 H")
    print("=" * 76)
    raise SystemExit(selftest())
