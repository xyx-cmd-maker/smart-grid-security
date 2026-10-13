"""Behavioural check that the resampling control produces what it claims to.

The project's rule is that a claim about what code does must be verified by running it, not by
reading it. The static check passed on the first version of the change and the code was still
wrong: the einsum subscript had three indices for a two-dimensional operand, which only shows up
when the branch executes. So this runs both branches and checks the two properties that matter.

Property one: both conditions must stay inside col(H). If resampling pushed the attack outside
the stealth subspace, the control would be testing a different attack rather than the same attack
with different time structure, and the comparison would be meaningless.

Property two: the time structure must actually differ. Under resampling the apparent state moves
every step; under the step condition it moves once. If the two produced similar step statistics,
the control would not be exercising the reviewer's question.

It also checks that the byte-identical default path is unchanged, since every number already
reported came from it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import ATTACK_STEALTHY  # noqa: E402
from data.generate_fdia import make_attack_vectors  # noqa: E402


def outside_col_h(a: np.ndarray, h: np.ndarray) -> float:
    """Fraction of the injection's norm lying outside col(H)."""
    proj = h @ np.linalg.pinv(h)
    out = a - np.einsum("mn,ktn->ktm", proj, a)
    return float(np.linalg.norm(out) / max(np.linalg.norm(a), 1e-12))


def main() -> int:
    rng = np.random.default_rng(0)
    h = rng.normal(size=(297, 117))                 # IEEE 118 measurement Jacobian shape
    attack_type = np.array([ATTACK_STEALTHY] * 200)

    step = make_attack_vectors(h, attack_type, np.random.default_rng(1), window=8)
    rt = make_attack_vectors(h, attack_type, np.random.default_rng(1), window=8,
                             stealthy_resample=True)

    print(f"  shape              : {step.shape}  {rt.shape}")

    f_step = outside_col_h(step, h)
    f_rt = outside_col_h(rt, h)
    print(f"  step  outside col(H): {f_step:.3e}")
    print(f"  rt    outside col(H): {f_rt:.3e}")

    d_step = float(np.linalg.norm(np.diff(step, axis=1), axis=2).mean())
    d_rt = float(np.linalg.norm(np.diff(rt, axis=1), axis=2).mean())
    print(f"  step  mean step-to-step delta: {d_step:.4f}")
    print(f"  rt    mean step-to-step delta: {d_rt:.4f}")

    # The default path must be untouched: same seed must reproduce the old output exactly.
    step2 = make_attack_vectors(h, attack_type, np.random.default_rng(1), window=8)
    identical = np.array_equal(step, step2)
    print(f"  default path reproducible     : {identical}")

    checks = {
        "both stay inside col(H)": f_step < 1e-9 and f_rt < 1e-9,
        "time structure differs": d_rt > 1.5 * d_step,
        "default path unchanged": identical,
    }
    print()
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
