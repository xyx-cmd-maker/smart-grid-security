"""Recompute the resolution figures at the largest observed seed spread and restate the sentence.

The manuscript said "the observed spread of 0.0035", which is neither a measured value nor the largest one. The three per-case spreads are
0.0027, 0.0031 and 0.0039. For a limit on what the design can resolve, the conservative choice is the largest, because a larger spread
means fewer differences are distinguishable; using a smaller one makes the stated limit look better than the data supports.

At 0.0039 the n = 3 detectable difference rises from 0.0107 to 0.0120, and the ten-seed figure from 0.0046 to 0.0052. The sentence is
rewritten to name the figure it uses and why, so a reader can check it against the per-case spreads rather than trusting a bare number.
"""
from __future__ import annotations

import math

from scipy import stats

SD_LARGEST = 0.0039
SPREADS = (0.0027, 0.0031, 0.0039)


def min_detectable_d(n: int, target: float = 0.80) -> float:
    df = 2 * (n - 1)
    crit = stats.t.ppf(0.975, df)

    def power(d: float) -> float:
        lam = d * math.sqrt(n / 2.0)
        return (1.0 - stats.nct.cdf(crit, df, lam)) + stats.nct.cdf(-crit, df, lam)

    lo, hi = 0.2, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if power(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


print(f"  per-case spreads: {', '.join(f'{s:.4f}' for s in SPREADS)}")
print(f"  largest used     : {SD_LARGEST:.4f}")
print()
print(f"  {'n/group':>8}{'d (SD)':>10}{'AUC diff @0.0039':>20}")
for n in (3, 10):
    d = min_detectable_d(n)
    print(f"  {n:>8}{d:>10.3f}{d * SD_LARGEST:>20.4f}")

d3 = min_detectable_d(3)
d10 = min_detectable_d(10)
print()
print(f"  manuscript now says: 0.0107 (n=3) and 0.0046 (n=10) at spread 0.0035")
print(f"  conservative values : {d3 * SD_LARGEST:.4f} (n=3) and {d10 * SD_LARGEST:.4f} (n=10) at 0.0039")
