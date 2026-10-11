"""Test the mechanism claim behind the meeting's reframing of the reorder component.

The teacher's account is: topology is not a parallel alternative to frequency ordering,
it is the *reason* frequency ordering works -- vertices with more connecting edges get
reused more often, so grouping tightly-connected buses into a community and then sorting
by descending frequency should help. The code does not implement that: it has two
independent permutation functions (BFS over the grid graph, and descending access
frequency) and compares them side by side. That is why the measurement came out
"topology 0.6730 < frequency 0.7133" and why it looked like topology had no value.

Two claims are separable and both are testable without training:

  H1  Access frequency correlates with vertex degree. If it does not, the mechanism the
      teacher describes has no substrate in this data, and no ordering can exploit it.

  H2  A community-aware frequency ordering beats plain frequency ordering. If it does,
      the component has real value that the current implementation simply fails to
      express. If it does not, the value is not there regardless of framing.

Both are cheap: no model, no training, just the observed id matrix, the adjacency, and
prefix-reuse arithmetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                    # noqa: E402
from data.dataset import build_spec                        # noqa: E402
import recad.reorder as R                                  # noqa: E402

np.random.seed(0)
d = load_contract(HERE / "data_cache/ieee118.npz")
spec = build_spec(d)
adj = np.asarray(d["adj"], dtype=np.float64)
sparse = np.asarray(d["sparse"])

print("=" * 88)
print("  Does topology explain frequency, or compete with it?")
print("=" * 88)
print()

# --- observed access counts per id ------------------------------------------
flat = sparse.reshape(-1)
num_ids = int(flat.max()) + 1
counts = np.bincount(flat, minlength=num_ids).astype(np.float64)
print(f"  ids observed      : {num_ids}")
print(f"  total accesses    : {int(counts.sum()):,}")
print(f"  distinct ids used : {int((counts > 0).sum())}")
print()

# --- H1: does degree correlate with access frequency? -----------------------
# The id -> node map is the model's own; recover it the same way the ordering does.
id_to_node = getattr(spec, "id_to_node", None)
if id_to_node is None:
    id_to_node = np.arange(num_ids) % adj.shape[0]
id_to_node = np.asarray(id_to_node).ravel()[:num_ids]

deg = adj.sum(axis=1)
node_deg = deg[id_to_node]
used = counts > 0

print("  " + "-" * 84)
print("  H1: does access frequency track vertex degree?")
print("  " + "-" * 84)
if used.sum() > 10 and np.std(node_deg[used]) > 0:
    r_pearson = float(np.corrcoef(node_deg[used], counts[used])[0, 1])
    # Spearman without scipy: rank both, then Pearson on ranks.
    def rank(a):
        o = np.argsort(a)
        r = np.empty_like(o, dtype=float)
        r[o] = np.arange(len(a))
        return r
    r_spear = float(np.corrcoef(rank(node_deg[used]), rank(counts[used]))[0, 1])
    print(f"  Pearson  (degree, count) = {r_pearson:+.4f}")
    print(f"  Spearman (degree, count) = {r_spear:+.4f}")
    verdict = ("SUPPORTED" if r_spear > 0.3 else
               "WEAK" if r_spear > 0.1 else "NOT SUPPORTED")
    print(f"  -> H1 {verdict}")
else:
    print("  insufficient variation to test")
    r_spear = float("nan")

# --- H2: community-aware frequency vs plain frequency -----------------------
print()
print("  " + "-" * 84)
print("  H2: does community-aware frequency ordering beat plain frequency?")
print("  " + "-" * 84)

ids = sparse
factors = None
for attr in ("sparse_factors", "factors", "cardinalities"):
    v = getattr(spec, attr, None)
    if v is not None:
        factors = list(np.asarray(v).ravel())
        break
if factors is None:
    factors = [int(ids.max()) + 1]
print(f"  multi-index factors: {factors}")

base = {k: R.build_permutation(k, num_ids=num_ids, ids=ids, adj=adj,
                               id_to_node=id_to_node)
        for k in ("none", "frequency", "topology")}

# Community-aware: split buses into connected communities, then order by frequency
# with community membership as the primary key (the meeting's "group close buses,
# then sort descending by frequency").
n_nodes = adj.shape[0]
deg_sorted = np.argsort(-deg)
communities = np.zeros(n_nodes, dtype=int)
c = 0
for v in deg_sorted:
    if communities[v] != 0 or v == deg_sorted[0]:
        continue
    c += 1
    stack = [int(v)]
    while stack:
        u = stack.pop()
        if communities[u] != 0:
            continue
        communities[u] = c
        for w in np.nonzero(adj[u])[0]:
            if communities[w] == 0:
                stack.append(int(w))
print(f"  communities found  : {c}")

node_comm = communities[id_to_node]
order_ids = np.arange(num_ids)
# primary key: -count (descending frequency); tie-break: community
community_freq = order_ids[np.lexsort((node_comm, -counts))]
community_perm = np.empty(num_ids, dtype=int)
community_perm[community_freq] = np.arange(num_ids)

print()
print(f"  {'ordering':<22}{'reuse rate':>13}{'contractions':>15}")
print("  " + "-" * 84)
rows = dict(base)
rows["community+freq"] = community_perm
for name, perm in rows.items():
    try:
        rr = R.prefix_reuse_rate(ids, perm, factors=factors)
        cc = R.contraction_counts_for_batch(ids, perm, factors=factors)
    except TypeError:
        rr = R.prefix_reuse_rate(ids, perm)
        cc = R.contraction_counts_for_batch(ids, perm)
    val = float(rr) if np.ndim(rr) == 0 else float(np.mean(rr))
    cval = float(cc) if np.ndim(cc) == 0 else float(np.mean(cc))
    print(f"  {name:<22}{val:>13.4f}{cval:>15,.0f}")

print()
print("  " + "-" * 84)
print("  How to read this")
print("  " + "-" * 84)
print("  H1 supported  -> the teacher's mechanism has a substrate; an ordering can exploit it.")
print("  H2 supported  -> community-aware frequency beats plain frequency, so the component")
print("                   has value the current BFS implementation fails to express.")
print("  H2 not supported -> the value is absent regardless of how the claim is framed.")
print()
