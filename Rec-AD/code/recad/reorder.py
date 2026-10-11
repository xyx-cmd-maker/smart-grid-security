"""Index reordering strategies for TT embedding tables.

Why this matters for the revision
----------------------------------
Section V.C of the original manuscript claims that "TT-compressed embeddings
effectively preserve the structural information of power flow equations"
without any quantitative support. Reviewer 1 (comment 6) asked for evidence.

This module supplies it in the strongest available form. The claim "structure
is preserved" is a representation-level statement and is easy to attack
("preserved is not the same as used"). The claim "structure is *used*" is
functional and falsifiable: reorder the embedding rows by grid topology and
compare against orderings that ignore topology. The measured gap IS the value
of the structural information.

Three strategies are provided so the comparison has a real control:

  * ``topology``  -- BFS over the grid graph, so electrically adjacent buses
                     land in adjacent embedding rows and share TT prefixes
  * ``random``    -- a uniform permutation; the honest null hypothesis
  * ``frequency`` -- most-accessed ids first; the standard industry heuristic,
                     which is what a reviewer would expect a system paper to
                     use as its baseline
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------

def bfs_order(adj: np.ndarray, start: int = 0) -> List[int]:
    """Breadth-first node order over the grid graph.

    BFS is the right traversal here rather than DFS: it orders buses by
    electrical distance from the slack bus, so nodes one hop apart stay one
    index apart, which is what makes TT prefixes collide.
    """
    n = int(adj.shape[0])
    if n == 0:
        return []
    adj_bool = np.asarray(adj) > 0
    visited = np.zeros(n, dtype=bool)
    order: List[int] = []

    # Start from the highest-degree node when the caller does not care, since
    # a leaf start makes the traversal degenerate into a line.
    if start is None:
        start = int(np.argmax(adj_bool.sum(axis=1)))
    queue = deque([int(start) % n])
    visited[int(start) % n] = True

    while queue:
        node = queue.popleft()
        order.append(node)
        for nbr in np.argsort(-adj_bool[node].astype(int) * 1.0):
            if adj_bool[node, nbr] and not visited[nbr]:
                visited[nbr] = True
                queue.append(int(nbr))

    # Disconnected components: append any remaining nodes in index order.
    for node in range(n):
        if not visited[node]:
            visited[node] = True
            order.append(node)
    return order


# ---------------------------------------------------------------------------
# Bijection construction
# ---------------------------------------------------------------------------

def random_permutation(num_ids: int, seed: int = 0) -> np.ndarray:
    """Uniformly random bijection, used as the null hypothesis.

    IMPORTANT: `seed` must be independent of whatever seed produced the id
    structure being tested. `default_rng(s).permutation(n)` and
    `default_rng(s).permutation(arr)` for a length-n array consume the same
    random stream and yield the *same* index permutation, so reusing the data
    seed here silently turns the "random" control into a copy of whatever
    structure the data happened to have. That failure mode makes the random
    baseline look far stronger than it is and hides the real effect.
    """
    rng = np.random.default_rng(seed)
    return rng.permutation(num_ids).astype(np.int64)


def frequency_permutation(ids: np.ndarray, num_ids: int) -> np.ndarray:
    """Order ids by descending access frequency.

    This is the caching heuristic a systems reviewer will expect as the
    non-topological baseline. Ids never observed are appended in index order.
    """
    flat = np.asarray(ids).reshape(-1)
    counts = np.bincount(flat, minlength=num_ids)
    order = np.argsort(-counts, kind="stable")
    perm = np.empty(num_ids, dtype=np.int64)
    perm[order] = np.arange(num_ids, dtype=np.int64)
    return perm


def topology_permutation(id_to_node: np.ndarray,
                         num_ids: int,
                         adj: np.ndarray,
                         start: Optional[int] = None) -> np.ndarray:
    """Order ids so that ids sharing a grid neighbourhood share a TT prefix.

    `id_to_node[i]` gives the bus that embedding row `i` belongs to. Rows are
    grouped by bus in BFS order, and within a bus by their original id so the
    mapping stays deterministic and reproducible.
    """
    id_to_node = np.asarray(id_to_node).reshape(-1)
    if len(id_to_node) != num_ids:
        raise ValueError(
            f"id_to_node has length {len(id_to_node)}, expected {num_ids}"
        )

    order = bfs_order(adj, start=0 if start is None else start)
    rank = np.empty(len(order), dtype=np.int64)
    for position, node in enumerate(order):
        rank[node] = position

    node_rank = rank[np.clip(id_to_node, 0, len(order) - 1)]
    # Sort by (bus BFS position, original id) for a stable, reproducible order.
    priority = node_rank.astype(np.int64) * (num_ids + 1) + np.arange(num_ids)
    sorted_ids = np.argsort(priority, kind="stable")

    perm = np.empty(num_ids, dtype=np.int64)
    perm[sorted_ids] = np.arange(num_ids, dtype=np.int64)
    return perm


def build_permutation(kind: str,
                      num_ids: int,
                      ids: Optional[np.ndarray] = None,
                      adj: Optional[np.ndarray] = None,
                      id_to_node: Optional[np.ndarray] = None,
                      seed: int = 0) -> np.ndarray:
    """Factory for the three strategies compared in the paper's reorder study."""
    kind = kind.lower()
    if kind in ("none", "identity"):
        return np.arange(num_ids, dtype=np.int64)
    if kind == "random":
        return random_permutation(num_ids, seed=seed)
    if kind == "frequency":
        if ids is None:
            raise ValueError("frequency ordering needs the observed id matrix")
        return frequency_permutation(ids, num_ids)
    if kind in ("topology", "bfs", "topology_bfs"):
        if adj is None or id_to_node is None:
            raise ValueError("topology ordering needs `adj` and `id_to_node`")
        return topology_permutation(id_to_node, num_ids, adj)
    raise KeyError(f"unknown ordering kind {kind!r}")


# ---------------------------------------------------------------------------
# Reuse measurement — the quantitative link to the TT table
# ---------------------------------------------------------------------------

def multi_index(ids: np.ndarray, factors: Sequence[int]) -> np.ndarray:
    """Row ids -> per-core multi-indices, C-order (the paper's Eq. (3))."""
    ids = np.asarray(ids).reshape(-1).astype(np.int64)
    d = len(factors)
    out = np.empty((len(ids), d), dtype=np.int64)
    for k in range(d):
        stride = int(np.prod(factors[k + 1:]))
        out[:, k] = (ids // stride) % factors[k]
    return out


def prefix_reuse_rate(ids: np.ndarray,
                      factors: Sequence[int],
                      prefix_len: Optional[int] = None) -> float:
    """Share of DISTINCT ids in a batch that share a leading multi-index.

    The basis must be distinct ids, not raw occurrences. Repeated occurrences
    of the same id are deduplicated by any sane lookup implementation and are
    not what the reuse mechanism exploits; the saving comes from *different*
    ids agreeing on their leading multi-indices so the prefix contraction can
    be computed once.

    Returns (D - P) / D where D is the number of distinct ids and P the number
    of distinct prefixes. 0 means every id has its own prefix.
    """
    ids = np.asarray(ids).reshape(-1)
    if len(ids) == 0:
        return 0.0
    d = len(factors)
    prefix_len = d - 1 if prefix_len is None else int(prefix_len)
    if prefix_len <= 0:
        return 0.0

    distinct = np.unique(ids)
    n_distinct = len(distinct)
    prefixes = multi_index(distinct, factors)[:, :prefix_len]
    n_prefixes = len(np.unique(prefixes, axis=0))
    return float((n_distinct - n_prefixes) / n_distinct) if n_distinct else 0.0


def contraction_counts_for_batch(ids: np.ndarray,
                                 factors: Sequence[int]) -> Tuple[int, int]:
    """(plain, reuse) contraction counts for one batch, over distinct ids.

    plain = D * (d-1)          every distinct id contracts all cores
    reuse = P * (d-2) + D      each distinct prefix contracts once, then one
                               final contraction per distinct id
    """
    ids = np.asarray(ids).reshape(-1)
    d = len(factors)
    if len(ids) == 0:
        return 0, 0
    distinct = np.unique(ids)
    n_distinct = len(distinct)
    prefixes = multi_index(distinct, factors)[:, :d - 1]
    n_prefixes = len(np.unique(prefixes, axis=0))
    plain = n_distinct * max(d - 1, 1)
    reused = n_prefixes * max(d - 2, 0) + n_distinct
    return int(plain), int(reused)


def reorder_batch(ids: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply a bijection to a batch of row ids."""
    ids = np.asarray(ids).reshape(-1).astype(np.int64)
    return perm[ids]


def evaluate_orderings(ids: np.ndarray,
                       num_ids: int,
                       factors: Sequence[int],
                       adj: Optional[np.ndarray] = None,
                       id_to_node: Optional[np.ndarray] = None,
                       kinds: Sequence[str] = ("none", "random", "frequency", "topology"),
                       batch_size: int = 4096,
                       seed: int = 0) -> List[Dict[str, float]]:
    """Compare reordering strategies by the reuse rate they induce.

    Rows are the paper's Table: reuse rate and the resulting contraction count
    for each strategy, on the same batch stream.
    """
    d = len(factors)
    results: List[Dict[str, float]] = []

    for kind in kinds:
        try:
            perm = build_permutation(kind, num_ids, ids=ids, adj=adj,
                                     id_to_node=id_to_node, seed=seed)
        except (ValueError, KeyError) as exc:
            results.append({"ordering": kind, "error": str(exc)})
            continue

        mapped = reorder_batch(ids, perm)
        n_batches = max(1, math.ceil(len(mapped) / batch_size))
        rates: List[float] = []
        plain_total = 0
        reuse_total = 0
        for b in range(n_batches):
            chunk = mapped[b * batch_size:(b + 1) * batch_size]
            if len(chunk) == 0:
                continue
            rates.append(prefix_reuse_rate(chunk, factors))
            plain_b, reuse_b = contraction_counts_for_batch(chunk, factors)
            plain_total += plain_b
            reuse_total += reuse_b

        mean_rate = float(np.mean(rates)) if rates else 0.0
        results.append({
            "ordering": kind,
            "reuse_rate": mean_rate,
            "contractions_plain": float(plain_total),
            "contractions_reuse": float(reuse_total),
            "contraction_ratio": (float(plain_total / reuse_total)
                                  if reuse_total else float("nan")),
            "n_batches": float(n_batches),
        })
    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> int:
    print("Index reordering self-test")
    print("=" * 62)

    n_bus = 64
    rows_per_bus = 1024
    num_ids = n_bus * rows_per_bus          # 65536
    # factors [64, 64, 16]: the (d-1)-prefix covers id // 16, so one bus
    # occupies exactly 1024/16 = 64 prefix groups.
    factors = [64, 64, 16]

    adj = np.zeros((n_bus, n_bus), dtype=np.float32)
    for i in range(n_bus - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    adj[0, n_bus - 1] = adj[n_bus - 1, 0] = 1.0
    for chord in range(0, n_bus, 8):
        adj[chord, (chord + 13) % n_bus] = adj[(chord + 13) % n_bus, chord] = 1.0

    order = bfs_order(adj)
    print(f"[1/4] BFS order (first 12): {order[:12]}")
    assert sorted(order) == list(range(n_bus)), "BFS must visit every node once"

    rng = np.random.default_rng(0)
    # Real embedding rows come from hashed cross features, so raw id order
    # carries no topology. Shuffling the id -> bus map is what makes the test
    # meaningful; without it the original order is already bus-contiguous.
    id_to_node = rng.permutation(np.repeat(np.arange(n_bus), rows_per_bus))
    bus_to_ids = np.argsort(id_to_node, kind="stable").reshape(n_bus, rows_per_bus)
    assert int(np.prod(factors)) == num_ids

    # Two access regimes. The distinction is the whole point of the study:
    # topology-aware ordering can only help when a batch concentrates on a
    # few buses. A batch that scans the whole grid has to touch every bus
    # regardless of how rows are laid out.
    def global_scan(n=200000):
        w = rng.dirichlet(np.ones(n_bus) * 0.3)
        bus = rng.choice(n_bus, size=n, p=w)
        pick = rng.integers(0, rows_per_bus, size=n)
        return bus_to_ids[bus, pick].astype(np.int64)

    def regional_scan(n=200000, span=4, batch_size=4096):
        """Each BATCH draws from one contiguous BFS window of `span` buses.

        The window must be fixed within a batch, not per sample: if every
        sample picks its own window, a 4096-sample batch still touches the
        whole grid and the regime collapses into the global scan. This models
        regional edge nodes that only see their own neighbourhood — the
        deployment setting the paper's positioning targets.
        """
        bus_by_rank = np.asarray(order)          # bus id at each BFS position
        n_windows = max(1, n_bus // span)
        blocks = []
        n_batches = int(np.ceil(n / batch_size))
        for _ in range(n_batches):
            w = int(rng.integers(0, n_windows))
            buses = bus_by_rank[(w * span + np.arange(span)) % n_bus]
            chosen = rng.choice(buses, size=batch_size)
            pick = rng.integers(0, rows_per_bus, size=batch_size)
            blocks.append(bus_to_ids[chosen, pick])
        return np.concatenate(blocks).astype(np.int64)

    print(f"[2/4] id space: {num_ids} rows, {n_bus} buses x {rows_per_bus} rows, "
          f"factors={factors}")
    print("      id -> bus mapping shuffled (raw order is topology-free)")

    print("[3/4] reuse rate by strategy and access regime:")
    print(f"      {'ordering':<12}{'global scan':>14}{'regional scan':>16}")
    # The permutation seed MUST differ from the data seed. Sharing seed 0 makes
    # `random` reproduce the block structure of the data and the comparison
    # silently collapses (see random_permutation's docstring).
    perm_seed = 987654321
    streams = {"global": global_scan(), "regional": regional_scan()}
    ids_probe = streams["regional"]
    summary: Dict[str, Dict[str, float]] = {}
    for label, stream in streams.items():
        res = evaluate_orderings(stream, num_ids, factors, adj=adj,
                                 id_to_node=id_to_node, batch_size=4096,
                                 seed=perm_seed)
        for r in res:
            if "error" in r:
                continue
            summary.setdefault(r["ordering"], {})[label] = r["reuse_rate"]

    for kind in ("none", "random", "frequency", "topology"):
        g = summary.get(kind, {}).get("global", float("nan"))
        rg = summary.get(kind, {}).get("regional", float("nan"))
        print(f"      {kind:<12}{g:>14.4f}{rg:>16.4f}")

    # Guard: an independent random permutation must NOT reproduce the bus-block
    # structure of the data. If it does, the seeds are entangled again.
    probe = build_permutation("random", num_ids, ids=ids_probe, adj=adj,
                              id_to_node=id_to_node, seed=perm_seed)
    mapped_probe = reorder_batch(ids_probe, probe)
    mi_probe = multi_index(np.unique(mapped_probe), factors)[:, 0]
    n_blocks = len(np.unique(mi_probe))
    print(f"      [guard] random permutation spans {n_blocks} core-0 blocks "
          f"(expect ~min(64, distinct/1024))")
    if n_blocks < 16:
        print()
        print("Self-test FAILED: the random permutation is correlated with the")
        print("data's id structure. Check that the permutation seed differs")
        print("from the seed used to build id_to_node.")
        return 1

    g_topo = summary.get("topology", {}).get("global", 0.0)
    g_rand = summary.get("random", {}).get("global", 0.0)
    r_topo = summary.get("topology", {}).get("regional", 0.0)
    r_rand = summary.get("random", {}).get("regional", 0.0)
    print()
    print(f"[4/4] topology - random:  global {g_topo - g_rand:+.4f}   "
          f"regional {r_topo - r_rand:+.4f}")

    if r_topo <= r_rand:
        print()
        print("Self-test FAILED: topology reordering must beat random when")
        print("batches concentrate on a few buses. Check the BFS order and the")
        print("id -> bus mapping.")
        return 1

    print()
    print("Self-test PASSED.")
    print()
    print("FINDING WORTH REPORTING IN THE PAPER: topology-aware reordering pays")
    print("off only when a batch concentrates on a few buses. Under a global")
    print("scan it is indistinguishable from a random permutation, because the")
    print("batch has to reach every bus no matter how rows are laid out.")
    print("Report both regimes; claiming a universal gain would be refutable")
    print("by any reviewer who tries the global-scan case.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
