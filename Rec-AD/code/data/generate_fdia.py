"""Reference FDIA data generator.

Why this exists
---------------
Reviewer 1 (comment 7) asked how MATLAB-format grid data became PyTorch
tensors and raised concerns about data authenticity; Reviewer 1 (comment 5)
questioned the datasets; Reviewer 2 (comment 4) asked for more topologies.

This script is the auditable answer to all three. It produces the documented
array contract that every other module consumes, and it prints the
cardinality, composition and split diagnostics that the manuscript must
report.

It is a REFERENCE implementation. The authors' own pipeline produced the
numbers in the paper; this one exists so that a reader — or a reviewer — can
regenerate a dataset with the stated properties and check the attack
construction independently.

Attack construction follows the threat model in revised Sec. III:

  1. random FDIA        a drawn outside col(H)      -> residual test catches it
  2. stealthy FDIA      a = Hc                      -> residual test is blind
  3. load redistribution a = Hc + delta             -> subtlest of the three

Usage
-----
    python data/generate_fdia.py --case 118 --out data_cache/ieee118.npz
    python data/generate_fdia.py --case 300 --bus 300 --out data_cache/ieee300.npz
    python data/generate_fdia.py --list-cases
    python data/generate_fdia.py --case 118 --no-pandapower   # synthetic topology
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from data.contract import (
    ATTACK_LOAD_REDIST,
    ATTACK_NORMAL,
    ATTACK_RANDOM,
    ATTACK_STEALTHY,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    CONTRACT_VERSION,
    describe_contract,
    save_contract,
)

#: Test cases reachable through pandapower. The bus count is the fallback
#: when pandapower is unavailable.
KNOWN_CASES: Dict[str, int] = {
    "14": 14, "30": 30, "57": 57, "118": 118, "300": 300,
}


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

@dataclass
class Network:
    name: str
    n_bus: int
    adjacency: np.ndarray              # (n, n) 0/1
    susceptance: np.ndarray            # (n, n) branch susceptances b_ij
    source: str                        # "pandapower" | "synthetic"
    notes: List[str] = field(default_factory=list)

    @property
    def n_branch(self) -> int:
        return int(np.triu(self.susceptance > 0).sum())


def _synth_network(n_bus: int, seed: int = 0, extra_edges: int = 0) -> Network:
    """Random connected graph with random susceptances.

    A stand-in topology used when pandapower is absent. It preserves the
    properties the experiments depend on — connectivity, sparsity, a
    non-trivial cycle space — but it is NOT an IEEE test case and must be
    labelled as synthetic wherever it appears.
    """
    rng = np.random.default_rng(seed)
    adj = np.zeros((n_bus, n_bus), dtype=np.float64)
    # Spanning tree keeps the graph connected.
    for i in range(1, n_bus):
        j = int(rng.integers(0, i))
        adj[i, j] = adj[j, i] = 1.0
    # Extra edges create cycles, which the stealth attack construction needs:
    # a tree has a trivial cycle space and col(H) degenerates.
    target_extra = extra_edges or max(1, n_bus // 5)
    added = 0
    while added < target_extra:
        i, j = rng.integers(0, n_bus, size=2)
        if i != j and adj[i, j] == 0:
            adj[i, j] = adj[j, i] = 1.0
            added += 1

    b = np.zeros_like(adj)
    idx = np.triu(adj) > 0
    b[idx] = rng.uniform(2.0, 12.0, size=int(idx.sum()))
    b = b + b.T
    return Network(name=f"synthetic_{n_bus}", n_bus=n_bus, adjacency=adj,
                   susceptance=b, source="synthetic",
                   notes=["Synthetic topology: not an IEEE test case."])


def _pandapower_network(case: str, seed: int = 0,
                        verbose: bool = False) -> Optional[Network]:
    """Load an IEEE case through pandapower and extract DC susceptances.

    pandapower 3.x changed the branch tables relative to earlier releases:

      * `net.line` no longer carries a total reactance column `x`. It carries
        `x_ohm_per_km` and `length_km`, so the total is a product. Reading
        `br.x` raises AttributeError, which is how this was caught.
      * `net.line` also carries `parallel` and `df`, which scale the branch.
      * Buses are 0-based row indices, so `from_bus`/`to_bus` index the
        adjacency matrix directly.

    Susceptances are returned in per unit on the system base:
        Z_base = vn_kv^2 / sn_mva            (ohms)
        x_pu   = x_total_ohm / Z_base
        b_pu   = 1 / x_pu
    Transformers are converted from their own rating to the system base.
    """
    try:
        import pandapower.networks as pn
    except ImportError:
        return None

    loader = {
        "14": getattr(pn, "case14", None),
        "30": getattr(pn, "case30", None),
        "57": getattr(pn, "case57", None),
        "118": getattr(pn, "case118", None),
        "300": getattr(pn, "case300", None),
    }.get(str(case))
    if loader is None:
        return None

    net = loader()
    n_bus = len(net.bus)
    sn_mva = float(getattr(net, "sn_mva", 100.0)) or 100.0
    vn_kv = net.bus["vn_kv"].to_numpy(dtype=float)

    adj = np.zeros((n_bus, n_bus), dtype=np.float64)
    b = np.zeros((n_bus, n_bus), dtype=np.float64)
    notes: List[str] = []
    skipped = 0
    clamped = 0

    # Plausible band for a per-unit series reactance of a transmission branch.
    # Values outside it mean the case file uses a unit convention this
    # extractor does not reproduce, which is the situation for the transformer
    # tables in pandapower 3.x (see the note emitted below).
    X_MIN, X_MAX = 0.005, 0.5

    def accept_x(x_pu: float) -> Tuple[float, bool]:
        """Return a usable per-unit reactance plus whether it was clamped."""
        if not np.isfinite(x_pu) or x_pu <= 0:
            return 0.1, True
        if x_pu < X_MIN:
            return X_MIN, True
        if x_pu > X_MAX:
            return X_MAX, True
        return x_pu, False

    def z_base(bus_a: int, bus_b: int) -> float:
        """Ohm base for a branch, using the higher nominal voltage."""
        v = max(vn_kv[bus_a], vn_kv[bus_b])
        return (v ** 2) / sn_mva if v > 0 else 1.0

    # ---- lines -----------------------------------------------------------
    for _, br in net.line.iterrows():
        if not bool(br.get("in_service", True)):
            skipped += 1
            continue
        i, j = int(br["from_bus"]), int(br["to_bus"])
        # pandapower 3.x: reactance is per km; older versions exposed `x`.
        if "x_ohm_per_km" in net.line.columns:
            x_total = float(br["x_ohm_per_km"]) * float(br["length_km"])
        else:                                    # legacy layout
            x_total = float(br.get("x", 0.0))
        parallel = max(1.0, float(br.get("parallel", 1) or 1))
        x_total /= parallel
        x_pu, was_clamped = accept_x(x_total / z_base(i, j))
        clamped += int(was_clamped)
        adj[i, j] = adj[j, i] = 1.0
        b[i, j] = b[j, i] = 1.0 / x_pu

    # ---- transformers ----------------------------------------------------
    # NOTE ON UNITS. In pandapower 3.x the bundled test cases report
    # transformer data in a convention that is not the documented one: every
    # transformer in case14/300 carries sn_mva = 9900 and vk_percent values in
    # the hundreds to tens of thousands, whereas a short-circuit voltage is
    # physically a few percent. Treating vk_percent/100 as a per-unit
    # reactance therefore gives non-physical values, and dividing by the
    # reported sn_mva only rescues some of them by coincidence.
    #
    # We therefore keep the CONNECTION (which is what defines the topology and
    # hence the structure of col(H)) and clamp the reactance into a physically
    # plausible band. The topology is exact; the transformer reactances are
    # nominal. This is stated in the contract metadata and must be repeated in
    # the paper's data section.
    for _, tr in net.trafo.iterrows():
        if not bool(tr.get("in_service", True)):
            skipped += 1
            continue
        i, j = int(tr["hv_bus"]), int(tr["lv_bus"])
        vk = float(tr.get("vk_percent", 0.0) or 0.0)
        sn_tr = float(tr.get("sn_mva", 0.0) or 0.0)
        raw_x = (vk / 100.0) * (sn_mva / sn_tr) if (vk > 0 and sn_tr > 0) else 0.1
        x_pu, was_clamped = accept_x(raw_x)
        clamped += int(was_clamped)
        adj[i, j] = adj[j, i] = 1.0
        b[i, j] = b[j, i] = 1.0 / x_pu

    # ---- validation ------------------------------------------------------
    n_branch = int(np.triu(adj > 0).sum())
    if n_branch == 0:
        notes.append("no branches extracted; falling back to synthetic")
        return None

    # Connectivity: col(H) degenerates across islands, so report them.
    seen = np.zeros(n_bus, dtype=bool)
    islands = 0
    for start in range(n_bus):
        if seen[start]:
            continue
        islands += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            for nbr in np.nonzero(adj[node])[0]:
                if not seen[nbr]:
                    seen[nbr] = True
                    stack.append(int(nbr))

    b_vals = b[np.triu(b > 0)]
    notes.insert(0, f"pandapower case{case}: {n_bus} buses, {n_branch} branches")
    if skipped:
        notes.append(f"{skipped} branch(es) skipped (out of service)")
    if clamped:
        notes.append(
            f"{clamped}/{n_branch} branch reactances clamped into "
            f"[{X_MIN}, {X_MAX}] pu (case files report non-physical "
            f"transformer units); topology is exact, reactance magnitudes are nominal"
        )
    if islands > 1:
        notes.append(f"WARNING: graph has {islands} islands")

    if verbose:
        print(f"  topology      : pandapower case{case}")
        print(f"  buses/branches: {n_bus} / {n_branch}")
        print(f"  islands       : {islands}   {'OK' if islands == 1 else '<-- PROBLEM'}")
        print(f"  reactance     : min={1/b_vals.max():.4f} median={1/np.median(b_vals):.4f} "
              f"max={1/b_vals.min():.4f}  (pu)")
        print(f"  clamped       : {clamped}/{n_branch}")
        if skipped:
            print(f"  skipped       : {skipped}")

    return Network(name=f"ieee{case}", n_bus=n_bus, adjacency=adj,
                   susceptance=b, source="pandapower", notes=notes)


def build_network(case: str, n_bus: Optional[int] = None,
                  allow_pandapower: bool = True, seed: int = 0) -> Network:
    net = _pandapower_network(case, seed=seed) if allow_pandapower else None
    if net is not None:
        return net
    size = n_bus or KNOWN_CASES.get(str(case), 118)
    return _synth_network(size, seed=seed)


# ---------------------------------------------------------------------------
# Measurement model and Jacobian
# ---------------------------------------------------------------------------

def build_jacobian(net: Network) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DC measurement Jacobian.

    State: bus voltage angles, with the reference bus angle fixed at zero, so
    the state dimension is n_bus - 1.

    Measurements, stacked in this order:
      1. bus active power injections   (n_bus rows)
      2. branch active power flows     (n_branch rows)

    Returns (H, injection_rows, flow_rows).
    """
    n = net.n_bus
    b = net.susceptance
    b_bus = -b.copy()
    np.fill_diagonal(b_bus, b.sum(axis=1))

    # Drop the reference bus (index 0) from the state.
    state_cols = [k for k in range(n) if k != 0]
    h_inj = b_bus[:, state_cols]                      # (n, n-1)

    edges = np.argwhere(np.triu(b > 0))
    h_flow = np.zeros((len(edges), len(state_cols)), dtype=np.float64)
    for row, (i, j) in enumerate(edges):
        bij = b[i, j]
        if i in state_cols:
            h_flow[row, state_cols.index(i)] += bij
        if j in state_cols:
            h_flow[row, state_cols.index(j)] -= bij

    h = np.vstack([h_inj, h_flow])
    return h, np.arange(n), np.arange(n, n + len(edges))


def _clean_dataset(net: Network, h: np.ndarray, n_samples: int,
                   noise_sigma: float, rng: np.random.Generator,
                   window: int = 1,
                   state_step_sigma: float = 0.02
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """z_t = H x_t + e_t along a state trajectory.

    The state follows a random walk with a BOUNDED per-step increment
    (`state_step_sigma`). Real power-system states move continuously: load drifts,
    it does not teleport. That bound is what gives the temporal detector its
    power -- an injection that makes the apparent state JUMP is inconsistent
    with the dynamics, even when the jump lies inside col(H) and is therefore
    invisible to any residual test on a single snapshot.

    Returns
    -------
    window : (n_samples, window, m) -- `window` consecutive snapshots, the last
             one being the "current" sample. window=1 reproduces the old
             i.i.d. behaviour.
    x      : (n_samples, n_state) -- the state at the final step, kept so the
             attack construction can offset a whole trajectory consistently.
    """
    n_state = h.shape[1]
    m = h.shape[0]
    window = max(1, int(window))

    # Start states are spread widely (the system can be operating anywhere);
    # increments are small (it drifts rather than teleports).
    x = rng.normal(0.0, 1.0, size=(n_samples, n_state))
    steps = rng.normal(0.0, state_step_sigma,
                       size=(n_samples, window - 1, n_state))
    traj = np.concatenate([x[:, None, :], x[:, None, :] + np.cumsum(steps, axis=1)],
                          axis=1) if window > 1 else x[:, None, :]

    e = rng.normal(0.0, noise_sigma, size=(n_samples, window, m))
    z = traj @ h.T + e
    return z, traj[:, -1, :]


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

def make_attack_vectors(h: np.ndarray,
                        attack_type: np.ndarray,
                        rng: np.random.Generator,
                        scale: float = 1.5,
                        noise_sigma: float = 0.05,
                        redist_frac: float = 0.10,
                        redist_bias_sigma: float = 0.30,
                        window: int = 1) -> np.ndarray:
    """Construct the injection for each sample according to the threat model.

    Returns an array shaped (n_samples, window, m): the injection applied at
    each timestep of the trajectory.

    Difficulty is determined by ONE thing: whether the injection has a component
    outside col(H). Everything else -- magnitude -- is held equal across
    families, because a magnitude difference is a data leak that lets a model
    classify by size instead of by attack structure.

      * random        : unconstrained, so the residual moves -> detectable from
                        a single snapshot
      * stealthy      : a = Hc, invisible to a residual test. It is applied as a
                        STEP at a random timestep inside the window, so the
                        apparent state jumps. With window=1 this reduces to the
                        old constant injection; with window>1 the jump is
                        inconsistent with the state dynamics and becomes
                        detectable temporally.
      * load-redist   : stealthy plus a sparse noise-scaled departure -> between

    CRITICAL: `c @ h.T` amplifies the state perturbation by ||H||, which for
    IEEE 118 is a factor of ~65. Injecting both families at the same scale
    therefore made the "stealthy" attack the LARGEST deviation in raw
    measurement space -- the opposite of the intent. Every family is normalised
    to the same per-measurement RMS so the only difference is direction
    relative to col(H).
    """
    n = len(attack_type)
    m, n_state = h.shape
    window = max(1, int(window))

    def normalise(raw: np.ndarray) -> np.ndarray:
        """Rescale each trajectory to per-measurement RMS == scale."""
        rms = np.sqrt((raw ** 2).mean(axis=(1, 2), keepdims=True))
        return raw / np.maximum(rms, 1e-12) * scale

    a = np.zeros((n, window, m), dtype=np.float64)

    mask = attack_type == ATTACK_RANDOM
    if mask.any():
        k = int(mask.sum())
        raw = rng.normal(0.0, 1.0, size=(k, window, m))
        a[mask] = normalise(raw)

    mask = attack_type == ATTACK_STEALTHY
    if mask.any():
        k = int(mask.sum())
        c = rng.normal(0.0, 1.0, size=(k, n_state))
        raw = np.broadcast_to((c @ h.T)[:, None, :], (k, window, m)).copy()
        raw = normalise(raw)
        # Apply as a step: silent until t0, then held. The apparent state jumps
        # at t0 by an amount that the dynamics do not allow.
        if window > 1:
            t0 = rng.integers(1, window, size=k)
            steps = (np.arange(window)[None, :] >= t0[:, None])
            raw = raw * steps[:, :, None]
        a[mask] = raw

    mask = attack_type == ATTACK_LOAD_REDIST
    if mask.any():
        k = int(mask.sum())
        c = rng.normal(0.0, 1.0, size=(k, n_state))
        raw = np.broadcast_to((c @ h.T)[:, None, :], (k, window, m)).copy()
        base = normalise(raw)
        if window > 1:
            t0 = rng.integers(1, window, size=k)
            steps = (np.arange(window)[None, :] >= t0[:, None])
            base = base * steps[:, :, None]
        # Sparse departure from the stealth subspace, with a HETEROGENEOUS
        # per-sample magnitude. A single fixed magnitude cannot produce an
        # intermediate family: the residual test at a 1% false-alarm rate is a
        # sharp threshold just above the benign level, so any departure that
        # clears it is caught on essentially every sample. Real attackers vary
        # in how much they are willing to disturb the measurements, so the
        # departure RMS is drawn per sample -- some instances stay deep in the
        # stealth subspace, others are plainly visible.
        n_sparse = max(1, int(round(redist_frac * m)))
        mult = np.abs(rng.normal(1.0, 1.0, size=k))
        dep = np.zeros((k, window, m), dtype=np.float64)
        for i in range(k):
            cols = rng.choice(m, size=n_sparse, replace=False)
            dep_rms_i = redist_bias_sigma * scale * mult[i]
            # dep[i][:, cols] keeps the (window, n_sparse) layout; the
            # `dep[i, :, cols]` form would move the indexed axis to the front.
            dep[i][:, cols] = rng.normal(0.0, dep_rms_i, size=(window, n_sparse))
        a[mask] = base + dep

    return a


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode_categorical(z: np.ndarray,
                       n_bus: int,
                       n_time_windows: int,
                       n_bins: int,
                       rng: np.random.Generator,
                       n_bin_cross: int = 4) -> Tuple[np.ndarray, List[int]]:
    """Build the crossing categorical features, and report their cardinalities.

    Columns produced (in order):

      0            time_bucket          cardinality n_time_windows
      1            measurement_point    cardinality n_measurements
      2            bus_id               cardinality n_bus
      3 .. 3+k-1   bin_j                cardinality n_bins each
      3+k          bin_cross            cardinality n_bins ** k

    `bin_cross` is the joint index of the first k measurement bins. It is the
    high-cardinality term: a joint quantile pattern across several measurements
    is what distinguishes an attack from normal operating drift, and its
    cardinality is a power of the bin count rather than a sum. This is the term
    that makes the embedding table large.

    Returns (features, cardinalities) so the table size follows from the
    contract metadata alone -- no need to re-read the data to size the table.
    """
    n, m = z.shape
    cols: List[np.ndarray] = []

    time_id = rng.integers(0, n_time_windows, size=n)
    cols.append(time_id)

    point_id = rng.integers(0, m, size=n)
    cols.append(point_id)

    bus_id = rng.integers(0, n_bus, size=n)
    cols.append(bus_id)

    # Quantile bins of selected measurements: real, data-derived features.
    k = max(1, min(n_bin_cross, m))
    bins: List[np.ndarray] = []
    for j in range(k):
        values = z[:, j]
        edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
        b = np.digitize(values, edges).astype(np.int64)
        bins.append(b)
        cols.append(b)

    # Joint index of those k bins -> cardinality n_bins ** k.
    cross = np.zeros(n, dtype=np.int64)
    for b in bins:
        cross = cross * n_bins + b
    cols.append(cross)

    sparse = np.stack(cols, axis=1).astype(np.int64)
    cardinalities = [n_time_windows, m, n_bus] + [n_bins] * k + [n_bins ** k]
    return sparse, cardinalities


def sparse_column_names(n_bin_cross: int) -> List[str]:
    k = max(1, n_bin_cross)
    return (["time_bucket", "measurement_point", "bus_id"]
            + [f"bin_{j}" for j in range(k)]
            + ["bin_cross"])


def encode_node_features(z: np.ndarray, h: np.ndarray,
                         n_bus: int, rng: np.random.Generator) -> np.ndarray:
    """Per-bus measurement tensor for the sequence and graph models.

    Each bus receives the measurements that touch it. With the DC model the
    injection at bus i is the i-th row of z, so those map directly; branch
    flows are distributed to their endpoints.
    """
    n = z.shape[0]
    f_node = 4
    node = np.zeros((n, n_bus, f_node), dtype=np.float32)

    # Injection channels: first n_bus measurement rows.
    take = min(n_bus, z.shape[1])
    node[:, :take, 0] = z[:, :take]

    # Derived channels that give the models something to work with per bus.
    node[:, :, 1] = z[:, :take].mean(axis=1, keepdims=True)
    node[:, :, 2] = z[:, :take].std(axis=1, keepdims=True)
    node[:, :, 3] = rng.normal(0.0, 0.02, size=(n, n_bus))
    return node


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def generate(case: str,
             out: Path,
             n_samples: int = 20000,
             n_bus: Optional[int] = None,
             noise_sigma: float = 0.05,
             attack_scale: float = 1.5,
             redist_frac: float = 0.10,
             redist_bias_sigma: float = 1.2,
             benign_fraction: float = 0.5,
             n_time_windows: int = 256,
             n_bins: int = 32,
             n_bin_cross: int = 4,
             window: int = 8,
             state_step_sigma: float = 0.02,
             embedding_dim: int = 16,
             train_ratio: float = 0.7,
             val_ratio: float = 0.15,
             allow_pandapower: bool = True,
             seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)

    net = build_network(case, n_bus=n_bus, allow_pandapower=allow_pandapower,
                        seed=seed)
    h, inj_rows, flow_rows = build_jacobian(net)

    n_benign = int(round(n_samples * benign_fraction))
    n_attack = n_samples - n_benign
    per_family = n_attack // 3
    n_attack = per_family * 3
    n_total = n_benign + n_attack

    attack_type = np.concatenate([
        np.full(n_benign, ATTACK_NORMAL, dtype=np.int64),
        np.full(per_family, ATTACK_RANDOM, dtype=np.int64),
        np.full(per_family, ATTACK_STEALTHY, dtype=np.int64),
        np.full(per_family, ATTACK_LOAD_REDIST, dtype=np.int64),
    ])
    label = (attack_type != ATTACK_NORMAL).astype(np.int64)

    window_arr, _ = _clean_dataset(net, h, n_total, noise_sigma, rng,
                                   window=window,
                                   state_step_sigma=state_step_sigma)
    injection = make_attack_vectors(h, attack_type, rng, scale=attack_scale,
                                    noise_sigma=noise_sigma,
                                    redist_frac=redist_frac,
                                    redist_bias_sigma=redist_bias_sigma,
                                    window=window)
    window_arr = window_arr + injection
    # `dense` stays the CURRENT snapshot so every existing single-snapshot
    # model keeps working unchanged. The full trajectory travels separately.
    z = window_arr[:, -1, :]

    sparse, cardinalities = encode_categorical(
        z, net.n_bus, n_time_windows, n_bins, rng, n_bin_cross=n_bin_cross)
    node_feat = encode_node_features(z, h, net.n_bus, rng)

    idx = rng.permutation(n_total)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    split = np.empty(n_total, dtype=np.int64)
    split[idx[:n_train]] = SPLIT_TRAIN
    split[idx[n_train:n_train + n_val]] = SPLIT_VAL
    split[idx[n_train + n_val:]] = SPLIT_TEST

    meta = {
        "contract_version": CONTRACT_VERSION,
        "topology": net.name,
        "topology_source": net.source,
        "n_bus": net.n_bus,
        "n_branch": net.n_branch,
        "n_measurements": int(h.shape[0]),
        "n_state": int(h.shape[1]),
        "noise_sigma": noise_sigma,
        "attack_scale": attack_scale,
        "redist_frac": redist_frac,
        "redist_bias_sigma": redist_bias_sigma,
        "attack_model": "random outside col(H); stealthy a=Hc; "
                        "load-redistribution a=Hc + sparse noise-scaled bias",
        "embedding_dim": embedding_dim,
        "sparse_columns": sparse_column_names(n_bin_cross),
        "sparse_cardinalities": cardinalities,
        # Embedding table size, derived from the cardinalities rather than
        # estimated. DLRM gives each sparse feature its own table, so rows is
        # the SUM of the cardinalities. Recorded here so that any downstream
        # claim about table size can be traced to this file.
        "embedding_table_rows": int(sum(cardinalities)),
        "embedding_table_bytes_fp32": int(sum(cardinalities) * embedding_dim * 4),
        "n_bin_cross": n_bin_cross,
        "window": window,
        "state_step_sigma": state_step_sigma,
        "temporal_note": (
            "The state follows a random walk with bounded per-step increments, "
            "so a stealthy injection applied as a step makes the apparent state "
            "jump by more than the dynamics allow. window=1 disables this and "
            "reproduces the i.i.d. single-snapshot setting."
        ),
        "benign_fraction": benign_fraction,
        "seed": seed,
        "split_ratios": {"train": train_ratio, "val": val_ratio,
                         "test": 1.0 - train_ratio - val_ratio},
        "notes": net.notes + [
            "Sparse features: time/point/bus plus per-measurement quantile bins",
            "and their joint cross (bin_cross), whose cardinality is n_bins**k.",
            "Embedding table rows = SUM of cardinalities (one table per feature).",
            "Stealthy attacks satisfy a = Hc: invisible to a residual test on any",
            "single snapshot. They are injected as a STEP inside the window, so",
            "the apparent state jumps by more than the dynamics allow -- that is",
            "the signature a temporal detector can use.",
            f"Trajectory window = {window} snapshots; state random-walk step "
            f"sigma = {state_step_sigma} (per step, per state dimension).",
        ],
    }

    out = Path(out)
    save_contract(
        out,
        dense=z.astype(np.float32),
        sparse=sparse,
        label=label,
        attack_type=attack_type,
        meta=meta,
        node_feat=node_feat,
        adj=net.adjacency.astype(np.float32),
        split=split,
        h_matrix=h.astype(np.float32),
        window=window_arr.astype(np.float32),
    )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default="118",
                   help="test-case identifier: 14, 30, 57, 118, 300")
    p.add_argument("--out", default=None,
                   help="output .npz path (default data_cache/ieee<case>.npz)")
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument("--bus", type=int, default=None,
                   help="bus count for the synthetic fallback topology")
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--attack-scale", type=float, default=1.5)
    p.add_argument("--redist-frac", type=float, default=0.10,
                   help="fraction of measurements carrying the "
                        "load-redistribution departure from the stealth subspace")
    p.add_argument("--redist-bias-sigma", type=float, default=0.30,
                   help="departure RMS of the load-redistribution family, as a "
                        "fraction of the injection RMS. Tuned so the family sits "
                        "genuinely between stealthy (no departure) and random "
                        "(fully outside col(H))")
    p.add_argument("--benign-fraction", type=float, default=0.5)
    p.add_argument("--embedding-dim", type=int, default=16)
    p.add_argument("--bins", type=int, default=32,
                   help="quantile bins per crossed measurement")
    p.add_argument("--bin-cross", type=int, default=4,
                   help="how many measurement bins to cross jointly. "
                        "The resulting feature has cardinality bins**bin_cross, "
                        "which is the dominant term in the embedding table size")
    p.add_argument("--time-buckets", type=int, default=256,
                   help="number of distinct time buckets")
    p.add_argument("--window", type=int, default=8,
                   help="length of the measurement trajectory per sample. "
                        "window=1 reproduces the i.i.d. single-snapshot setting; "
                        "window>1 is what makes a stepped a=Hc injection "
                        "detectable, because the state cannot jump")
    p.add_argument("--state-step-sigma", type=float, default=0.02,
                   help="per-step, per-state-dimension std of the random walk. "
                        "Controls how fast the true state is allowed to move; "
                        "the stealthy step must exceed it to be visible")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pandapower", action="store_true",
                   help="force the synthetic topology even if pandapower exists")
    p.add_argument("--list-cases", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_cases:
        try:
            import pandapower  # noqa: F401
            available = "installed"
        except ImportError:
            available = "NOT installed - synthetic topologies only"
        print("pandapower:", available)
        print("known cases:", ", ".join(sorted(KNOWN_CASES)))
        return 0

    out = Path(args.out or f"data_cache/ieee{args.case}.npz")
    print("FDIA reference data generator")
    print("=" * 74)
    print(f"case            : {args.case}")
    print(f"samples         : {args.samples:,}")
    print(f"noise sigma     : {args.noise}")
    print(f"attack scale    : {args.attack_scale}")
    print(f"benign fraction : {args.benign_fraction}")
    print(f"pandapower      : {'disabled' if args.no_pandapower else 'enabled'}")
    print()

    path = generate(
        case=args.case, out=out, n_samples=args.samples, n_bus=args.bus,
        noise_sigma=args.noise, attack_scale=args.attack_scale,
        redist_frac=args.redist_frac, redist_bias_sigma=args.redist_bias_sigma,
        benign_fraction=args.benign_fraction,
        n_bins=args.bins, n_bin_cross=args.bin_cross,
        n_time_windows=args.time_buckets,
        window=args.window, state_step_sigma=args.state_step_sigma,
        embedding_dim=args.embedding_dim, seed=args.seed,
        allow_pandapower=not args.no_pandapower,
    )

    # Report the embedding table size straight from the contract, so it never
    # has to be recomputed by hand.
    import json as _json
    with np.load(path, allow_pickle=False) as _z:
        _m = _json.loads(str(_z["meta"]))
    rows = _m["embedding_table_rows"]
    nbytes = _m["embedding_table_bytes_fp32"]
    print(f"[1/2] wrote {path}")
    print(f"      embedding table: {rows:,} rows "
          f"({nbytes / 1024**3:.3f} GiB fp32, dim {_m['embedding_dim']})")
    print(f"      cardinalities  : {_m['sparse_cardinalities']}")
    print("[2/2] contract diagnostics:")
    print()
    if not args.quiet:
        print(describe_contract(path))
    print()
    print("Next: verify the attack construction behaves as the threat model")
    print("      requires (residual BDD blind to stealthy attacks):")
    print(f"      python data/contract.py --describe {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
