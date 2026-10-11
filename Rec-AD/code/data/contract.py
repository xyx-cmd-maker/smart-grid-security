"""Data contract between the FDIA data pipeline and every baseline.

Why this module exists
----------------------
Reviewer 1 (comment 7) asked how MATPOWER topology and measurements became
PyTorch tensors, and questioned data authenticity. Reviewer 1 (comment 5)
asked what role each dataset plays. Both are answered by making the data
format explicit, versioned, and self-describing.

Reviewer 1 (comment 5) and the paper's own Table II also disagree: an
IEEE 118-bus system has at most 118 bus IDs, yet Table II reports an
embedding table with 19.53M rows. `describe_sparse_cardinality()` prints the
real breakdown so that number can be explained in the manuscript rather than
defended after the fact.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow `python data/contract.py` as well as `python -m data.contract`.
# Running a script directly puts its own directory on sys.path, not the
# project root, which would break `from data.dataset import ...`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CONTRACT_VERSION = "1.0"

ATTACK_NORMAL = 0
ATTACK_RANDOM = 1        # random FDIA, detectable by residual BDD
ATTACK_STEALTHY = 2      # a = Hc, bypasses residual BDD
ATTACK_LOAD_REDIST = 3   # load-redistribution / targeted FDIA

ATTACK_NAMES = {
    ATTACK_NORMAL: "normal",
    ATTACK_RANDOM: "random_fdia",
    ATTACK_STEALTHY: "stealthy_fdia_a_eq_Hc",
    ATTACK_LOAD_REDIST: "load_redistribution",
}

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2

#: key -> (ndim, dtype, required)
SCHEMA: Dict[str, Tuple[int, str, bool]] = {
    "dense":       (2, "float32", True),
    "sparse":      (2, "int64",   True),
    "node_feat":   (3, "float32", False),
    "adj":         (2, "float32", False),
    # Measurement Jacobian of the linearised power-flow model, if available.
    # Shape (n_measurements, n_states). Enables the model-based chi-square BDD
    # variant; without it the data-driven subspace residual is used instead.
    "h_matrix":    (2, "float32", False),
    "label":       (1, "int64",   True),
    "attack_type": (1, "int64",   True),
    "split":       (1, "int64",   False),
    "meta":        (0, "str",     True),
}


class ContractError(ValueError):
    """Raised when a data file violates the contract."""


# ---------------------------------------------------------------------------
# Save / load / validate
# ---------------------------------------------------------------------------

def save_contract(path: str | Path,
                  dense: np.ndarray,
                  sparse: np.ndarray,
                  label: np.ndarray,
                  attack_type: np.ndarray,
                  meta: Dict,
                  node_feat: Optional[np.ndarray] = None,
                  adj: Optional[np.ndarray] = None,
                  split: Optional[np.ndarray] = None,
                  h_matrix: Optional[np.ndarray] = None,
                  window: Optional[np.ndarray] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "dense": np.asarray(dense, dtype=np.float32),
        "sparse": np.asarray(sparse, dtype=np.int64),
        "label": np.asarray(label, dtype=np.int64).reshape(-1),
        "attack_type": np.asarray(attack_type, dtype=np.int64).reshape(-1),
        "meta": json.dumps(meta, ensure_ascii=False),
    }
    if node_feat is not None:
        payload["node_feat"] = np.asarray(node_feat, dtype=np.float32)
    if adj is not None:
        payload["adj"] = np.asarray(adj, dtype=np.float32)
    if h_matrix is not None:
        payload["h_matrix"] = np.asarray(h_matrix, dtype=np.float32)
    if split is not None:
        payload["split"] = np.asarray(split, dtype=np.int64).reshape(-1)
    if window is not None:
        # (n_samples, window, m): the measurement trajectory. `dense` is its
        # final snapshot, so single-snapshot models keep working unchanged.
        payload["window"] = np.asarray(window, dtype=np.float32)

    np.savez_compressed(path, **payload)
    return path


def load_contract(path: str | Path, validate: bool = True) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data contract not found: {path}\n"
            f"Produce it with your pipeline, or run "
            f"`python data/generate_fdia.py --case 118 --out {path}`."
        )
    with np.load(path, allow_pickle=False) as npz:
        data = {k: npz[k] for k in npz.files}

    if "meta" in data:
        data["meta"] = str(data["meta"])
        try:
            data["meta_parsed"] = json.loads(data["meta"])
        except json.JSONDecodeError:
            data["meta_parsed"] = {}
    if validate:
        validate_contract(data)
    return data


def validate_contract(data: Dict) -> None:
    problems: List[str] = []

    for key, (ndim, dtype, required) in SCHEMA.items():
        if key not in data:
            if required:
                problems.append(f"missing required key {key!r}")
            continue
        arr = data[key]
        if key == "meta":
            continue
        if arr.ndim != ndim:
            problems.append(f"{key!r}: expected {ndim} dims, got {arr.ndim}")
        if str(arr.dtype) != dtype:
            problems.append(f"{key!r}: expected dtype {dtype}, got {arr.dtype}")

    if problems:
        raise ContractError("Contract violations:\n  - " + "\n  - ".join(problems))

    n = len(data["label"])
    for key in ("dense", "sparse", "attack_type", "split", "node_feat"):
        if key in data and len(data[key]) != n:
            raise ContractError(
                f"{key!r} has {len(data[key])} rows but label has {n}"
            )

    labels = np.unique(data["label"])
    if not set(labels.tolist()).issubset({0, 1}):
        raise ContractError(f"label must be binary 0/1, found {labels.tolist()}")

    unknown = set(np.unique(data["attack_type"]).tolist()) - set(ATTACK_NAMES)
    if unknown:
        raise ContractError(f"unknown attack_type values: {sorted(unknown)}")

    if data["sparse"].size and int(data["sparse"].min()) < 0:
        raise ContractError("sparse contains negative ids")

    if data["label"].sum() == 0:
        raise ContractError("no positive (attacked) samples; nothing to detect")


# ---------------------------------------------------------------------------
# Diagnostics — these feed directly into the manuscript
# ---------------------------------------------------------------------------

@dataclass
class CardinalityReport:
    """Explains the embedding-table row count reported in the paper.

    Table II says IEEE 118-Bus has a 19.53M-row embedding table. With 118
    buses that is impossible if the sparse ids are bus ids. This report shows
    what the ids actually are, so Sec. II.A can describe them correctly and
    the R1.5/R1.7 responses can cite a real breakdown.
    """

    n_samples: int
    n_sparse_columns: int
    per_column_cardinality: List[int]
    total_distinct_ids: int
    table_rows_if_joint: int
    rows_per_sample_mean: float

    def render(self) -> str:
        lines = [
            "Sparse feature cardinality report",
            "=" * 46,
            f"samples                     : {self.n_samples:,}",
            f"sparse columns              : {self.n_sparse_columns}",
            f"distinct ids (union)        : {self.total_distinct_ids:,}",
            f"embedding rows (table size) : {self.table_rows_if_joint:,}",
            f"mean ids per sample         : {self.rows_per_sample_mean:.2f}",
            "",
            "per-column cardinality:",
        ]
        for i, card in enumerate(self.per_column_cardinality):
            lines.append(f"  column {i:>2} : {card:>12,} distinct ids")
        lines += [
            "",
            "Interpretation note for the manuscript:",
            "  If the sparse ids were bus ids, the row count would be bounded by",
            "  the number of buses. A much larger count means the ids are",
            "  cross/binned features (e.g. time-window x measurement-point x bin),",
            "  which must be stated explicitly in Sec. II.A.",
        ]
        return "\n".join(lines)


def describe_sparse_cardinality(data: Dict) -> CardinalityReport:
    sparse = data["sparse"]
    n_rows = int(sparse.shape[0])
    n_cols = int(sparse.shape[1])

    # A single shared embedding table indexes on the raw id, so table size is
    # the union of ids across columns; per-column cardinalities are reported
    # separately because that is what a reader needs to reproduce the setup.
    per_column = [int(len(np.unique(sparse[:, c]))) for c in range(n_cols)]
    union = int(len(np.unique(sparse)))

    return CardinalityReport(
        n_samples=n_rows,
        n_sparse_columns=n_cols,
        per_column_cardinality=per_column,
        total_distinct_ids=union,
        table_rows_if_joint=union,
        rows_per_sample_mean=float(n_cols),
    )


def describe_attacks(data: Dict) -> str:
    """Attack-type composition — required reporting for R3.1/R1.7."""
    at = data["attack_type"]
    label = data["label"]
    total = len(at)
    lines = [
        "Attack composition",
        "=" * 46,
        f"total samples               : {total:,}",
        f"positive (attacked)         : {int(label.sum()):,} "
        f"({100.0 * label.mean():.2f}%)",
        "",
    ]
    for code, name in ATTACK_NAMES.items():
        mask = at == code
        count = int(mask.sum())
        pct = 100.0 * count / total if total else 0.0
        lines.append(f"  {name:<28} {count:>10,}  ({pct:5.2f}%)")

    lines.append("")
    lines.append("Consistency check (label must equal 'attack_type != 0'):")
    expected = (at != ATTACK_NORMAL).astype(np.int64)
    n_mismatch = int((expected != label).sum())
    lines.append(
        f"  mismatches: {n_mismatch}"
        + ("  OK" if n_mismatch == 0 else "  <-- FIX BEFORE SUBMITTING")
    )
    return "\n".join(lines)


def describe_splits(data: Dict) -> str:
    if "split" not in data:
        return "No `split` array present; loaders will split by ratio."
    split = data["split"]
    lines = ["Split composition", "=" * 46]
    for code, name in ((SPLIT_TRAIN, "train"), (SPLIT_VAL, "val"), (SPLIT_TEST, "test")):
        mask = split == code
        n = int(mask.sum())
        pos = int(data["label"][mask].sum())
        rate = 100.0 * pos / n if n else 0.0
        lines.append(f"  {name:<6} n={n:>10,}  positives={pos:>9,}  ({rate:5.2f}%)")
    lines.append("")
    lines.append("Attack-type coverage per split (report this in the paper):")
    header = "    " + "".join(f"{n[:12]:>14}" for n in ATTACK_NAMES.values())
    lines.append(header)
    for code, name in ((SPLIT_TRAIN, "train"), (SPLIT_VAL, "val"), (SPLIT_TEST, "test")):
        mask = split == code
        row = f"  {name:<6}"
        for at in ATTACK_NAMES:
            row += f"{int(((data['attack_type'] == at) & mask).sum()):>14,}"
        lines.append(row)
    return "\n".join(lines)


def describe_contract(path: str | Path) -> str:
    data = load_contract(path)
    meta = data.get("meta_parsed", {})
    blocks = [
        f"Contract file : {path}",
        f"Version       : {meta.get('contract_version', CONTRACT_VERSION)}",
        "",
    ]
    if meta:
        blocks.append("Metadata")
        blocks.append("=" * 46)
        for key in sorted(meta):
            blocks.append(f"  {key:<28} {meta[key]}")
        blocks.append("")
    blocks.append(describe_sparse_cardinality(data).render())
    blocks.append("")
    blocks.append(describe_attacks(data))
    blocks.append("")
    blocks.append(describe_splits(data))

    if "node_feat" in data:
        nf = data["node_feat"]
        blocks.append("")
        blocks.append(
            f"Graph tensors : node_feat {tuple(nf.shape)}, "
            f"adj {tuple(data['adj'].shape) if 'adj' in data else 'absent'}"
        )
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Self-test — validates the toolchain without any real data
# ---------------------------------------------------------------------------

def make_synthetic_contract(n_samples: int = 4000,
                            n_bus: int = 14,
                            dense_dim: int = 6,
                            sparse_cols: int = 7,
                            n_state: int = 2,
                            noise_sigma: float = 0.05,
                            attack_scale: float = 1.5,
                            seed: int = 0) -> Dict:
    """Synthetic FDIA dataset built with a real ``a = Hc`` structure.

    Not a scientific dataset, but not arbitrary either: measurements are
    generated as ``z = H x + e`` with a low-dimensional state, so the
    legitimate measurement subspace is genuinely narrow. Attacks are then
    constructed the way the literature defines them:

      * random FDIA  -- ``a`` drawn outside ``col(H)``; residual BDD catches it
      * stealthy FDIA -- ``a = Hc`` inside ``col(H)``; residual BDD cannot
      * load redistribution -- ``a = Hc`` plus a small operating-point shift

    This makes the self-test a genuine check of the paper's central premise
    rather than a "does it run" smoke test: a correct BDD implementation must
    score near-perfectly on random attacks and near-zero on stealthy ones.
    """
    rng = np.random.default_rng(seed)

    n_normal = n_samples // 2
    n_each_attack = (n_samples - n_normal) // 3
    labels = np.concatenate([
        np.zeros(n_normal, dtype=np.int64),
        np.ones(n_each_attack * 3, dtype=np.int64),
    ])
    attack_type = np.concatenate([
        np.full(n_normal, ATTACK_NORMAL, dtype=np.int64),
        np.full(n_each_attack, ATTACK_RANDOM, dtype=np.int64),
        np.full(n_each_attack, ATTACK_STEALTHY, dtype=np.int64),
        np.full(n_each_attack, ATTACK_LOAD_REDIST, dtype=np.int64),
    ])
    n = len(labels)

    # Measurement Jacobian: n_meas measurements, n_state states.
    h_matrix = rng.normal(0.0, 1.0, size=(dense_dim, n_state)).astype(np.float64)
    # Orthonormalise so the column space is well conditioned.
    h_matrix, _ = np.linalg.qr(h_matrix)

    x_true = rng.normal(0.0, 1.0, size=(n, n_state))
    noise = rng.normal(0.0, noise_sigma, size=(n, dense_dim))
    dense = x_true @ h_matrix.T + noise

    # ---- attack injection ------------------------------------------------
    attack_vec = np.zeros_like(dense)

    random_mask = attack_type == ATTACK_RANDOM
    attack_vec[random_mask] = rng.normal(
        0.0, attack_scale, size=(int(random_mask.sum()), dense_dim))

    stealth_mask = attack_type == ATTACK_STEALTHY
    c_stealth = rng.normal(0.0, attack_scale, size=(int(stealth_mask.sum()), n_state))
    # a = Hc lies inside col(H): the residual is unchanged, so BDD is blind.
    attack_vec[stealth_mask] = c_stealth @ h_matrix.T

    load_mask = attack_type == ATTACK_LOAD_REDIST
    c_load = rng.normal(0.0, attack_scale, size=(int(load_mask.sum()), n_state))
    # Stealthy component plus a small bias from the shifted operating point.
    bias = 0.15 * rng.normal(0.0, 1.0, size=(int(load_mask.sum()), dense_dim))
    attack_vec[load_mask] = c_load @ h_matrix.T + bias

    dense = dense + attack_vec

    # ---- per-bus view for the sequence and graph models ------------------
    # The injection must be a *systematic* offset on the targeted buses, not
    # zero-mean noise: sequence and graph models pool across buses, and a
    # zero-mean perturbation averages away under that pooling, leaving no
    # learnable signal at all.
    node_feat_dim = 3
    node_feat = np.zeros((n, n_bus, node_feat_dim), dtype=np.float32)
    node_feat[:, :, 0] = x_true[:, 0:1]
    node_feat[:, :, 1] = x_true[:, 1:2]
    node_feat[:, :, 2] = rng.normal(0.0, 1.0, size=(n, n_bus))
    node_feat += rng.normal(0.0, 0.02, size=node_feat.shape)

    # Injection magnitude per attack family, giving the difficulty gradient
    # the threat model describes: random is loud, load redistribution is the
    # subtlest, stealthy sits between them but stays inside col(H) so that it
    # defeats the residual test.
    inj_scale = {
        ATTACK_RANDOM: attack_scale,
        ATTACK_STEALTHY: attack_scale * 0.6,
        ATTACK_LOAD_REDIST: attack_scale * 0.35,
    }
    for code, scale in inj_scale.items():
        rows = np.where(attack_type == code)[0]
        if len(rows) == 0:
            continue
        per_sample = rng.normal(0.0, scale, size=(len(rows), node_feat_dim))
        for k, i in enumerate(rows):
            buses = rng.choice(n_bus, size=max(1, n_bus // 4), replace=False)
            # Same offset on every targeted bus: a coherent false reading.
            node_feat[i, buses, :] += per_sample[k].astype(np.float32)

    # ---- cross/binned categorical features -------------------------------
    # These are what Table II's 19.53M-row embedding table really indexes:
    # a cross of time window x measurement point x bin, NOT bus ids.
    time_ids = rng.integers(0, 256, size=(n, 1))
    point_ids = rng.integers(0, dense_dim * 3, size=(n, 1))
    bin_ids = rng.integers(0, 32, size=(n, sparse_cols - 2))
    sparse = np.concatenate([time_ids, point_ids, bin_ids], axis=1).astype(np.int64)

    # ---- ring topology ---------------------------------------------------
    adj = np.zeros((n_bus, n_bus), dtype=np.float32)
    for i in range(n_bus - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    adj[0, n_bus - 1] = adj[n_bus - 1, 0] = 1.0

    # ---- splits ----------------------------------------------------------
    idx = rng.permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    split = np.empty(n, dtype=np.int64)
    split[idx[:n_train]] = SPLIT_TRAIN
    split[idx[n_train:n_train + n_val]] = SPLIT_VAL
    split[idx[n_train + n_val:]] = SPLIT_TEST

    meta = {
        "contract_version": CONTRACT_VERSION,
        "topology": f"synthetic_{n_bus}bus",
        "source": "make_synthetic_contract (SELFTEST ONLY - not for the paper)",
        "seed": seed,
        "n_bus": n_bus,
        "n_state": n_state,
        "noise_sigma": noise_sigma,
        "attack_scale": attack_scale,
        "attack_model": "random outside col(H); stealthy a=Hc; load-redistribution a=Hc+bias",
    }

    return {
        "dense": dense.astype(np.float32), "sparse": sparse, "label": labels,
        "attack_type": attack_type, "split": split,
        "node_feat": node_feat, "adj": adj, "h_matrix": h_matrix.astype(np.float32),
        "meta": json.dumps(meta), "meta_parsed": meta,
    }


def _selftest(tmp_dir: Path = Path("data_cache")) -> int:
    print("Rec-AD data contract self-test")
    print("=" * 60)

    data = make_synthetic_contract()
    out = save_contract(
        tmp_dir / "_selftest.npz",
        dense=data["dense"], sparse=data["sparse"], label=data["label"],
        attack_type=data["attack_type"], node_feat=data["node_feat"],
        adj=data["adj"], split=data["split"], meta=data["meta_parsed"],
        h_matrix=data["h_matrix"],
    )
    print(f"[1/5] wrote synthetic contract -> {out}")

    loaded = load_contract(out)
    print("[2/5] validate_contract passed")

    n_attacks = len(np.unique(loaded["attack_type"]))
    assert n_attacks == 4, f"expected 4 attack classes, found {n_attacks}"
    print(f"[3/5] attack classes present: {n_attacks}")

    from data.dataset import build_loaders, build_spec, loader_arrays
    # standardize=False is deliberate: the chi-square test relies on the
    # geometry of a = Hc, and per-dimension rescaling maps col(H) to col(DH),
    # which would make the stealthy attacks spuriously detectable. The
    # synthetic measurements are already generated in the space where H
    # applies. See baselines/bdd.py for the full explanation.
    loaders = build_loaders(loaded, batch_size=256, num_workers=0,
                            standardize=False)
    batch = next(iter(loaders["train"]))
    expected = {"dense", "sparse", "node_feat", "adj", "label", "attack_type"}
    missing = expected - set(batch)
    assert not missing, f"batch missing keys: {missing}"
    print(f"[4/5] loader OK: batch dense={tuple(batch['dense'].shape)} "
          f"sparse={tuple(batch['sparse'].shape)} "
          f"node_feat={tuple(batch['node_feat'].shape)} "
          f"adj={tuple(batch['adj'].shape)}")

    # ---- premise check ---------------------------------------------------
    # A correct residual BDD must catch random attacks and miss a = Hc
    # stealthy attacks. If this fails, either the detector or the attack
    # construction is wrong, and every downstream claim is unsafe.
    from baselines.bdd import BDDDetector
    from evaluate import per_attack_type_metrics

    spec = build_spec(loaded)
    bdd = BDDDetector(spec, {"baselines": {"bdd": {"alpha": 0.01}}},
                      h_matrix=loaded["h_matrix"])
    info = bdd.fit(loaders["train"], loaders["val"])
    scores = bdd.predict_proba(loaders["test"])
    _, _, y_test, at_test = loader_arrays(loaders["test"])
    stats = per_attack_type_metrics(y_test, scores, at_test, threshold=1.0 - 0.01)

    rnd = stats["detect_rate_random"]
    stl = stats["detect_rate_stealthy"]
    fp = stats["detect_rate_normal_fp"]
    print(f"[5/5] a=Hc premise check (BDD, subspace={info['subspace_dim']}, "
          f"dof={info['dof']}):")
    print(f"      random attacks detected     : {rnd:.3f}  (expect high)")
    print(f"      stealthy attacks detected   : {stl:.3f}  (expect low)")
    print(f"      benign false-positive rate  : {fp:.3f}  (expect ~0.01)")

    problems = []
    if rnd < 0.8:
        problems.append(f"BDD missed random attacks (detect rate {rnd:.2f})")
    if stl > 0.3:
        problems.append(f"BDD caught stealthy a=Hc attacks (detect rate {stl:.2f}) "
                        f"- the attack construction is not stealthy")
    if fp > 0.10:
        problems.append(f"BDD false-positive rate too high ({fp:.2f}) "
                        f"- threshold calibration is off")

    print()
    print(describe_contract(out))
    print()
    if problems:
        print("Self-test FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Self-test PASSED. Toolchain wired correctly, and the a=Hc premise")
    print("holds: residual BDD is blind to stealthy attacks by construction.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true",
                        help="build a synthetic contract and verify the toolchain")
    parser.add_argument("--describe", metavar="NPZ",
                        help="print the full diagnostic report for a contract file")
    parser.add_argument("--tmp-dir", default="data_cache",
                        help="where the self-test writes its scratch file")
    args = parser.parse_args()

    if args.describe:
        print(describe_contract(args.describe))
        return 0
    if args.selftest:
        return _selftest(Path(args.tmp_dir))

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
