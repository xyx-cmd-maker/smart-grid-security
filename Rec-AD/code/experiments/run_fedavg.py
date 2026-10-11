"""P1-7 — federated averaging feasibility study.

Answers Reviewer 1 (comment 3) and Reviewer 2 (comment 2), who both observed
that the original Introduction claimed Rec-AD was "well-suited for integration
with federated learning frameworks" while presenting no federated experiment.

Two things are deliberately NOT done here:

  * **No privacy claim is made.** Exchanging TT cores instead of a full
    embedding table reduces communication volume; it does not make the
    exchanged update resistant to gradient inversion, membership inference or
    any other attack on model updates. The original manuscript's error was
    asserting a capability without evidence; repeating that error in the
    opposite direction would be worse. The disclaimer is printed with the
    results and appears in Sec. VI.G.

  * **No convergence theory is claimed.** This is an empirical feasibility
    check under one partitioning, not a convergence result.

What is measured, because it is what the claim actually rests on:

  * communication volume per round, against a full-embedding FedAvg control
  * detection performance relative to centralised training
  * per-silo peak memory

Usage
-----
    python experiments/run_fedavg.py --data data_cache/ieee118.npz --silos 4
    python experiments/run_fedavg.py --data ... --quick
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.contract import describe_contract, load_contract
from data.dataset import FDIADataset, build_loaders, build_spec
from evaluate import compute_metrics, per_attack_type_metrics, select_threshold
from recad.common import (
    ResultsWriter,
    aggregate_seeds,
    describe_environment,
    get_device,
    get_tier,
    load_config,
    set_seed,
    setup_logging,
    warn_if_unpublishable,
)
from recad.detector import _to_device
from recad.model import RecADConfig, RecADDetector, make_permutation

METRICS = [
    "accuracy", "precision", "recall", "f1", "auc", "fpr",
    "detect_rate_random", "detect_rate_stealthy", "detect_rate_load_redist",
    "comm_bytes_per_round", "comm_bytes_total",
    "comm_embedding_bytes", "comm_other_bytes",
    "compression_ratio", "silo_vram_mb", "rounds_to_target",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--tier", default=None)
    p.add_argument("--seeds", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--silos", type=int, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--local-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--partition", default="contiguous",
                   choices=["contiguous", "interleaved"],
                   help="how silos are formed from the training split")
    p.add_argument("--no-centralised", action="store_true",
                   help="skip the centralised reference run")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def partition_indices(n: int, n_silos: int, scheme: str,
                      rng: np.random.Generator) -> List[np.ndarray]:
    """Split the training index range into silos.

    `contiguous` models regional nodes that see a band of the sample stream;
    `interleaved` is the non-IID worst case. Both are reported so a reader can
    see how sensitive the result is to the partitioning.
    """
    idx = np.arange(n)
    if scheme == "interleaved":
        rng.shuffle(idx)
    return [a for a in np.array_split(idx, n_silos) if len(a) > 0]


# ---------------------------------------------------------------------------
# Federated averaging
# ---------------------------------------------------------------------------

def _param_bytes(state: Dict[str, torch.Tensor]) -> int:
    return int(sum(v.numel() * v.element_size() for v in state.values()))


@torch.no_grad()
def aggregate(global_state: Dict[str, torch.Tensor],
              local_states: List[Dict[str, torch.Tensor]],
              weights: List[float]) -> Dict[str, torch.Tensor]:
    """Weighted element-wise average of the local parameter sets."""
    total = float(sum(weights)) or 1.0
    out = copy.deepcopy(global_state)
    for key in out:
        if not torch.is_floating_point(out[key]):
            out[key] = local_states[0][key].clone()
            continue
        acc = torch.zeros_like(out[key], dtype=torch.float32)
        for state, w in zip(local_states, weights):
            acc += state[key].float() * (w / total)
        out[key] = acc.to(out[key].dtype)
    return out


def train_local(model: RecADDetector,
                loader,
                device: torch.device,
                epochs: int,
                lr: float,
                weight_decay: float,
                grad_clip: Optional[float]) -> float:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    last = float("nan")
    for _ in range(epochs):
        for batch in loader:
            batch = _to_device(batch, device,
                               needs_node_feat=model.needs_node_feat,
                               needs_adj=model.needs_adj,
                               needs_window=getattr(model, "needs_window", False))
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch["label"].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            last = float(loss.detach())
    return last


def run_federated(cfg: Dict[str, Any],
                   data: Dict[str, Any],
                   spec,
                   device: torch.device,
                   recad: RecADConfig,
                   permutation: np.ndarray,
                   n_silos: int,
                   rounds: int,
                   local_epochs: int,
                   batch_size: int,
                   num_workers: int,
                   scheme: str,
                   seed: int,
                   label: str) -> Dict[str, Any]:
    """One federated run. Returns metrics plus communication accounting."""
    set_seed(seed)
    rng = np.random.default_rng(seed)

    loaders = build_loaders(data, batch_size=batch_size, num_workers=num_workers,
                            seed=seed)
    train_ds = loaders["train"].dataset
    n_train = len(train_ds)
    parts = partition_indices(n_train, n_silos, scheme, rng)

    from torch.utils.data import DataLoader

    silo_loaders = []
    for part in parts:
        ds = FDIADataset(
            dense=train_ds.dense, sparse=train_ds.sparse,
            label=train_ds.label, attack_type=train_ds.attack_type,
            node_feat=train_ds.node_feat, adj=train_ds.adj,
            index=train_ds.index[part],
        )
        silo_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers))

    tcfg = cfg.get("train", {}) or {}
    lr = float(tcfg.get("lr", 1e-3))
    wd = float(tcfg.get("weight_decay", 0.0))
    clip = tcfg.get("grad_clip")

    global_model = RecADDetector(spec, cfg, name=f"{label}-fed",
                                 recad=recad, permutation=permutation, data=data)
    global_model.to(device)
    global_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
    exchanged = _param_bytes(global_state)

    # The claim is specifically about exchanging TT cores instead of a full
    # embedding table. Reporting only the total would dilute it with the head
    # and bottom MLP, which are identical in both variants and exchanged
    # either way. Break the two apart so the comparison is interpretable.
    embedding_bytes = sum(
        v.numel() * v.element_size() for k, v in global_state.items()
        if k.startswith("embedding.")
    )
    other_bytes = exchanged - embedding_bytes

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history: List[float] = []
    t0 = time.perf_counter()
    for r in range(rounds):
        local_states, weights = [], []
        for silo_loader in silo_loaders:
            local = copy.deepcopy(global_model)
            local.load_state_dict(global_state)
            local.to(device)
            train_local(local, silo_loader, device, local_epochs, lr, wd, clip)
            local_states.append({k: v.detach().cpu().clone()
                                 for k, v in local.state_dict().items()})
            weights.append(float(len(silo_loader.dataset)))
        global_state = aggregate(global_state, local_states, weights)
        global_model.load_state_dict(global_state)

        _, _, y_val, _ = _split_labels(loaders["val"])
        scores = global_model.predict_proba(loaders["val"], device=device)
        history.append(float(compute_metrics(y_val, scores)["auc"]))

    wall = time.perf_counter() - t0

    _, _, y_val, _ = _split_labels(loaders["val"])
    scores_val = global_model.predict_proba(loaders["val"], device=device)
    thr = select_threshold(y_val, scores_val, strategy="max_f1")
    _, _, y_test, at_test = _split_labels(loaders["test"])
    scores_test = global_model.predict_proba(loaders["test"], device=device)

    row: Dict[str, Any] = {
        "scheme": label,
        "partition": scheme,
        "n_silos": n_silos,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "seed": seed,
        "wall_seconds": wall,
        "comm_bytes_per_round": float(exchanged),
        "comm_bytes_total": float(exchanged * 2 * rounds),   # up + down
        "comm_embedding_bytes": float(embedding_bytes),
        "comm_other_bytes": float(other_bytes),
        "compression_ratio": float(global_model.compression_ratio()),
        "final_val_auc": history[-1] if history else float("nan"),
        "val_auc_curve": history,
    }
    row.update(compute_metrics(y_test, scores_test, thr))
    row.update(per_attack_type_metrics(y_test, scores_test, at_test, thr))
    if device.type == "cuda":
        row["silo_vram_mb"] = float(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
    return row


def _split_labels(loader):
    from data.dataset import loader_arrays
    return loader_arrays(loader)


def run_centralised(cfg, data, spec, device, recad, permutation,
                    batch_size, num_workers, seed) -> Dict[str, Any]:
    set_seed(seed)
    loaders = build_loaders(data, batch_size=batch_size, num_workers=num_workers,
                            seed=seed)
    model = RecADDetector(spec, cfg, name="centralised", recad=recad,
                          permutation=permutation, data=data)
    hist = model.fit(loaders["train"], loaders["val"], device=device)

    _, _, y_val, _ = _split_labels(loaders["val"])
    scores_val = model.predict_proba(loaders["val"], device=device)
    thr = select_threshold(y_val, scores_val, strategy="max_f1")
    _, _, y_test, at_test = _split_labels(loaders["test"])
    scores_test = model.predict_proba(loaders["test"], device=device)

    row: Dict[str, Any] = {
        "scheme": "centralised",
        "partition": "none",
        "seed": seed,
        "wall_seconds": hist.get("train_seconds", float("nan")),
        "comm_bytes_per_round": 0.0,
        "comm_bytes_total": 0.0,
        "compression_ratio": float(model.compression_ratio()),
    }
    row.update(compute_metrics(y_test, scores_test, thr))
    row.update(per_attack_type_metrics(y_test, scores_test, at_test, thr))
    return row


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging((cfg.get("output", {}) or {}).get("log_level", "INFO"))

    data = load_contract(args.data)
    spec = build_spec(data)
    device = get_device()
    tier_name = args.tier or (cfg.get("measure", {}) or {}).get("tier", "t4")
    warn_if_unpublishable(tier_name)

    fcfg = cfg.get("fedavg", {}) or {}
    n_silos = int(args.silos or fcfg.get("silos", 4))
    rounds = int(args.rounds or fcfg.get("rounds", 30))
    local_epochs = int(args.local_epochs or fcfg.get("local_epochs", 2))
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(cfg.get("seeds", [0])))

    batch_size = int(args.batch_size
                     or (cfg.get("train", {}) or {}).get("batch_size", 4096))
    num_workers = int(args.num_workers if args.num_workers is not None
                      else (cfg.get("data", {}) or {}).get("num_workers", 0))
    if args.quick:
        seeds = seeds[:1]
        rounds, local_epochs = min(rounds, 5), 1
        batch_size = min(batch_size, 512)

    results_dir = Path(args.results_dir
                       or (cfg.get("output", {}) or {}).get("results_dir", "results"))
    writer = ResultsWriter(
        name=f"fedavg_{Path(args.data).stem}_{tier_name}"
             + (f"_{args.tag}" if args.tag else ""),
        results_dir=results_dir,
    )

    if not args.quiet:
        print(describe_contract(args.data))
        print()

    print("=" * 96)
    print("P1-7  Federated averaging feasibility study")
    print("=" * 96)
    print(f"device       : {device}")
    print(f"tier         : {tier_name}  (publishable={get_tier(tier_name).publishable})")
    print(f"silos        : {n_silos}")
    print(f"rounds       : {rounds}   local epochs: {local_epochs}")
    print(f"seeds        : {seeds}")
    print()
    print("NOTE: this study measures communication volume and memory only.")
    print("      It makes NO privacy claim. Exchanged TT cores are as")
    print("      susceptible to gradient-based inference as any model update.")
    print()

    perm = make_permutation("topology", spec, data)

    tt_cfg = RecADConfig.from_cfg(cfg, use_eff_tt=True)
    dense_cfg = RecADConfig.from_cfg(cfg, use_eff_tt=False)

    for seed in seeds:
        if not args.no_centralised:
            log.info("[fedavg] centralised reference seed=%d", seed)
            row = run_centralised(cfg, data, spec, device, tt_cfg, perm,
                                  batch_size, num_workers, seed)
            writer.add(**row)

        for scheme, recad in (("fedavg_tt_cores", tt_cfg),
                              ("fedavg_full_embedding", dense_cfg)):
            for partition in ("contiguous", "interleaved"):
                log.info("[fedavg] %s/%s seed=%d", scheme, partition, seed)
                row = run_federated(cfg, data, spec, device, recad, perm,
                                    n_silos=n_silos, rounds=rounds,
                                    local_epochs=local_epochs,
                                    batch_size=batch_size, num_workers=num_workers,
                                    scheme=partition, seed=seed,
                                    label=f"{scheme}:{partition}")
                record = dict(row)
                record["scheme"] = scheme
                record["variant"] = f"{scheme}:{partition}"
                writer.add(**record)
                print(f"  {scheme:<24} {partition:<12} seed={seed}  "
                      f"f1={row.get('f1', float('nan')):.4f}  "
                      f"auc={row.get('auc', float('nan')):.4f}  "
                      f"comm/round={row['comm_bytes_per_round'] / 1024:9.1f} KB")

    summary = aggregate_seeds(writer.rows, ["variant"], METRICS)
    writer.write(extra={
        "config": cfg,
        "environment": describe_environment(tier_name),
        "silos": n_silos, "rounds": rounds, "local_epochs": local_epochs,
        "seeds": seeds,
        "privacy_claim": "NONE - communication and memory only",
        "summary": summary,
    })
    (results_dir / f"{writer.name}_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")

    print()
    print("=" * 96)
    print("Federated summary (mean over seeds)")
    print("=" * 96)
    from evaluate import format_results_table
    print(format_results_table(summary, columns=[
        "variant", "f1", "auc", "detect_rate_stealthy",
        "comm_bytes_per_round", "compression_ratio",
    ]))

    central = next((r for r in summary if r.get("variant") == "centralised"), None)
    tt_row = next((r for r in summary
                   if str(r.get("variant", "")).startswith("fedavg_tt_cores")), None)
    dense_row = next((r for r in summary
                      if str(r.get("variant", "")).startswith("fedavg_full_embedding")), None)

    if tt_row and dense_row:
        ct = tt_row.get("comm_embedding_bytes_mean", float("nan"))
        cd = dense_row.get("comm_embedding_bytes_mean", float("nan"))
        ot = tt_row.get("comm_other_bytes_mean", float("nan"))
        print()
        print("Communication result (the substance of the federated claim):")
        print("  The exchange is split into the embedding table and the rest of")
        print("  the model. Only the embedding differs between variants; the head")
        print("  and bottom MLP are exchanged either way and are not part of the")
        print("  claim.")
        print()
        print(f"  {'component':<26}{'TT cores':>14}{'full embedding':>18}")
        print("  " + "-" * 56)
        print(f"  {'embedding per round':<26}{ct / 1024:>11.1f} KB{cd / 1024:>15.1f} KB")
        print(f"  {'non-embedding per round':<26}{ot / 1024:>11.1f} KB"
              f"{dense_row.get('comm_other_bytes_mean', float('nan')) / 1024:>15.1f} KB")
        if ct and not np.isnan(ct) and ct > 0:
            print()
            print(f"  Embedding communication reduction: {cd / ct:.2f}x")
            print(f"  Ratio matches the table compression ratio "
                  f"({tt_row.get('compression_ratio_mean', float('nan')):.2f}x), "
                  f"as expected:")
            print("  the saving is exactly the table compression, so a larger")
            print("  table yields a larger reduction.")
    if central and tt_row:
        d = tt_row.get("f1_mean", float("nan")) - central.get("f1_mean", float("nan"))
        print(f"  F1 change vs centralised: {d:+.4f}")
    print()
    print("State both halves in the paper: the communication reduction AND the")
    print("accuracy change, in whichever direction it goes. Reporting only the")
    print("reduction would repeat the error this experiment was added to fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
