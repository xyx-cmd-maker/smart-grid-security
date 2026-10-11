"""Single construction point for every baseline.

Comparisons are only meaningful if every method is built the same way from the
same config. Centralising construction here also makes it obvious which
method families participate, which is the list Reviewer 1 (comment 8) asked
for and did not find in Table III.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from recad.detector import DataSpec, DetectorBase

#: Baseline name -> (family, human-readable reference for the paper)
BASELINE_FAMILIES: Dict[str, tuple] = {
    "BDD":         ("physics",    "Chi-square / largest-normalised-residual bad data detection"),
    "TemporalResidual": ("physics_temporal",
                         "Innovation test on the measurement trajectory (Kalman-style)"),
    "SVM":         ("classical_ml", "RBF-kernel support vector machine"),
    "RF":          ("classical_ml", "Random forest"),
    "Autoencoder": ("deep",       "Reconstruction-error anomaly detector"),
    "LSTM":        ("deep",       "Recurrent sequence classifier over measurement trajectories"),
    "CNN":         ("deep",       "1-D convolutional sequence classifier"),
    "GCN":         ("gnn",        "Graph convolutional network"),
    "GAT":         ("gnn",        "Graph attention network"),
    "Transformer": ("deep",       "Transformer encoder over measurement trajectories"),
}

#: Order used in result tables. Physics first, then ML, then deep, then GNN.
DEFAULT_ORDER: List[str] = [
    "BDD", "TemporalResidual", "SVM", "RF", "Autoencoder",
    "LSTM", "CNN", "Transformer", "GCN", "GAT",
]


def needs_graph(name: str) -> bool:
    return name in ("GCN", "GAT")


def build_detector(name: str,
                   spec: DataSpec,
                   cfg: Dict[str, Any],
                   data: Optional[Dict[str, Any]] = None) -> DetectorBase:
    """Instantiate one detector by name.

    `data` is only needed by BDD, which uses the measurement Jacobian when the
    contract provides one. Passing it for every call keeps the signature
    uniform.
    """
    key = name.upper() if name.upper() in ("BDD", "SVM", "RF") else name

    if key == "BDD":
        h = None
        if data is not None and "h_matrix" in data:
            h = np.asarray(data["h_matrix"], dtype=np.float64)
        return _bdd(spec, cfg, h)
    if key in ("TemporalResidual", "TEMPORALRESIDUAL"):
        from baselines.temporal import TemporalInnovationDetector
        h = None
        if data is not None and "h_matrix" in data:
            h = np.asarray(data["h_matrix"], dtype=np.float64)
        return TemporalInnovationDetector(spec, cfg, name="TemporalResidual",
                                          h_matrix=h)
    if key == "SVM":
        from baselines.ml import SVMDetector
        return SVMDetector(spec, cfg, name="SVM")
    if key == "RF":
        from baselines.ml import RandomForestDetector
        return RandomForestDetector(spec, cfg, name="RF")
    if key == "Autoencoder":
        from baselines.autoencoder import AutoencoderDetector
        # Pass H so the residual input mode is available here too; without it the
        # Autoencoder would silently fall back to raw measurements while the rest of
        # the residual condition ran with residuals.
        h_flat = None
        if data is not None and "h_matrix" in data:
            h_flat = np.asarray(data["h_matrix"], dtype=np.float64)
        return AutoencoderDetector(spec, cfg, name="Autoencoder", h_matrix=h_flat)
    # Sequence models take the Jacobian so they can build the physics-informed
    # token (raw / residual / innovation). Without it they see raw values only,
    # on which they measure at chance against a stealthy injection.
    h_for_seq = None
    if data is not None and "h_matrix" in data:
        h_for_seq = np.asarray(data["h_matrix"], dtype=np.float64)

    if key == "LSTM":
        from baselines.lstm import LSTMDetector
        return LSTMDetector(spec, cfg, name="LSTM", h_matrix=h_for_seq)
    if key == "CNN":
        from baselines.cnn import CNNDetector
        return CNNDetector(spec, cfg, name="CNN", h_matrix=h_for_seq)
    if key == "Transformer":
        from baselines.transformer import TransformerDetector
        return TransformerDetector(spec, cfg, name="Transformer",
                                   h_matrix=h_for_seq)
    if key in ("GCN", "GAT"):
        from baselines.gnn import GNNDetector
        return GNNDetector(spec, cfg, name=name, variant=key.lower())

    raise KeyError(f"Unknown baseline {name!r}. Known: {sorted(BASELINE_FAMILIES)}")


def _bdd(spec: DataSpec, cfg: Dict[str, Any], h_matrix) -> DetectorBase:
    from baselines.bdd import BDDDetector
    return BDDDetector(spec, cfg, name="BDD", h_matrix=h_matrix)


def enabled_baselines(cfg: Dict[str, Any],
                      data: Optional[Dict[str, Any]] = None,
                      include: Optional[List[str]] = None) -> List[str]:
    """Return the enabled baseline names, in reporting order.

    Graph models are dropped automatically when the contract carries no
    `node_feat`, so a partially built dataset still produces a usable run
    instead of a crash deep inside a model.
    """
    bcfg = cfg.get("baselines", {}) or {}
    has_graph = data is not None and "node_feat" in data and "adj" in data
    has_window = data is not None and "window" in data
    # Sequence models accept either token order: the temporal window when the
    # contract has one, otherwise the per-bus node features.
    has_sequence = (data is None) or has_window or ("node_feat" in data)

    names: List[str] = []
    if include:
        names = list(include)
    else:
        if (bcfg.get("bdd", {}) or {}).get("enabled", True):
            names.append("BDD")
        # The temporal control only makes sense when there is a trajectory.
        if has_window and (bcfg.get("temporal", {}) or {}).get("enabled", True):
            names.append("TemporalResidual")
        if (bcfg.get("svm", {}) or {}).get("enabled", True):
            names.append("SVM")
        if (bcfg.get("rf", {}) or {}).get("enabled", True):
            names.append("RF")
        if (bcfg.get("autoencoder", {}) or {}).get("enabled", True):
            names.append("Autoencoder")
        if (bcfg.get("lstm", {}) or {}).get("enabled", True):
            names.append("LSTM")
        if (bcfg.get("cnn", {}) or {}).get("enabled", True):
            names.append("CNN")
        if (bcfg.get("transformer", {}) or {}).get("enabled", True):
            names.append("Transformer")
        gnn_cfg = bcfg.get("gnn", {}) or {}
        if gnn_cfg.get("enabled", True):
            for variant in gnn_cfg.get("variants", ["gcn", "gat"]):
                names.append(variant.upper())

    if data is not None:
        dropped = []
        if not has_graph:
            dropped += [n for n in names if needs_graph(n)]
        if not has_sequence:
            dropped += [n for n in names if n in ("LSTM", "CNN", "Transformer")]
        if not has_window:
            dropped += [n for n in names if n == "TemporalResidual"]
        if dropped:
            import logging
            logging.getLogger("recad").warning(
                "Contract lacks the inputs these baselines need; dropping: %s",
                sorted(set(dropped)),
            )
        names = [n for n in names if n not in dropped]

    unknown = [n for n in names if n not in BASELINE_FAMILIES]
    if unknown:
        raise KeyError(f"Unknown baselines requested: {unknown}")

    order = {name: i for i, name in enumerate(DEFAULT_ORDER)}
    return sorted(names, key=lambda n: order.get(n, 99))
