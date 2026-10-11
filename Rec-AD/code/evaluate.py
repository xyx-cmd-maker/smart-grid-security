"""Unified evaluation: detection metrics + per-attack-type breakdown.

Reviewer 3 (comment 1) asked for a defined detection objective and threat
scenarios; Reviewer 1 (comment 2) asked the paper to stop treating FDIA as a
generic binary classification problem. Both are answered by reporting
detection quality *per attack type* rather than as a single aggregate number.
The stealthy ``a = Hc`` column is the one that matters: it is the attack that
bypasses traditional bad data detection.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from data.contract import (
    ATTACK_LOAD_REDIST,
    ATTACK_NAMES,
    ATTACK_NORMAL,
    ATTACK_RANDOM,
    ATTACK_STEALTHY,
)

#: Metrics that go into every result row and every paper table.
CORE_METRICS = ("accuracy", "precision", "recall", "f1", "auc", "fpr")


def compute_metrics(y_true: np.ndarray,
                    y_score: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    """Detection metrics at a fixed operating threshold.

    Precision and FPR are included deliberately. The paper previously
    reported only accuracy/recall/F1, but false alarms drive operator alert
    fatigue in a control room and are a real deployment constraint.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / max(len(y_true), 1)
    f1 = 2 * precision * recall / (precision + recall + eps)
    fpr = fp / (fp + tn + eps)

    out = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_samples": int(len(y_true)),
        "threshold": float(threshold),
    }
    out["auc"] = float(_auc(y_true, y_score))
    out["ap"] = float(_average_precision(y_true, y_score))
    out["tpr_at_fpr_1pct"] = float(_tpr_at_fpr(y_true, y_score, 0.01))
    return out


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_score)
    except ImportError:  # pragma: no cover
        return _auc_rank(y_true, y_score)


def _auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney U formulation, used when scikit-learn is unavailable."""
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_score))
    except ImportError:  # pragma: no cover
        return float("nan")


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Detection rate achievable at a fixed false-positive rate.

    This is the metric a security reviewer expects: what fraction of attacks
    do you catch if you can only tolerate `target_fpr` false alarms?
    """
    neg = y_score[y_true == 0]
    pos = y_score[y_true == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - target_fpr)
    return float((pos >= thr).mean())


def select_threshold(y_true: np.ndarray,
                     y_score: np.ndarray,
                     strategy: str = "max_f1",
                     target_fpr: Optional[float] = None,
                     val_scores: Optional[np.ndarray] = None) -> float:
    """Pick an operating threshold on VALIDATION data.

    Choosing the threshold on validation and freezing it before touching the
    test split is what keeps the reported test numbers honest.
    """
    src_y, src_s = y_true, (val_scores if val_scores is not None else y_score)
    if strategy == "target_fpr" and target_fpr is not None:
        neg = src_s[src_y == 0]
        if len(neg) == 0:
            return 0.5
        return float(np.quantile(neg, 1.0 - target_fpr))

    if strategy == "max_f1":
        grid = np.unique(np.quantile(src_s, np.linspace(0.01, 0.99, 99)))
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = compute_metrics(src_y, src_s, float(t))["f1"]
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t

    return 0.5


def per_attack_type_metrics(y_true: np.ndarray,
                            y_score: np.ndarray,
                            attack_type: np.ndarray,
                            threshold: float,
                            prefix: str = "") -> Dict[str, float]:
    """Detection rate for each attack family, plus the benign false-alarm rate.

    This produces the table that carries the paper's central security claim:
    traditional BDD catches `random_fdia` and misses `stealthy_fdia_a_eq_Hc`,
    while the learned detectors recover the stealthy cases.
    """
    attack_type = np.asarray(attack_type).ravel()
    y_score = np.asarray(y_score).ravel()
    detected = (y_score >= threshold).astype(int)
    p = f"{prefix}_" if prefix else ""

    out: Dict[str, float] = {}

    normal = attack_type == ATTACK_NORMAL
    if normal.any():
        out[f"{p}detect_rate_normal_fp"] = float(detected[normal].mean())
        out[f"{p}n_normal"] = float(normal.sum())

    for name, code in (("random", ATTACK_RANDOM),
                       ("stealthy", ATTACK_STEALTHY),
                       ("load_redist", ATTACK_LOAD_REDIST)):
        mask = attack_type == code
        if mask.any():
            out[f"{p}detect_rate_{name}"] = float(detected[mask].mean())
            out[f"{p}n_{name}"] = float(mask.sum())
    return out


def evaluate_detector(model,
                      loaders: Dict[str, Any],
                      attack_type_test: np.ndarray,
                      threshold_strategy: str = "max_f1",
                      target_fpr: Optional[float] = None,
                      device=None) -> Dict[str, Any]:
    """Full evaluation: val thresholds, test metrics, per-attack breakdown."""
    from data.dataset import loader_arrays

    _, _, y_val, at_val = loader_arrays(loaders["val"])
    scores_val = model.predict_proba(loaders["val"], device=device)

    thr = select_threshold(y_val, scores_val, strategy=threshold_strategy,
                           target_fpr=target_fpr)

    _, _, y_test, at_test = loader_arrays(loaders["test"])
    scores_test = model.predict_proba(loaders["test"], device=device)

    row: Dict[str, Any] = {"threshold_strategy": threshold_strategy}
    row.update({f"val_{k}": v for k, v in
                compute_metrics(y_val, scores_val, thr).items()})
    row.update(compute_metrics(y_test, scores_test, thr))
    row.update(per_attack_type_metrics(y_test, scores_test, at_test, thr))

    # Same per-family rates, but at a COMMON operating point.
    #
    # The rates above use this run's own calibrated threshold, so each model sits at a
    # slightly different false-alarm rate (measured spread across runs: 0.0085 to
    # 0.0210). Comparing per-family detection across models then confounds the detector
    # with the threshold. Setting the threshold at the 99th percentile of benign test
    # scores gives every model a 1% false-alarm rate on the same data, which is the
    # only way the per-family numbers are comparable across rows.
    #
    # Reported under a distinct prefix so the two are never confused: `detect_rate_*`
    # stays as the calibrated-operating-point number, `detect_rate_*_at_fpr1pct` is the
    # common-operating-point one. AGENTS.md 4.8: name the frame of reference or the
    # comparison is invalid.
    benign_test = scores_test[at_test == ATTACK_NORMAL] if at_test is not None else None
    if benign_test is not None and benign_test.size > 0:
        thr_common = float(np.quantile(benign_test, 0.99))
        row["threshold_at_fpr1pct"] = thr_common
        row.update(per_attack_type_metrics(y_test, scores_test, at_test,
                                           thr_common, prefix="unified"))

    row.update(model.extra_metrics(y_test, scores_test, at_test))
    row["n_parameters"] = model.n_parameters()
    row["model_bytes"] = model.footprint_bytes()
    return row, scores_val, scores_test, y_val, y_test


def format_results_table(rows: list[Dict[str, Any]],
                         columns: Optional[list[str]] = None) -> str:
    """Render a result list as the fixed-width table used in the paper draft.

    Accepts either raw per-seed rows (`accuracy`) or aggregated rows
    (`accuracy_mean`), so the same renderer serves both.
    """
    if not rows:
        return "(no results)"
    columns = columns or ["model", "accuracy", "precision", "recall", "f1", "auc",
                          "fpr", "detect_rate_random", "detect_rate_stealthy",
                          "detect_rate_load_redist"]

    # Map each requested column to the key actually present in the rows.
    resolved: list[tuple[str, str]] = []
    for col in columns:
        if col in rows[0]:
            resolved.append((col, col))
        elif f"{col}_mean" in rows[0]:
            resolved.append((col, f"{col}_mean"))
    if not resolved:
        return "(no matching columns)"

    labels = [label for label, _ in resolved]
    # Column width must consider the data, not just the header, or long cell
    # names get silently truncated and the table becomes unreadable.
    widths = []
    for (_, key), label in zip(resolved, labels):
        widest = len(label)
        for row in rows:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            widest = max(widest, len(str(value)))
        widths.append(min(widest, 26))

    header = "  ".join(label[:w].ljust(w) for label, w in zip(labels, widths))
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = []
        for (_, key), w in zip(resolved, widths):
            v = row.get(key, "")
            if isinstance(v, float):
                v = f"{v:.4f}"
            cells.append(str(v)[:w].ljust(w))
        lines.append("  ".join(cells))
    return "\n".join(lines)
