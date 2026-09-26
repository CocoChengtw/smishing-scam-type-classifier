"""Evaluation: per-class metrics, bootstrap CIs, slices, and review routing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def bootstrap_ci(
    y_true, y_pred, groups=None, n_boot: int = 1000, alpha: float = 0.05, seed: int = 13
) -> tuple[float, float, float]:
    """Macro-F1 point estimate with a percentile bootstrap CI.

    If `groups` is given, resample whole template clusters (cluster bootstrap),
    because messages within a template are not independent observations.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    point = macro_f1(y_true, y_pred)
    if groups is None:
        units = [np.array([i]) for i in range(len(y_true))]
    else:
        groups = np.asarray(groups)
        order = np.argsort(groups, kind="stable")
        _, starts = np.unique(groups[order], return_index=True)
        units = np.split(order, starts[1:])
    n = len(units)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        idx = np.concatenate([units[i] for i in pick])
        stats.append(macro_f1(y_true[idx], y_pred[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def per_class_report(y_true, y_pred, labels) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": s}, index=labels
    ).round(3)


def slice_report(df: pd.DataFrame, y_pred, slice_col: str, min_support: int = 100) -> pd.DataFrame:
    """Macro-F1 per slice (e.g. language), for slices with enough support."""
    out = []
    y_pred = np.asarray(y_pred)
    for value, part in df.groupby(slice_col):
        if len(part) < min_support:
            continue
        idx = df.index.get_indexer(part.index)
        out.append({slice_col: value, "n": len(part),
                    "macro_f1": round(macro_f1(part["label"], y_pred[idx]), 3)})
    return pd.DataFrame(out).sort_values("n", ascending=False).reset_index(drop=True)


def review_routing_curve(y_true, proba: np.ndarray, classes, thresholds=None) -> pd.DataFrame:
    """Auto-label only confident predictions and send the rest to human review.

    For each confidence threshold report the share handled automatically
    (coverage) and the accuracy / macro-F1 on that automated share.
    """
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    conf = proba.max(axis=1)
    pred = classes[proba.argmax(axis=1)]
    thresholds = thresholds if thresholds is not None else [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    rows = []
    for t in thresholds:
        keep = conf >= t
        if keep.sum() == 0:
            continue
        rows.append({
            "threshold": t,
            "coverage": round(float(keep.mean()), 3),
            "accuracy": round(float((pred[keep] == y_true[keep]).mean()), 3),
            "macro_f1": round(macro_f1(y_true[keep], pred[keep]), 3),
        })
    return pd.DataFrame(rows)
