from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int


def binary_label(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        values = numeric.astype(int).to_numpy()
    else:
        mapped = series.astype(str).str.lower().map({"benign": 0, "phishing": 1})
        if mapped.isna().any():
            raise ValueError("labels must be 0/1 or benign/phishing")
        values = mapped.astype(int).to_numpy()
    if not set(np.unique(values)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    return values


def confusion_counts(truth: np.ndarray, pred: np.ndarray) -> Confusion:
    truth = np.asarray(truth, dtype=int)
    pred = np.asarray(pred, dtype=int)
    return Confusion(
        tp=int(((truth == 1) & (pred == 1)).sum()),
        fp=int(((truth == 0) & (pred == 1)).sum()),
        tn=int(((truth == 0) & (pred == 0)).sum()),
        fn=int(((truth == 1) & (pred == 0)).sum()),
    )


def binary_metrics(
    truth: np.ndarray,
    score: np.ndarray,
    threshold: float,
    sample_weight: np.ndarray | None = None,
) -> dict:
    truth = np.asarray(truth, dtype=int)
    score = np.asarray(score, dtype=float)
    if truth.shape != score.shape:
        raise ValueError("truth and score arrays must have the same shape")
    pred = (score >= float(threshold)).astype(int)
    weight = np.ones(len(truth), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weight.shape != truth.shape:
        raise ValueError("sample weights must align with truth")

    benign = truth == 0
    false_positive_weight = float(weight[benign & (pred == 1)].sum())
    benign_weight = float(weight[benign].sum())
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(truth, pred, sample_weight=weight, zero_division=0)),
        "recall": float(recall_score(truth, pred, sample_weight=weight, zero_division=0)),
        "f1": float(f1_score(truth, pred, sample_weight=weight, zero_division=0)),
        "fpr": false_positive_weight / benign_weight if benign_weight else float("nan"),
        "average_precision": float(average_precision_score(truth, score, sample_weight=weight)),
    }


def recall_at_fpr_budget(truth: np.ndarray, score: np.ndarray, fpr_budget: float) -> dict:
    truth = np.asarray(truth, dtype=int)
    score = np.asarray(score, dtype=float)
    benign_scores = pd.Series(score[truth == 0])
    if benign_scores.empty:
        raise ValueError("cannot compute R@FPR without benign examples")
    threshold = float(benign_scores.quantile(1.0 - float(fpr_budget), interpolation="higher"))
    pred = (score >= threshold).astype(int)
    counts = confusion_counts(truth, pred)
    positives = counts.tp + counts.fn
    benign = counts.fp + counts.tn
    return {
        "threshold": threshold,
        "recall": counts.tp / positives if positives else float("nan"),
        "fpr": counts.fp / benign if benign else float("nan"),
        "tp": counts.tp,
        "fp": counts.fp,
        "tn": counts.tn,
        "fn": counts.fn,
    }
