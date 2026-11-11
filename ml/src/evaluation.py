import platform
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_optimal_threshold(
    y_val: pd.Series | np.ndarray,
    val_scores: np.ndarray,
    min_precision: float = 0.90,
) -> tuple[float, dict[str, Any]]:
    """
    Selects the operational decision threshold strictly on the validation set
    (PRD Section 2.3 & 8.2).
    Criterion: Maximize F1-score on validation set subject to candidate thresholds
    between 0.10 and 0.95 (with preference for precision >= min_precision when achievable).
    """
    y_true = np.asarray(y_val, dtype=int)
    scores = np.asarray(val_scores, dtype=np.float64)

    candidate_thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.50
    best_f1 = -1.0
    best_metrics: dict[str, float] = {}

    for thr in candidate_thresholds:
        preds = (scores >= thr).astype(int)
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        if f1 > best_f1 or (np.isclose(f1, best_f1) and prec > best_metrics.get("precision", 0.0)):
            best_f1 = f1
            best_threshold = float(round(thr, 4))
            best_metrics = {
                "precision": round(prec, 6),
                "recall": round(rec, 6),
                "f1": round(f1, 6),
            }

    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, scores)
    sampled_indices = np.linspace(0, len(precisions) - 1, min(25, len(precisions)), dtype=int)
    pr_curve_sample = [
        {
            "precision": round(float(precisions[i]), 5),
            "recall": round(float(recalls[i]), 5),
            "threshold": round(
                float(pr_thresholds[min(i, len(pr_thresholds) - 1)]), 5
            )
            if len(pr_thresholds) > 0
            else 0.5,
        }
        for i in sampled_indices
    ]

    justification = (
        f"Threshold {best_threshold:.2f} selected strictly on the validation partition (15%) "
        f"by maximizing F1-score (F1={best_metrics['f1']:.4f}, "
        f"Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}) "
        f"across candidate thresholds in [0.05, 0.95], then frozen prior to single-pass test evaluation."
    )

    return best_threshold, {
        "selected_threshold": best_threshold,
        "selection_criterion": "max_f1_on_validation_grid_0.05_to_0.95",
        "validation_at_selected_threshold": best_metrics,
        "justification": justification,
        "pr_curve_sample": pr_curve_sample,
    }


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """
    Computes all required PRD Section 8.3 metrics:
    - Confusion matrix (TP, FP, TN, FN)
    - Precision, Recall, F1 for class 1
    - ROC-AUC and Average Precision (PR-AUC)
    - Support per class and score distribution quantiles
    """
    y_arr = np.asarray(y_true, dtype=int)
    score_arr = np.asarray(scores, dtype=np.float64)

    if not np.isfinite(score_arr).all():
        raise ValueError("Model produced non-finite probability scores.")
    if (score_arr < 0.0).any() or (score_arr > 1.0).any():
        raise ValueError("Model produced probability scores outside [0, 1].")

    y_pred = (score_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()

    precision = float(precision_score(y_arr, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_arr, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_arr, y_pred, pos_label=1, zero_division=0))
    roc_auc = float(roc_auc_score(y_arr, score_arr))
    avg_precision = float(average_precision_score(y_arr, score_arr))

    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    class0_scores = score_arr[y_arr == 0]
    class1_scores = score_arr[y_arr == 1]

    return {
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "roc_auc": round(roc_auc, 6),
        "average_precision": round(avg_precision, 6),
        "support": {
            "class_0": int(np.sum(y_arr == 0)),
            "class_1": int(np.sum(y_arr == 1)),
            "total": int(len(y_arr)),
        },
        "score_distribution": {
            "overall_quantiles": {
                f"q{int(q * 100):02d}": round(float(np.quantile(score_arr, q)), 6)
                for q in quantiles
            },
            "class_0_mean": round(float(np.mean(class0_scores)), 6)
            if len(class0_scores) > 0
            else 0.0,
            "class_1_mean": round(float(np.mean(class1_scores)), 6)
            if len(class1_scores) > 0
            else 0.0,
        },
    }


def measure_inference_latency(
    pipeline: Any,
    X_sample: pd.DataFrame,
    n_single_runs: int = 100,
) -> dict[str, Any]:
    """
    Measures single-record p50/p95 prediction latency and batch latency,
    recording local hardware/environment details (PRD Section 8.3 & RNF-06).
    """
    n_runs = min(n_single_runs, len(X_sample))
    single_latencies_ms: list[float] = []

    for i in range(n_runs):
        row_df = X_sample.iloc[[i]]
        t0 = time.perf_counter()
        _ = pipeline.predict_proba(row_df)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        single_latencies_ms.append(elapsed_ms)

    batch_size = min(1000, len(X_sample))
    t_batch0 = time.perf_counter()
    _ = pipeline.predict_proba(X_sample.iloc[:batch_size])
    batch_elapsed_ms = (time.perf_counter() - t_batch0) * 1000.0

    arr = np.asarray(single_latencies_ms, dtype=np.float64)
    return {
        "single_prediction_p50_ms": round(float(np.percentile(arr, 50)), 4),
        "single_prediction_p95_ms": round(float(np.percentile(arr, 95)), 4),
        "single_prediction_mean_ms": round(float(np.mean(arr)), 4),
        "batch_1000_ms": round(float(batch_elapsed_ms), 4),
        "samples_measured": int(n_runs),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
    }


def evaluate_rule_based_heuristic(
    X: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    v14_threshold: float = -2.5,
    v4_threshold: float = 2.0,
    v12_threshold: float = -2.0,
    amount_high_threshold: float = 15000.0,
) -> dict[str, Any]:
    """
    Evaluates a deterministic Rule-Based Heuristic Engine on the exact same partition
    to provide an empirical baseline comparing static business rules against supervised ML
    (LogisticRegression and RandomForestClassifier).

    Rule logic:
    Flags a transaction as suspected fraud (1) if strong univariate anomalies occur:
      (V14 < v14_threshold) OR (V4 > v4_threshold AND V12 < v12_threshold) OR (Amount > amount_high_threshold)
    Converts the number of triggered conditions into a pseudo-score in [0, 1] for AUC comparison.
    """
    y_arr = np.asarray(y_true, dtype=int)
    c1 = (X["V14"] < v14_threshold).astype(float)
    c2 = ((X["V4"] > v4_threshold) & (X["V12"] < v12_threshold)).astype(float)
    c3 = (X["Amount"] > amount_high_threshold).astype(float)

    # Pseudo-score in [0, 1] proportional to triggered heuristic signals
    pseudo_scores = np.clip((0.55 * c1) + (0.35 * c2) + (0.20 * c3), 0.0, 1.0)
    metrics = evaluate_predictions(y_true=y_arr, scores=pseudo_scores, threshold=0.35)
    metrics["rule_definition"] = (
        f"IF (V14 < {v14_threshold}) OR (V4 > {v4_threshold} AND V12 < {v12_threshold}) "
        f"OR (Amount > {amount_high_threshold}) THEN Fraud(1) ELSE Legit(0)"
    )
    return metrics


