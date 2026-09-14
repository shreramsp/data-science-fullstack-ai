"""Model evaluation teaching simulation: confusion matrix, Type I/II errors,
precision/recall tradeoff, ROC-AUC, and cost-sensitive thresholding.

A classifier's continuous scores for the positive and negative class are
modelled as two overlapping Gaussians. Moving the decision threshold slides
along this synthetic score distribution and every downstream metric is
recomputed live, which is exactly what a real threshold sweep does.
"""
from __future__ import annotations

import numpy as np


def generate_scores(n_pos: int = 300, n_neg: int = 300, separation: float = 1.5,
                     seed: int = 42):
    """Synthetic classifier scores. Positive-class scores are centered higher
    than negative-class scores by `separation` standard deviations; larger
    separation => easier classification problem => higher achievable AUC."""
    rng = np.random.default_rng(seed)
    neg_scores = rng.normal(loc=0.0, scale=1.0, size=n_neg)
    pos_scores = rng.normal(loc=separation, scale=1.0, size=n_pos)
    y_true = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    y_score = np.concatenate([neg_scores, pos_scores])
    return y_true, y_score


def confusion_counts(y_true: np.ndarray, y_score: np.ndarray, threshold: float):
    """Counts for a binary classifier at a given decision threshold.

    TP: predicted positive, actually positive
    FP: predicted positive, actually negative  -> Type I error
    FN: predicted negative, actually positive  -> Type II error
    TN: predicted negative, actually negative
    """
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # a.k.a. sensitivity, TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "precision": precision, "recall": recall, "specificity": specificity,
        "fpr": fpr, "accuracy": accuracy, "f1": f1,
    }


def expected_cost(tp: int, fp: int, fn: int, tn: int,
                   cost_fp: float, cost_fn: float) -> float:
    """Total cost implied by a cost matrix that only penalizes the two error
    types (correct predictions cost 0)."""
    return fp * cost_fp + fn * cost_fn


def roc_curve(y_true: np.ndarray, y_score: np.ndarray, n_thresholds: int = 200):
    """Manual ROC curve: FPR vs TPR swept across thresholds spanning the
    observed score range, plus the trapezoidal AUC estimate."""
    thresholds = np.linspace(y_score.min() - 1e-6, y_score.max() + 1e-6, n_thresholds)
    tprs, fprs = [], []
    for t in thresholds[::-1]:  # sweep from low threshold (predict-all-positive) to high
        tp, fp, fn, tn = confusion_counts(y_true, y_score, t)
        m = metrics_from_counts(tp, fp, fn, tn)
        tprs.append(m["recall"])
        fprs.append(m["fpr"])
    fprs = np.array(fprs)
    tprs = np.array(tprs)
    auc = float(np.trapezoid(tprs, fprs))
    return fprs, tprs, auc
