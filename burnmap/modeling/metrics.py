"""
Classification metrics computed locally from Python lists.

All functions take plain Python lists (not ee objects).
Metrics are computed after .getInfo() pulls data from GEE.

Includes: confusion matrix, F1, precision, recall, kappa,
balanced accuracy, AUC, Cohen's d, probability stats,
feature importance formatting.
"""

from __future__ import annotations
import math


# ── Confusion matrix ────────────────────────────────────────────────

def confusion_matrix(y_true: list, y_pred: list) -> dict:
    """
    Binary confusion matrix from label lists.

    Returns dict with TP, FP, TN, FN counts.
    """
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        else:
            fn += 1
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


# ── Core metrics ────────────────────────────────────────────────────

def compute_basic_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute F1, precision, recall, kappa, balanced accuracy
    from binary predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    n = tp + fp + tn + fn

    # Precision / recall / F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)

    # Cohen's kappa
    po = (tp + tn) / n if n > 0 else 0.0
    p_yes = ((tp + fp) / n) * ((tp + fn) / n) if n > 0 else 0.0
    p_no = ((tn + fn) / n) * ((tn + fp) / n) if n > 0 else 0.0
    pe = p_yes + p_no
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    # Balanced accuracy
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2.0

    # Predicted positive rate
    pred_pos_rate = (tp + fp) / n if n > 0 else 0.0

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "kappa": kappa,
        "balanced_accuracy": balanced_accuracy,
        "specificity": specificity,
        "pred_pos_rate": pred_pos_rate,
        **cm,
    }


def compute_auc(y_true: list, y_prob: list) -> float:
    """
    Compute AUC via the Wilcoxon–Mann–Whitney statistic.
    Works on plain Python lists — no sklearn required.
    """
    pairs = sorted(zip(y_prob, y_true), key=lambda x: x[0])
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = sum(1 for _, y in pairs if y == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Count how many negative examples are ranked below each positive
    cum_neg = 0
    auc_sum = 0.0
    for prob, label in pairs:
        if label == 0:
            cum_neg += 1
        else:
            auc_sum += cum_neg

    return auc_sum / (n_pos * n_neg)


def compute_all_metrics(y_true: list, y_pred: list,
                        y_prob: list | None = None) -> dict:
    """
    Compute all classification metrics.

    Parameters
    ----------
    y_true : list of true labels (0/1)
    y_pred : list of binary predictions (0/1)
    y_prob : list of predicted probabilities (optional, for AUC + Cohen's d)

    Returns
    -------
    dict with all metrics: f1, precision, recall, kappa,
    balanced_accuracy, specificity, pred_pos_rate, AUC,
    plus probability stats if y_prob provided.
    """
    result = compute_basic_metrics(y_true, y_pred)

    if y_prob is not None:
        result["auc"] = compute_auc(y_true, y_prob)
        prob_stats = class_probability_stats(y_true, y_prob)
        result.update(prob_stats)
        result["cohens_d"] = cohens_d(
            [p for y, p in zip(y_true, y_prob) if y == 1],
            [p for y, p in zip(y_true, y_prob) if y == 0],
        )
    else:
        result["auc"] = float("nan")
        result["cohens_d"] = float("nan")

    return result


# ── Probability analysis ────────────────────────────────────────────

def class_probability_stats(y_true: list, y_prob: list) -> dict:
    """
    Per-class mean and std of predicted probabilities.
    """
    pos_probs = [p for y, p in zip(y_true, y_prob) if y == 1]
    neg_probs = [p for y, p in zip(y_true, y_prob) if y == 0]

    def _mean_std(vals):
        if not vals:
            return 0.0, 0.0
        m = sum(vals) / len(vals)
        if len(vals) < 2:
            return m, 0.0
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        return m, math.sqrt(var)

    pos_mean, pos_std = _mean_std(pos_probs)
    neg_mean, neg_std = _mean_std(neg_probs)

    return {
        "pos_prob_mean": pos_mean,
        "pos_prob_std": pos_std,
        "neg_prob_mean": neg_mean,
        "neg_prob_std": neg_std,
    }


def cohens_d(group1: list, group2: list) -> float:
    """
    Cohen's d effect size between two groups.
    Uses pooled standard deviation.
    """
    if len(group1) < 2 or len(group2) < 2:
        return 0.0

    m1 = sum(group1) / len(group1)
    m2 = sum(group2) / len(group2)

    var1 = sum((x - m1) ** 2 for x in group1) / (len(group1) - 1)
    var2 = sum((x - m2) ** 2 for x in group2) / (len(group2) - 1)

    pooled_std = math.sqrt(
        ((len(group1) - 1) * var1 + (len(group2) - 1) * var2)
        / (len(group1) + len(group2) - 2)
    )

    if pooled_std == 0:
        return 0.0

    return (m1 - m2) / pooled_std


# ── Feature importance ──────────────────────────────────────────────

def format_importance(importance_dict: dict, top_k: int | None = None) -> list[dict]:
    """
    Sort feature importances descending, optionally keep top K.

    Parameters
    ----------
    importance_dict : dict mapping band_name → importance_value
    top_k           : if set, keep only top K features

    Returns
    -------
    list of {"feature": name, "importance": value} dicts, sorted descending.
    """
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        sorted_items = sorted_items[:top_k]
    return [{"feature": k, "importance": v} for k, v in sorted_items]