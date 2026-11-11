"""
District-held-out cross-validation utilities.

Handles:
  - Deterministic assignment of districts to K folds (per seed)
  - Train/test district splits for each fold
  - Inner K-fold assignment for threshold selection
  - Threshold sweep to pick optimal classification threshold

All functions return plain Python objects (lists, dicts).
No ee.batch.Export calls — that belongs in pipelines/.
"""

import random
import burnmap.config as config


def assign_folds(district_names: list[str], k: int = config.K_FOLDS,
                 seed: int = 13) -> dict:
    """
    Deterministically assign districts to K folds.

    Parameters
    ----------
    district_names : list of district name strings (e.g. from ADM2_NAME)
    k              : number of folds
    seed           : random seed for shuffle

    Returns
    -------
    dict with keys:
        assignments : list of (district_name, fold_id) tuples
        fold_map    : dict mapping district_name → fold_id
        folds       : dict mapping fold_id → list of district names
        seed        : the seed used
        k           : number of folds
    """
    names_sorted = sorted(district_names)
    rng = random.Random(seed)
    shuffled = names_sorted.copy()
    rng.shuffle(shuffled)

    fold_map = {}
    folds = {i: [] for i in range(k)}
    for idx, name in enumerate(shuffled):
        fold_id = idx % k
        fold_map[name] = fold_id
        folds[fold_id].append(name)

    assignments = [(name, fold_map[name]) for name in names_sorted]

    return {
        "assignments": assignments,
        "fold_map": fold_map,
        "folds": folds,
        "seed": seed,
        "k": k,
    }


def get_fold_split(fold_info: dict, test_fold: int) -> dict:
    """
    Return train/test district lists for a given held-out fold.

    Parameters
    ----------
    fold_info : dict from assign_folds()
    test_fold : which fold to hold out (0-indexed)

    Returns
    -------
    dict with keys:
        train_districts : list of district names in training set
        test_districts  : list of district names in test set
    """
    train = []
    test = []
    for name, fid in fold_info["fold_map"].items():
        if fid == test_fold:
            test.append(name)
        else:
            train.append(name)
    return {
        "train_districts": sorted(train),
        "test_districts": sorted(test),
    }


def inner_fold_assignment(train_districts: list[str],
                          k: int = config.INNER_K_FOLDS,
                          seed: int = 13) -> dict:
    """
    Assign training districts to inner folds for threshold selection.
    Same logic as assign_folds but on the training subset.
    """
    return assign_folds(train_districts, k=k, seed=seed)


def sweep_thresholds(y_true: list, y_prob: list,
                     thresholds: list[float] = None,
                     mode: str = config.CHOOSE_MODE) -> dict:
    """
    Find the threshold that maximizes the chosen metric.

    Parameters
    ----------
    y_true      : list of true labels (0/1)
    y_prob      : list of predicted probabilities
    thresholds  : list of thresholds to try
    mode        : "KAPPA" | "F1" | "ACC"

    Returns
    -------
    dict with keys:
        chosen_thresh : best threshold
        best_score    : score at best threshold
        all_scores    : list of (threshold, score) tuples
    """
    # Import here to avoid circular dependency at module level
    import burnmap.modeling.metrics as met

    if thresholds is None:
        thresholds = config.THRESHOLDS

    best_thresh = 0.5
    best_score = -1.0
    all_scores = []

    for thr in thresholds:
        y_pred = [1 if p >= thr else 0 for p in y_prob]
        m = met.compute_basic_metrics(y_true, y_pred)

        if mode == "KAPPA":
            score = m["kappa"]
        elif mode == "F1":
            score = m["f1"]
        else:
            score = m["balanced_accuracy"]

        all_scores.append((thr, score))
        if score > best_score:
            best_score = score
            best_thresh = thr

    return {
        "chosen_thresh": best_thresh,
        "best_score": best_score,
        "all_scores": all_scores,
    }


def build_fold_manifest_rows(seeds: list[int] = None,
                             district_names: list[str] = None,
                             k: int = config.K_FOLDS) -> list[dict]:
    """
    Build a flat list of rows for the fold manifest CSV.
    Each row: {seed, district, fold_id}.
    Useful for exporting as ee.FeatureCollection.
    """
    if seeds is None:
        seeds = config.SEEDS

    rows = []
    for seed in seeds:
        fi = assign_folds(district_names, k=k, seed=seed)
        for name, fid in fi["assignments"]:
            rows.append({"seed": seed, "district": name, "fold_id": fid})
    return rows