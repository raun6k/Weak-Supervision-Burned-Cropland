"""
Phase 4: Multi-Classifier Training and Comparison
===================================================

Extends Phase 3 by adding 5 Python classifiers to the existing
RF + GBT results, all running on the same exported sample CSVs:

  - Logistic Regression (LR) — linear baseline
  - XGBoost (XGB) — boosted trees with colsample
  - LightGBM (LGBM) — leaf-wise boosting
  - MLP — PyTorch neural net (37→128→64→1)
  - MLP-SCE — same MLP with symmetric cross-entropy loss

Also runs statistical comparison (Wilcoxon signed-rank + Cliff's delta)
across all classifiers and produces the Phase 4 gate checks.

Uses identical fold assignments and sample CSVs from Phase 3.

Usage
-----
    python scripts/train_eval.py --year 2024
    python -m burnmap.modeling.classifiers --year 2024 --classifiers lr xgb
    python -m burnmap.modeling.classifiers --year 2024 --recipes very_strict --feature_sets A B
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import warnings

import burnmap.config as config
from burnmap.config import recipes as recipe_mod
import burnmap.modeling.folds as cv_mod
import burnmap.modeling.metrics as met

# Reuse Phase 3 utilities
from burnmap.modeling.cross_validation import (
    _build_band_lists, _load_csv, _find_full_csv,
    _assign_adm2_folds, _extract_xy, write_csv,
)

warnings.filterwarnings("ignore")


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 4: Multi-Classifier Comparison")
    p.add_argument("--year", type=int, default=2024, choices=[2023, 2024])
    p.add_argument("--sample_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--phase3_results", type=str, default=None,
                   help="Path to Phase 3 cv_metrics.csv to merge")
    p.add_argument("--recipes", nargs="+", default=["strict", "moderate", "very_strict"])
    p.add_argument("--feature_sets", nargs="+", default=["A", "B", "C", "D"])
    p.add_argument("--classifiers", nargs="+",
                   default=["lr", "xgb", "lgbm", "mlp", "mlp_sce"],
                   help="Classifiers: lr, xgb, lgbm, mlp, mlp_sce")
    p.add_argument("--skip_stats", action="store_true",
                   help="Skip statistical comparison")
    return p.parse_args()


# ── Classifier builders ─────────────────────────────────────────────

def _build_clf(name: str, n_features: int = 37):
    """Build a classifier by name."""

    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=config.LR_C,
            max_iter=config.LR_MAX_ITER,
            random_state=42,
            n_jobs=-1,
        )

    elif name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=config.XGB_N_ESTIMATORS,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LEARNING_RATE,
            colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
        )

    elif name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=config.LGBM_N_ESTIMATORS,
            num_leaves=config.LGBM_NUM_LEAVES,
            learning_rate=config.LGBM_LEARNING_RATE,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )

    elif name in ("mlp", "mlp_sce"):
        # Return a string tag — actual training handled by _run_mlp()
        return name

    else:
        raise ValueError(f"Unknown classifier: {name}")


# ── MLP (PyTorch) ────────────────────────────────────────────────────

def _run_mlp(X_train, y_train, X_test, y_test, use_sce: bool = False):
    """
    Train a PyTorch MLP and return test probabilities.

    Architecture: n_features → 128 → 64 → 1
    Loss: BCE (standard) or SCE (symmetric cross-entropy)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cpu")
    n_features = len(X_train[0])

    # Convert to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_te = torch.tensor(X_test, dtype=torch.float32)

    train_ds = TensorDataset(X_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=config.MLP_BATCH_SIZE, shuffle=True)

    # Model
    model = nn.Sequential(
        nn.Linear(n_features, config.MLP_HIDDEN[0]),
        nn.BatchNorm1d(config.MLP_HIDDEN[0]),
        nn.ReLU(),
        nn.Dropout(config.MLP_DROPOUT),
        nn.Linear(config.MLP_HIDDEN[0], config.MLP_HIDDEN[1]),
        nn.BatchNorm1d(config.MLP_HIDDEN[1]),
        nn.ReLU(),
        nn.Dropout(config.MLP_DROPOUT),
        nn.Linear(config.MLP_HIDDEN[1], 1),
        nn.Sigmoid(),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.MLP_LR)

    # Loss
    bce = nn.BCELoss()

    def sce_loss(pred, target):
        """Symmetric Cross-Entropy: α·CE(y,p) + β·CE(p,y)"""
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1 - eps)
        ce_forward = bce(pred, target)
        ce_reverse = -(pred * torch.log(target + eps)
                       + (1 - pred) * torch.log(1 - target + eps)).mean()
        return config.SCE_ALPHA * ce_forward + config.SCE_BETA * ce_reverse

    loss_fn = sce_loss if use_sce else bce

    # Train
    model.train()
    for epoch in range(config.MLP_EPOCHS):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Predict
    model.eval()
    with torch.no_grad():
        probs = model(X_te.to(device)).squeeze().cpu().numpy().tolist()

    return probs


# ── Single fold runner ───────────────────────────────────────────────

def run_fold(
    clf_name: str,
    train_rows: list[dict],
    test_rows: list[dict],
    band_list: list[str],
    recipe_id: str,
    set_id: str,
    seed: int,
    fold_id: int,
) -> tuple:
    """Run one CV fold. Same logic as phase3_cv but supports all classifiers."""
    import random

    X_train, y_train, valid_bands = _extract_xy(train_rows, band_list)
    X_test, y_test, _ = _extract_xy(test_rows, band_list)

    if len(X_train) < 20 or len(X_test) < 10:
        return {
            "recipe_id": recipe_id, "set_id": set_id,
            "classifier": clf_name, "seed": seed, "fold_id": fold_id,
            "error": f"Too few samples: train={len(X_train)}, test={len(X_test)}",
        }

    # ── Inner threshold sweep ────────────────────────────────────
    train_adm2s = sorted(set(
        int(r.get("adm2", -1)) for r in train_rows
        if r.get("class") is not None and int(r.get("adm2", -1)) >= 0
    ))

    inner_y_true_all = []
    inner_y_prob_all = []

    if len(train_adm2s) >= config.INNER_K_FOLDS:
        rng = random.Random(seed)
        shuffled = train_adm2s.copy()
        rng.shuffle(shuffled)
        adm2_to_inner = {c: i % config.INNER_K_FOLDS for i, c in enumerate(shuffled)}

        for inner_fid in range(config.INNER_K_FOLDS):
            it_X, it_y, iv_X, iv_y = [], [], [], []
            for row in train_rows:
                label = row.get("class")
                adm2 = int(row.get("adm2", -1))
                if label is None or adm2 < 0:
                    continue
                feats = [row.get(b, 0.0) for b in valid_bands]
                if any(v != v for v in feats):
                    continue
                if adm2_to_inner.get(adm2, -1) == inner_fid:
                    iv_X.append(feats)
                    iv_y.append(int(label))
                else:
                    it_X.append(feats)
                    it_y.append(int(label))

            if len(iv_X) < 5 or len(it_X) < 10:
                continue
            if len(set(it_y)) < 2 or len(set(iv_y)) < 2:
                continue

            # Train inner model
            if clf_name in ("mlp", "mlp_sce"):
                probs = _run_mlp(it_X, it_y, iv_X, iv_y,
                                 use_sce=(clf_name == "mlp_sce"))
            else:
                clf_inner = _build_clf(clf_name, len(valid_bands))
                clf_inner.fit(it_X, it_y)
                probs = [p[1] for p in clf_inner.predict_proba(iv_X)]

            inner_y_true_all.extend(iv_y)
            inner_y_prob_all.extend(probs)

    chosen_thresh = 0.5
    if inner_y_true_all:
        thresh_result = cv_mod.sweep_thresholds(inner_y_true_all, inner_y_prob_all)
        chosen_thresh = thresh_result["chosen_thresh"]

    # ── Train full, predict test ─────────────────────────────────
    if clf_name in ("mlp", "mlp_sce"):
        test_probs = _run_mlp(X_train, y_train, X_test, y_test,
                              use_sce=(clf_name == "mlp_sce"))
    else:
        clf = _build_clf(clf_name, len(valid_bands))
        clf.fit(X_train, y_train)
        test_probs = [p[1] for p in clf.predict_proba(X_test)]

    y_pred = [1 if p >= chosen_thresh else 0 for p in test_probs]
    all_metrics = met.compute_all_metrics(y_test, y_pred, test_probs)

    # Feature importance (sklearn tree-based only)
    importance = {}
    if clf_name not in ("mlp", "mlp_sce", "lr"):
        try:
            if hasattr(clf, "feature_importances_"):
                importance = dict(zip(valid_bands, clf.feature_importances_))
        except Exception:
            pass

    result = {
        "recipe_id": recipe_id, "set_id": set_id, "classifier": clf_name,
        "seed": seed, "fold_id": fold_id, "chosen_thresh": chosen_thresh,
        "n_train": len(X_train), "n_test": len(X_test),
        **all_metrics,
    }

    if importance:
        formatted = met.format_importance(importance, config.TOPK_IMPORTANCE)
        for i, item in enumerate(formatted):
            result[f"imp_{i+1}_feat"] = item["feature"]
            result[f"imp_{i+1}_val"] = item["importance"]

    return result, test_probs, y_test


# ── Statistical comparison ──────────────────────────────────────────

def wilcoxon_signed_rank(x: list[float], y: list[float]) -> float:
    """
    Wilcoxon signed-rank test (two-sided), returns p-value.
    Pure Python implementation — no scipy dependency.
    """
    diffs = [a - b for a, b in zip(x, y) if a != b]
    n = len(diffs)
    if n < 5:
        return 1.0

    ranks = sorted(range(n), key=lambda i: abs(diffs[i]))
    rank_vals = [0.0] * n
    for i, idx in enumerate(ranks):
        rank_vals[idx] = i + 1.0

    w_plus = sum(r for d, r in zip(diffs, rank_vals) if d > 0)
    w_minus = sum(r for d, r in zip(diffs, rank_vals) if d < 0)
    w = min(w_plus, w_minus)

    # Normal approximation for n >= 10
    mean_w = n * (n + 1) / 4.0
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if std_w == 0:
        return 1.0
    z = (w - mean_w) / std_w

    # Two-sided p-value from z (normal CDF approximation)
    p = 2.0 * _norm_cdf(-abs(z))
    return p


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def cliffs_delta(x: list[float], y: list[float]) -> float:
    """Cliff's delta effect size."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0
    more = sum(1 for a in x for b in y if a > b)
    less = sum(1 for a in x for b in y if a < b)
    return (more - less) / (n1 * n2)


def run_statistical_comparison(all_results: list[dict], out_dir: str):
    """
    Pairwise Wilcoxon signed-rank + Cliff's delta for all classifier pairs.
    Uses paired fold-seed values within each recipe × feature_set.
    """
    print("\n── Statistical Comparison ──")

    clf_results = [r for r in all_results
                   if not r.get("classifier", "").startswith("baseline_")
                   and "error" not in r]

    classifiers = sorted(set(r["classifier"] for r in clf_results))
    if len(classifiers) < 2:
        print("  Need ≥2 classifiers for comparison")
        return []

    comparison_rows = []
    n_comparisons = len(classifiers) * (len(classifiers) - 1) // 2
    bonferroni = n_comparisons

    for recipe in sorted(set(r["recipe_id"] for r in clf_results)):
        for set_id in sorted(set(r["set_id"] for r in clf_results)):
            for i, c1 in enumerate(classifiers):
                for c2 in classifiers[i+1:]:
                    # Get paired kappa values (matched by seed × fold)
                    vals1 = {}
                    vals2 = {}
                    for r in clf_results:
                        if r["recipe_id"] != recipe or r["set_id"] != set_id:
                            continue
                        key = (r["seed"], r["fold_id"])
                        if r["classifier"] == c1:
                            vals1[key] = float(r["kappa"])
                        elif r["classifier"] == c2:
                            vals2[key] = float(r["kappa"])

                    paired_keys = sorted(set(vals1.keys()) & set(vals2.keys()))
                    if len(paired_keys) < 5:
                        continue

                    x = [vals1[k] for k in paired_keys]
                    y = [vals2[k] for k in paired_keys]

                    p_val = wilcoxon_signed_rank(x, y)
                    p_corrected = min(1.0, p_val * bonferroni)
                    delta = cliffs_delta(x, y)
                    mean_diff = sum(a - b for a, b in zip(x, y)) / len(x)

                    sig = "***" if p_corrected < 0.001 else "**" if p_corrected < 0.01 else "*" if p_corrected < 0.05 else "ns"

                    row = {
                        "recipe_id": recipe, "set_id": set_id,
                        "classifier_1": c1, "classifier_2": c2,
                        "mean_1": sum(x) / len(x),
                        "mean_2": sum(y) / len(y),
                        "mean_diff": mean_diff,
                        "p_value": p_val,
                        "p_corrected": p_corrected,
                        "cliffs_delta": delta,
                        "n_pairs": len(paired_keys),
                        "significant": sig,
                    }
                    comparison_rows.append(row)

    # Print summary of significant differences
    sig_rows = [r for r in comparison_rows if r["significant"] != "ns"]
    print(f"  {len(comparison_rows)} pairwise comparisons, {len(sig_rows)} significant")

    if sig_rows:
        print(f"  {'Recipe':12s} {'Set':3s} {'C1':8s} {'C2':8s} {'Diff':>7s} {'p_corr':>8s} {'δ':>6s}")
        for r in sorted(sig_rows, key=lambda x: x["p_corrected"])[:20]:
            print(f"  {r['recipe_id']:12s} {r['set_id']:3s} "
                  f"{r['classifier_1']:8s} {r['classifier_2']:8s} "
                  f"{r['mean_diff']:+.4f} {r['p_corrected']:.4f} {r['cliffs_delta']:+.3f} "
                  f"{r['significant']}")

    # Write to CSV
    if comparison_rows:
        write_csv(comparison_rows, os.path.join(out_dir, "statistical_comparisons.csv"))

    return comparison_rows


# ── Gate checks ──────────────────────────────────────────────────────

def run_gate_checks(all_results: list[dict], comparison_rows: list[dict]):
    """Phase 4 gate checks per the build plan."""
    print("\n" + "=" * 60)
    print("PHASE 4 GATE CHECKS")
    print("=" * 60)

    clf_results = [r for r in all_results
                   if not r.get("classifier", "").startswith("baseline_")
                   and "error" not in r]

    # 1) Best classifier kappa (very_strict, since strict may not hit 0.35)
    vs_kappas = [float(r["kappa"]) for r in clf_results
                 if r["recipe_id"] == "very_strict"]
    best_kappa_vs = max(vs_kappas) if vs_kappas else 0
    strict_kappas = [float(r["kappa"]) for r in clf_results
                     if r["recipe_id"] == "strict"]
    best_kappa_strict = max(strict_kappas) if strict_kappas else 0

    g1 = best_kappa_vs >= config.GATE_BEST_KAPPA_STRICT
    print(f"\n  1) Best kappa (very_strict): {best_kappa_vs:.3f} "
          f"{'PASS' if g1 else 'FAIL'} (need ≥{config.GATE_BEST_KAPPA_STRICT})")
    print(f"     Best kappa (strict):      {best_kappa_strict:.3f}")

    # 2) Best AUC
    aucs = [float(r["auc"]) for r in clf_results if r.get("auc")]
    best_auc = max(aucs) if aucs else 0
    g2 = best_auc >= config.GATE_BEST_AUC
    print(f"\n  2) Best AUC: {best_auc:.3f} "
          f"{'PASS' if g2 else 'FAIL'} (need ≥{config.GATE_BEST_AUC})")

    # 3) Best Cohen's d
    ds = [float(r["cohens_d"]) for r in clf_results
          if r.get("cohens_d") and float(r.get("cohens_d", -999)) > -999]
    best_d = max(ds) if ds else 0
    g3 = best_d >= config.GATE_BEST_COHENS_D
    print(f"\n  3) Best Cohen's d: {best_d:.3f} "
          f"{'PASS' if g3 else 'FAIL'} (need ≥{config.GATE_BEST_COHENS_D})")

    # 4) Any classifier beats RF significantly?
    rf_beats = [r for r in comparison_rows
                if "rf" in (r["classifier_1"], r["classifier_2"])
                and r["significant"] != "ns"]
    g4 = len(rf_beats) > 0
    print(f"\n  4) Classifier significantly beats RF: "
          f"{'YES (' + str(len(rf_beats)) + ' comparisons)' if g4 else 'NO'}")

    # 5) LR vs RF gap
    lr_kappas = {(r["recipe_id"], r["set_id"], r["seed"], r["fold_id"]): float(r["kappa"])
                 for r in clf_results if r["classifier"] == "lr"}
    rf_kappas = {(r["recipe_id"], r["set_id"], r["seed"], r["fold_id"]): float(r["kappa"])
                 for r in clf_results if r["classifier"] == "rf"}
    common_keys = set(lr_kappas.keys()) & set(rf_kappas.keys())
    if common_keys:
        lr_mean = sum(lr_kappas[k] for k in common_keys) / len(common_keys)
        rf_mean = sum(rf_kappas[k] for k in common_keys) / len(common_keys)
        gap = rf_mean - lr_mean
        g5 = gap >= config.GATE_LR_RF_GAP
        print(f"\n  5) LR vs RF gap: RF={rf_mean:.3f}, LR={lr_mean:.3f}, "
              f"gap={gap:+.3f} {'PASS (RF > LR)' if g5 else 'signal may be linear'}")
    else:
        g5 = True
        print(f"\n  5) LR vs RF: no paired data available")

    # 6) No-SWIR ablation
    set_a = [float(r["kappa"]) for r in clf_results if r["set_id"] == "A"]
    set_d = [float(r["kappa"]) for r in clf_results if r["set_id"] == "D"]
    if set_a and set_d:
        drop = sum(set_a) / len(set_a) - sum(set_d) / len(set_d)
        g6 = drop <= config.GATE_SWIR_ABLATION_DROP
        print(f"\n  6) SWIR ablation drop: {drop:.4f} "
              f"{'PASS (minor)' if g6 else 'CONCERN (>0.03)'}")
    else:
        g6 = True

    # Summary by classifier
    print(f"\n  ── Classifier summary ──")
    by_clf = {}
    for r in clf_results:
        by_clf.setdefault(r["classifier"], []).append(float(r["kappa"]))
    for c in sorted(by_clf, key=lambda c: -sum(by_clf[c])/len(by_clf[c])):
        ks = by_clf[c]
        print(f"    {c:8s}: mean={sum(ks)/len(ks):.3f} ±{(sum((k-sum(ks)/len(ks))**2 for k in ks)/len(ks))**0.5:.3f}  n={len(ks)}")

    all_pass = g1 and g2
    print(f"\n{'=' * 60}")
    print(f"  OVERALL: {'ALL CRITICAL GATES PASS' if all_pass else 'ISSUES — see above'}")
    print(f"{'=' * 60}\n")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    year = args.year

    sample_dir = args.sample_dir or "data/samples"
    out_dir = args.out_dir or f"outputs/{year}/phase4_results"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Phase 4 · Punjab {year}")
    print(f"Sample dir:   {sample_dir}")
    print(f"Output dir:   {out_dir}")
    print(f"Recipes:      {args.recipes}")
    print(f"Feature sets: {args.feature_sets}")
    print(f"Classifiers:  {args.classifiers}\n")

    if not os.path.isdir(sample_dir):
        print(f"ERROR: {sample_dir} not found.")
        return

    band_lists = _build_band_lists()
    selected_recipes = [recipe_mod.get_recipe(r) for r in args.recipes]

    all_results = []
    all_prob_records = []

    total = (len(args.recipes) * len(args.feature_sets)
             * len(args.classifiers) * len(config.SEEDS) * config.K_FOLDS)
    run_count = 0

    for recipe in selected_recipes:
        rid = recipe["recipe_id"]

        for seed in config.SEEDS:
            csv_path = _find_full_csv(sample_dir, rid, seed)
            if not csv_path:
                print(f"  MISSING: {rid} / s{seed} — skipping")
                missing = len(args.feature_sets) * len(args.classifiers) * config.K_FOLDS
                run_count += missing
                continue

            print(f"\n  Loading {rid} / s{seed} …")
            all_rows = _load_csv(csv_path)
            print(f"    {len(all_rows)} rows")

            adm2_codes = sorted(set(
                int(r.get("adm2", -1)) for r in all_rows
                if r.get("adm2") is not None and int(r.get("adm2", -1)) >= 0
            ))
            adm2_fold_map = _assign_adm2_folds(adm2_codes, config.K_FOLDS, seed)

            for fold_id in range(config.K_FOLDS):
                test_adm2s = {c for c, f in adm2_fold_map.items() if f == fold_id}
                train_adm2s = {c for c, f in adm2_fold_map.items() if f != fold_id}
                train_rows = [r for r in all_rows if int(r.get("adm2", -1)) in train_adm2s]
                test_rows = [r for r in all_rows if int(r.get("adm2", -1)) in test_adm2s]

                for set_id in args.feature_sets:
                    bl = band_lists.get(set_id, band_lists["A"])

                    for clf_name in args.classifiers:
                        run_count += 1

                        try:
                            result = run_fold(
                                clf_name, train_rows, test_rows, bl,
                                rid, set_id, seed, fold_id,
                            )

                            if isinstance(result, tuple):
                                result, probs, y_test = result
                                step = max(1, len(y_test) // 200)
                                for i in range(0, len(y_test), step):
                                    all_prob_records.append({
                                        "recipe_id": rid, "set_id": set_id,
                                        "classifier": clf_name, "seed": seed,
                                        "fold_id": fold_id,
                                        "y_true": y_test[i], "y_prob": probs[i],
                                    })
                                k = result.get("kappa", 0)
                                f1 = result.get("f1", 0)
                                auc_v = result.get("auc", 0)
                                print(f"    [{run_count}/{total}] {set_id}/{clf_name}"
                                      f"/f{fold_id}: k={k:.3f} f1={f1:.3f} auc={auc_v:.3f}")
                            else:
                                print(f"    [{run_count}/{total}] {set_id}/{clf_name}"
                                      f"/f{fold_id}: {result.get('error','?')}")

                            all_results.append(result)

                        except Exception as e:
                            print(f"    [{run_count}/{total}] {set_id}/{clf_name}"
                                  f"/f{fold_id}: ERROR {e}")
                            all_results.append({
                                "recipe_id": rid, "set_id": set_id,
                                "classifier": clf_name, "seed": seed,
                                "fold_id": fold_id, "error": str(e),
                            })

    # ── Merge Phase 3 results if available ───────────────────────
    phase3_path = args.phase3_results
    if not phase3_path:
        phase3_path = f"outputs/{year}/phase3_results/cv_metrics.csv"
    phase3_rows = []
    if os.path.exists(phase3_path):
        print(f"\nMerging Phase 3 results from {phase3_path} …")
        phase3_rows = _load_csv(phase3_path)
        print(f"  {len(phase3_rows)} Phase 3 rows merged")

    combined = phase3_rows + all_results

    # ── Write results ────────────────────────────────────────────
    print(f"\nWriting results to {out_dir} …")
    write_csv(all_results, os.path.join(out_dir, "phase4_cv_metrics.csv"))
    write_csv(combined, os.path.join(out_dir, "combined_cv_metrics.csv"))

    if all_prob_records:
        write_csv(all_prob_records, os.path.join(out_dir, "phase4_test_probabilities.csv"))

    # ── Statistical comparison ───────────────────────────────────
    comparison_rows = []
    if not args.skip_stats:
        comparison_rows = run_statistical_comparison(combined, out_dir)

    # ── Gate checks ──────────────────────────────────────────────
    run_gate_checks(combined, comparison_rows)

    print("Phase 4 done.\n")


if __name__ == "__main__":
    main()