"""
Phase 3 — Step 2: Cross-Validation on Exported Samples
========================================================

Loads full-Punjab CSVs (one per recipe × seed) exported by
phase3_export, then splits by district/fold and subsets columns
for feature sets B/C/D locally. Runs RF + GBT via scikit-learn.

No GEE dependency — everything runs in Python.

Expects CSVs in: data/samples/
Writes results to: outputs/<year>/phase3_results/

Usage
-----
    python scripts/train_eval.py --year 2024
    python scripts/train_eval.py --year 2024 --sample_dir /path/to/csvs
    python scripts/train_eval.py --year 2024 --recipes strict --feature_sets A
    python -m burnmap.modeling.cross_validation --year 2024 --classifiers rf
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import burnmap.config as config
from burnmap.config import recipes as recipe_mod
import burnmap.modeling.folds as cv_mod
import burnmap.modeling.metrics as met


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 — CV on Exported Samples")
    p.add_argument("--year", type=int, default=2024, choices=[2023, 2024])
    p.add_argument("--sample_dir", type=str, default=None,
                   help="Directory with exported CSVs (default: data/samples)")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Directory for results (default: outputs/<year>/phase3_results)")
    p.add_argument("--recipes", nargs="+", default=["strict", "moderate", "very_strict"])
    p.add_argument("--feature_sets", nargs="+", default=["A", "B", "C", "D"])
    p.add_argument("--classifiers", nargs="+", default=["rf", "gbt"])
    p.add_argument("--skip_baselines", action="store_true")
    return p.parse_args()


# ── Band lists (no ee dependency) ────────────────────────────────────

def _build_band_lists() -> dict:
    """Build band name lists for each feature set, pure Python."""
    _S2 = config.S2_BANDS
    _IX = ["NDVI", "NDMI", "NDRE"]
    _SWIR = ["B11", "B12"]
    _ns = [b for b in _S2 if b not in _SWIR]

    def _pre(bands):   return [f"{b}_pre" for b in bands]
    def _post(bands):  return [f"{b}_post" for b in bands]
    # dB12 and dNDVI gate the weak labels, so they are never predictors.
    _LABEL_VARS = {"dB12", "dNDVI"}
    def _delta(bands): return [f"d{b}" for b in bands if f"d{b}" not in _LABEL_VARS]
    def _ipre():       return [f"{ix}_pre" for ix in _IX]
    def _ipost():      return [f"{ix}_post" for ix in _IX]
    def _idelta():     return [f"d{ix}" for ix in _IX if f"d{ix}" not in _LABEL_VARS]

    return {
        "A": _pre(_S2) + _post(_S2) + _delta(_S2) + _ipre() + _ipost() + _idelta(),
        "B": _delta(_S2) + _idelta(),
        "C": _delta(_S2) + _idelta() + _pre(_S2) + _ipre(),
        "D": _pre(_ns) + _post(_ns) + _delta(_ns) + _ipre() + _ipost() + _idelta(),
        "E": _pre(_S2) + _post(_S2) + _delta(_S2) + _ipre() + _ipost() + _idelta()
             + ["VV_pre", "VH_pre", "VV_post", "VH_post", "dVV", "dVH"],
    }


# ── CSV loading ──────────────────────────────────────────────────────

def _load_csv(path: str) -> list[dict]:
    """Load CSV into list of dicts with numeric conversion."""
    with open(path, "r") as f:
        rows = []
        for row in csv.DictReader(f):
            clean = {}
            for k, v in row.items():
                try:
                    clean[k] = float(v)
                except (ValueError, TypeError):
                    clean[k] = v
            rows.append(clean)
    return rows


def _first_match(sample_dir: str, patterns: list[str]) -> str | None:
    """Return the first existing sample, preferring the public filename."""
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(sample_dir, pattern)))
        if matches:
            return matches[0]
    return None


def _find_full_csv(sample_dir: str, recipe_id: str, seed: int) -> str | None:
    """Find the Punjab 2024 sample CSV for a given recipe and seed.

    The recipe id is matched as its own token so ``strict`` does not
    also select ``very_strict``.
    """
    return _first_match(sample_dir, [
        f"punjab_2024_{recipe_id}_A_s{seed}.csv",
        f"*phase3_{recipe_id}_A_s{seed}_full.csv",
    ])


def _find_baseline_csv(sample_dir: str, recipe_id: str,
                       baseline_type: str, seed: int) -> str | None:
    """Find the Punjab 2024 baseline CSV for a recipe, baseline, and seed."""
    return _first_match(sample_dir, [
        f"punjab_2024_{recipe_id}_baseline_{baseline_type}_s{seed}.csv",
        f"*phase3_{recipe_id}_baseline_{baseline_type}_s{seed}_full.csv",
    ])


# ── District code map from fold manifest ─────────────────────────────

def _load_fold_manifest(sample_dir: str) -> list[dict]:
    """Load fold manifest CSV."""
    matches = glob.glob(os.path.join(sample_dir, "*_fold_manifest.csv"))
    if matches:
        return _load_csv(matches[0])
    return []


def _build_district_code_map_from_data(rows: list[dict],
                                       manifest: list[dict]) -> dict:
    """
    Build {adm2_code: district_name} mapping.
    Uses fold manifest for name→code, and sample data for available codes.
    """
    # From manifest we get district names
    # From sample data we get adm2 codes
    # We need to match them. Since the manifest doesn't have codes,
    # we'll assign folds by adm2 code directly using the same logic.
    adm2_codes = sorted(set(int(r.get("adm2", -1)) for r in rows if r.get("adm2") is not None))
    adm2_codes = [c for c in adm2_codes if c >= 0]
    return adm2_codes


def _split_by_adm2(rows: list[dict], adm2_codes: list[int],
                   fold_map: dict) -> dict:
    """
    Split rows into {fold_id: [rows]} based on adm2 → fold assignment.
    fold_map: {adm2_code: fold_id}
    """
    folds = {}
    for r in rows:
        adm2 = int(r.get("adm2", -1))
        fid = fold_map.get(adm2)
        if fid is not None:
            folds.setdefault(fid, []).append(r)
    return folds


def _assign_adm2_folds(adm2_codes: list[int], k: int, seed: int) -> dict:
    """
    Assign adm2 codes to folds deterministically.
    Same logic as cv.assign_folds but on integer codes.
    Returns {adm2_code: fold_id}.
    """
    import random
    codes_sorted = sorted(adm2_codes)
    rng = random.Random(seed)
    shuffled = codes_sorted.copy()
    rng.shuffle(shuffled)
    return {code: i % k for i, code in enumerate(shuffled)}


# ── Feature extraction ───────────────────────────────────────────────

def _extract_xy(rows: list[dict], band_list: list[str]) -> tuple:
    """Extract (X, y) from rows, filtering NaN."""
    available = set(rows[0].keys()) if rows else set()
    valid_bands = [b for b in band_list if b in available]

    X, y = [], []
    for row in rows:
        label = row.get("class")
        if label is None:
            continue
        features = [row.get(b, 0.0) for b in valid_bands]
        if any(v != v for v in features):
            continue
        X.append(features)
        y.append(int(label))
    return X, y, valid_bands


# ── Scikit-learn classifiers ─────────────────────────────────────────

def _build_clf(name: str):
    """Build a scikit-learn classifier matching GEE equivalent."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=config.RF_N_TREES,
            max_features="sqrt",
            min_samples_leaf=config.RF_MIN_LEAF_POP,
            random_state=42,
            n_jobs=-1,
        )
    elif name == "gbt":
        return GradientBoostingClassifier(
            n_estimators=config.GBT_N_TREES,
            learning_rate=config.GBT_SHRINKAGE,
            max_leaf_nodes=config.GBT_MAX_NODES,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")


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
    """
    Run one CV fold entirely in Python.
    Inner threshold sweep splits training data by adm2 code.
    """
    X_train, y_train, valid_bands = _extract_xy(train_rows, band_list)
    X_test, y_test, _ = _extract_xy(test_rows, band_list)

    if len(X_train) < 20 or len(X_test) < 10:
        return {
            "recipe_id": recipe_id, "set_id": set_id,
            "classifier": clf_name, "seed": seed, "fold_id": fold_id,
            "error": f"Too few samples: train={len(X_train)}, test={len(X_test)}",
        }

    # ── Inner threshold sweep by adm2 ────────────────────────────
    import random

    train_adm2s = sorted(set(int(r.get("adm2", -1)) for r in train_rows
                             if r.get("class") is not None and int(r.get("adm2", -1)) >= 0))

    inner_y_true_all = []
    inner_y_prob_all = []

    if len(train_adm2s) >= config.INNER_K_FOLDS:
        rng = random.Random(seed)
        shuffled = train_adm2s.copy()
        rng.shuffle(shuffled)
        adm2_to_inner = {code: i % config.INNER_K_FOLDS for i, code in enumerate(shuffled)}

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

            clf_inner = _build_clf(clf_name)
            clf_inner.fit(it_X, it_y)
            probs = [p[1] for p in clf_inner.predict_proba(iv_X)]
            inner_y_true_all.extend(iv_y)
            inner_y_prob_all.extend(probs)

    if inner_y_true_all:
        thresh_result = cv_mod.sweep_thresholds(inner_y_true_all, inner_y_prob_all)
        chosen_thresh = thresh_result["chosen_thresh"]
    else:
        chosen_thresh = 0.5

    # ── Train full, predict test ─────────────────────────────────
    clf = _build_clf(clf_name)
    clf.fit(X_train, y_train)
    test_probs = [p[1] for p in clf.predict_proba(X_test)]
    y_pred = [1 if p >= chosen_thresh else 0 for p in test_probs]

    all_metrics = met.compute_all_metrics(y_test, y_pred, test_probs)

    importance = {}
    if hasattr(clf, "feature_importances_"):
        importance = dict(zip(valid_bands, clf.feature_importances_))

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


# ── Baselines ────────────────────────────────────────────────────────

def compute_baselines(
    dnbr_rows: list[dict] | None,
    test_adm2s: set[int],
    recipe: dict,
    recipe_id: str,
    seed: int,
    fold_id: int,
) -> list[dict]:
    """Compute the recipe-rule and fixed-dNBR baselines on held-out districts."""
    results = []

    if dnbr_rows:
        test_dnbr = [r for r in dnbr_rows if int(r.get("adm2", -1)) in test_adm2s]
        y_true = [int(r["class"]) for r in test_dnbr if r.get("class") is not None]
        y_prob = [float(r["dNBR_baseline"]) for r in test_dnbr
                  if r.get("dNBR_baseline") is not None]

        if y_true and len(y_true) == len(y_prob):
            # Recipe rule
            y_pred = [1 if p >= recipe["burnDNBRThr"] else 0 for p in y_prob]
            m1 = met.compute_all_metrics(y_true, y_pred, y_prob)
            results.append({
                "classifier": "baseline_recipe_rule",
                "recipe_id": recipe_id, "seed": seed, "fold_id": fold_id,
                "set_id": "baseline", "chosen_thresh": recipe["burnDNBRThr"],
                **m1,
            })

            # Fixed dNBR
            y_pred2 = [1 if p >= config.FIXED_DNBR_THRESHOLD else 0 for p in y_prob]
            m2 = met.compute_all_metrics(y_true, y_pred2, y_prob)
            results.append({
                "classifier": "baseline_fixed_dnbr",
                "recipe_id": recipe_id, "seed": seed, "fold_id": fold_id,
                "set_id": "baseline", "chosen_thresh": config.FIXED_DNBR_THRESHOLD,
                **m2,
            })

    return results


# ── Gate checks ──────────────────────────────────────────────────────

def run_gate_checks(all_results, baseline_results):
    """Phase 3 gate checks."""
    print("\n" + "=" * 60)
    print("PHASE 3 GATE CHECKS")
    print("=" * 60)

    all_pass = True
    clf_results = [r for r in all_results
                   if not r.get("classifier", "").startswith("baseline_")
                   and "error" not in r]

    configs = set()
    rows_per = {}
    for r in clf_results:
        key = (r["recipe_id"], r["set_id"], r["classifier"])
        configs.add(key)
        rows_per[key] = rows_per.get(key, 0) + 1

    incomplete = {k: v for k, v in rows_per.items()
                  if v < config.GATE_CV_ROWS_PER_CONFIG}
    g1 = len(incomplete) == 0
    print(f"\n  1) Completeness: {len(clf_results)} rows, {len(configs)} configs")
    if incomplete:
        for k, v in incomplete.items():
            print(f"     {k}: {v}/{config.GATE_CV_ROWS_PER_CONFIG}")
    print(f"     {'PASS' if g1 else 'FAIL'}")
    if not g1:
        all_pass = False

    errors = [r for r in all_results if "error" in r]
    g2 = len(errors) == 0
    print(f"\n  2) Errors: {len(errors)}  {'PASS' if g2 else 'FAIL'}")
    if not g2:
        all_pass = False

    bl_types = set(r["classifier"] for r in baseline_results)
    expected = {"baseline_recipe_rule", "baseline_fixed_dnbr"}
    g3 = expected.issubset(bl_types)
    print(f"\n  3) Baselines: {bl_types}  {'PASS' if g3 else 'FAIL'}")
    if not g3:
        all_pass = False

    if clf_results:
        kappas = [r["kappa"] for r in clf_results if "kappa" in r]
        if kappas:
            print(f"\n  Summary: mean kappa = {sum(kappas)/len(kappas):.3f}, "
                  f"max = {max(kappas):.3f}")
            by_clf = {}
            for r in clf_results:
                by_clf.setdefault(r["classifier"], []).append(r["kappa"])
            for c, ks in sorted(by_clf.items()):
                print(f"    {c}: mean={sum(ks)/len(ks):.3f}, max={max(ks):.3f}, n={len(ks)}")

    print(f"\n{'=' * 60}")
    print(f"  OVERALL: {'ALL PASS' if all_pass else 'ISSUES'}")
    print(f"{'=' * 60}\n")
    return {"pass": all_pass}


# ── Results I/O ──────────────────────────────────────────────────────

def write_csv(results: list[dict], path: str):
    """Write results to CSV."""
    if not results:
        return
    all_keys = []
    seen = set()
    for r in results:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for row in results:
            clean = {k: (-999 if isinstance(row.get(k), float) and row.get(k) != row.get(k)
                         else row.get(k, ""))
                     for k in all_keys}
            w.writerow(clean)
    print(f"  Written: {path} ({len(results)} rows)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    year = args.year

    sample_dir = args.sample_dir or "data/samples"
    out_dir = args.out_dir or f"outputs/{year}/phase3_results"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Phase 3 CV · Punjab {year}")
    print(f"Sample dir:   {sample_dir}")
    print(f"Output dir:   {out_dir}")
    print(f"Recipes:      {args.recipes}")
    print(f"Feature sets: {args.feature_sets}")
    print(f"Classifiers:  {args.classifiers}\n")

    if not os.path.isdir(sample_dir):
        print(f"ERROR: {sample_dir} not found.")
        print(f"Download CSVs from Google Drive first.")
        return

    band_lists = _build_band_lists()
    selected_recipes = [recipe_mod.get_recipe(r) for r in args.recipes]

    all_cv_results = []
    all_baseline_results = []
    all_prob_records = []

    total = (len(args.recipes) * len(args.feature_sets)
             * len(args.classifiers) * len(config.SEEDS) * config.K_FOLDS)
    run_count = 0

    for recipe in selected_recipes:
        rid = recipe["recipe_id"]

        for seed in config.SEEDS:
            # Load full-Punjab CSV once per (recipe × seed)
            csv_path = _find_full_csv(sample_dir, rid, seed)
            if not csv_path:
                print(f"  MISSING: {rid} / s{seed} full CSV — skipping")
                missing_runs = len(args.feature_sets) * len(args.classifiers) * config.K_FOLDS
                run_count += missing_runs
                for _ in range(missing_runs):
                    all_cv_results.append({
                        "recipe_id": rid, "set_id": "?", "classifier": "?",
                        "seed": seed, "fold_id": -1, "error": "Missing CSV",
                    })
                continue

            print(f"\n  Loading {rid} / s{seed} …")
            all_rows = _load_csv(csv_path)
            print(f"    {len(all_rows)} rows loaded")

            # Get adm2 codes from data
            adm2_codes = sorted(set(
                int(r.get("adm2", -1)) for r in all_rows
                if r.get("adm2") is not None and int(r.get("adm2", -1)) >= 0
            ))
            print(f"    {len(adm2_codes)} districts")

            # Load baseline data once
            dnbr_rows = None
            if not args.skip_baselines:
                dnbr_path = _find_baseline_csv(sample_dir, rid, "dnbr", seed)
                if dnbr_path:
                    dnbr_rows = _load_csv(dnbr_path)

            # Assign adm2 codes to folds
            adm2_fold_map = _assign_adm2_folds(adm2_codes, config.K_FOLDS, seed)

            for fold_id in range(config.K_FOLDS):
                # Split rows by fold
                test_adm2s = {c for c, f in adm2_fold_map.items() if f == fold_id}
                train_adm2s = {c for c, f in adm2_fold_map.items() if f != fold_id}

                train_rows = [r for r in all_rows if int(r.get("adm2", -1)) in train_adm2s]
                test_rows = [r for r in all_rows if int(r.get("adm2", -1)) in test_adm2s]

                # Baselines (once per fold, not per feature set)
                if not args.skip_baselines and dnbr_rows:
                    baselines = compute_baselines(
                        dnbr_rows, test_adm2s,
                        recipe, rid, seed, fold_id,
                    )
                    all_baseline_results.extend(baselines)

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
                                auc = result.get("auc", 0)
                                print(f"    [{run_count}/{total}] {set_id}/{clf_name}"
                                      f"/f{fold_id}: k={k:.3f} f1={f1:.3f} auc={auc:.3f}")
                            else:
                                print(f"    [{run_count}/{total}] {set_id}/{clf_name}"
                                      f"/f{fold_id}: {result.get('error','?')}")

                            all_cv_results.append(result)

                        except Exception as e:
                            print(f"    [{run_count}/{total}] {set_id}/{clf_name}"
                                  f"/f{fold_id}: ERROR {e}")
                            all_cv_results.append({
                                "recipe_id": rid, "set_id": set_id,
                                "classifier": clf_name, "seed": seed,
                                "fold_id": fold_id, "error": str(e),
                            })

    # ── Write results ────────────────────────────────────────────
    print(f"\nWriting results to {out_dir} …")
    combined = all_cv_results + all_baseline_results
    write_csv(combined, os.path.join(out_dir, "cv_metrics.csv"))

    if all_prob_records:
        write_csv(all_prob_records, os.path.join(out_dir, "test_probabilities.csv"))

    # ── Gate checks ──────────────────────────────────────────────
    run_gate_checks(
        [r for r in all_cv_results if "error" not in r],
        all_baseline_results,
    )

    print("Phase 3 CV done.\n")


if __name__ == "__main__":
    main()