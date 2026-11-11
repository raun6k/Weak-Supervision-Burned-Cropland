"""
Phase 5 — Step 2: Independent Validation Analysis
===================================================

Runs entirely locally on exported GeoTIFFs and CSVs:

  5.1 MODIS MCD64A1 comparison (district-level R², pixel-level agreement)
  5.2 Government fire-count concordance (Spearman ρ)
  5.5 Per-district error analysis (from CV metrics)

Expects:
  - Probability GeoTIFFs in outputs/<year>/phase5_maps/
  - MODIS burn mask GeoTIFF
  - ADM2 district raster GeoTIFF
  - District lookup CSV (adm2_code,district)
  - CV metrics CSV from combined Phase 3/4 results
  - District fire-count CSV (district, fire_count)
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
from typing import Any

import burnmap.config as config
import burnmap.modeling.metrics as met


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 — Validation Analysis")
    p.add_argument("--year", type=int, default=2024, choices=[2023, 2024])
    p.add_argument("--cv_metrics", type=str, default=None,
                   help="Path to combined CV metrics CSV")
    p.add_argument("--map_dir", type=str, default=None,
                   help="Directory with probability GeoTIFFs")
    p.add_argument("--adm2_tif", type=str, default=None,
                   help="ADM2 raster GeoTIFF aligned to probability map")
    p.add_argument("--district_lookup", type=str, default=None,
                   help="District lookup CSV with columns adm2_code,district")
    p.add_argument("--caqm_csv", type=str, default="data/external/fire_counts.csv",
                   help="CAQM/IARI district fire count CSV (columns: district, fire_count)")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="Override raster threshold. Default: median chosen_thresh of best config")
    p.add_argument("--skip_raster", action="store_true",
                   help="Skip raster-based analysis (MODIS pixel comparison)")
    return p.parse_args()


# ── CSV helpers ──────────────────────────────────────────────────────

def _load_csv(path: str) -> list[dict[str, Any]]:
    with open(path, newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            clean = {}
            for k, v in row.items():
                if v is None or v == "":
                    clean[k] = ""
                    continue
                try:
                    if isinstance(v, str) and v.strip().isdigit():
                        clean[k] = int(v)
                    else:
                        clean[k] = float(v)
                except (ValueError, TypeError):
                    clean[k] = v
            rows.append(clean)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: str):
    if not rows:
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {path} ({len(rows)} rows)")


# ── General helpers ──────────────────────────────────────────────────

def _find_first(candidates: list[str]) -> str | None:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _normalize_name(name: str) -> str:
    s = str(name or "").strip().lower().replace("&", "and")
    for ch in ["-", ".", ",", "'", "(", ")"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    alias = {
        "sahibzada ajit singh nagar": "sas nagar",
        "sas nagar": "sas nagar",
        "shahid bhagat singh nagar": "sbs nagar",
        "nawan shehar": "sbs nagar",
        "nawanshahr": "sbs nagar",
        "sbs nagar": "sbs nagar",
        "sri muktsar sahib": "muktsar",
        "muktsar": "muktsar",
        "rup nagar": "rupnagar",
        "ropar": "rupnagar",
        "rupnagar": "rupnagar",
        "taran taran": "tarn taran",
        "tarn taran": "tarn taran",
        "fatehgarh sahib": "fatehgarh sahib",
    }
    return alias.get(s, s)


def _load_district_lookup(path: str | None) -> tuple[dict[int, str], list[int]]:
    if not path or not os.path.exists(path):
        return {}, []
    rows = _load_csv(path)
    code_to_name = {}
    for r in rows:
        code = r.get("adm2_code", r.get("ADM2_CODE", r.get("adm2", "")))
        name = r.get("district", r.get("ADM2_NAME", r.get("name", "")))
        if code != "" and name:
            code_to_name[int(code)] = str(name)
    return code_to_name, sorted(code_to_name.keys())


def _selected_cv_rows(cv_metrics_path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _load_csv(cv_metrics_path)
    clf_rows = [
        r for r in rows
        if not str(r.get("classifier", "")).startswith("baseline_") and "error" not in r
    ]
    selected = [
        r for r in clf_rows
        if r.get("recipe_id") == config.BEST_RECIPE
        and r.get("classifier") == config.BEST_CLASSIFIER
        and r.get("set_id") == config.BEST_FEATURE_SET
    ]
    return clf_rows, selected


def _phase5_threshold(selected_rows: list[dict[str, Any]], override: float | None) -> float:
    if override is not None:
        return float(override)
    vals = [float(r.get("chosen_thresh", 0.5)) for r in selected_rows if r.get("chosen_thresh", "") != ""]
    if not vals:
        return 0.5
    return float(statistics.median(vals))


def _r2(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxx = sum((v - mx) ** 2 for v in x)
    syy = sum((v - my) ** 2 for v in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    if sxx <= 0 or syy <= 0:
        return 0.0
    r = sxy / math.sqrt(sxx * syy)
    return r * r


# ── 5.1 MODIS District-Level Comparison (from raster) ────────────────

def modis_district_comparison(prob_tif: str, modis_tif: str, adm2_tif: str,
                              threshold: float, out_dir: str,
                              district_lookup: dict[int, str] | None = None):
    """Compare thresholded Phase 5 map against MODIS at MODIS native grid."""
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import reproject, Resampling
    except ImportError:
        print("  WARNING: rasterio/numpy not installed, skipping raster analysis")
        print("  Install with: pip install rasterio numpy")
        return None

    for path in [prob_tif, modis_tif, adm2_tif]:
        if not path or not os.path.exists(path):
            print(f"  WARNING: required raster not found: {path}")
            return None

    print(f"  Loading probability map: {prob_tif}")
    print(f"  Loading MODIS mask: {modis_tif}")
    print(f"  Loading ADM2 raster: {adm2_tif}")
    print(f"  Threshold (selected): {threshold:.3f}")

    with rasterio.open(prob_tif) as src_prob, rasterio.open(modis_tif) as src_mod, rasterio.open(adm2_tif) as src_adm2:
        prob = src_prob.read(1, masked=True)
        modis = src_mod.read(1, masked=True)

        src_bin = prob.astype("float32").filled(-9999.0)
        src_bin = np.where(src_bin >= 0, (src_bin >= threshold).astype("float32"), -9999.0)
        prob_on_modis = np.full((src_mod.height, src_mod.width), -9999.0, dtype="float32")
        reproject(
            source=src_bin,
            destination=prob_on_modis,
            src_transform=src_prob.transform,
            src_crs=src_prob.crs,
            src_nodata=-9999.0,
            dst_transform=src_mod.transform,
            dst_crs=src_mod.crs,
            dst_nodata=-9999.0,
            resampling=Resampling.average,
        )
        ours = np.where(prob_on_modis == -9999.0, -9999.0, (prob_on_modis >= 0.5).astype("uint8"))

        mod_arr = modis.astype("float32").filled(-9999.0)
        mod_bin = np.where(mod_arr == -9999.0, -9999.0, (mod_arr > 0).astype("uint8"))

        adm2_on_modis = np.full((src_mod.height, src_mod.width), -1, dtype="int32")
        reproject(
            source=src_adm2.read(1),
            destination=adm2_on_modis,
            src_transform=src_adm2.transform,
            src_crs=src_adm2.crs,
            dst_transform=src_mod.transform,
            dst_crs=src_mod.crs,
            dst_nodata=-1,
            resampling=Resampling.nearest,
        )

        valid = (ours != -9999.0) & (mod_bin != -9999.0) & (adm2_on_modis > 0)
        if not np.any(valid):
            print("  WARNING: no valid overlap pixels for MODIS comparison")
            return None

        y_true = mod_bin[valid].astype(int).tolist()
        y_pred = ours[valid].astype(int).tolist()
        pixel_metrics = met.compute_basic_metrics(y_true, y_pred)

        pixel_area_km2 = abs(src_mod.transform.a * src_mod.transform.e) / 1_000_000.0
        overall_ours_area = float(np.sum(ours[valid] == 1) * pixel_area_km2)
        overall_modis_area = float(np.sum(mod_bin[valid] == 1) * pixel_area_km2)
        area_ratio = overall_ours_area / overall_modis_area if overall_modis_area > 0 else float("inf")

        rows = []
        district_codes = sorted(int(c) for c in np.unique(adm2_on_modis[valid]) if int(c) > 0)
        for code in district_codes:
            mask = valid & (adm2_on_modis == code)
            if not np.any(mask):
                continue
            ours_count = int(np.sum(ours[mask] == 1))
            modis_count = int(np.sum(mod_bin[mask] == 1))
            rows.append({
                "adm2_code": code,
                "district": (district_lookup or {}).get(code, str(code)),
                "ours_burn_pixels_500m": ours_count,
                "modis_burn_pixels_500m": modis_count,
                "ours_burn_area_km2": ours_count * pixel_area_km2,
                "modis_burn_area_km2": modis_count * pixel_area_km2,
                "pixel_count_500m": int(np.sum(mask)),
            })

    district_r2 = _r2(
        [float(r["ours_burn_area_km2"]) for r in rows],
        [float(r["modis_burn_area_km2"]) for r in rows],
    ) if rows else 0.0

    _write_csv(rows, os.path.join(out_dir, "district_modis_comparison.csv"))
    _write_csv([{
        **pixel_metrics,
        "threshold": threshold,
        "district_r2": district_r2,
        "overall_ours_burn_area_km2": overall_ours_area,
        "overall_modis_burn_area_km2": overall_modis_area,
        "burned_area_ratio": area_ratio,
        "n_valid_modis_pixels": int(sum(valid.ravel())),
    }], os.path.join(out_dir, "pixel_modis_agreement.csv"))

    print(f"  District R² vs MODIS: {district_r2:.3f}")
    print(f"  Burned area ratio (ours / MODIS): {area_ratio:.3f}")
    print(f"  Pixel agreement: kappa={pixel_metrics['kappa']:.3f}, F1={pixel_metrics['f1']:.3f}")

    return {
        "district_r2": district_r2,
        "burned_area_ratio": area_ratio,
        "pixel_metrics": pixel_metrics,
        "district_rows": rows,
        "threshold": threshold,
    }


# ── 5.2 Government Fire-Count Concordance ────────────────────────────

def spearman_rank(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0

    def _rank(vals):
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = _rank(x), _rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1))


def fire_count_concordance(caqm_csv_path: str | None, district_rows: list[dict[str, Any]] | None,
                           out_dir: str):
    if not caqm_csv_path or not os.path.exists(caqm_csv_path):
        print("\n── 5.2 Fire-Count Concordance ──")
        print("  CAQM fire count CSV not provided. Skipping.")
        print("  To run this: provide --caqm_csv with columns: district, fire_count")
        return None
    if not district_rows:
        print("\n── 5.2 Fire-Count Concordance ──")
        print("  District burned-area table not available yet. Run raster MODIS comparison first.")
        return None

    print("\n── 5.2 Government Fire-Count Concordance ──")
    caqm = _load_csv(caqm_csv_path)
    product_map = {
        _normalize_name(r.get("district", "")): float(r.get("ours_burn_area_km2", 0.0))
        for r in district_rows
        if r.get("district", "")
    }

    joined = []
    for r in caqm:
        name = r.get("district", r.get("District", r.get("ADM2_NAME", "")))
        fire_count = r.get("fire_count", r.get("Fire_Count", r.get("count", "")))
        if not name or fire_count == "":
            continue
        norm = _normalize_name(str(name))
        if norm in product_map:
            joined.append({
                "district": str(name),
                "normalized_district": norm,
                "fire_count": float(fire_count),
                "ours_burn_area_km2": product_map[norm],
            })

    if len(joined) < 3:
        print("  WARNING: too few district matches between CAQM and product names")
        return None

    rho = spearman_rank(
        [float(r["ours_burn_area_km2"]) for r in joined],
        [float(r["fire_count"]) for r in joined],
    )
    _write_csv(joined, os.path.join(out_dir, "district_fire_count_join.csv"))
    print(f"  Matched districts: {len(joined)}")
    print(f"  Spearman rho: {rho:.3f}")
    return {"rho": rho, "n_matched": len(joined), "rows": joined}


# ── 5.4 Visual validation ────────────────────────────────────────────


# ── 5.5 Per-District Error Analysis ──────────────────────────────────

def per_district_analysis(cv_metrics_path: str, out_dir: str,
                          district_lookup_path: str | None = None):
    print("\n── 5.5 Per-District Error Analysis ──")
    if not os.path.exists(cv_metrics_path):
        print(f"  WARNING: {cv_metrics_path} not found")
        return None

    clf_rows, selected = _selected_cv_rows(cv_metrics_path)
    if not clf_rows:
        print("  No classifier results found")
        return None

    if not selected:
        selected = clf_rows
        print("  WARNING: best config rows not found; falling back to all classifier rows")

    code_to_name, district_codes = _load_district_lookup(district_lookup_path)
    print(f"  Using {len(selected)} results from {config.BEST_RECIPE}/{config.BEST_CLASSIFIER}/{config.BEST_FEATURE_SET}")

    fold_rows = []
    if district_codes:
        import random
        for row in sorted(selected, key=lambda r: (int(r.get("seed", -1)), int(r.get("fold_id", -1)))):
            seed = int(row.get("seed", -1))
            fold_id = int(row.get("fold_id", -1))
            codes_sorted = sorted(district_codes)
            shuffled = codes_sorted.copy()
            rng = random.Random(seed)
            rng.shuffle(shuffled)
            fold_map = {code: i % config.K_FOLDS for i, code in enumerate(shuffled)}
            test_codes = sorted([c for c, f in fold_map.items() if f == fold_id])
            test_names = [code_to_name.get(c, str(c)) for c in test_codes]
            fold_rows.append({
                "seed": seed,
                "fold_id": fold_id,
                "mean_kappa": float(row.get("kappa", 0.0)),
                "mean_n_test": int(row.get("n_test", 0)),
                "test_districts": "; ".join(test_names),
                "n_configs": 1,
            })
    else:
        for row in selected:
            fold_rows.append({
                "seed": int(row.get("seed", -1)),
                "fold_id": int(row.get("fold_id", -1)),
                "mean_kappa": float(row.get("kappa", 0.0)),
                "mean_n_test": int(row.get("n_test", 0)),
                "test_districts": "",
                "n_configs": 1,
            })

    print(f"\n  {'Seed':>6s} {'Fold':>4s} {'Kappa':>7s} {'N_test':>7s} {'Test Districts'}")
    print(f"  {'-'*80}")
    for r in fold_rows:
        print(f"  {str(r['seed']):>6s} {str(r['fold_id']):>4s} {r['mean_kappa']:7.3f} {r['mean_n_test']:7.0f}   {r['test_districts']}")

    if fold_rows:
        kappas = [r["mean_kappa"] for r in fold_rows]
        mean_all = sum(kappas) / len(kappas)
        std_all = (sum((k - mean_all) ** 2 for k in kappas) / len(kappas)) ** 0.5
        print(f"\n  Overall: mean={mean_all:.3f} ±{std_all:.3f}, range=[{min(kappas):.3f}, {max(kappas):.3f}]")

    _write_csv(fold_rows, os.path.join(out_dir, "per_fold_kappa.csv"))

    print("\n  ── Classifier × Recipe × Feature Set summary ──")
    summary_rows = []
    classifiers = sorted(set(r.get("classifier") for r in clf_rows))
    for recipe in ["strict", "moderate", "very_strict"]:
        for clf in classifiers:
            for sid in ["A", "B", "C", "D"]:
                subset = [
                    r for r in clf_rows
                    if r.get("recipe_id") == recipe
                    and r.get("classifier") == clf
                    and r.get("set_id") == sid
                ]
                if not subset:
                    continue
                ks = [float(r["kappa"]) for r in subset]
                aucs = [float(r.get("auc", 0)) for r in subset]
                f1s = [float(r.get("f1", 0)) for r in subset]
                bas = [float(r.get("balanced_accuracy", 0)) for r in subset]
                summary_rows.append({
                    "recipe": recipe,
                    "classifier": clf,
                    "feature_set": sid,
                    "mean_kappa": sum(ks) / len(ks),
                    "std_kappa": (sum((k - sum(ks) / len(ks)) ** 2 for k in ks) / len(ks)) ** 0.5,
                    "mean_auc": sum(aucs) / len(aucs) if aucs else 0,
                    "mean_f1": sum(f1s) / len(f1s) if f1s else 0,
                    "mean_bal_acc": sum(bas) / len(bas) if bas else 0,
                    "n": len(ks),
                })

    _write_csv(summary_rows, os.path.join(out_dir, "classifier_summary_table.csv"))
    summary_rows.sort(key=lambda r: -r["mean_kappa"])
    print(f"\n  Top 10 configurations:")
    print(f"  {'Recipe':12s} {'Clf':8s} {'Set':3s} {'Kappa':>7s} {'±':>5s} {'AUC':>6s} {'F1':>6s}")
    print(f"  {'-'*50}")
    for r in summary_rows[:10]:
        print(
            f"  {r['recipe']:12s} {r['classifier']:8s} {r['feature_set']:3s} "
            f"{r['mean_kappa']:7.3f} {r['std_kappa']:5.3f} "
            f"{r['mean_auc']:6.3f} {r['mean_f1']:6.3f}"
        )

    return fold_rows


# ── Gate checks ──────────────────────────────────────────────────────

def run_gate_checks(fold_rows, cv_metrics_path: str, raster_result=None,
                    fire_result=None):
    print("\n" + "=" * 60)
    print("PHASE 5 GATE CHECKS")
    print("=" * 60)

    if raster_result:
        r2_ok = raster_result["district_r2"] >= config.GATE_MODIS_DISTRICT_R2
        ratio_ok = raster_result["burned_area_ratio"] <= config.GATE_BURNED_AREA_RATIO
        print(f"\n  1) District R² vs MODIS: {raster_result['district_r2']:.3f}  {'PASS' if r2_ok else 'FAIL'}")
        print(f"  2) Burned area ratio: {raster_result['burned_area_ratio']:.3f}  {'PASS' if ratio_ok else 'FAIL'}")
    else:
        print(f"\n  1) District R² vs MODIS: PENDING (need probability/MODIS/ADM2 GeoTIFFs)")
        print(f"  2) Burned area ratio: PENDING (need probability/MODIS/ADM2 GeoTIFFs)")

    if fire_result:
        rho = fire_result["rho"]
        print(f"  3) Spearman vs CAQM: {rho:.3f}  {'PASS' if rho >= config.GATE_SPEARMAN_CAQM else 'FAIL'}")
    else:
        print(f"  3) Spearman vs CAQM: PENDING (need CAQM fire count data)")

    if fold_rows:
        kappas = [r["mean_kappa"] for r in fold_rows]
        min_k, max_k = min(kappas), max(kappas)
        std_k = (sum((k - sum(kappas) / len(kappas)) ** 2 for k in kappas) / len(kappas)) ** 0.5
        print(f"\n  Per-fold selected-config kappa range: [{min_k:.3f}, {max_k:.3f}], std={std_k:.3f}")
        if min_k < 0.10:
            print("     WARNING: some folds have very low kappa — inspect sparse districts")

    if os.path.exists(cv_metrics_path):
        rows = _load_csv(cv_metrics_path)
        clf_rows = [r for r in rows if not str(r.get("classifier", "")).startswith("baseline_") and "error" not in r]
        if clf_rows:
            best_k = max(float(r.get("kappa", 0)) for r in clf_rows)
            best_auc = max(float(r.get("auc", 0)) for r in clf_rows)
            print(f"\n  Best single-run kappa across all configs: {best_k:.3f}")
            print(f"  Best single-run AUC across all configs: {best_auc:.3f}")

    print(f"\n{'=' * 60}\n")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    year = args.year
    out_dir = args.out_dir or f"outputs/{year}/phase5_results"
    os.makedirs(out_dir, exist_ok=True)

    cv_path = args.cv_metrics or _find_first([
        f"outputs/{year}/combined_results/all_cv_metrics.csv",
        f"outputs/{year}/phase4_results/combined_cv_metrics.csv",
        f"outputs/{year}/phase3_results/cv_metrics.csv",
    ])

    map_dir = args.map_dir or f"outputs/{year}/phase5_maps"
    prob_tif = _find_first(glob.glob(os.path.join(map_dir, f"*{config.BEST_RECIPE}*_{config.BEST_FEATURE_SET}_prob*.tif"))
                           or glob.glob(os.path.join(map_dir, f"*{config.BEST_RECIPE}*prob*.tif")))
    modis_tif = _find_first(glob.glob(os.path.join(map_dir, "*MODIS*burn*.tif")))
    adm2_tif = args.adm2_tif or _find_first(glob.glob(os.path.join(map_dir, "*adm2*.tif")))
    district_lookup_path = args.district_lookup or _find_first(glob.glob(os.path.join(map_dir, "*district_lookup*.csv")))

    print(f"Phase 5 Validation · Punjab {year}")
    print(f"CV metrics:  {cv_path or 'NOT FOUND'}")
    print(f"Map dir:     {map_dir}")
    print(f"Output dir:  {out_dir}\n")

    clf_rows, selected_rows = ([], [])
    if cv_path and os.path.exists(cv_path):
        clf_rows, selected_rows = _selected_cv_rows(cv_path)
    threshold = _phase5_threshold(selected_rows, args.threshold)
    print(f"Selected Phase 5 config: {config.BEST_RECIPE}/{config.BEST_CLASSIFIER}/{config.BEST_FEATURE_SET}")
    print(f"Selected threshold: {threshold:.3f}\n")

    fold_rows = None
    if cv_path and os.path.exists(cv_path):
        fold_rows = per_district_analysis(cv_path, out_dir, district_lookup_path)
    else:
        print("  CV metrics not found — run merge_phase4.py first")

    raster_result = None
    if not args.skip_raster:
        if prob_tif and modis_tif and adm2_tif:
            print("\n── 5.1 MODIS District-Level Comparison ──")
            lookup_map, _ = _load_district_lookup(district_lookup_path)
            raster_result = modis_district_comparison(prob_tif, modis_tif, adm2_tif,
                                                      threshold, out_dir, lookup_map)
        else:
            print("\n── 5.1 MODIS Comparison ──")
            print("  Missing one or more required rasters:")
            print(f"    probability: {prob_tif or 'NOT FOUND'}")
            print(f"    MODIS:       {modis_tif or 'NOT FOUND'}")
            print(f"    ADM2:        {adm2_tif or 'NOT FOUND'}")

    fire_result = fire_count_concordance(
        args.caqm_csv,
        raster_result["district_rows"] if raster_result else None,
        out_dir,
    )

    run_gate_checks(fold_rows, cv_path or "", raster_result, fire_result)
    print("Phase 5 validation done.\n")


if __name__ == "__main__":
    main()
