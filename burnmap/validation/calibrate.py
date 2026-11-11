"""
Phase 5 — Threshold calibration sweep
=====================================

Sweeps one or more probability thresholds over the exported Phase 5 map for the
selected config and evaluates each threshold against:
  - MODIS district-level comparison
  - MODIS pixel-level agreement
  - CAQM district fire-count concordance (optional but recommended)

Outputs:
  - outputs/<year>/phase5_results/threshold_calibration_summary.csv
  - outputs/<year>/phase5_results/threshold_calibration_best.txt
  - optional per-threshold district tables under
    outputs/<year>/phase5_results/threshold_calibration_details/

Usage
-----
    python scripts/calibrate.py --year 2024 --caqm_csv data/external/fire_counts.csv
    python scripts/calibrate.py --year 2024 --thresholds 0.3 0.35 0.4 0.45 0.5
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import shutil
from typing import Any

import burnmap.config as config
import burnmap.validation.validate as p5v


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 — Threshold calibration sweep")
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
                   help="CAQM/IARI district fire count CSV (district, fire_count)")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Directory for threshold calibration outputs")
    p.add_argument("--thresholds", nargs="+", type=float, default=None,
                   help="Explicit thresholds to test")
    p.add_argument("--step", type=float, default=0.05,
                   help="Step size for auto-generated threshold sweep")
    p.add_argument("--min_threshold", type=float, default=0.10)
    p.add_argument("--max_threshold", type=float, default=0.90)
    p.add_argument("--write_details", action="store_true",
                   help="Write per-threshold district comparison tables")
    return p.parse_args()


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


def _threshold_list(args, selected_rows):
    if args.thresholds:
        vals = sorted({round(float(t), 4) for t in args.thresholds})
        return vals
    vals = []
    t = float(args.min_threshold)
    # robust float stepping
    while t <= float(args.max_threshold) + 1e-9:
        vals.append(round(t, 4))
        t += float(args.step)
    cv_t = p5v._phase5_threshold(selected_rows, None)
    if round(cv_t, 4) not in vals:
        vals.append(round(cv_t, 4))
    return sorted(set(vals))


def _score_row(r: dict[str, Any]) -> tuple:
    """Ranking key for recommendation.

    Prefer thresholds that pass the area-ratio gate first, then maximize district
    R^2, then Spearman rho, then pixel F1. If none pass the area-ratio gate,
    prefer ratios closer to 1 on a log scale, then higher R^2, rho, and F1.
    """
    area_ratio = float(r.get("burned_area_ratio", float("inf")))
    r2 = float(r.get("district_r2", 0.0))
    rho = float(r.get("spearman_rho", -1.0))
    f1 = float(r.get("pixel_f1", 0.0))
    pass_area = bool(r.get("pass_area_ratio", False))
    pass_r2 = bool(r.get("pass_district_r2", False))
    pass_rho = bool(r.get("pass_spearman", False))
    gate_count = int(pass_area) + int(pass_r2) + int(pass_rho)
    log_dist = abs(math.log(area_ratio, 2)) if area_ratio > 0 else float("inf")
    return (gate_count, int(pass_area), -log_dist, r2, rho, f1, -float(r.get("threshold", 0)))


def main():
    args = parse_args()
    year = args.year

    map_dir = args.map_dir or f"outputs/{year}/phase5_maps"
    out_dir = args.out_dir or f"outputs/{year}/phase5_results"
    os.makedirs(out_dir, exist_ok=True)
    details_dir = os.path.join(out_dir, "threshold_calibration_details")
    os.makedirs(details_dir, exist_ok=True)

    cv_path = args.cv_metrics or p5v._find_first([
        f"outputs/{year}/combined_results/all_cv_metrics.csv",
        f"outputs/{year}/phase4_results/combined_cv_metrics.csv",
        f"outputs/{year}/phase3_results/cv_metrics.csv",
    ])
    if not cv_path:
        raise SystemExit("CV metrics CSV not found.")

    clf_rows, selected_rows = p5v._selected_cv_rows(cv_path)
    if not selected_rows:
        raise SystemExit(
            f"Selected config rows not found for {config.BEST_RECIPE}/{config.BEST_CLASSIFIER}/{config.BEST_FEATURE_SET}."
        )
    cv_selected_thr = p5v._phase5_threshold(selected_rows, None)

    prob_tif = p5v._find_first([
        os.path.join(map_dir, f"*{config.BEST_RECIPE}*_{config.BEST_FEATURE_SET}_prob*.tif"),
        os.path.join(map_dir, f"*{config.BEST_RECIPE}*prob*.tif"),
    ])
    if prob_tif and "*" in prob_tif:
        prob_tif = None
    if not prob_tif:
        prob_matches = glob.glob(os.path.join(map_dir, f"*{config.BEST_RECIPE}*prob*.tif"))
        prob_tif = prob_matches[0] if prob_matches else None

    modis_tif = p5v._find_first([
        os.path.join(map_dir, "*MODIS*.tif"),
    ])
    if modis_tif and "*" in modis_tif:
        modis_tif = None
    if not modis_tif:
        modis_matches = glob.glob(os.path.join(map_dir, "*MODIS*.tif"))
        modis_tif = modis_matches[0] if modis_matches else None

    adm2_tif = args.adm2_tif or None
    if not adm2_tif:
        adm2_matches = glob.glob(os.path.join(map_dir, "*adm2*.tif"))
        adm2_tif = adm2_matches[0] if adm2_matches else None

    district_lookup_path = args.district_lookup or None
    if not district_lookup_path:
        lookup_matches = glob.glob(os.path.join(map_dir, "*district_lookup*.csv"))
        district_lookup_path = lookup_matches[0] if lookup_matches else None

    if not prob_tif or not modis_tif or not adm2_tif:
        raise SystemExit("Required Phase 5 rasters not found. Need probability map, MODIS burn raster, and ADM2 raster.")

    district_lookup, _ = p5v._load_district_lookup(district_lookup_path) if district_lookup_path else ({}, [])
    thresholds = _threshold_list(args, selected_rows)

    print(f"Phase 5 threshold calibration · Punjab {year}")
    print(f"Selected config: {config.BEST_RECIPE}/{config.BEST_CLASSIFIER}/{config.BEST_FEATURE_SET}")
    print(f"CV-selected threshold: {cv_selected_thr:.3f}")
    print(f"Testing thresholds: {', '.join(f'{t:.2f}' for t in thresholds)}")
    print(f"Map dir:   {map_dir}")
    print(f"Out dir:   {out_dir}\n")

    results = []
    for thr in thresholds:
        print(f"-- Threshold {thr:.3f} --")
        res = p5v.modis_district_comparison(
            prob_tif=prob_tif,
            modis_tif=modis_tif,
            adm2_tif=adm2_tif,
            threshold=thr,
            out_dir=out_dir,
            district_lookup=district_lookup,
        )
        if not res:
            continue
        fire = p5v.fire_count_concordance(args.caqm_csv, res.get("district_rows"), out_dir)

        # Preserve per-threshold outputs written by phase5_validate helpers
        suffix = str(thr).replace(".", "p")
        for base_name in ["district_modis_comparison.csv", "pixel_modis_agreement.csv", "district_fire_count_join.csv"]:
            src = os.path.join(out_dir, base_name)
            if os.path.exists(src):
                stem, ext = os.path.splitext(base_name)
                dst = os.path.join(details_dir, f"{stem}_threshold_{suffix}{ext}")
                shutil.copyfile(src, dst)

        pixel = res["pixel_metrics"]
        row = {
            "threshold": thr,
            "district_r2": float(res.get("district_r2", 0.0)),
            "burned_area_ratio": float(res.get("burned_area_ratio", float("inf"))),
            "pixel_kappa": float(pixel.get("kappa", 0.0)),
            "pixel_f1": float(pixel.get("f1", 0.0)),
            "pixel_precision": float(pixel.get("precision", 0.0)),
            "pixel_recall": float(pixel.get("recall", 0.0)),
            "spearman_rho": float(fire.get("rho", float("nan"))) if fire else float("nan"),
            "matched_districts": int(fire.get("n_matched", 0)) if fire else 0,
        }
        row["pass_district_r2"] = row["district_r2"] >= config.GATE_MODIS_DISTRICT_R2
        row["pass_area_ratio"] = row["burned_area_ratio"] <= config.GATE_BURNED_AREA_RATIO
        row["pass_spearman"] = (not math.isnan(row["spearman_rho"])) and row["spearman_rho"] >= config.GATE_SPEARMAN_CAQM
        row["n_gates_passed"] = int(row["pass_district_r2"]) + int(row["pass_area_ratio"]) + int(row["pass_spearman"])
        results.append(row)

        if args.write_details and res.get("district_rows"):
            detail_rows = []
            for dr in res["district_rows"]:
                dr = dict(dr)
                dr["threshold"] = thr
                detail_rows.append(dr)
            detail_path = os.path.join(details_dir, f"district_modis_threshold_{str(thr).replace('.', 'p')}.csv")
            _write_csv(detail_rows, detail_path)
        print()

    if not results:
        raise SystemExit("No threshold results computed.")

    results_sorted = sorted(results, key=_score_row, reverse=True)
    summary_path = os.path.join(out_dir, "threshold_calibration_summary.csv")
    _write_csv(results_sorted, summary_path)

    best = results_sorted[0]
    note_lines = [
        f"Selected config: {config.BEST_RECIPE}/{config.BEST_CLASSIFIER}/{config.BEST_FEATURE_SET}",
        f"CV-selected threshold: {cv_selected_thr:.3f}",
        f"Recommended calibrated threshold: {best['threshold']:.3f}",
        "",
        f"district_r2 = {best['district_r2']:.3f}",
        f"burned_area_ratio = {best['burned_area_ratio']:.3f}",
        f"pixel_kappa = {best['pixel_kappa']:.3f}",
        f"pixel_f1 = {best['pixel_f1']:.3f}",
        f"spearman_rho = {best['spearman_rho']:.3f}" if not math.isnan(best['spearman_rho']) else "spearman_rho = nan",
        f"gates_passed = {best['n_gates_passed']}/3",
        f"pass_district_r2 = {best['pass_district_r2']}",
        f"pass_area_ratio = {best['pass_area_ratio']}",
        f"pass_spearman = {best['pass_spearman']}",
    ]
    best_txt = os.path.join(out_dir, "threshold_calibration_best.txt")
    with open(best_txt, "w") as f:
        f.write("\n".join(note_lines) + "\n")
    print(f"  Written: {best_txt}")

    print("\nTop threshold candidates:")
    print(f"  {'thr':>5s} {'R2':>6s} {'ratio':>8s} {'rho':>6s} {'pxF1':>6s} {'gates':>5s}")
    print(f"  {'-'*42}")
    for r in results_sorted[:10]:
        rho_txt = "nan" if math.isnan(r['spearman_rho']) else f"{r['spearman_rho']:.3f}"
        print(f"  {r['threshold']:5.2f} {r['district_r2']:6.3f} {r['burned_area_ratio']:8.3f} {rho_txt:>6s} {r['pixel_f1']:6.3f} {int(r['n_gates_passed']):5d}")

    print("\nRecommended threshold:")
    print(f"  {best['threshold']:.3f}")
    print(f"  district_r2={best['district_r2']:.3f}, burned_area_ratio={best['burned_area_ratio']:.3f}, ")
    print(f"  pixel_f1={best['pixel_f1']:.3f}, spearman_rho={best['spearman_rho'] if not math.isnan(best['spearman_rho']) else 'nan'}")
    print()


if __name__ == "__main__":
    main()
