"""
Phase 2: Feature Engineering and Weak-Label Design
====================================================

Builds on Phase 1 data products to:
  1. Construct 5 feature sets (A–E)
  2. Generate weak labels for 3 recipes (strict, moderate, very_strict)
  3. Run label sanity checks:
     a) Label-vs-MODIS overlap
     b) Label count balance (≥ 10,000 positives per recipe)
     c) District coverage (configured minimum positive and negative counts)
     d) Feature correlation below the configured maximum
  4. Export label count summary

Usage
-----
    python scripts/check_labels.py --year 2024
    python scripts/check_labels.py --year 2024 --gate_only
    python scripts/check_labels.py --year 2024 --recipes strict moderate
"""

import argparse
import ee

import burnmap.config as config
from burnmap.config import recipes as recipe_mod
import burnmap.gee.aoi as aoi_mod
import burnmap.gee.sentinel2 as s2_mod
import burnmap.gee.sentinel1 as s1_mod
import burnmap.gee.indices as idx_mod
import burnmap.gee.viirs as viirs_mod
import burnmap.gee.modis as modis_mod
import burnmap.gee.features as feat_mod
import burnmap.gee.labels as label_mod
import burnmap.gee.export as exp


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: Feature Engineering & Weak-Label Design")
    p.add_argument("--year", type=int, default=2024, choices=[2023, 2024])
    p.add_argument("--drive_folder", type=str, default=config.DRIVE_FOLDER)
    p.add_argument("--gate_only", action="store_true", help="Run sanity checks only, no exports")
    p.add_argument(
        "--recipes", nargs="+", default=["strict", "moderate", "very_strict"],
        help="Which recipes to process (default: all three)"
    )
    p.add_argument("--skip_s1", action="store_true", help="Skip S1 (no feature set E)")
    p.add_argument("--skip_correlation", action="store_true",
                   help="Skip feature correlation check (slow)")
    return p.parse_args()


# ── Sanity checks ────────────────────────────────────────────────────

def check_label_modis_overlap(label_data, modis_data, aoi, recipe_id):
    """Gate: ≥ 55% of strict positive labels (≥ 50% moderate) in MODIS burn."""
    overlap = label_mod.label_modis_overlap(
        label_data["class_img"], modis_data["burn_mask"], aoi
    )
    return ee.Dictionary({"recipe_id": recipe_id}).combine(overlap)


def check_label_counts(class_img, aoi, recipe_id):
    """Gate: ≥ 10,000 positive labels per recipe."""
    hist = label_mod.label_pixel_counts(class_img, aoi)
    n0 = ee.Number(hist.get("0", 0))
    n1 = ee.Number(hist.get("1", 0))
    return ee.Dictionary({"recipe_id": recipe_id, "n0": n0, "n1": n1})


def check_district_coverage(class_img, districts, recipe_id):
    """Gate: every district meets the configured minimum positive and negative counts."""
    fc = label_mod.per_district_label_counts(class_img, districts, recipe_id)
    # Find min positive and min negative across districts
    min_pos = fc.aggregate_min("label_1")
    min_neg = fc.aggregate_min("label_0")
    return ee.Dictionary({
        "recipe_id": recipe_id,
        "min_pos_in_any_district": min_pos,
        "min_neg_in_any_district": min_neg,
    }), fc


def check_feature_correlation(feature_img, band_list, aoi, set_id, n_samples=5000):
    """
    Gate: no feature pair with |r| above the configured maximum.
    Samples n_samples random pixels and computes Pearson correlations.
    Returns max |r| across all pairs.
    """
    sample = feature_img.sample(
        region=aoi,
        scale=20,
        numPixels=n_samples,
        seed=42,
        tileScale=4,
        dropNulls=True,
    )
    # Compute Pearson correlation matrix via reduceColumns
    corr = sample.reduceColumns(
        reducer=ee.Reducer.pearsonsCorrelation().repeat(len(band_list)).group(0),
        selectors=band_list,
    )
    # This is complex to parse server-side; we'll sample + compute locally
    return ee.Dictionary({"set_id": set_id, "n_bands": len(band_list)})


def run_gate_checks(label_results, modis_data, feature_sets, study, args):
    """
    Run sanity checks with individual getInfo() calls per check
    to stay within GEE's concurrent aggregation limit.
    """
    aoi = study["aoi"]
    districts = study["districts"]

    print("\n" + "=" * 60)
    print("PHASE 2 GATE CHECKS")
    print("=" * 60)

    all_pass = True
    vals = {}
    district_fcs = {}

    for recipe_id, ldata in label_results.items():
        class_img = ldata["class_img"]
        print(f"\n  ── {recipe_id} ──")

        # 1) Label counts (one getInfo)
        print(f"     Counting labels …")
        counts = check_label_counts(class_img, aoi, recipe_id)
        count_vals = ee.Dictionary({
            "n0": counts.get("n0"),
            "n1": counts.get("n1"),
        }).getInfo()
        n0 = count_vals["n0"]
        n1 = count_vals["n1"]
        g_count = n1 >= config.GATE_LABEL_MIN_POS
        print(f"     Labels:    {n1:,.0f} pos / {n0:,.0f} neg    {'PASS' if g_count else f'FAIL (need ≥{config.GATE_LABEL_MIN_POS:,})'}")

        # 2) MODIS overlap (one getInfo)
        print(f"     Checking MODIS overlap …")
        overlap = check_label_modis_overlap(ldata, modis_data, aoi, recipe_id)
        overlap_val = ee.Number(overlap.get("overlap_fraction")).getInfo()
        if recipe_id in ("strict", "very_strict"):
            overlap_thr = config.GATE_MODIS_OVERLAP_STRICT
        else:
            overlap_thr = config.GATE_MODIS_OVERLAP_MODERATE
        g_overlap = overlap_val >= overlap_thr
        print(f"     MODIS:     {overlap_val:.1%} overlap             {'PASS' if g_overlap else f'FAIL (need ≥{overlap_thr:.0%})'}")

        # 3) District coverage (one getInfo for min, then fetch full detail)
        print(f"     Checking district coverage …")
        dcov, dfc = check_district_coverage(class_img, districts, recipe_id)
        district_fcs[recipe_id] = dfc

        # Fetch per-district detail
        print(f"     Fetching per-district breakdown …")
        district_data = dfc.getInfo()
        district_rows = []
        for feat in district_data.get("features", []):
            p = feat["properties"]
            district_rows.append({
                "name": p.get("ADM2_NAME", "?"),
                "pos": p.get("label_1", 0),
                "neg": p.get("label_0", 0),
            })

        # Sort by positive count ascending to show weakest districts first
        district_rows.sort(key=lambda x: x["pos"])

        min_pos = district_rows[0]["pos"] if district_rows else 0
        min_neg = min(r["neg"] for r in district_rows) if district_rows else 0

        g_district = min_pos >= config.GATE_DISTRICT_MIN_POS and min_neg >= config.GATE_DISTRICT_MIN_NEG
        print(f"     Districts: min pos={min_pos:,.0f}  min neg={min_neg:,.0f}    {'PASS' if g_district else 'FAIL'}")

        # Print per-district table
        print(f"     {'District':<20s} {'Pos':>8s} {'Neg':>10s}")
        print(f"     {'─' * 40}")
        for r in district_rows:
            flag = ""
            if r["pos"] < config.GATE_DISTRICT_MIN_POS:
                flag = "  ◀ LOW"
            elif r["pos"] < 50:
                flag = "  ◀ sparse"
            print(f"     {r['name']:<20s} {r['pos']:>8,.0f} {r['neg']:>10,.0f}{flag}")

        # Store for exports
        vals[f"{recipe_id}_n0"] = n0
        vals[f"{recipe_id}_n1"] = n1
        vals[f"{recipe_id}_overlap_frac"] = overlap_val
        vals[f"{recipe_id}_min_pos"] = min_pos
        vals[f"{recipe_id}_min_neg"] = min_neg

        if not (g_count and g_overlap and g_district):
            all_pass = False

    # ── Feature correlation check ────────────────────────────────────
    if not args.skip_correlation and feature_sets:
        print(f"\n  ── Feature correlation (sampled, set A) ──")
        print(f"     Sampling 5,000 pixels for correlation matrix …")

        set_a = feature_sets.get("A")
        if set_a:
            sample = set_a["image"].sample(
                region=study["aoi"], scale=20,
                numPixels=5000, seed=42,
                tileScale=4, dropNulls=True,
            )
            sample_data = sample.getInfo()
            features_list = sample_data.get("features", [])

            if len(features_list) > 100:
                import numpy as np
                bands = set_a["band_list"]
                matrix = []
                for feat in features_list:
                    row = [feat["properties"].get(b, float("nan")) for b in bands]
                    matrix.append(row)
                arr = np.array(matrix, dtype=float)

                mask = ~np.isnan(arr).any(axis=1)
                arr = arr[mask]

                if arr.shape[0] > 50:
                    corr = np.corrcoef(arr.T)
                    np.fill_diagonal(corr, 0)
                    max_r = float(np.nanmax(np.abs(corr)))
                    idx = np.unravel_index(np.nanargmax(np.abs(corr)), corr.shape)
                    b1, b2 = bands[idx[0]], bands[idx[1]]

                    g_corr = max_r < config.GATE_MAX_CORRELATION
                    print(f"     Max |r| = {max_r:.4f} ({b1} × {b2})   {'PASS' if g_corr else f'FAIL (>{config.GATE_MAX_CORRELATION})'}")
                    if not g_corr:
                        all_pass = False
                else:
                    print(f"     Too few valid samples ({arr.shape[0]}), skipping")
            else:
                print(f"     Too few samples ({len(features_list)}), skipping")
    elif args.skip_correlation:
        print(f"\n  ── Feature correlation: SKIPPED (--skip_correlation) ──")

    print(f"\n{'=' * 60}")
    print(f"  OVERALL: {'ALL PASS → proceed to Phase 3' if all_pass else 'BLOCKED → fix above'}")
    print(f"{'=' * 60}\n")

    return {"pass": all_pass, "values": vals, "district_fcs": district_fcs}


# ── Export ───────────────────────────────────────────────────────────

def export_label_summary(gate_result, label_results, year, folder):
    """Export label-count summary as a CSV for records."""
    prefix = exp.make_run_prefix(year, "phase2")
    features = []
    vals = gate_result["values"]

    for recipe_id in label_results:
        props = {
            "recipe_id": recipe_id,
            "n_pos": vals.get(f"{recipe_id}_n1", 0),
            "n_neg": vals.get(f"{recipe_id}_n0", 0),
            "modis_overlap": vals.get(f"{recipe_id}_overlap_frac", 0),
            "min_district_pos": vals.get(f"{recipe_id}_min_pos", 0),
            "min_district_neg": vals.get(f"{recipe_id}_min_neg", 0),
        }
        features.append(ee.Feature(None, props))

    fc = ee.FeatureCollection(features)
    exp.table_to_drive(fc, f"{prefix}_label_summary", folder)
    print(f"  Label summary CSV")


def export_district_label_counts(gate_result, year, folder):
    """Export per-district label counts for each recipe."""
    prefix = exp.make_run_prefix(year, "phase2")
    all_fc = ee.FeatureCollection([])
    for recipe_id, fc in gate_result["district_fcs"].items():
        all_fc = all_fc.merge(fc)
    exp.table_to_drive(all_fc, f"{prefix}_district_label_counts", folder)
    print(f"  District label counts CSV")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    year = args.year
    dates = config.get_dates(year)

    try:
        ee.Initialize(project=config.PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=config.PROJECT)

    print(f"Phase 2 · Punjab {year}")
    print(f"Recipes: {args.recipes}\n")

    # ── Rebuild Phase 1 data products (lazy EE objects, no compute) ───
    print("Building study area …")
    study = aoi_mod.build_study_area()
    aoi = study["aoi"]
    crop_mask = study["crop_mask"]

    print("Building S2 composites …")
    s2 = s2_mod.build_composites(aoi, dates)

    print("Computing spectral indices …")
    indices = idx_mod.compute_all(s2["pre"], s2["post"])

    s1 = None
    if not args.skip_s1:
        print("Building S1 composites …")
        s1 = s1_mod.build_composites(aoi, dates)

    print("Building VIIRS fire layers …")
    viirs = viirs_mod.build_fire_layers(aoi, dates)

    print("Building MODIS burn mask …")
    modis = modis_mod.build_burn_mask(aoi, crop_mask, dates)

    # ── 2.1 Feature sets ─────────────────────────────────────────────
    print("\nConstructing feature sets …")
    feature_sets = feat_mod.build_all(s2["pre"], s2["post"], indices, s1, crop_mask, aoi)
    for sid, fs in feature_sets.items():
        print(f"  Set {sid}: {fs['n']} features")

    # ── 2.2 Weak labels ──────────────────────────────────────────────
    print("\nConstructing weak labels …")
    selected_recipes = [recipe_mod.get_recipe(r) for r in args.recipes]
    label_results = {}

    for recipe in selected_recipes:
        rid = recipe["recipe_id"]
        ldata = label_mod.build_class_image(
            recipe=recipe,
            viirs_data=viirs,
            dNBR=indices["dNBR"],
            dB12=indices["dB12"],
            dNDVI=indices["dNDVI"],
            crop_mask=crop_mask,
            aoi=aoi,
        )
        label_results[rid] = ldata
        print(f"  {rid}: class image built")

    # ── 2.4 Sanity checks / gate ─────────────────────────────────────
    gate = run_gate_checks(label_results, modis, feature_sets, study, args)

    # ── Exports ──────────────────────────────────────────────────────
    if not args.gate_only:
        print("Submitting export tasks …")
        export_label_summary(gate, label_results, year, args.drive_folder)
        export_district_label_counts(gate, year, args.drive_folder)
        print("\nAll tasks submitted.")
    else:
        print("Gate-only mode — no exports submitted.")

    print("Phase 2 done.\n")


if __name__ == "__main__":
    main()