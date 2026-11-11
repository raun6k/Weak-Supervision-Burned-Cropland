"""
Phase 3 — Step 1: Export Samples (Optimized)
=============================================

Exports ONE full-Punjab stratified sample per (recipe × seed),
using feature set A (37 bands — superset of B, C, D). Fold
splitting and feature-set subsetting happen locally in phase3_cv.

This reduces the export to 18 tasks:
  - 9 sample exports  (3 recipes × 3 seeds)
  - 9 dNBR baselines  (3 recipes × 3 seeds)

MODIS is an external calibration reference, not a district-held-out
classifier baseline.

Usage
-----
    python scripts/export_samples.py --year 2024
    python scripts/export_samples.py --year 2024 --status
    python scripts/export_samples.py --year 2024 --recipes strict
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
import burnmap.gee.features as feat_mod
import burnmap.gee.labels as label_mod
import burnmap.gee.sampling as samp_mod
import burnmap.gee.export as exp


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 — Export CV Samples (Optimized)")
    p.add_argument("--year", type=int, default=2024, choices=[2023, 2024])
    p.add_argument("--drive_folder", type=str, default=config.DRIVE_FOLDER)
    p.add_argument("--recipes", nargs="+", default=["strict", "moderate", "very_strict"])
    p.add_argument("--skip_s1", action="store_true", help="Skip S1 (no set E)")
    p.add_argument("--include_e", action="store_true",
                   help="Also export set E (S1+S2) samples separately")
    p.add_argument("--status", action="store_true",
                   help="Check status of running tasks, don't submit new ones")
    return p.parse_args()


# ── Task status ──────────────────────────────────────────────────────

def check_task_status():
    """Print status of Phase 3 GEE tasks."""
    tasks = ee.data.getTaskList()
    phase3 = [t for t in tasks
              if "phase3" in t.get("description", "")]

    if not phase3:
        print("No Phase 3 tasks found.")
        return

    by_state = {}
    for t in phase3:
        state = t.get("state", "UNKNOWN")
        by_state.setdefault(state, []).append(t)

    total = len(phase3)
    completed = len(by_state.get("COMPLETED", []))
    print(f"\nPhase 3 tasks: {completed}/{total} completed")
    for state, tl in sorted(by_state.items()):
        print(f"  {state}: {len(tl)}")
        if state == "FAILED":
            for t in tl[:5]:
                print(f"    {t['description']}: {t.get('error_message', '?')}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    year = args.year
    dates = config.get_dates(year)
    folder = args.drive_folder
    prefix = exp.make_run_prefix(year, "phase3")

    try:
        ee.Initialize(project=config.PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=config.PROJECT)

    if args.status:
        check_task_status()
        return

    print(f"Phase 3 Export (Optimized) · Punjab {year}")
    print(f"Recipes:      {args.recipes}")
    print(f"Seeds:        {config.SEEDS}")
    print(f"Drive folder: {folder}")
    print(f"Strategy:     1 full-Punjab sample per (recipe × seed)")
    print(f"              Feature set A (superset), split locally\n")

    # ── Build lazy EE objects ────────────────────────────────────
    print("Building study area …")
    study = aoi_mod.build_study_area()
    aoi = study["aoi"]
    crop_mask = study["crop_mask"]

    print("Building S2 composites …")
    s2 = s2_mod.build_composites(aoi, dates)

    print("Computing spectral indices …")
    indices = idx_mod.compute_all(s2["pre"], s2["post"])

    s1 = None
    if not args.skip_s1 and args.include_e:
        print("Building S1 composites …")
        s1 = s1_mod.build_composites(aoi, dates)

    print("Building VIIRS fire layers …")
    viirs = viirs_mod.build_fire_layers(aoi, dates)

    # Build feature set A (superset of B, C, D)
    print("\nConstructing feature set A …")
    all_sets = feat_mod.build_all(s2["pre"], s2["post"], indices, s1, crop_mask, aoi)
    set_a = all_sets["A"]
    print(f"  Set A: {set_a['n']} features")

    # Weak labels
    print("\nConstructing weak labels …")
    selected_recipes = [recipe_mod.get_recipe(r) for r in args.recipes]
    label_results = {}
    for recipe in selected_recipes:
        rid = recipe["recipe_id"]
        ldata = label_mod.build_class_image(
            recipe=recipe, viirs_data=viirs,
            dNBR=indices["dNBR"], dB12=indices["dB12"],
            dNDVI=indices["dNDVI"],
            crop_mask=crop_mask, aoi=aoi,
        )
        label_results[rid] = ldata
        print(f"  {rid}: class image built")

    # District names for fold manifest
    print("\nFetching district names …")
    district_names = sorted(
        study["districts"].aggregate_array("ADM2_NAME").getInfo()
    )
    print(f"  {len(district_names)} districts")

    # ── Export fold manifest ─────────────────────────────────────
    import burnmap.modeling.folds as cv_mod
    print("\nExporting fold manifest …")
    rows = cv_mod.build_fold_manifest_rows(
        seeds=config.SEEDS, district_names=district_names,
    )
    manifest_fc = ee.FeatureCollection([ee.Feature(None, r) for r in rows])
    exp.table_to_drive(manifest_fc, f"{prefix}_fold_manifest", folder)

    # ── Submit sample exports ────────────────────────────────────
    task_count = 0
    print("\nSubmitting exports …")

    for recipe in selected_recipes:
        rid = recipe["recipe_id"]
        class_img = label_results[rid]["class_img"]

        for seed in config.SEEDS:
            # Full-Punjab sample with ALL districts, set A features
            desc = f"{prefix}_{rid}_A_s{seed}_full"
            sample_fc = samp_mod.stratified_sample(
                class_img=class_img,
                feature_img=set_a["image"],
                band_list=set_a["band_list"],
                districts=study["districts"],
                district_names=district_names,
                n_per_class=config.N_PER_CLASS,
                seed=seed,
                adm2_img=study["adm2_img"],
            )
            # Annotate with metadata
            sample_fc = samp_mod.annotate_sample(
                sample_fc, rid, "A", seed, -1, "full"
            )
            exp.table_to_drive(sample_fc, desc, folder)
            task_count += 1
            print(f"  {rid} / s{seed} / full ({task_count} tasks)")

            # dNBR baseline sample (full Punjab, for recipe + fixed baselines)
            dnbr_fc = samp_mod.stratified_sample(
                class_img=class_img,
                feature_img=indices["dNBR"].rename("dNBR_baseline"),
                band_list=["dNBR_baseline"],
                districts=study["districts"],
                district_names=district_names,
                n_per_class=config.N_PER_CLASS,
                seed=seed,
                adm2_img=study["adm2_img"],
            )
            dnbr_fc = samp_mod.annotate_sample(
                dnbr_fc, rid, "baseline_dnbr", seed, -1, "full"
            )
            exp.table_to_drive(dnbr_fc, f"{prefix}_{rid}_baseline_dnbr_s{seed}_full", folder)
            task_count += 1

    # Set E exports if requested
    if args.include_e and "E" in all_sets:
        set_e = all_sets["E"]
        print(f"\n  Set E: {set_e['n']} features")
        for recipe in selected_recipes:
            rid = recipe["recipe_id"]
            class_img = label_results[rid]["class_img"]
            for seed in config.SEEDS:
                desc = f"{prefix}_{rid}_E_s{seed}_full"
                sample_fc = samp_mod.stratified_sample(
                    class_img=class_img,
                    feature_img=set_e["image"],
                    band_list=set_e["band_list"],
                    districts=study["districts"],
                    district_names=district_names,
                    n_per_class=config.N_PER_CLASS,
                    seed=seed,
                    adm2_img=study["adm2_img"],
                )
                sample_fc = samp_mod.annotate_sample(
                    sample_fc, rid, "E", seed, -1, "full"
                )
                exp.table_to_drive(sample_fc, desc, folder)
                task_count += 1
                print(f"  {rid} / E / s{seed} / full ({task_count} tasks)")

    print(f"\n{'=' * 60}")
    print(f"  {task_count} export tasks submitted")
    print(f"  Estimated time: ~{task_count * 3} minutes ({task_count * 3 / 60:.1f} hours)")
    print(f"  Check progress: python scripts/export_samples.py --year {year} --status")
    print("  Once complete, download CSVs to data/samples/")
    print(f"  Then run: python scripts/train_eval.py --year {year}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()