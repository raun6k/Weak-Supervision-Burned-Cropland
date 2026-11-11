"""
Phase 6 — Step 1: Export Generalizability Samples
===================================================

Exports samples for three generalizability experiments:

  6.1 Multi-year: Punjab 2023, using the same seasonal windows as 2024
  6.2 Haryana transfer: train Punjab → predict Haryana, plus Haryana CV
  6.3 S1+S2 fusion: feature set E samples for Punjab 2024

All use the best recipe (very_strict) and best classifier (RF).

Patched behavior:
  - supports selecting which multiyear seasons to export
  - preflights VIIRS detections for each multiyear year
  - cleanly skips years that fail the Phase 1 VIIRS gate instead of
    submitting tasks that immediately fail

Usage
-----
    python scripts/export_transfer_samples.py --year 2024
    python scripts/export_transfer_samples.py --year 2024 --experiments multiyear
    python scripts/export_transfer_samples.py --year 2024 --experiments haryana s1
    python scripts/export_transfer_samples.py --year 2024 --multiyear_years 2023
    python scripts/export_transfer_samples.py --year 2024 --status
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
import burnmap.gee.sampling as samp_mod
import burnmap.gee.export as exp


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 6 — Export Generalizability Samples")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--drive_folder", type=str, default=config.DRIVE_FOLDER)
    p.add_argument(
        "--experiments",
        nargs="+",
        default=["multiyear", "haryana", "s1"],
        help="Which experiments: multiyear, haryana, s1",
    )
    p.add_argument(
        "--multiyear_years",
        nargs="+",
        type=int,
        default=[2023],
        choices=[2023, 2024],
        help="Which Punjab years to export for the multiyear experiment",
    )
    p.add_argument(
        "--allow_zero_viirs",
        action="store_true",
        help="Submit exports even if the Phase 1 VIIRS gate is zero for a multiyear year",
    )
    p.add_argument("--status", action="store_true")
    return p.parse_args()


def check_task_status():
    tasks = ee.data.getTaskList()
    phase6 = [t for t in tasks if "phase6" in t.get("description", "")]
    if not phase6:
        print("No Phase 6 tasks found.")
        return
    by_state = {}
    for t in phase6:
        by_state.setdefault(t.get("state", "?"), []).append(t)
    done = len(by_state.get("COMPLETED", []))
    print(f"\nPhase 6 tasks: {done}/{len(phase6)} completed")
    for state, tl in sorted(by_state.items()):
        print(f"  {state}: {len(tl)}")
        if state == "FAILED":
            for t in tl[:3]:
                print(f"    {t['description']}: {t.get('error_message', '?')}")


def _build_data_products(aoi, dates, crop_mask, include_s1=False):
    """Build S2 composites, indices, VIIRS, MODIS for a given AOI + dates."""
    s2 = s2_mod.build_composites(aoi, dates)
    indices = idx_mod.compute_all(s2["pre"], s2["post"])
    viirs = viirs_mod.build_fire_layers(aoi, dates)
    modis = modis_mod.build_burn_mask(aoi, crop_mask, dates)

    s1 = None
    if include_s1:
        s1 = s1_mod.build_composites(aoi, dates)

    return s2, indices, viirs, modis, s1


def _export_full_sample(class_img, feature_img, band_list, districts,
                        district_names, adm2_img, seed, recipe_id,
                        set_id, prefix, folder):
    """Export one full-region stratified sample."""
    sample_fc = samp_mod.stratified_sample(
        class_img=class_img,
        feature_img=feature_img,
        band_list=band_list,
        districts=districts,
        district_names=district_names,
        n_per_class=config.N_PER_CLASS,
        seed=seed,
        adm2_img=adm2_img,
    )
    sample_fc = samp_mod.annotate_sample(
        sample_fc, recipe_id, set_id, seed, -1, "full"
    )
    desc = f"{prefix}_{recipe_id}_{set_id}_s{seed}_full"
    exp.table_to_drive(sample_fc, desc, folder)
    return desc


def _safe_number_info(num_obj, default=None):
    """Resolve an ee.Number to Python, returning default on failure."""
    try:
        val = num_obj.getInfo()
        return default if val is None else val
    except Exception:
        return default


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    folder = args.drive_folder
    recipe = recipe_mod.get_recipe(config.BEST_RECIPE)
    rid = recipe["recipe_id"]

    try:
        ee.Initialize(project=config.PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=config.PROJECT)

    if args.status:
        check_task_status()
        return

    print("Phase 6 Export · Generalizability")
    print(f"Best config: {config.BEST_CLASSIFIER}/{rid}/set {config.BEST_FEATURE_SET}")
    print(f"Experiments: {args.experiments}")
    if "multiyear" in args.experiments:
        print(f"Multi-year seasons: {args.multiyear_years}")
    print()

    task_count = 0

    # ── 6.1 Multi-year: Punjab 2023 ─────────────────────────────────
    if "multiyear" in args.experiments:
        print("── 6.1 Multi-year replication ──")

        for year in args.multiyear_years:
            dates = config.get_dates(year)
            prefix = exp.make_run_prefix(year, "phase6")

            print(f"\n  Building Punjab {year} data products …")
            study = aoi_mod.build_study_area()
            aoi = study["aoi"]
            crop_mask = study["crop_mask"]

            s2, indices, viirs, modis, _ = _build_data_products(aoi, dates, crop_mask)

            # Phase-1-style preflight: do not submit exports for years with no VIIRS fire signal
            det_hi = _safe_number_info(viirs["detection_count_hi"], default=None)
            uniq_hi = _safe_number_info(viirs["unique_pixel_count_hi"], default=None)
            if det_hi is not None and uniq_hi is not None:
                print(f"    VIIRS preflight: {det_hi} detections ({uniq_hi} unique pixels)")
            else:
                print("    VIIRS preflight: unavailable (continuing)")

            if not args.allow_zero_viirs and det_hi == 0:
                print(
                    f"    SKIP {year}: VIIRS detections are zero, so the weak-label recipe has no fire anchor."
                )
                print(
                    "    This is treated as a data-availability limitation for this year, not a Phase 6 code failure."
                )
                continue

            all_sets = feat_mod.build_all(s2["pre"], s2["post"], indices, None, crop_mask, aoi)
            fset = all_sets[config.BEST_FEATURE_SET]

            ldata = label_mod.build_class_image(
                recipe=recipe, viirs_data=viirs,
                dNBR=indices["dNBR"], dB12=indices["dB12"],
                dNDVI=indices["dNDVI"],
                crop_mask=crop_mask, aoi=aoi,
            )

            district_names = sorted(
                study["districts"].aggregate_array("ADM2_NAME").getInfo()
            )

            for seed in config.SEEDS:
                _export_full_sample(
                    ldata["class_img"], fset["image"], fset["band_list"],
                    study["districts"], district_names, study["adm2_img"],
                    seed, rid, config.BEST_FEATURE_SET, prefix, folder,
                )
                task_count += 1
                print(f"    {year} / s{seed} ({task_count} tasks)")

    # ── 6.2 Haryana transfer ─────────────────────────────────────
    if "haryana" in args.experiments:
        print("\n── 6.2 Haryana transfer ──")
        dates = config.get_dates(2024)

        print("  Building Haryana study area …")
        haryana = aoi_mod.build_study_area(
            adm0=config.ADM0, adm1=config.ADM1_TRANSFER
        )
        h_aoi = haryana["aoi"]
        h_crop = haryana["crop_mask"]
        prefix = exp.make_run_prefix(2024, "phase6_haryana")

        print("  Building Haryana data products …")
        s2, indices, viirs, modis, _ = _build_data_products(h_aoi, dates, h_crop)

        all_sets = feat_mod.build_all(s2["pre"], s2["post"], indices, None, h_crop, h_aoi)
        fset = all_sets[config.BEST_FEATURE_SET]

        ldata = label_mod.build_class_image(
            recipe=recipe, viirs_data=viirs,
            dNBR=indices["dNBR"], dB12=indices["dB12"],
            dNDVI=indices["dNDVI"],
            crop_mask=h_crop, aoi=h_aoi,
        )

        h_district_names = sorted(
            haryana["districts"].aggregate_array("ADM2_NAME").getInfo()
        )
        print(f"  {len(h_district_names)} Haryana districts")

        for seed in config.SEEDS:
            _export_full_sample(
                ldata["class_img"], fset["image"], fset["band_list"],
                haryana["districts"], h_district_names, haryana["adm2_img"],
                seed, rid, config.BEST_FEATURE_SET, prefix, folder,
            )
            task_count += 1
            print(f"    Haryana / s{seed} ({task_count} tasks)")

    # ── 6.3 S1+S2 fusion (Punjab 2024) ───────────────────────────
    if "s1" in args.experiments:
        print("\n── 6.3 S1+S2 fusion ──")
        dates = config.get_dates(2024)
        prefix = exp.make_run_prefix(2024, "phase6_s1")

        print("  Building Punjab 2024 with S1 …")
        study = aoi_mod.build_study_area()
        aoi = study["aoi"]
        crop_mask = study["crop_mask"]

        s2, indices, viirs, modis, s1 = _build_data_products(
            aoi, dates, crop_mask, include_s1=True
        )

        all_sets = feat_mod.build_all(s2["pre"], s2["post"], indices, s1, crop_mask, aoi)

        if "E" not in all_sets:
            print("  WARNING: Set E not available (S1 build failed)")
        else:
            fset_e = all_sets["E"]
            print(f"  Set E: {fset_e['n']} features")

            ldata = label_mod.build_class_image(
                recipe=recipe, viirs_data=viirs,
                dNBR=indices["dNBR"], dB12=indices["dB12"],
                dNDVI=indices["dNDVI"],
                crop_mask=crop_mask, aoi=aoi,
            )

            district_names = sorted(
                study["districts"].aggregate_array("ADM2_NAME").getInfo()
            )

            for seed in config.SEEDS:
                _export_full_sample(
                    ldata["class_img"], fset_e["image"], fset_e["band_list"],
                    study["districts"], district_names, study["adm2_img"],
                    seed, rid, "E", prefix, folder,
                )
                task_count += 1
                print(f"    S1+S2 / s{seed} ({task_count} tasks)")

    print(f"\n{'=' * 60}")
    print(f"  {task_count} export tasks submitted")
    print("  Check: python scripts/export_transfer_samples.py --status")
    print(f"  Then download CSVs and run: python scripts/transfer.py --year {args.year}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
