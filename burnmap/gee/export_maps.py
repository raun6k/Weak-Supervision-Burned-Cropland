"""
Phase 5 — Step 1: Export Full-Punjab Probability Maps
======================================================

Trains the best classifier (from Phase 4) on ALL Punjab labeled
pixels and exports full-Punjab probability surfaces as GeoTIFFs.

Exports:
  - Probability map per recipe (strict, moderate, very_strict)
  - MODIS burn mask GeoTIFF (for pixel-level comparison)
  - ADM2 district raster GeoTIFF (for district-level zonal analysis)
  - District lookup CSV (ADM2 code ↔ district name)

Usage
-----
    python scripts/export_maps.py --year 2024
    python scripts/export_maps.py --year 2024 --status
    python scripts/export_maps.py --year 2024 --recipes very_strict
"""

import argparse
import ee

import burnmap.config as config
from burnmap.config import recipes as recipe_mod
import burnmap.gee.aoi as aoi_mod
import burnmap.gee.sentinel2 as s2_mod
import burnmap.gee.indices as idx_mod
import burnmap.gee.viirs as viirs_mod
import burnmap.gee.modis as modis_mod
import burnmap.gee.features as feat_mod
import burnmap.gee.labels as label_mod
import burnmap.gee.export as exp


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 — Export Probability Maps")
    p.add_argument("--year", type=int, default=2024, choices=[2023, 2024])
    p.add_argument("--drive_folder", type=str, default=config.DRIVE_FOLDER)
    p.add_argument("--recipes", nargs="+", default=config.MAP_RECIPES)
    p.add_argument("--skip_modis", action="store_true")
    p.add_argument("--skip_adm2", action="store_true",
                   help="Skip ADM2 raster + district lookup exports")
    p.add_argument("--status", action="store_true")
    return p.parse_args()


# ── Task status ──────────────────────────────────────────────────────

def check_task_status():
    tasks = ee.data.getTaskList()
    phase5 = [t for t in tasks if "phase5" in t.get("description", "")]
    if not phase5:
        print("No Phase 5 tasks found.")
        return
    by_state = {}
    for t in phase5:
        by_state.setdefault(t.get("state", "?"), []).append(t)
    total = len(phase5)
    done = len(by_state.get("COMPLETED", []))
    print(f"\nPhase 5 tasks: {done}/{total} completed")
    for state, tl in sorted(by_state.items()):
        print(f"  {state}: {len(tl)}")
        if state == "FAILED":
            for t in tl[:5]:
                print(f"    {t['description']}: {t.get('error_message', '?')}")


# ── Helpers ──────────────────────────────────────────────────────────

def _build_classifier():
    """Build the best classifier for full-map prediction."""
    if config.BEST_CLASSIFIER == "rf":
        return ee.Classifier.smileRandomForest(
            numberOfTrees=config.RF_N_TREES,
            bagFraction=config.RF_BAG_FRAC,
            minLeafPopulation=config.RF_MIN_LEAF_POP,
        ).setOutputMode("PROBABILITY")
    if config.BEST_CLASSIFIER == "gbt":
        return ee.Classifier.smileGradientTreeBoost(
            numberOfTrees=config.GBT_N_TREES,
            shrinkage=config.GBT_SHRINKAGE,
            maxNodes=config.GBT_MAX_NODES,
        ).setOutputMode("PROBABILITY")
    raise ValueError(f"GEE doesn't support {config.BEST_CLASSIFIER} natively")


def _train_probability_image(recipe: dict, fset: dict, indices: dict, viirs: dict,
                             crop_mask, aoi):
    """Train on every labeled cropland pixel and return a probability image.

    Unlabeled pixels are masked in the class image and dropped. Omitting
    ``numPixels`` asks Earth Engine to sample all remaining pixels.
    """
    ldata = label_mod.build_class_image(
        recipe=recipe,
        viirs_data=viirs,
        dNBR=indices["dNBR"],
        dB12=indices["dB12"],
        dNDVI=indices["dNDVI"],
        crop_mask=crop_mask,
        aoi=aoi,
    )
    class_img = ldata["class_img"]
    stack = fset["image"].toFloat().addBands(class_img.select("class").toInt16())
    train_fc = stack.sample(
        region=aoi,
        scale=config.EXPORT_SCALE,
        seed=42,
        dropNulls=True,
        tileScale=4,
    )
    trained = _build_classifier().train(
        features=train_fc,
        classProperty="class",
        inputProperties=ee.List(fset["band_list"]),
    )
    prob_map = (
        fset["image"]
        .classify(trained, "probability")
        .updateMask(crop_mask)
        .clip(aoi)
        .toFloat()
    )
    return prob_map


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    year = args.year
    dates = config.get_dates(year)
    folder = args.drive_folder
    prefix = exp.make_run_prefix(year, "phase5")

    try:
        ee.Initialize(project=config.PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=config.PROJECT)

    if args.status:
        check_task_status()
        return

    print(f"Phase 5 Export · Punjab {year}")
    print(f"Best config: {config.BEST_CLASSIFIER} / {config.BEST_RECIPE} / set {config.BEST_FEATURE_SET}")
    print(f"Recipes for maps: {args.recipes}")
    print("Training sample: all labeled cropland pixels")
    print(f"Drive folder: {folder}\n")

    print("Building study area …")
    study = aoi_mod.build_study_area()
    aoi = study["aoi"]
    crop_mask = study["crop_mask"]

    print("Building S2 composites …")
    s2 = s2_mod.build_composites(aoi, dates)

    print("Computing spectral indices …")
    indices = idx_mod.compute_all(s2["pre"], s2["post"])

    print("Building VIIRS fire layers …")
    viirs = viirs_mod.build_fire_layers(aoi, dates)

    print("Building MODIS burn mask …")
    modis = modis_mod.build_burn_mask(aoi, crop_mask, dates)

    print(f"\nConstructing feature set {config.BEST_FEATURE_SET} …")
    all_sets = feat_mod.build_all(s2["pre"], s2["post"], indices, None, crop_mask, aoi)
    fset = all_sets[config.BEST_FEATURE_SET]
    print(f"  {fset['n']} features")

    task_count = 0
    prob_maps = {}

    print("\nSubmitting probability map exports …")
    selected_recipes = [recipe_mod.get_recipe(r) for r in args.recipes]
    for recipe in selected_recipes:
        rid = recipe["recipe_id"]
        prob_map = _train_probability_image(recipe, fset, indices, viirs, crop_mask, aoi)
        prob_maps[rid] = prob_map
        desc = f"{prefix}_{rid}_{config.BEST_FEATURE_SET}_prob"
        exp.image_to_drive(prob_map, desc, aoi, folder, config.FULL_MAP_SCALE)
        task_count += 1
        print(f"  {rid} probability map ({task_count} tasks)")

    if not args.skip_modis:
        modis_export = modis["burn_mask"].unmask(0).clip(aoi).toFloat()
        desc_m = f"{prefix}_MODIS_burn"
        exp.image_to_drive(modis_export, desc_m, aoi, folder, config.MODIS_SCALE)
        task_count += 1
        print(f"  MODIS burn mask ({task_count} tasks)")

    if not args.skip_adm2:
        adm2_export = study["adm2_img"].rename("adm2").clip(aoi).toInt32()
        desc_adm2 = f"{prefix}_adm2"
        exp.image_to_drive(adm2_export, desc_adm2, aoi, folder, config.FULL_MAP_SCALE)
        task_count += 1
        print(f"  ADM2 district raster ({task_count} tasks)")

        district_lookup = study["districts"].map(
            lambda f: ee.Feature(None, {
                "adm2_code": f.get("ADM2_CODE"),
                "district": f.get("ADM2_NAME"),
                "adm1": f.get("ADM1_NAME"),
                "adm0": f.get("ADM0_NAME"),
            })
        )
        desc_lookup = f"{prefix}_district_lookup"
        exp.table_to_drive(district_lookup, desc_lookup, folder)
        task_count += 1
        print(f"  District lookup CSV ({task_count} tasks)")

    print(f"\n{'=' * 60}")
    print(f"  {task_count} export tasks submitted")
    print(f"  Check progress: python scripts/export_maps.py --year {year} --status")
    print(f"  Once complete, download GeoTIFFs + CSVs from Drive")
    print(f"  Then run: python scripts/calibrate.py --year {year}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
