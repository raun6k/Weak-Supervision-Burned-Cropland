"""
Phase 1: Study Area, Data Assembly, and Compositing
====================================================

Orchestrates core modules to:
  1. Build Punjab AOI + cropland mask
  2. Create S2 pre/post median composites (Cloud Score+ masking)
  3. Compute spectral indices (NDVI, NDMI, NDRE, NBR, dB12)
  4. Create S1 pre/post median composites
  5. Build VIIRS fire-context layer
  6. Build MODIS MCD64A1 binary burn mask
  7. Run gate checks (coverage, fire count, MODIS pixels)
  8. Export everything to Google Drive

Usage
-----
    python scripts/assemble_data.py --year 2024
    python scripts/assemble_data.py --year 2023 --gate_only
    python scripts/assemble_data.py --year 2024 --skip_s1
"""

import argparse
import json
import ee

import burnmap.config as config
import burnmap.gee.aoi as aoi_mod
import burnmap.gee.sentinel2 as s2_mod
import burnmap.gee.sentinel1 as s1_mod
import burnmap.gee.indices as idx_mod
import burnmap.gee.viirs as viirs_mod
import burnmap.gee.modis as modis_mod
import burnmap.gee.export as exp


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: Data Assembly & Compositing")
    p.add_argument("--year", type=int, default=2024, choices=[2023, 2024])
    p.add_argument("--drive_folder", type=str, default=config.DRIVE_FOLDER)
    p.add_argument("--export_scale", type=int, default=config.EXPORT_SCALE)
    p.add_argument("--gate_only", action="store_true", help="Run gate checks only, no exports")
    p.add_argument("--skip_s1", action="store_true", help="Skip Sentinel-1 exports")
    p.add_argument("--skip_modis", action="store_true", help="Skip MODIS burn mask export")
    return p.parse_args()


# ── Gate checks ──────────────────────────────────────────────────────

def run_gate_checks(s2, viirs, modis, crop_mask, study_area):
    """
    Batch all gate-check computations into one getInfo() call
    to minimise round-trips to the EE servers.
    """
    aoi = study_area["aoi"]

    # S2 coverage: fraction of cropland pixels with valid data
    crop_count = crop_mask.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi,
        scale=20, maxPixels=1e10, tileScale=4,
    ).get("Map")

    pre_valid = s2["pre"].select("B4").mask().updateMask(crop_mask).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi,
        scale=20, maxPixels=1e10, tileScale=4,
    ).get("B4")

    post_valid = s2["post"].select("B4").mask().updateMask(crop_mask).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi,
        scale=20, maxPixels=1e10, tileScale=4,
    ).get("B4")

    vals = ee.Dictionary({
        "crop_px":          crop_count,
        "pre_valid":        pre_valid,
        "post_valid":       post_valid,
        "viirs_detections": viirs["detection_count_hi"],
        "viirs_unique_px":  viirs["unique_pixel_count_hi"],
        "modis_burn":       modis["burned_pixel_count"],
    }).getInfo()

    cpx = vals["crop_px"] or 1
    pre_pct  = vals["pre_valid"]  / cpx * 100
    post_pct = vals["post_valid"] / cpx * 100

    g1 = pre_pct >= config.GATE_S2_COVERAGE_PCT and post_pct >= config.GATE_S2_COVERAGE_PCT
    g2 = vals["viirs_detections"] >= config.GATE_VIIRS_MIN_COUNT
    g3 = vals["modis_burn"] > 0

    print("\n" + "=" * 60)
    print("PHASE 1 GATE CHECKS")
    print("=" * 60)
    print(f"\n  1) S2 coverage   pre {pre_pct:.1f}%  post {post_pct:.1f}%   {'PASS' if g1 else 'FAIL  ← extend window or loosen cloud filter'}")
    print(f"  2) VIIRS fires   {vals['viirs_detections']:,.0f} detections ({vals['viirs_unique_px']:,.0f} unique pixels)   {'PASS' if g2 else 'FAIL  ← verify date filters'}")
    print(f"  3) MODIS burned  {vals['modis_burn']:,.0f} px               {'PASS' if g3 else 'FAIL  ← check BurnDate band'}")
    overall = g1 and g2 and g3
    print(f"\n  OVERALL: {'ALL PASS → proceed to Phase 2' if overall else 'BLOCKED → fix above'}")
    print("=" * 60 + "\n")

    return {
        "pass": overall,
        "s2_pre_pct": round(pre_pct, 2),
        "s2_post_pct": round(post_pct, 2),
        "viirs_detections": vals["viirs_detections"],
        "viirs_unique_pixels": vals["viirs_unique_px"],
        "modis_burned": vals["modis_burn"],
    }


# ── Export orchestration ─────────────────────────────────────────────

def export_all(year, s2, indices, s1, viirs, modis, study_area, gate, args):
    prefix = exp.make_run_prefix(year, "phase1")
    folder = args.drive_folder
    sc = args.export_scale
    aoi_geom = study_area["aoi"]
    tasks = []

    # S2 composites
    tasks.append(exp.image_to_drive(s2["pre"],  f"{prefix}_S2_pre",  aoi_geom, folder, sc))
    tasks.append(exp.image_to_drive(s2["post"], f"{prefix}_S2_post", aoi_geom, folder, sc))
    print(f"  S2 pre/post composites")

    # Spectral index stack
    idx_stack = idx_mod.delta_stack(indices)
    tasks.append(exp.image_to_drive(idx_stack, f"{prefix}_indices", aoi_geom, folder, sc))
    print(f"  Spectral index stack (dNBR, dNDVI, dNDMI, dNDRE, dB12)")

    # S1
    if s1 is not None:
        tasks.append(exp.image_to_drive(s1["pre"],  f"{prefix}_S1_pre",  aoi_geom, folder, config.S1_EXPORT_SCALE))
        tasks.append(exp.image_to_drive(s1["post"], f"{prefix}_S1_post", aoi_geom, folder, config.S1_EXPORT_SCALE))
        print(f"  S1 pre/post composites")

    # VIIRS
    tasks.append(exp.image_to_drive(viirs["fire_hi"], f"{prefix}_VIIRS_fire", aoi_geom, folder, config.VIIRS_SCALE))
    print(f"  VIIRS fire layer")

    # MODIS
    if modis is not None:
        burn_export = modis["burn_mask"].unmask(0).clip(aoi_geom)
        tasks.append(exp.image_to_drive(burn_export, f"{prefix}_MODIS_burn", aoi_geom, folder, config.MODIS_SCALE))
        print(f"  MODIS burn mask")

    # Run manifest
    manifest = {
        "run_id": prefix, "year": year,
        "project": config.PROJECT,
        "adm0": config.ADM0, "adm1": config.ADM1,
        **config.get_dates(year),
        "cloud_mask_method": config.CLOUD_MASK_METHOD,
        "csplus_thr": config.CSPLUS_CLEAR_THR,
        "export_scale": sc,
        **(gate or {}),
    }
    manifest["config_json"] = json.dumps(manifest, default=str, sort_keys=True)
    fc = ee.FeatureCollection([ee.Feature(None, manifest)])
    tasks.append(exp.table_to_drive(fc, f"{prefix}_manifest", folder))
    print(f"  Run manifest CSV")

    return tasks


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    year = args.year
    dates = config.get_dates(year)

    # Init EE
    try:
        ee.Initialize(project=config.PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=config.PROJECT)

    print(f"Phase 1 · Punjab {year}")
    print(f"Drive: {args.drive_folder}  Scale: {args.export_scale}m\n")

    # ---- Build everything ----
    print("Building study area …")
    study = aoi_mod.build_study_area()

    print("Building S2 composites …")
    s2 = s2_mod.build_composites(study["aoi"], dates)

    print("Computing spectral indices …")
    indices = idx_mod.compute_all(s2["pre"], s2["post"])

    s1 = None
    if not args.skip_s1:
        print("Building S1 composites …")
        s1 = s1_mod.build_composites(study["aoi"], dates)

    print("Building VIIRS fire layers …")
    viirs = viirs_mod.build_fire_layers(study["aoi"], dates)

    modis = None
    if not args.skip_modis:
        print("Building MODIS burn mask …")
        modis = modis_mod.build_burn_mask(study["aoi"], study["crop_mask"], dates)

    # ---- Gate checks ----
    gate = None
    if modis is not None:
        gate = run_gate_checks(s2, viirs, modis, study["crop_mask"], study)

    # ---- Exports ----
    if not args.gate_only:
        print("Submitting export tasks …")
        export_all(year, s2, indices, s1, viirs, modis, study, gate, args)
        print("\nAll tasks submitted. Check GEE Tasks tab + Google Drive.")
    else:
        print("Gate-only mode — no exports submitted.")

    print("Phase 1 done.\n")


if __name__ == "__main__":
    main()