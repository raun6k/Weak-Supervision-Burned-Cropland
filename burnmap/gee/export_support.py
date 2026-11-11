"""
Phase 5 — Support exports only
==============================

Exports only the Phase 5 support artifacts that are still missing,
without rerunning the probability-map or MODIS export tasks.

Default exports:
  - ADM2 district raster GeoTIFF
  - District lookup CSV

Usage
-----
    python scripts/export_maps.py --year 2024
    python scripts/export_maps.py --year 2024 --skip_adm2
    python scripts/export_maps.py --year 2024 --skip_lookup
    python scripts/export_maps.py --year 2024 --status
"""

import argparse
import ee

import burnmap.config as config
import burnmap.gee.aoi as aoi_mod
import burnmap.gee.export as exp


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 — Support exports only")
    p.add_argument("--year", type=int, default=2024, choices=[2023, 2024])
    p.add_argument("--drive_folder", type=str, default=config.DRIVE_FOLDER)
    p.add_argument("--skip_adm2", action="store_true")
    p.add_argument("--skip_lookup", action="store_true")
    p.add_argument("--status", action="store_true")
    return p.parse_args()


def check_task_status():
    tasks = ee.data.getTaskList()
    support = [t for t in tasks if "phase5" in t.get("description", "") and ("adm2" in t.get("description", "") or "district_lookup" in t.get("description", ""))]
    if not support:
        print("No Phase 5 support-export tasks found.")
        return
    by_state = {}
    for t in support:
        by_state.setdefault(t.get("state", "?"), []).append(t)
    total = len(support)
    done = len(by_state.get("COMPLETED", []))
    print(f"\nPhase 5 support tasks: {done}/{total} completed")
    for state, tl in sorted(by_state.items()):
        print(f"  {state}: {len(tl)}")
        if state == "FAILED":
            for t in tl[:5]:
                print(f"    {t['description']}: {t.get('error_message', '?')}")


def main():
    args = parse_args()
    year = args.year
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

    print(f"Phase 5 Support Export · Punjab {year}")
    print(f"Drive folder: {folder}\n")

    study = aoi_mod.build_study_area()
    aoi = study["aoi"]

    task_count = 0

    if not args.skip_adm2:
        adm2_export = study["adm2_img"].rename("adm2").clip(aoi).toInt32()
        desc_adm2 = f"{prefix}_adm2"
        exp.image_to_drive(adm2_export, desc_adm2, aoi, folder, config.FULL_MAP_SCALE)
        task_count += 1
        print(f"  ADM2 district raster ({task_count} tasks)")

    if not args.skip_lookup:
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
    print(f"  {task_count} support export tasks submitted")
    print(f"  Check progress: python scripts/export_maps.py --year {year} --status")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
