#!/usr/bin/env python3
"""Export statewide probability maps and district support rasters. Formerly phase 5 export."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from burnmap.gee import export_maps, export_support


def main():
    p = argparse.ArgumentParser(
        description="Export probability maps, then ADM2 support rasters"
    )
    p.add_argument("--year", type=int, default=2024, choices=[2023, 2024])
    p.add_argument("--drive_folder", default="")
    p.add_argument("--recipes", nargs="+", default=None)
    p.add_argument("--skip_modis", action="store_true")
    p.add_argument("--skip_adm2", action="store_true")
    p.add_argument("--skip_lookup", action="store_true")
    p.add_argument("--status", action="store_true")
    args = p.parse_args()

    map_argv = ["export_maps", "--year", str(args.year)]
    if args.drive_folder:
        map_argv += ["--drive_folder", args.drive_folder]
    if args.recipes:
        map_argv += ["--recipes", *args.recipes]
    if args.skip_modis:
        map_argv.append("--skip_modis")
    if args.skip_adm2:
        map_argv.append("--skip_adm2")
    if args.status:
        map_argv.append("--status")
    sys.argv = map_argv
    export_maps.main()

    support_argv = ["export_support", "--year", str(args.year)]
    if args.drive_folder:
        support_argv += ["--drive_folder", args.drive_folder]
    if args.status:
        support_argv.append("--status")
    if args.skip_adm2:
        support_argv.append("--skip_adm2")
    if args.skip_lookup:
        support_argv.append("--skip_lookup")
    sys.argv = support_argv
    export_support.main()


if __name__ == "__main__":
    main()
