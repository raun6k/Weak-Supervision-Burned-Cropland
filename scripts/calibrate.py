#!/usr/bin/env python3
"""Compare the exported map with MODIS and sweep the probability threshold.

Formerly phase 5 validation followed by the threshold calibration sweep.
Map GeoTIFFs are read from ``outputs/<year>/phase5_maps``.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from burnmap.validation import calibrate, validate


def _validate_argv(user: list[str]) -> list[str]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--cv_metrics", default=None)
    p.add_argument("--map_dir", default=None)
    p.add_argument("--adm2_tif", default=None)
    p.add_argument("--district_lookup", default=None)
    p.add_argument("--caqm_csv", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--skip_raster", action="store_true")
    args, _ = p.parse_known_args(user)
    argv = ["validate", "--year", str(args.year)]
    optional = {
        "--cv_metrics": args.cv_metrics,
        "--map_dir": args.map_dir,
        "--adm2_tif": args.adm2_tif,
        "--district_lookup": args.district_lookup,
        "--caqm_csv": args.caqm_csv,
        "--out_dir": args.out_dir,
    }
    for flag, value in optional.items():
        if value:
            argv += [flag, value]
    if args.threshold is not None:
        argv += ["--threshold", str(args.threshold)]
    if args.skip_raster:
        argv.append("--skip_raster")
    return argv


def main():
    user = sys.argv[1:]
    if "--help" in user or "-h" in user:
        sys.argv = ["calibrate", "--help"]
        calibrate.main()
        return
    sys.argv = _validate_argv(user)
    validate.main()
    sys.argv = ["calibrate", *user]
    calibrate.main()


if __name__ == "__main__":
    main()
