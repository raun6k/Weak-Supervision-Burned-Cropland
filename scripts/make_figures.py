#!/usr/bin/env python3
"""Draw the six standalone manuscript figures using their default inputs.

Study-area and weak-label figures need Earth Engine unless the Figure 1
local cache is present. The other four figures read tables and GeoTIFFs
written by ``scripts/paper_tables.py`` and ``scripts/calibrate.py``.
Override a single figure with ``python -m burnmap.figures.<module>``.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from burnmap.figures import (
    burned_area_map,
    calibration,
    district_error,
    recipe_quality,
    study_area,
    weak_labels,
)


def main():
    argparse.ArgumentParser(
        description=(
            "Draw the six manuscript figures with their default inputs. "
            "Override one figure with python -m burnmap.figures.<module>."
        )
    ).parse_args()
    for module in (
        study_area,
        weak_labels,
        calibration,
        district_error,
        recipe_quality,
        burned_area_map,
    ):
        sys.argv = [module.__name__]
        module.main()


if __name__ == "__main__":
    main()
