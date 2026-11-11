#!/usr/bin/env python3
"""Export Haryana, 2023, and Sentinel-1 samples. Formerly phase 6 export.

The default multiyear export is Punjab 2023, the only earlier year compared with 2024 in the paper.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from burnmap.gee.export_transfer import main


if __name__ == "__main__":
    main()
