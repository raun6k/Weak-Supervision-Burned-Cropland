#!/usr/bin/env python3
"""Score Haryana transfer, 2023 replication, and Sentinel-1 fusion. Formerly phase 6 CV."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from burnmap.modeling.transfer import main


if __name__ == "__main__":
    main()
