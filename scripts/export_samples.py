#!/usr/bin/env python3
"""Export stratified training samples to Google Drive. Formerly phase 3 export."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from burnmap.gee.export_samples import main


if __name__ == "__main__":
    main()
