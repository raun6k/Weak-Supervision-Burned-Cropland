#!/usr/bin/env python3
"""Write the paper tables and the figures produced with them. Formerly phase 7."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from burnmap.reporting.paper import main


if __name__ == "__main__":
    main()
