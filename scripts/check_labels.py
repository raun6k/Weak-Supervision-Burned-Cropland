#!/usr/bin/env python3
"""Build weak labels and run label gates. Formerly phase 2."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from burnmap.gee.weak_labels import main


if __name__ == "__main__":
    main()
