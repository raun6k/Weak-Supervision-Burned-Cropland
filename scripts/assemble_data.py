#!/usr/bin/env python3
"""Assemble Punjab composites and run coverage gates. Formerly phase 1."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from burnmap.gee.assemble import main


if __name__ == "__main__":
    main()
