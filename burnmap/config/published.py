"""Map configuration reported in the paper.

The external calibration sweep selects probability 0.60 for the
published Punjab 2024 map (very_strict / rf / feature set A).
"""

# ── Phase 5: Validation ─────────────────────────────────────────────
# Best config from Phase 4 (used for full-Punjab maps)
BEST_CLASSIFIER  = "rf"
BEST_RECIPE      = "very_strict"
BEST_FEATURE_SET = "A"
MAP_RECIPES      = ["strict", "moderate", "very_strict"]

# Full-map sampling (more points for production map)
FULL_MAP_SCALE = 20  # export probability map at 20m


# Threshold selected by the external MODIS and fire-count sweep.
CALIBRATED_THRESHOLD = 0.60
