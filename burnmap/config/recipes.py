"""
Weak-label recipe definitions.

Each recipe is a dict of thresholds / buffer distances that
control how positive (burned) and negative (unburned) labels
are generated from VIIRS fire detections + spectral indices.
"""

STRICT = {
    "recipe_id":          "strict",
    "viirs_mode":         "hi",
    "burnDNBRThr":        0.22,
    "useSwirGate":        True,
    "swirGateThr":        200,
    "unburnedFarMeters":  8000,
    "nearFireMeters":     2000,
    "hardNeg_dNBR_Max":   0.05,
    "hardNeg_dB12_Max":   100,
    "ndviDecreaseGate":   None,   # not used in strict
}

MODERATE = {
    "recipe_id":          "moderate",
    "viirs_mode":         "hi",
    "burnDNBRThr":        0.18,
    "useSwirGate":        True,
    "swirGateThr":        200,
    "unburnedFarMeters":  8000,
    "nearFireMeters":     3000,
    "hardNeg_dNBR_Max":   0.06,
    "hardNeg_dB12_Max":   120,
    "ndviDecreaseGate":   None,
}

VERY_STRICT = {
    "recipe_id":          "very_strict",
    "viirs_mode":         "hi",
    "burnDNBRThr":        0.28,
    "useSwirGate":        True,
    "swirGateThr":        250,
    "unburnedFarMeters":  10000,
    "nearFireMeters":     1500,
    "hardNeg_dNBR_Max":   0.04,
    "hardNeg_dB12_Max":   80,
    "ndviDecreaseGate":   0.10,  # new multi-spectral constraint
}

# Convenient list for iteration
ALL_RECIPES  = [STRICT, MODERATE, VERY_STRICT]
PRIMARY_RECIPES = [STRICT, MODERATE]  # backwards-compatible with iteration 1


def get_recipe(name: str) -> dict:
    """Look up a recipe by its recipe_id string."""
    for r in ALL_RECIPES:
        if r["recipe_id"] == name:
            return r
    raise ValueError(f"Unknown recipe: {name}. Choose from {[r['recipe_id'] for r in ALL_RECIPES]}")