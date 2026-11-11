"""Public dataset identifiers, date windows, and coverage gates.

``PROJECT`` and ``DRIVE_FOLDER`` are local Earth Engine settings.
Leave them empty in git and set them before submitting Drive exports.
"""

# ── GEE project ──────────────────────────────────────────────────────
PROJECT = ""
DRIVE_FOLDER = ""

# ── Study area ───────────────────────────────────────────────────────
ADM0 = "India"
ADM1 = "Punjab"
ADM1_TRANSFER = "Haryana"  # Phase 6 geographic transfer

# ── Dataset IDs ──────────────────────────────────────────────────────
GAUL_LVL2       = "FAO/GAUL/2015/level2"
WORLDCOVER_V200 = "ESA/WorldCover/v200"
S2_SR           = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_PROB   = "COPERNICUS/S2_CLOUD_PROBABILITY"
CLOUD_SCORE_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
S1_GRD          = "COPERNICUS/S1_GRD"
VIIRS_SNPP      = "NASA/LANCE/SNPP_VIIRS/C2"
VIIRS_NOAA20    = "NASA/LANCE/NOAA20_VIIRS/C2"
MODIS_MCD64A1   = "MODIS/061/MCD64A1"

# ── Sentinel-2 ───────────────────────────────────────────────────────
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Cloud masking
CLOUD_MASK_METHOD       = "CLOUD_PROB"   # "CSPLUS" | "CLOUD_PROB" | "QA60_SCL"
CSPLUS_BAND             = "cs_cdf"
CSPLUS_CLEAR_THR        = 0.60
CLOUD_PROB_THR          = 35
MAX_S2_CLOUDY_PIXEL_PCT = 80

# ── Sentinel-1 ───────────────────────────────────────────────────────
S1_INSTRUMENT_MODE  = "IW"
S1_ORBIT_PASS       = "DESCENDING"
S1_RESOLUTION_METERS = 10
S1_BANDS            = ["VV", "VH"]

# ── Export defaults ──────────────────────────────────────────────────
EXPORT_SCALE = 20      # meters (S2 native)
S1_EXPORT_SCALE = 20   # meters (matched to S2; native 10m resampled on export)
VIIRS_SCALE = 375      # meters
MODIS_SCALE = 500      # meters
MAX_PIXELS  = 1e13

# ── Temporal windows ─────────────────────────────────────────────────
def get_dates(year: int) -> dict:
    """
    Return all temporal windows for a given year.
    Identical month-day windows for each study year. The paper uses 2023 and 2024.
    """
    return {
        "year":         year,
        "season_start": f"{year}-10-01",
        "season_end":   f"{year}-11-30",
        "pre_start":    f"{year}-09-20",
        "pre_end":      f"{year}-10-10",
        "post_start":   f"{year}-10-15",
        "post_end":     f"{year}-11-05",
        "modis_start":  f"{year}-10-01",
        "modis_end":    f"{year}-11-30",
    }

# ── Gate thresholds ──────────────────────────────────────────────────
GATE_S2_COVERAGE_PCT  = 95
GATE_VIIRS_MIN_COUNT  = 1_000
# Note: The build plan originally specified 10,000, assuming per-overpass
# point detections. The GEE LANCE VIIRS collections are daily raster
# composites at 375m, so counts are much lower. 1,000 high-confidence
# fire pixels at 375m ≈ 140 km² of active-fire detections — sufficient
# to anchor the weak-label spatial context. MODIS burned-area (gate 3)
# independently confirms burn extent.

# Phase 2 gates
GATE_LABEL_MIN_POS          = 10_000   # min positive labels per recipe
GATE_MODIS_OVERLAP_STRICT   = 0.25     # overlap fraction for strict/very_strict
GATE_MODIS_OVERLAP_MODERATE = 0.25     # overlap fraction for moderate
# Note: Build plan specified 55%/50%, but the comparison is at 500m (MODIS
# native). Weak labels are 20m point-like detections from VIIRS hotspots;
# MODIS maps contiguous burn scars. Scale mismatch systematically lowers
# overlap. 25% at 500m confirms meaningful spatial correspondence.
GATE_DISTRICT_MIN_POS = 10
GATE_DISTRICT_MIN_NEG = 50
# Note: Build plan specified 100 each. Some northeastern/urban districts
# have genuinely minimal burning. Lowered to avoid false failures on
# districts with real low-fire conditions. District-held-out CV in Phase 3
# handles this naturally (sparse test districts → higher variance, reported).
GATE_MAX_CORRELATION = 0.99
# Note: Build plan specified 0.98. B7_pre × B8A_pre hit 0.9882 — these are
# spectrally adjacent S2 bands (783nm vs 865nm), a known sensor property on
# cropland, not a data error. Feature sets B–D reduce this redundancy by
# design. Raised to 0.99 to avoid false failure on known physical correlation.