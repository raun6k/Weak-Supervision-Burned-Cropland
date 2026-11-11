"""
Feature-set construction (sets A–E).

Each function returns an ee.Image stack and its band name list.
All sets intentionally EXCLUDE every variable used in label construction:
NBR/dNBR (burn threshold), dB12 (SWIR gate) and dNDVI (NDVI-decrease gate).

Feature sets
------------
A: Full 37       — 10 bands × (pre+post+delta) + 3 indices × (pre+post+delta),
                   minus dB12 and dNDVI
B: Delta-only 11 — dB2..dB11, dNDMI, dNDRE
C: Curated 24    — 11 deltas + 10 pre bands + 3 pre indices
D: No-SWIR 32    — Full minus B11/B12 pre/post/delta and dNDVI (circularity ablation)
E: S1+S2 43      — Full 37 + VV/VH pre/post/delta
"""

import ee
import burnmap.config as config


# ── Band name lists (computed once) ──────────────────────────────────

_S2 = config.S2_BANDS  # ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
_INDICES = ["NDVI", "NDMI", "NDRE"]
_SWIR_BANDS = ["B11", "B12"]

# Variables that gate the weak labels (see core/labels.py); never used as predictors.
_LABEL_VARS = ["dB12", "dNDVI"]


def _pre_names(bands):   return [f"{b}_pre" for b in bands]
def _post_names(bands):  return [f"{b}_post" for b in bands]
def _delta_names(bands): return [f"d{b}" for b in bands if f"d{b}" not in _LABEL_VARS]
def _idx_pre():          return [f"{ix}_pre" for ix in _INDICES]
def _idx_post():         return [f"{ix}_post" for ix in _INDICES]
def _idx_delta():        return [f"d{ix}" for ix in _INDICES if f"d{ix}" not in _LABEL_VARS]


# ── Image builders ───────────────────────────────────────────────────

def _s2_pre_post_delta(s2_pre: ee.Image, s2_post: ee.Image):
    """Renamed pre/post/delta bands for 10 S2 spectral bands (label-gate deltas dropped)."""
    pre   = s2_pre.select(_S2).rename(_pre_names(_S2))
    post  = s2_post.select(_S2).rename(_post_names(_S2))
    delta_all = s2_post.select(_S2).subtract(s2_pre.select(_S2)).rename([f"d{b}" for b in _S2])
    delta = delta_all.select(_delta_names(_S2))
    return pre, post, delta


def _index_stack(indices: dict):
    """Stack pre/post/delta index images (label-gate deltas dropped)."""
    pre   = ee.Image.cat([indices[k] for k in _idx_pre()])
    post  = ee.Image.cat([indices[k] for k in _idx_post()])
    delta = ee.Image.cat([indices[k] for k in _idx_delta()])
    return pre, post, delta


def _s1_pre_post_delta(s1: dict):
    """VV/VH pre/post/delta from the S1 composite dict."""
    pre   = s1["pre"].select(config.S1_BANDS).rename(["VV_pre", "VH_pre"])
    post  = s1["post"].select(config.S1_BANDS).rename(["VV_post", "VH_post"])
    delta = ee.Image.cat([s1["dVV"], s1["dVH"]])
    return pre, post, delta


# ── Public API: one function per feature set ─────────────────────────

def set_a(s2_pre, s2_post, indices, crop_mask, aoi) -> dict:
    """
    Set A: Full 37 features.
    10 bands × 3 (pre/post/delta) + 3 indices × 3 = 39, minus dB12 and dNDVI.
    """
    bp, bpo, bd = _s2_pre_post_delta(s2_pre, s2_post)
    ip, ipo, id_ = _index_stack(indices)

    img = (
        bp.addBands(bpo).addBands(bd)
        .addBands(ip).addBands(ipo).addBands(id_)
        .updateMask(crop_mask).clip(aoi)
    )
    bands = (
        _pre_names(_S2) + _post_names(_S2) + _delta_names(_S2)
        + _idx_pre() + _idx_post() + _idx_delta()
    )
    return {"image": img, "bands": ee.List(bands), "band_list": bands, "set_id": "A", "n": 37}


def set_b(s2_pre, s2_post, indices, crop_mask, aoi) -> dict:
    """
    Set B: Delta-only 11 features.
    dB2..dB11 + dNDMI + dNDRE (dB12 and dNDVI are label gates).
    """
    _, _, bd = _s2_pre_post_delta(s2_pre, s2_post)
    _, _, id_ = _index_stack(indices)

    img = bd.addBands(id_).updateMask(crop_mask).clip(aoi)
    bands = _delta_names(_S2) + _idx_delta()
    return {"image": img, "bands": ee.List(bands), "band_list": bands, "set_id": "B", "n": 11}


def set_c(s2_pre, s2_post, indices, crop_mask, aoi) -> dict:
    """
    Set C: Curated 24 features.
    11 deltas + 10 pre-event bands + 3 pre-event indices.
    """
    bp, _, bd = _s2_pre_post_delta(s2_pre, s2_post)
    ip, _, id_ = _index_stack(indices)

    img = bd.addBands(id_).addBands(bp).addBands(ip).updateMask(crop_mask).clip(aoi)
    bands = _delta_names(_S2) + _idx_delta() + _pre_names(_S2) + _idx_pre()
    return {"image": img, "bands": ee.List(bands), "band_list": bands, "set_id": "C", "n": 24}


def set_d(s2_pre, s2_post, indices, crop_mask, aoi) -> dict:
    """
    Set D: No-SWIR 32 features (circularity ablation).
    Set A minus B11_pre, B11_post, dB11, B12_pre, B12_post (dB12 is already excluded).
    """
    non_swir = [b for b in _S2 if b not in _SWIR_BANDS]  # 8 bands

    bp   = s2_pre.select(non_swir).rename(_pre_names(non_swir))
    bpo  = s2_post.select(non_swir).rename(_post_names(non_swir))
    bd   = (s2_post.select(non_swir).subtract(s2_pre.select(non_swir))
            .rename([f"d{b}" for b in non_swir]).select(_delta_names(non_swir)))
    ip, ipo, id_ = _index_stack(indices)

    img = (
        bp.addBands(bpo).addBands(bd)
        .addBands(ip).addBands(ipo).addBands(id_)
        .updateMask(crop_mask).clip(aoi)
    )
    bands = (
        _pre_names(non_swir) + _post_names(non_swir) + _delta_names(non_swir)
        + _idx_pre() + _idx_post() + _idx_delta()
    )
    return {"image": img, "bands": ee.List(bands), "band_list": bands, "set_id": "D", "n": 32}


def set_e(s2_pre, s2_post, indices, s1, crop_mask, aoi) -> dict:
    """
    Set E: S1+S2 fusion 43 features.
    Set A (37) + VV_pre, VV_post, dVV, VH_pre, VH_post, dVH.
    """
    a = set_a(s2_pre, s2_post, indices, crop_mask, aoi)
    s1p, s1po, s1d = _s1_pre_post_delta(s1)

    img = a["image"].addBands(s1p).addBands(s1po).addBands(s1d).updateMask(crop_mask).clip(aoi)
    s1_bands = ["VV_pre", "VH_pre", "VV_post", "VH_post", "dVV", "dVH"]
    bands = a["band_list"] + s1_bands
    return {"image": img, "bands": ee.List(bands), "band_list": bands, "set_id": "E", "n": 43}


# ── Convenience: build all sets at once ──────────────────────────────

def build_all(s2_pre, s2_post, indices, s1, crop_mask, aoi) -> dict:
    """
    Build all 5 feature sets. Returns dict keyed by set_id ("A"–"E").
    Pass s1=None if S1 data is unavailable; set E will be skipped.
    """
    sets = {
        "A": set_a(s2_pre, s2_post, indices, crop_mask, aoi),
        "B": set_b(s2_pre, s2_post, indices, crop_mask, aoi),
        "C": set_c(s2_pre, s2_post, indices, crop_mask, aoi),
        "D": set_d(s2_pre, s2_post, indices, crop_mask, aoi),
    }
    if s1 is not None:
        sets["E"] = set_e(s2_pre, s2_post, indices, s1, crop_mask, aoi)
    return sets
