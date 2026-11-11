"""
Sentinel-2 cloud masking and median compositing.

Supports three masking strategies (configured in config.py):
  CSPLUS     – Cloud Score+ cs_cdf ≥ threshold, combined with SCL + QA60
  CLOUD_PROB – S2 cloud probability ≤ threshold, combined with SCL + QA60
  QA60_SCL   – QA60 + SCL only (no external product)
"""

import ee
import burnmap.config as config


# ── Pixel-level masks ────────────────────────────────────────────────

def _scl_mask(img: ee.Image) -> ee.Image:
    """Mask cloud shadow (3), med/high cloud (8,9), cirrus (10), snow (11)."""
    scl = img.select("SCL")
    return (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )


def _qa60_mask(img: ee.Image) -> ee.Image:
    """Mask opaque clouds (bit 10) and cirrus (bit 11) from QA60."""
    qa = img.select("QA60")
    return (
        qa.bitwiseAnd(1 << 10).eq(0)
        .And(qa.bitwiseAnd(1 << 11).eq(0))
    )


# ── Collection builders ─────────────────────────────────────────────

def _base_collection(aoi: ee.Geometry, date_start: str, date_end: str) -> ee.ImageCollection:
    """Filtered S2 SR collection: bounds + date range + cloud %."""
    return (
        ee.ImageCollection(config.S2_SR)
        .filterBounds(aoi)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", config.MAX_S2_CLOUDY_PIXEL_PCT))
    )


def _build_csplus(aoi, date_start, date_end):
    s2 = _base_collection(aoi, date_start, date_end)
    cs = (
        ee.ImageCollection(config.CLOUD_SCORE_PLUS)
        .filterBounds(aoi)
        .filterDate(date_start, date_end)
    )
    joined = ee.Join.saveFirst("csplus").apply(
        primary=s2, secondary=cs,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    def _mask(img):
        img = ee.Image(img)
        cs_img = ee.Image(img.get("csplus"))
        clear = cs_img.select(config.CSPLUS_BAND).gte(config.CSPLUS_CLEAR_THR)
        mask = clear.And(_scl_mask(img)).And(_qa60_mask(img))
        return (
            img.updateMask(mask)
            .select(config.S2_BANDS)
            .copyProperties(img, ["system:time_start", "system:index"])
        )

    return ee.ImageCollection(joined).map(_mask)


def _build_cloud_prob(aoi, date_start, date_end):
    s2 = _base_collection(aoi, date_start, date_end)
    cp = (
        ee.ImageCollection(config.S2_CLOUD_PROB)
        .filterBounds(aoi)
        .filterDate(date_start, date_end)
    )
    joined = ee.Join.saveFirst("cloudprob").apply(
        primary=s2, secondary=cp,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    def _mask(img):
        img = ee.Image(img)
        cp_img = ee.Image(img.get("cloudprob"))
        clear = cp_img.select("probability").lte(config.CLOUD_PROB_THR)
        mask = clear.And(_scl_mask(img)).And(_qa60_mask(img))
        return (
            img.updateMask(mask)
            .select(config.S2_BANDS)
            .copyProperties(img, ["system:time_start", "system:index"])
        )

    return ee.ImageCollection(joined).map(_mask)


def _build_qa60_only(aoi, date_start, date_end):
    s2 = _base_collection(aoi, date_start, date_end)

    def _mask(img):
        img = ee.Image(img)
        mask = _qa60_mask(img).And(_scl_mask(img))
        return (
            img.updateMask(mask)
            .select(config.S2_BANDS)
            .copyProperties(img, ["system:time_start", "system:index"])
        )

    return s2.map(_mask)


# ── Public API ───────────────────────────────────────────────────────

def masked_collection(aoi: ee.Geometry, dates: dict) -> ee.ImageCollection:
    """
    Return a cloud-masked S2 ImageCollection spanning pre_start → post_end.

    Parameters
    ----------
    aoi   : ee.Geometry
    dates : dict with keys pre_start, post_end (at minimum)
    """
    builders = {
        "CSPLUS":     _build_csplus,
        "CLOUD_PROB": _build_cloud_prob,
        "QA60_SCL":   _build_qa60_only,
    }
    build_fn = builders[config.CLOUD_MASK_METHOD]
    return build_fn(aoi, dates["pre_start"], dates["post_end"])


def build_composites(aoi: ee.Geometry, dates: dict) -> dict:
    """
    Build pre-event and post-event median composites.

    Returns
    -------
    dict with keys:
        collection – the full masked ImageCollection
        pre        – ee.Image median composite (original band names)
        post       – ee.Image median composite (original band names)
    """
    col = masked_collection(aoi, dates)
    pre  = col.filterDate(dates["pre_start"], dates["pre_end"]).median().clip(aoi)
    post = col.filterDate(dates["post_start"], dates["post_end"]).median().clip(aoi)
    return {"collection": col, "pre": pre, "post": post}