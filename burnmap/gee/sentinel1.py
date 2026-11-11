"""
Sentinel-1 GRD compositing.

IW mode, configurable orbit pass, VV + VH polarisations.
Pre/post median composites + delta bands.
"""

import ee
import burnmap.config as config


def _s1_collection(aoi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    """Filtered S1 GRD collection."""
    s1 = (
        ee.ImageCollection(config.S1_GRD)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", config.S1_INSTRUMENT_MODE))
        .filter(ee.Filter.eq("resolution_meters", config.S1_RESOLUTION_METERS))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(config.S1_BANDS)
    )
    if config.S1_ORBIT_PASS:
        s1 = s1.filter(ee.Filter.eq("orbitProperties_pass", config.S1_ORBIT_PASS))
    return s1


def build_composites(aoi: ee.Geometry, dates: dict) -> dict:
    """
    Build pre/post S1 median composites and delta bands.

    Returns
    -------
    dict with keys:
        pre, post          – ee.Image (VV, VH)
        dVV, dVH           – ee.Image delta bands
        pre_count, post_count – ee.Number image counts
    """
    pre_col  = _s1_collection(aoi, dates["pre_start"], dates["pre_end"])
    post_col = _s1_collection(aoi, dates["post_start"], dates["post_end"])

    pre  = pre_col.median().clip(aoi)
    post = post_col.median().clip(aoi)

    dVV = post.select("VV").subtract(pre.select("VV")).rename("dVV")
    dVH = post.select("VH").subtract(pre.select("VH")).rename("dVH")

    return {
        "pre":  pre,
        "post": post,
        "dVV":  dVV,
        "dVH":  dVH,
        "pre_count":  pre_col.size(),
        "post_count": post_col.size(),
    }