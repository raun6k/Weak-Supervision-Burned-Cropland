"""
Study area definition: Punjab (and optionally Haryana) districts,
AOI geometry, cropland mask, and district raster.
"""

import ee
import burnmap.config as config


def load_districts(adm0: str = config.ADM0, adm1: str = config.ADM1) -> ee.FeatureCollection:
    """Load level-2 admin districts for a given state."""
    return (
        ee.FeatureCollection(config.GAUL_LVL2)
        .filter(ee.Filter.eq("ADM0_NAME", adm0))
        .filter(ee.Filter.eq("ADM1_NAME", adm1))
    )


def get_aoi(districts: ee.FeatureCollection) -> ee.Geometry:
    """Dissolve districts into a single AOI geometry."""
    return districts.geometry()


def rasterize_districts(districts: ee.FeatureCollection) -> ee.Image:
    """Rasterize ADM2_CODE so each pixel carries its district ID."""
    return (
        districts
        .reduceToImage(properties=["ADM2_CODE"], reducer=ee.Reducer.first())
        .rename("adm2")
    )


def load_cropland_mask(aoi: ee.Geometry) -> ee.Image:
    """ESA WorldCover v200 class 40 (cropland), clipped to AOI."""
    wc = ee.ImageCollection(config.WORLDCOVER_V200).first().select("Map")
    return wc.eq(40).clip(aoi)


def build_study_area(adm0: str = config.ADM0, adm1: str = config.ADM1) -> dict:
    """
    One-call convenience: returns everything downstream modules need.

    Returns
    -------
    dict with keys: districts, aoi, adm2_img, crop_mask
    """
    districts = load_districts(adm0, adm1)
    aoi = get_aoi(districts)
    return {
        "districts": districts,
        "aoi":       aoi,
        "adm2_img":  rasterize_districts(districts),
        "crop_mask": load_cropland_mask(aoi),
    }