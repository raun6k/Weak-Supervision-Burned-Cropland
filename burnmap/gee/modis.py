"""
MODIS MCD64A1 burned-area reference data.

Loads monthly burn-date layers for the target Oct–Nov window
and creates a binary burn mask on cropland.
"""

import ee
import burnmap.config as config


def build_burn_mask(aoi: ee.Geometry, crop_mask: ee.Image, dates: dict) -> dict:
    """
    Binary MODIS burn mask: 1 where BurnDate > 0 in Oct–Nov, on cropland.

    Returns
    -------
    dict with keys:
        burn_mask        – ee.Image (0/1, masked to cropland)
        burn_date_max    – ee.Image (raw max BurnDate, unmasked)
        burned_pixel_count – ee.Number (for gate check)
    """
    modis = (
        ee.ImageCollection(config.MODIS_MCD64A1)
        .filterDate(dates["modis_start"], dates["modis_end"])
        .select("BurnDate")
    )

    burn_date_max = modis.max().clip(aoi)
    burn_mask = burn_date_max.gt(0).rename("modis_burned").updateMask(crop_mask)

    burned_px = ee.Number(
        burn_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=config.MODIS_SCALE,
            maxPixels=1e9,
            tileScale=4,
        ).get("modis_burned")
    )

    return {
        "burn_mask":         burn_mask,
        "burn_date_max":     burn_date_max,
        "burned_pixel_count": burned_px,
    }