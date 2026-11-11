"""
VIIRS active-fire data: merged SNPP + NOAA-20.

Builds:
  - binary fire images (high-confidence and nominal+high)
  - buffered fire regions
  - distance-to-fire rasters
  - fire detection counts for gate checks (two methods):
      • detection_count — sum of per-image fire pixels across all dates
        (each overpass contributes separately; this is the true detection
        count that matches official VIIRS fire-count statistics)
      • unique_pixel_count — number of unique 375m pixels that fired at
        least once (from the max-composite; always ≤ detection_count)

This patched version handles empty VIIRS collections gracefully for years
or regions with no available detections in the selected window. Instead of
crashing on Image.unmask() with a 0-band image, it returns a zero-valued
placeholder fire image and zero detection counts.
"""

import ee
import burnmap.config as config


def _merged_collection(aoi: ee.Geometry, dates: dict) -> ee.ImageCollection:
    """Merge SNPP and NOAA-20 VIIRS over the season window."""
    return (
        ee.ImageCollection(config.VIIRS_SNPP)
        .merge(ee.ImageCollection(config.VIIRS_NOAA20))
        .filterBounds(aoi)
        .filterDate(dates["season_start"], dates["season_end"])
    )


def _zero_fire_image(aoi: ee.Geometry) -> ee.Image:
    """
    Return a 1-band zero fire image with a stable 375 m projection.

    This prevents downstream Image.unmask() / reduceRegion() failures when
    the filtered VIIRS collection is empty for a given year or region.
    """
    return (
        ee.Image.constant(0)
        .rename("fire")
        .toByte()
        .reproject(crs="EPSG:4326", scale=config.VIIRS_SCALE)
        .clip(aoi)
    )


def _fire_image(collection: ee.ImageCollection, mode: str, aoi: ee.Geometry) -> ee.Image:
    """
    Create a binary fire max-composite.
    mode = "hi"      → confidence == 2 (high only)
    mode = "hi_nom"  → confidence >= 1 (nominal + high)

    If the collection is empty, return a zero-valued placeholder image
    instead of a 0-band image.
    """
    if mode == "hi":
        mapped = collection.map(lambda img: img.select("confidence").eq(2).rename("fire"))
    else:
        mapped = collection.map(lambda img: img.select("confidence").gte(1).rename("fire"))

    return ee.Image(
        ee.Algorithms.If(
            collection.size().gt(0),
            mapped.max().unmask(0).clip(aoi),
            _zero_fire_image(aoi),
        )
    )


def _make_fire_dist(fire_img: ee.Image, aoi: ee.Geometry):
    """Create a 500m-buffered fire region and distance-to-fire raster."""
    proj = fire_img.projection()
    fire_buffer = fire_img.focal_max(radius=500, units="meters").clip(aoi)
    fire_reproj = fire_buffer.selfMask().reproject(crs=proj, scale=config.VIIRS_SCALE)
    dist = (
        fire_reproj.distance(ee.Kernel.euclidean(20000, "meters"))
        .unmask(1e9)
        .reproject(crs=proj, scale=config.VIIRS_SCALE)
        .rename("distFire")
    )
    return fire_buffer, dist


def _count_detections(collection: ee.ImageCollection, mode: str,
                      aoi: ee.Geometry) -> ee.Number:
    """
    Count total fire detections across ALL images (per-overpass).

    Each image contributes its own fire pixels independently,
    so a pixel that burns on 5 different dates counts as 5 detections.
    This matches how official VIIRS fire-count statistics work.

    If the collection is empty, return 0.
    """
    if mode == "hi":
        mapped = collection.map(lambda img: img.select("confidence").eq(2).rename("fire"))
    else:
        mapped = collection.map(lambda img: img.select("confidence").gte(1).rename("fire"))

    return ee.Number(
        ee.Algorithms.If(
            collection.size().gt(0),
            ee.Number(
                mapped.sum().clip(aoi).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=aoi,
                    scale=config.VIIRS_SCALE,
                    maxPixels=1e9,
                    tileScale=4,
                ).get("fire")
            ),
            0,
        )
    )


def _count_unique_pixels(fire_max_img: ee.Image, aoi: ee.Geometry) -> ee.Number:
    """
    Count unique 375m pixels that fired at least once (from max-composite).
    Always ≤ detection_count.
    """
    return ee.Number(
        fire_max_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=config.VIIRS_SCALE,
            maxPixels=1e9,
            tileScale=4,
        ).get("fire")
    )


def build_fire_layers(aoi: ee.Geometry, dates: dict) -> dict:
    """
    Build all VIIRS fire products.

    Returns
    -------
    dict with keys:
        fire_hi, fire_hinom              – ee.Image binary max-composites
        buffer_hi, buffer_hinom          – ee.Image buffered fire regions
        dist_hi, dist_hinom              – ee.Image distance-to-fire
        detection_count_hi               – ee.Number total detections across
                                           all overpasses (gate check metric)
        unique_pixel_count_hi            – ee.Number unique spatial pixels
        collection                       – raw merged ImageCollection
    """
    col = _merged_collection(aoi, dates)

    fire_hi    = _fire_image(col, "hi", aoi)
    fire_hinom = _fire_image(col, "hi_nom", aoi)

    buf_hi,    dist_hi    = _make_fire_dist(fire_hi, aoi)
    buf_hinom, dist_hinom = _make_fire_dist(fire_hinom, aoi)

    detection_count_hi = _count_detections(col, "hi", aoi)
    unique_pixel_count_hi = _count_unique_pixels(fire_hi, aoi)

    return {
        "fire_hi": fire_hi,       "fire_hinom": fire_hinom,
        "buffer_hi": buf_hi,      "buffer_hinom": buf_hinom,
        "dist_hi": dist_hi,       "dist_hinom": dist_hinom,
        "detection_count_hi":     detection_count_hi,
        "unique_pixel_count_hi":  unique_pixel_count_hi,
        "collection": col,
    }


def fire_context_for_recipe(viirs: dict, recipe: dict):
    """
    Select the appropriate fire buffer and distance image
    for a given weak-label recipe.
    """
    if recipe["viirs_mode"] == "hi":
        return viirs["buffer_hi"], viirs["dist_hi"]
    return viirs["buffer_hinom"], viirs["dist_hinom"]
