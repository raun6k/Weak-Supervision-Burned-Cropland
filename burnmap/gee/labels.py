"""
Weak-label construction from VIIRS fire context + spectral rules.

Each recipe defines thresholds for:
  - Positive labels: near fire + dNBR above threshold (+ optional SWIR/NDVI gates)
  - Negative labels: far from fire OR near fire with low spectral change

The class image returned has:
  1 = weak positive (burned)
  0 = weak negative (unburned)
  masked = unlabelled (neither rule triggers)
"""

import ee
import burnmap.gee.viirs as viirs_mod


def build_class_image(
    recipe: dict,
    viirs_data: dict,
    dNBR: ee.Image,
    dB12: ee.Image,
    dNDVI: ee.Image,
    crop_mask: ee.Image,
    aoi: ee.Geometry,
) -> dict:
    """
    Construct weak-label class image for a given recipe.

    Parameters
    ----------
    recipe     : dict from recipes.py (strict / moderate / very_strict)
    viirs_data : dict from viirs.build_fire_layers()
    dNBR       : ee.Image (pre minus post NBR) — for label rules only
    dB12       : ee.Image (post minus pre B12)
    dNDVI      : ee.Image (post minus pre NDVI)
    crop_mask  : ee.Image (1 = cropland)
    aoi        : ee.Geometry

    Returns
    -------
    dict with keys:
        class_img    – ee.Image (0/1, masked to labelled pixels only)
        fire_buffer  – ee.Image (buffered fire context)
        dist_to_fire – ee.Image (distance to nearest fire)
    """
    fire_buffer, dist_to_fire = viirs_mod.fire_context_for_recipe(viirs_data, recipe)

    # Unmask dNBR so unlabelled areas don't cause nulls in comparisons
    dNBR_safe = dNBR.unmask(-999)

    # ── Positive (burned) mask ───────────────────────────────────────
    burned = fire_buffer.eq(1).And(dNBR_safe.gt(recipe["burnDNBRThr"]))

    # Optional SWIR gate
    if recipe["useSwirGate"]:
        burned = burned.And(dB12.gt(recipe["swirGateThr"]))

    # Optional NDVI decrease gate (new in very_strict)
    if recipe.get("ndviDecreaseGate") is not None:
        # dNDVI = post - pre; a decrease means dNDVI < 0
        # Gate: require NDVI drop > threshold (i.e. dNDVI < -threshold)
        burned = burned.And(dNDVI.lt(-recipe["ndviDecreaseGate"]))

    # ── Negative (unburned) mask ─────────────────────────────────────
    # Far from fire
    unburned_far = dist_to_fire.gt(recipe["unburnedFarMeters"])

    # Near fire but spectrally quiet (hard negatives)
    unburned_near = (
        dist_to_fire.lt(recipe["nearFireMeters"])
        .And(dNBR_safe.lt(recipe["hardNeg_dNBR_Max"]))
        .And(dB12.lt(recipe["hardNeg_dB12_Max"]))
    )
    unburned = unburned_far.Or(unburned_near)

    # ── Class image ──────────────────────────────────────────────────
    class_img = (
        ee.Image(0)
        .where(burned, 1)
        .rename("class")
        .toInt16()
        .updateMask(burned.Or(unburned))
        .updateMask(crop_mask)
        .clip(aoi)
    )

    return {
        "class_img":    class_img,
        "fire_buffer":  fire_buffer,
        "dist_to_fire": dist_to_fire,
    }


def label_pixel_counts(class_img: ee.Image, aoi: ee.Geometry) -> ee.Dictionary:
    """
    Count total positive and negative labelled pixels at coarse scale.
    Returns ee.Dictionary with keys "0" and "1".
    """
    return ee.Dictionary(
        class_img.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=aoi,
            scale=120,  # coarse for speed; exact counts come from sampling
            maxPixels=1e8,
            tileScale=4,
        ).get("class")
    )


def per_district_label_counts(
    class_img: ee.Image,
    districts: ee.FeatureCollection,
    recipe_id: str,
) -> ee.FeatureCollection:
    """
    Count positive and negative labels per district.
    Returns FeatureCollection with one feature per district,
    each carrying label_0, label_1, and recipe_id properties.
    """
    def count_in_district(district):
        geom = district.geometry()
        hist = ee.Dictionary(
            class_img.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=geom,
                scale=120,
                maxPixels=1e8,
                tileScale=4,
            ).get("class")
        )
        n0 = ee.Number(hist.get("0", 0))
        n1 = ee.Number(hist.get("1", 0))
        return district.set({
            "recipe_id": recipe_id,
            "label_0": n0,
            "label_1": n1,
        })

    return districts.map(count_in_district)


def label_modis_overlap(
    class_img: ee.Image,
    modis_burn_mask: ee.Image,
    aoi: ee.Geometry,
) -> ee.Dictionary:
    """
    Compute overlap between positive weak labels and MODIS burn mask.
    Both are compared at 500m scale (MODIS native) to avoid expensive
    reprojection. This slightly underestimates overlap but is fast and
    stays within GEE aggregation limits.
    Returns dict with total_pos_labels, pos_in_modis, overlap_fraction.
    """
    pos_labels = class_img.eq(1).selfMask()
    modis_binary = modis_burn_mask.unmask(0)

    # Reduce both at 500m — positive labels get aggregated (any pos label
    # in a 500m pixel counts), then intersected with MODIS
    total_pos = ee.Number(pos_labels.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi,
        scale=500, maxPixels=1e8, tileScale=4,
    ).get("class"))

    overlap = pos_labels.And(modis_binary.eq(1))
    overlap_count = ee.Number(overlap.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi,
        scale=500, maxPixels=1e8, tileScale=4,
    ).get("class"))

    frac = overlap_count.divide(total_pos.max(1))

    return ee.Dictionary({
        "total_pos_labels": total_pos,
        "pos_in_modis": overlap_count,
        "overlap_fraction": frac,
    })