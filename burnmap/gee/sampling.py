"""
Stratified sampling helpers for cross-validation.

Samples labeled pixels within specific districts for
train/test splits. Returns ee.FeatureCollection objects
with features, class label, district name, and coordinates.

Core module: returns ee objects, NEVER calls ee.batch.Export.
"""

import ee
import burnmap.config as config


def _district_filter(districts: ee.FeatureCollection,
                     district_names: list[str]) -> ee.FeatureCollection:
    """Filter a districts FeatureCollection to a list of names."""
    return districts.filter(ee.Filter.inList("ADM2_NAME", district_names))


def stratified_sample(
    class_img: ee.Image,
    feature_img: ee.Image,
    band_list: list[str],
    districts: ee.FeatureCollection,
    district_names: list[str],
    n_per_class: int,
    seed: int,
    adm2_img: ee.Image,
) -> ee.FeatureCollection:
    """
    Stratified sample of labeled pixels within specified districts.

    Samples n_per_class pixels for class 0 and class 1, restricted
    to pixels within the given district geometries. Each sample point
    carries all feature bands, class label, district code, and lat/lon.

    Parameters
    ----------
    class_img      : ee.Image with band "class" (0/1, masked elsewhere)
    feature_img    : ee.Image with all feature bands
    band_list      : list of feature band names
    districts      : ee.FeatureCollection of all districts
    district_names : list of district names to sample from
    n_per_class    : number of pixels per class
    seed           : random seed for reproducibility
    adm2_img       : ee.Image with "adm2" band (district codes)

    Returns
    -------
    ee.FeatureCollection with properties:
        all feature bands, "class", "adm2", ".geo" (coordinates)
    """
    subset = _district_filter(districts, district_names)
    region = subset.geometry()

    # Stack features (float) + class (must stay integer for stratifiedSample)
    # + district ID. Cast features to float first, then add integer bands.
    stack = (
        feature_img.toFloat()
        .addBands(class_img.select("class").toInt16())
        .addBands(adm2_img.select("adm2").toInt32())
    )

    sample = stack.stratifiedSample(
        numPoints=n_per_class,
        classBand="class",
        region=region,
        scale=config.EXPORT_SCALE,
        seed=seed,
        dropNulls=True,
        tileScale=4,
        geometries=True,  # retain point coordinates
    )

    return sample


def sample_for_fold(
    class_img: ee.Image,
    feature_img: ee.Image,
    band_list: list[str],
    districts: ee.FeatureCollection,
    adm2_img: ee.Image,
    train_districts: list[str],
    test_districts: list[str],
    n_per_class_train: int = config.N_PER_CLASS,
    n_per_class_test: int = config.N_PER_CLASS,
    seed: int = 13,
) -> dict:
    """
    Sample train and test sets for one CV fold.

    Returns
    -------
    dict with keys:
        train : ee.FeatureCollection
        test  : ee.FeatureCollection
    """
    train = stratified_sample(
        class_img, feature_img, band_list, districts,
        train_districts, n_per_class_train, seed, adm2_img,
    )
    test = stratified_sample(
        class_img, feature_img, band_list, districts,
        test_districts, n_per_class_test, seed, adm2_img,
    )
    return {"train": train, "test": test}


def annotate_sample(
    sample: ee.FeatureCollection,
    recipe_id: str,
    set_id: str,
    seed: int,
    fold_id: int,
    split: str,
) -> ee.FeatureCollection:
    """
    Add metadata properties to a sample FeatureCollection
    for provenance tracking in exported CSVs.
    """
    def _add_props(feat):
        feat = ee.Feature(feat)
        return feat.set({
            "recipe_id": recipe_id,
            "set_id": set_id,
            "seed": seed,
            "fold_id": fold_id,
            "split": split,
        })

    return sample.map(_add_props)