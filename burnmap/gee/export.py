"""
Generic GEE export helpers.

Thin wrappers around ee.batch.Export that enforce consistent
naming conventions and folder structure.
"""

import ee
import burnmap.config as config


def image_to_drive(
    image: ee.Image,
    description: str,
    aoi: ee.Geometry,
    folder: str = config.DRIVE_FOLDER,
    scale: int = config.EXPORT_SCALE,
    max_pixels: float = config.MAX_PIXELS,
) -> ee.batch.Task:
    """Export an ee.Image to Google Drive as GeoTIFF."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=aoi,
        scale=scale,
        maxPixels=max_pixels,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def table_to_drive(
    collection: ee.FeatureCollection,
    description: str,
    folder: str = config.DRIVE_FOLDER,
) -> ee.batch.Task:
    """Export an ee.FeatureCollection to Google Drive as CSV."""
    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description,
        folder=folder,
        fileFormat="CSV",
    )
    task.start()
    return task


def make_run_prefix(year: int, phase: str) -> str:
    """Consistent naming: punjab_2024_phase1, punjab_2024_phase3, etc."""
    return f"punjab_{year}_{phase}"