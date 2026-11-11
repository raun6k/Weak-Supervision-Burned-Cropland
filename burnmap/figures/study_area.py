#!/usr/bin/env python3
"""
Reworked Figure 1: study-area map for the burned-cropland paper.

Improvements over the first draft
---------------------------------
1) Uses a true three-panel layout:
   (a) India locator with Punjab and Haryana highlighted
   (b) Punjab cropland and district boundaries
   (c) VIIRS active-fire context over the same Punjab extent
2) Improves locator context by showing the India outline and direct labels.
3) Separates cropland and VIIRS into different panels for cleaner reading.
4) Uses a clearer visual hierarchy: Punjab outline > district boundaries.
5) Adds a compact colorbar for the VIIRS 5-km contextual count surface.

Example
-------
python make_fig1_study_area_reworked.py \
  --year 2024 \
  --out outputs/figures/fig1_study_area.png
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
from typing import Tuple

import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import rasterio
import requests
from requests.exceptions import ReadTimeout, ConnectionError as RequestsConnectionError
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from rasterio.transform import array_bounds
from scipy.ndimage import gaussian_filter

import burnmap.config as config
import burnmap.gee.aoi as aoi_mod
import burnmap.gee.viirs as viirs_mod


# ---------------------------------------------------------------------
# EE helpers
# ---------------------------------------------------------------------

def init_ee() -> None:
    try:
        ee.Initialize(project=config.PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=config.PROJECT)


def geom_to_region(geom: ee.Geometry):
    return geom.bounds(1).coordinates().getInfo()


def geom_to_extent(geom: ee.Geometry) -> Tuple[float, float, float, float]:
    coords = geom.bounds(1).coordinates().getInfo()[0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (min(xs), max(xs), min(ys), max(ys))


def fetch_thumb(img: ee.Image, region: ee.Geometry, dims: int = 1200, retries: int = 3) -> np.ndarray:
    """Fetch an EE thumbnail with retries and adaptive downsizing.

    Whole-state RGB composites can occasionally time out at large dimensions.
    If the request times out, retry with a slightly smaller image.
    """
    attempt_dims = dims
    last_err = None
    for attempt in range(retries):
        url = img.getThumbURL(
            {
                "region": geom_to_region(region),
                "dimensions": attempt_dims,
                "format": "png",
            }
        )
        try:
            resp = requests.get(url, timeout=(30, 240))
            resp.raise_for_status()
            return np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGBA"))
        except (ReadTimeout, RequestsConnectionError) as e:
            last_err = e
            attempt_dims = max(700, int(attempt_dims * 0.8))
            print(f"Thumbnail request timed out; retrying with dimensions={attempt_dims}...")
    raise last_err


# ---------------------------------------------------------------------
# Cartographic helpers
# ---------------------------------------------------------------------

def add_panel_label(ax, label: str):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.9),
    )


def draw_north_arrow(ax, xy=(0.90, 0.80), size=0.10, color="black"):
    x, y = xy
    ax.text(
        x,
        y + size + 0.015,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=color,
    )
    arrow = FancyArrowPatch(
        (x, y),
        (x, y + size),
        transform=ax.transAxes,
        arrowstyle="simple",
        mutation_scale=16,
        color=color,
        linewidth=0.6,
    )
    ax.add_patch(arrow)


def add_scale_bar(ax, extent: Tuple[float, float, float, float], km: int = 100):
    xmin, xmax, ymin, ymax = extent
    lat = 0.5 * (ymin + ymax)
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat))
    deg = km / km_per_deg_lon

    x0 = xmin + 0.08 * (xmax - xmin)
    x1 = x0 + deg
    y = ymin + 0.08 * (ymax - ymin)
    tick = 0.012 * (ymax - ymin)

    ax.plot([x0, x1], [y, y], color="black", lw=1.8, solid_capstyle="butt")
    ax.plot([x0, x0], [y - tick, y + tick], color="black", lw=1.2)
    ax.plot([x1, x1], [y - tick, y + tick], color="black", lw=1.2)
    ax.text((x0 + x1) / 2, y + 0.035 * (ymax - ymin), f"{km} km", ha="center", va="bottom", fontsize=9)


def draw_local_boundaries(ax, year: int):
    districts_path = Path(f"data/fixtures/fig1_layers/fig1_{year}_punjab_districts.geojson")
    boundary_path = Path(f"data/fixtures/fig1_layers/fig1_{year}_punjab_boundary.geojson")

    def plot_boundary(geom, color: str, linewidth: float, zorder: int):
        boundary = geom.boundary
        geoms = getattr(boundary, "geoms", [boundary])
        for line in geoms:
            if line is None or line.is_empty:
                continue
            xs, ys = line.xy
            ax.plot(xs, ys, color=color, linewidth=linewidth, zorder=zorder)

    if districts_path.exists():
        for geom in gpd.read_file(districts_path).geometry:
            plot_boundary(geom, color="#666666", linewidth=0.35, zorder=8)
    if boundary_path.exists():
        for geom in gpd.read_file(boundary_path).geometry:
            plot_boundary(geom, color="#1f1f1f", linewidth=0.8, zorder=9)


# ---------------------------------------------------------------------
# Layer preparation
# ---------------------------------------------------------------------

def load_local_viirs_density(year: int):
    path = Path(f"data/fixtures/fig1_layers/fig1_{year}_viirs_fire_density_5000m.tif")
    if not path.exists():
        return None

    with rasterio.open(path) as ds:
        raw = ds.read(1).astype("float32")
        west, south, east, north = array_bounds(ds.height, ds.width, ds.transform)

    density = np.where(np.isfinite(raw) & (raw > 0), raw, 0)
    density = gaussian_filter(density, sigma=1.3)
    density = np.ma.masked_less_equal(density, 0.05)
    return {
        "arr": density,
        "extent": (west, east, south, north),
        "vmax": 7.0,
    }


def build_locator_panel() -> tuple[np.ndarray, tuple[float, float, float, float], dict]:
    india_fc = ee.FeatureCollection(config.GAUL_LVL2).filter(ee.Filter.eq("ADM0_NAME", config.ADM0))
    india_geom = india_fc.geometry().dissolve().simplify(1000)

    punjab_fc = aoi_mod.load_districts(config.ADM0, config.ADM1)
    haryana_fc = aoi_mod.load_districts(config.ADM0, config.ADM1_TRANSFER)
    punjab = punjab_fc.geometry().dissolve().simplify(500)
    haryana = haryana_fc.geometry().dissolve().simplify(500)

    india_fill = ee.Image.constant(255).toByte().paint(india_geom, 1).visualize(
        min=0, max=1, palette=["#ffffff", "#d9d9d9"]
    )
    haryana_fill = ee.Image().byte().paint(haryana, 1).visualize(min=0, max=1, palette=["#fdd49e"])
    punjab_fill = ee.Image().byte().paint(punjab, 1).visualize(min=0, max=1, palette=["#d95f0e"])
    india_outline = ee.Image().byte().paint(india_geom, 1, 1).visualize(min=0, max=1, palette=["#666666"])
    state_outlines = (
        ee.Image().byte().paint(haryana, 1, 1).visualize(min=0, max=1, palette=["#8c8c8c"])
        .blend(ee.Image().byte().paint(punjab, 1, 1).visualize(min=0, max=1, palette=["#222222"]))
    )

    img = india_fill.blend(haryana_fill).blend(punjab_fill).blend(india_outline).blend(state_outlines)
    arr = fetch_thumb(img, india_geom, dims=800)
    extent = geom_to_extent(india_geom)

    label_xy = {
        "Punjab": ee.Feature(punjab.centroid(1)).geometry().coordinates().getInfo(),
        "Haryana": ee.Feature(haryana.centroid(1)).geometry().coordinates().getInfo(),
    }
    return arr, extent, label_xy


def build_main_layers(year: int):
    dates = config.get_dates(year)
    study = aoi_mod.build_study_area(config.ADM0, config.ADM1)
    districts = study["districts"]
    aoi = study["aoi"]
    crop_mask = study["crop_mask"]

    extent_geom = aoi.buffer(25000).bounds(1)
    extent = geom_to_extent(extent_geom)

    # Panel (b): cropland + district boundaries.
    base = ee.Image.constant(255).toByte().clip(extent_geom).visualize(palette=["#f7f7f7"])
    crop = crop_mask.selfMask().visualize(palette=["#c7e9c0"])
    crop_context = crop_mask.selfMask().visualize(palette=["#edf8e9"])
    district_lines = ee.Image().byte().paint(districts, 1, 1).visualize(min=0, max=1, palette=["#6b6b6b"])
    aoi_outline = ee.Image().byte().paint(aoi, 1, 2).visualize(min=0, max=1, palette=["#1f1f1f"])
    cropland_panel = base.blend(crop).blend(district_lines).blend(aoi_outline)

    # Panel (c): smoothed active-fire count surface.
    viirs_layers = viirs_mod.build_fire_layers(aoi, dates)
    fire_collection = viirs_layers["collection"]
    fire_count = ee.Image(
        ee.Algorithms.If(
            fire_collection.size().gt(0),
            fire_collection
            .map(lambda img: img.select("confidence").eq(2).rename("fire"))
            .sum()
            .unmask(0)
            .toFloat()
            .clip(aoi),
            ee.Image.constant(0).rename("fire").toFloat().clip(aoi),
        )
    )

    # Use the same 5 km contextual count used by the weak-label framing. A
    # light-to-dark sequential ramp keeps isolated low-count context subtle
    # while making clustered fire activity visually dominant.
    fire_density = fire_count.reduceNeighborhood(
        reducer=ee.Reducer.sum(),
        kernel=ee.Kernel.circle(radius=5000, units="meters"),
        skipMasked=False,
    ).rename("fire_density_5km")
    fire_density = fire_density.where(fire_density.gt(6), 6)

    viirs_palette = [
        "#f1eef6", "#d7b5d8", "#df65b0", "#dd1c77", "#980043", "#67001f"
    ]
    viirs_overlay = fire_density.updateMask(fire_density.gt(0)).visualize(
        min=1,
        max=6,
        palette=viirs_palette,
    )
    local_viirs_density = load_local_viirs_density(year)
    if local_viirs_density:
        viirs_panel = base.blend(crop_context).blend(district_lines).blend(aoi_outline)
    else:
        viirs_panel = base.blend(crop_context).blend(viirs_overlay).blend(district_lines).blend(aoi_outline)

    return {
        "cropland_arr": fetch_thumb(cropland_panel, extent_geom, dims=1100),
        "viirs_arr": fetch_thumb(viirs_panel, extent_geom, dims=1000),
        "extent": extent,
        "viirs_cmap": mpl.colors.LinearSegmentedColormap.from_list("viirs_context", viirs_palette),
        "viirs_norm": mpl.colors.Normalize(
            vmin=0.25 if local_viirs_density else 1,
            vmax=local_viirs_density["vmax"] if local_viirs_density else 6,
        ),
        "viirs_density": local_viirs_density,
        "year": year,
    }


# ---------------------------------------------------------------------
# Figure composition
# ---------------------------------------------------------------------

def compose_figure(locator_arr, locator_extent, label_xy, main_layers, out_path: Path, dpi: int = 300):
    fig = plt.figure(figsize=(14.2, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 1.28])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    for ax in (ax0, ax1, ax2):
        ax.set_facecolor("#f7f7f7")

    # (a) Locator
    ax0.imshow(locator_arr, extent=locator_extent, origin="upper")
    ax0.set_title("Northwestern India locator", fontsize=12)
    add_panel_label(ax0, "(a)")
    ax0.set_xticks([])
    ax0.set_yticks([])

    pxy = label_xy["Punjab"]
    hxy = label_xy["Haryana"]
    ax0.text(pxy[0] - 1.1, pxy[1] + 0.8, "Punjab", fontsize=10, fontweight="bold", color="#7f2704")
    ax0.text(hxy[0] + 0.25, hxy[1] - 0.7, "Haryana", fontsize=10, color="#7a7a7a")
    locator_handles = [
        mpatches.Patch(facecolor="#d95f0e", edgecolor="#222222", label="Punjab"),
        mpatches.Patch(facecolor="#fdd49e", edgecolor="#8c8c8c", label="Haryana"),
    ]
    ax0.legend(handles=locator_handles, loc="lower left", frameon=True, framealpha=0.95, fontsize=9)

    # (b) Cropland and districts
    ax1.imshow(main_layers["cropland_arr"], extent=main_layers["extent"], origin="upper")
    ax1.set_title("Punjab cropland and districts", fontsize=12)
    add_panel_label(ax1, "(b)")
    ax1.set_xticks([])
    ax1.set_yticks([])
    add_scale_bar(ax1, main_layers["extent"], km=100)
    draw_north_arrow(ax1, xy=(0.92, 0.78), size=0.10)
    ax1.legend(
        handles=[mpatches.Patch(facecolor="#c7e9c0", edgecolor="none", label="ESA WorldCover cropland")],
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        fontsize=9,
    )

    # (c) VIIRS context
    im = ax2.imshow(main_layers["viirs_arr"], extent=main_layers["extent"], origin="upper")
    if main_layers.get("viirs_density"):
        ax2.imshow(
            main_layers["viirs_density"]["arr"],
            extent=main_layers["viirs_density"]["extent"],
            origin="upper",
            cmap=main_layers["viirs_cmap"],
            norm=main_layers["viirs_norm"],
            alpha=0.68,
            interpolation="bilinear",
        )
        draw_local_boundaries(ax2, main_layers["year"])
    ax2.set_title(f"VIIRS active-fire context, Oct-Nov {main_layers['year']}", fontsize=12)
    add_panel_label(ax2, "(c)")
    ax2.set_xticks([])
    ax2.set_yticks([])
    add_scale_bar(ax2, main_layers["extent"], km=100)
    draw_north_arrow(ax2, xy=(0.90, 0.78), size=0.10)

    sm = mpl.cm.ScalarMappable(norm=main_layers["viirs_norm"], cmap=main_layers["viirs_cmap"])
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, fraction=0.048, pad=0.03)
    if main_layers.get("viirs_density"):
        cbar.set_ticks([0.25, 1, 2, 4, 7])
        cbar.set_ticklabels(["0.25", "1", "2", "4", "7+"])
        cbar.set_label("Smoothed VIIRS high-confidence fire density (5 km)", fontsize=10)
    else:
        cbar.set_ticks([1, 2, 3, 4, 5, 6])
        cbar.set_ticklabels(["1", "2", "3", "4", "5", "6+"])
        cbar.set_label("VIIRS high-confidence fire count within 5 km", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle("Figure 1. Study area for post-harvest burned-cropland mapping", fontsize=14, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--out", type=Path, default=Path("outputs/figures/fig1_study_area.png"))
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_ee()
    locator_arr, locator_extent, label_xy = build_locator_panel()
    main_layers = build_main_layers(args.year)
    compose_figure(locator_arr, locator_extent, label_xy, main_layers, args.out, dpi=args.dpi)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
