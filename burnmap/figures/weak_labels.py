#!/usr/bin/env python3
"""
Reworked Figure 3: weak-label construction schematic from real EE layers.

Improvements over the first draft
---------------------------------
1) Uses a smoother near-fire-zone rendering based on distance-to-fire,
   rather than blocky buffer tiles.
2) Switches the dNBR panel to a neutral-to-warm burn-sensitive palette.
3) Makes the spectral-gate panel a true pass mask on a neutral background.
4) Uses a lighter negative-class fill so positives dominate the label panel.
5) Adds flow arrows and a compact rule summary so the figure reads as a
   schematic rather than a simple montage.

Example
-------
python make_fig3_weak_label_schematic_reworked.py \
  --year 2024 \
  --recipe very_strict \
  --district Firozpur \
  --half-size-km 18 \
  --out figures/fig3_weak_labels.png
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Tuple

import ee
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import requests
from matplotlib.patches import FancyArrowPatch
from PIL import Image

import burnmap.config as config
from burnmap.config import recipes
import burnmap.gee.aoi as aoi_mod
import burnmap.gee.indices as indices_mod
import burnmap.gee.labels as labels_mod
import burnmap.gee.sentinel2 as sentinel2_mod
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


def fetch_thumb(img: ee.Image, region: ee.Geometry, dims: int = 900) -> np.ndarray:
    url = img.getThumbURL(
        {
            "region": region.bounds(1).coordinates().getInfo(),
            "dimensions": dims,
            "format": "png",
        }
    )
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGBA"))


# ---------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------

def get_crop_region(
    district_name: str | None,
    center_lon: float | None,
    center_lat: float | None,
    half_size_km: float,
    districts: ee.FeatureCollection,
) -> ee.Geometry:
    if center_lon is not None and center_lat is not None:
        center = ee.Geometry.Point([center_lon, center_lat])
    elif district_name:
        district = districts.filter(ee.Filter.eq("ADM2_NAME", district_name)).first()
        center = ee.Feature(district).geometry().centroid(1)
    else:
        raise ValueError("Provide either --district or both --center-lon and --center-lat")

    return center.buffer(half_size_km * 1000).bounds(1)


def build_layers(year: int, recipe_name: str, crop_region: ee.Geometry) -> dict:
    dates = config.get_dates(year)
    study = aoi_mod.build_study_area(config.ADM0, config.ADM1)
    crop_mask = study["crop_mask"]
    aoi = study["aoi"]

    comps = sentinel2_mod.build_composites(aoi, dates)
    idx = indices_mod.compute_all(comps["pre"], comps["post"])
    viirs = viirs_mod.build_fire_layers(aoi, dates)
    recipe = recipes.get_recipe(recipe_name)

    label_pack = labels_mod.build_class_image(
        recipe=recipe,
        viirs_data=viirs,
        dNBR=idx["dNBR"],
        dB12=idx["dB12"],
        dNDVI=idx["dNDVI"],
        crop_mask=crop_mask,
        aoi=aoi,
    )

    fire_buffer, dist_to_fire = viirs_mod.fire_context_for_recipe(viirs, recipe)
    fire_points = viirs["fire_hi"]

    spectral_gate = idx["dNBR"].gt(recipe["burnDNBRThr"])
    if recipe["useSwirGate"]:
        spectral_gate = spectral_gate.And(idx["dB12"].gt(recipe["swirGateThr"]))
    if recipe.get("ndviDecreaseGate") is not None:
        spectral_gate = spectral_gate.And(idx["dNDVI"].lt(-recipe["ndviDecreaseGate"]))

    return {
        "recipe": recipe,
        "crop_region": crop_region,
        "crop_mask": crop_mask.clip(crop_region),
        "fire_points": fire_points.clip(crop_region),
        "fire_buffer": fire_buffer.clip(crop_region),
        "dist_to_fire": dist_to_fire.clip(crop_region),
        "dNBR": idx["dNBR"].clip(crop_region),
        "dB12": idx["dB12"].clip(crop_region),
        "dNDVI": idx["dNDVI"].clip(crop_region),
        "spectral_gate": spectral_gate.clip(crop_region),
        "class_img": label_pack["class_img"].clip(crop_region),
    }


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------

def make_panel_images(layers: dict) -> dict:
    region = layers["crop_region"]
    recipe = layers["recipe"]

    bg = ee.Image.constant(255).toByte().clip(region).visualize(palette=["#f7f7f7"])

    # (a) Cropland mask
    crop_panel = bg.blend(layers["crop_mask"].selfMask().visualize(palette=["#4daf62"]))

    # (b) VIIRS detections + smoother near-fire zone from distance-to-fire
    # Create a smooth near-fire distance band for visualization only.
    near_dist = (
        layers["dist_to_fire"]
        .updateMask(layers["dist_to_fire"].lte(recipe["nearFireMeters"]))
        .focal_mean(radius=1200, units="meters")
        .resample("bilinear")
    )
    near_vis = near_dist.visualize(
        min=0,
        max=recipe["nearFireMeters"],
        palette=["#fff7ec", "#fdd49e", "#f16913"],
    )
    fire_vis = layers["fire_points"].selfMask().visualize(palette=["#c81d11"])
    viirs_panel = bg.blend(near_vis).blend(fire_vis)

    # (c) dNBR in a neutral-to-warm burn palette
    dnbr_panel = layers["dNBR"].visualize(
        min=0.0,
        max=0.60,
        palette=["#f7f7f7", "#fee8c8", "#fdbb84", "#e34a33", "#7f0000"],
    )

    # (d) Positive spectral gates: pass mask only on neutral background
    spectral_panel = bg.blend(layers["spectral_gate"].selfMask().visualize(palette=["#2ca25f"]))

    # (e) Final labels with lighter negatives
    class_img = layers["class_img"]
    pos = class_img.eq(1).selfMask().visualize(palette=["#d7301f"])
    neg = class_img.eq(0).selfMask().visualize(palette=["#5aa0d8"])
    unl = layers["crop_mask"].And(class_img.mask().Not()).selfMask().visualize(palette=["#d9d9d9"])
    class_panel = bg.blend(unl).blend(neg).blend(pos)

    return {
        "crop": fetch_thumb(crop_panel, region),
        "viirs": fetch_thumb(viirs_panel, region),
        "dnbr": fetch_thumb(dnbr_panel, region),
        "spectral": fetch_thumb(spectral_panel, region),
        "labels": fetch_thumb(class_panel, region),
    }


def _arrow_between_axes(fig, ax_from, ax_to, start=(0.5, 0.02), end=(0.5, 0.98), color="0.35"):
    start_disp = ax_from.transAxes.transform(start)
    end_disp = ax_to.transAxes.transform(end)
    start_fig = fig.transFigure.inverted().transform(start_disp)
    end_fig = fig.transFigure.inverted().transform(end_disp)
    arrow = FancyArrowPatch(
        start_fig,
        end_fig,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.2,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    fig.add_artist(arrow)


def compose_figure(panels: dict, recipe: dict, out_path: Path, dpi: int = 300) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.2), constrained_layout=False)
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.06, top=0.82, wspace=0.055, hspace=0.28)
    axs = axes.ravel()

    titles = [
        "(a) Cropland mask",
        "(b) VIIRS detections\nand near-fire zone",
        "(c) dNBR",
        "(d) Positive spectral gates",
        "(e) Final weak labels",
    ]
    keys = ["crop", "viirs", "dnbr", "spectral", "labels"]

    for ax, key, title in zip(axs[:5], keys, titles):
        ax.imshow(panels[key])
        ax.set_title(title, fontsize=11, pad=7)
        ax.set_xticks([])
        ax.set_yticks([])

    # Compact summary / legend panel
    axs[5].axis("off")
    axs[5].set_title("(f) Recipe and legend", fontsize=11, pad=8)
    summary_lines = [
        f"Recipe: {recipe['recipe_id']}",
        f"Near fire: {recipe['nearFireMeters']/1000:.1f} km",
        f"Far from fire: {recipe['unburnedFarMeters']/1000:.1f} km",
        f"dNBR > {recipe['burnDNBRThr']:.2f}",
    ]
    if recipe["useSwirGate"]:
        summary_lines.append(f"dB12 > {recipe['swirGateThr']}")
    if recipe.get("ndviDecreaseGate") is not None:
        summary_lines.append(f"dNDVI < -{recipe['ndviDecreaseGate']:.2f}")
    summary_text = "\n".join(summary_lines)
    axs[5].text(
        0.04,
        0.94,
        summary_text,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.75"),
    )

    legend_handles = [
        mpatches.Patch(facecolor="#d7301f", edgecolor="none", label="Weak positive"),
        mpatches.Patch(facecolor="#5aa0d8", edgecolor="none", label="Weak negative"),
        mpatches.Patch(facecolor="#d9d9d9", edgecolor="none", label="Unlabeled"),
        mpatches.Patch(facecolor="#fdbb84", edgecolor="none", label="Near-fire zone"),
        mpatches.Patch(facecolor="#2ca25f", edgecolor="none", label="Passes positive spectral gates"),
    ]
    axs[5].legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        fontsize=9,
        borderpad=0.8,
        labelspacing=0.7,
    )

    fig.suptitle("Weak-label construction schematic from real EE layers", fontsize=14, y=0.965)
    axs[4].text(
        0.5,
        0.94,
        "near-fire context\n+ burn-consistent spectral gates",
        transform=axs[4].transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#fffaf2", edgecolor="#d9b38c", alpha=0.88),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2024)
    p.add_argument(
        "--recipe",
        type=str,
        default="very_strict",
        choices=[r["recipe_id"] for r in recipes.ALL_RECIPES],
    )
    p.add_argument("--district", type=str, default="Firozpur", help="ADM2_NAME for auto-centering")
    p.add_argument("--center-lon", type=float, default=None)
    p.add_argument("--center-lat", type=float, default=None)
    p.add_argument("--half-size-km", type=float, default=18.0)
    p.add_argument("--out", type=Path, default=Path("outputs/figures/fig3_weak_labels.png"))
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_ee()

    districts = aoi_mod.load_districts(config.ADM0, config.ADM1)
    crop_region = get_crop_region(args.district, args.center_lon, args.center_lat, args.half_size_km, districts)
    layers = build_layers(args.year, args.recipe, crop_region)
    panels = make_panel_images(layers)
    compose_figure(panels, layers["recipe"], args.out, dpi=args.dpi)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
