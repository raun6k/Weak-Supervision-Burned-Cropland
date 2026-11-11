#!/usr/bin/env python3
"""
Create final burned-area product map figure for Punjab.

Figure panels:
(a) Probability surface
(b) Final thresholded burned-area product
(c) MODIS burned-area reference
(d) Agreement/disagreement map at MODIS scale

Expected inputs in outputs/2024/phase5_maps/:
    *very_strict*_A_prob*.tif
    *MODIS*.tif
    *adm2*.tif

Outputs:
    outputs/2024/paper/fig_final_burned_area_map.png
    outputs/2024/paper/fig_final_burned_area_map.pdf
    outputs/2024/paper/fig_final_burned_area_map_caption.txt
"""

from __future__ import annotations

from pathlib import Path
import argparse
import glob
import tempfile

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

import rasterio
from rasterio.enums import Resampling
from rasterio.plot import plotting_extent
from rasterio.vrt import WarpedVRT


def find_first(patterns: list[str]) -> Path | None:
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return Path(matches[0])
    return None


def downsample_shape(width: int, height: int, max_dim: int) -> tuple[int, int]:
    scale = max(width / max_dim, height / max_dim, 1.0)
    out_w = max(1, int(round(width / scale)))
    out_h = max(1, int(round(height / scale)))
    return out_h, out_w


def read_raster_for_plot(
    path: Path,
    max_dim: int = 1800,
    resampling: Resampling = Resampling.average,
) -> tuple[np.ndarray, tuple[float, float, float, float], rasterio.Affine, rasterio.crs.CRS]:
    with rasterio.open(path) as src:
        out_h, out_w = downsample_shape(src.width, src.height, max_dim)
        arr = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=resampling,
            masked=True,
        )

        transform = src.transform * src.transform.scale(
            src.width / out_w,
            src.height / out_h,
        )
        extent = plotting_extent(arr, transform)

        return arr, extent, transform, src.crs


def make_boundary_mask(adm2_arr: np.ndarray) -> np.ndarray:
    """
    Create a simple raster boundary mask from district-ID raster.
    This avoids needing geopandas/shapefiles.
    """
    arr = np.asarray(adm2_arr)
    if np.ma.isMaskedArray(adm2_arr):
        filled = adm2_arr.filled(-9999)
    else:
        filled = arr.copy()

    valid = filled > 0
    b = np.zeros(filled.shape, dtype=bool)

    diff_v = (filled[:-1, :] != filled[1:, :]) & valid[:-1, :] & valid[1:, :]
    b[:-1, :] |= diff_v
    b[1:, :] |= diff_v

    diff_h = (filled[:, :-1] != filled[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    b[:, :-1] |= diff_h
    b[:, 1:] |= diff_h

    return b


def overlay_boundaries(ax, adm2_arr, extent, alpha: float = 0.55, linewidth_like: bool = False):
    """
    Overlay district boundaries derived from the ADM2 raster.
    """
    boundary = make_boundary_mask(adm2_arr)
    boundary_masked = np.ma.masked_where(~boundary, boundary)

    cmap = ListedColormap([(0, 0, 0, 1)])
    ax.imshow(
        boundary_masked,
        extent=extent,
        origin="upper",
        cmap=cmap,
        interpolation="nearest",
        alpha=alpha,
        zorder=10,
    )


def make_agreement_at_modis_scale(
    prob_tif: Path,
    modis_tif: Path,
    adm2_tif: Path,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """
    Build agreement map on the MODIS grid.

    Categories:
        0 = no data
        1 = neither burned
        2 = both burned
        3 = mapped only
        4 = MODIS only
    """
    with rasterio.open(prob_tif) as src_prob, rasterio.open(modis_tif) as src_mod, rasterio.open(adm2_tif) as src_adm2:
        with tempfile.TemporaryDirectory() as tmpdir:
            binary_tif = Path(tmpdir) / "native_thresholded_burn.tif"
            profile = src_prob.profile.copy()
            profile.update(dtype="uint8", nodata=255, count=1, compress="lzw")

            # Threshold at native probability-map support without materializing
            # the full native raster in memory, then aggregate to MODIS.
            with rasterio.open(binary_tif, "w", **profile) as dst_binary:
                for _, window in src_prob.block_windows(1):
                    prob_block = src_prob.read(1, window=window, masked=True)
                    prob_data = prob_block.filled(np.nan).astype("float32")
                    binary_block = np.full(prob_data.shape, 255, dtype="uint8")
                    valid = np.isfinite(prob_data)
                    binary_block[valid] = (prob_data[valid] >= threshold).astype("uint8")
                    dst_binary.write(binary_block, 1, window=window)

            with rasterio.open(binary_tif) as src_binary:
                with WarpedVRT(
                    src_binary,
                    crs=src_mod.crs,
                    transform=src_mod.transform,
                    width=src_mod.width,
                    height=src_mod.height,
                    resampling=Resampling.average,
                    nodata=255,
                    src_nodata=255,
                    dtype="float32",
                ) as vrt:
                    mapped_fraction = vrt.read(1, masked=True, out_dtype="float32").filled(-9999.0)

        # Majority/area-fraction rule at MODIS scale.
        mapped = np.where(
            mapped_fraction == -9999.0,
            -1,
            (mapped_fraction >= 0.5).astype("int16"),
        )

        modis = src_mod.read(1, masked=True).astype("float32")
        modis_filled = modis.filled(-9999.0)
        modis_binary = np.where(
            modis_filled == -9999.0,
            -1,
            (modis_filled > 0).astype("int16"),
        )

        # Reproject district raster to MODIS grid for boundaries.
        with WarpedVRT(
            src_adm2,
            crs=src_mod.crs,
            transform=src_mod.transform,
            width=src_mod.width,
            height=src_mod.height,
            resampling=Resampling.nearest,
            nodata=-9999,
            src_nodata=-9999,
        ) as vrt:
            adm2_modis = vrt.read(1, masked=True).filled(-9999).astype("int32")

        valid = (mapped >= 0) & (modis_binary >= 0) & (adm2_modis > 0)

        agreement = np.zeros(mapped.shape, dtype="uint8")
        agreement[valid & (mapped == 0) & (modis_binary == 0)] = 1
        agreement[valid & (mapped == 1) & (modis_binary == 1)] = 2
        agreement[valid & (mapped == 1) & (modis_binary == 0)] = 3
        agreement[valid & (mapped == 0) & (modis_binary == 1)] = 4

        extent = plotting_extent(agreement, src_mod.transform)

        return agreement, adm2_modis, extent


def make_figure(
    prob_tif: Path,
    modis_tif: Path,
    adm2_tif: Path,
    out_dir: Path,
    threshold: float = 0.60,
    max_dim: int = 1800,
    dpi: int = 300,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Read display versions.
    prob_plot, prob_extent, _, _ = read_raster_for_plot(
        prob_tif,
        max_dim=max_dim,
        resampling=Resampling.average,
    )

    adm2_plot, adm2_extent, _, _ = read_raster_for_plot(
        adm2_tif,
        max_dim=max_dim,
        resampling=Resampling.nearest,
    )

    modis_plot, modis_extent, _, _ = read_raster_for_plot(
        modis_tif,
        max_dim=max_dim,
        resampling=Resampling.nearest,
    )

    # Final binary map for display.
    prob_filled = prob_plot.filled(np.nan) if np.ma.isMaskedArray(prob_plot) else prob_plot
    final_burn = np.where(np.isfinite(prob_filled), (prob_filled >= threshold).astype("float32"), np.nan)

    # MODIS binary for display.
    modis_filled = modis_plot.filled(np.nan) if np.ma.isMaskedArray(modis_plot) else modis_plot
    modis_bin = np.where(np.isfinite(modis_filled), (modis_filled > 0).astype("float32"), np.nan)

    # Agreement map at MODIS scale.
    agreement, adm2_modis, agreement_extent = make_agreement_at_modis_scale(
        prob_tif=prob_tif,
        modis_tif=modis_tif,
        adm2_tif=adm2_tif,
        threshold=threshold,
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(8.0, 7.8),
        constrained_layout=False,
    )
    axes = axes.ravel()

    # Panel (a): probability surface
    ax = axes[0]
    im0 = ax.imshow(
        prob_plot,
        extent=prob_extent,
        origin="upper",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    overlay_boundaries(ax, adm2_plot, adm2_extent, alpha=0.45)
    ax.set_title("(a) Burned probability surface", loc="left", fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im0, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label("Probability", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Panel (b): final thresholded map
    ax = axes[1]
    mapped_burn_color = "#8b0015"
    binary_cmap = ListedColormap(["white", mapped_burn_color])
    binary_norm = BoundaryNorm([-0.5, 0.5, 1.5], binary_cmap.N)
    im1 = ax.imshow(
        final_burn,
        extent=prob_extent,
        origin="upper",
        cmap=binary_cmap,
        norm=binary_norm,
        interpolation="nearest",
    )
    overlay_boundaries(ax, adm2_plot, adm2_extent, alpha=0.55)
    ax.set_title(f"(b) Final burned-area product (t = {threshold:.2f})", loc="left", fontsize=10, fontweight="bold")

    # Panel (c): MODIS burned mask
    ax = axes[2]
    modis_cmap = ListedColormap(["white", "#4d4d4d"])
    modis_norm = BoundaryNorm([-0.5, 0.5, 1.5], modis_cmap.N)
    im2 = ax.imshow(
        modis_bin,
        extent=modis_extent,
        origin="upper",
        cmap=modis_cmap,
        norm=modis_norm,
        interpolation="nearest",
    )
    # Boundary overlay on MODIS panel can use MODIS-scale agreement boundaries.
    overlay_boundaries(ax, adm2_modis, agreement_extent, alpha=0.55)
    ax.set_title("(c) MODIS MCD64A1 burned area", loc="left", fontsize=10, fontweight="bold")

    # Panel (d): agreement/disagreement
    ax = axes[3]
    # 0 no data, 1 neither, 2 both burned, 3 mapped only, 4 MODIS only
    agreement_cmap = ListedColormap([
        "white",      # no data
        "#f3f3f3",    # neither burned
        "#5e3c99",    # both burned
        "#fdb863",    # mapped only
        "#4393c3",    # MODIS only
    ])
    agreement_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], agreement_cmap.N)
    im3 = ax.imshow(
        agreement,
        extent=agreement_extent,
        origin="upper",
        cmap=agreement_cmap,
        norm=agreement_norm,
        interpolation="nearest",
    )
    overlay_boundaries(ax, adm2_modis, agreement_extent, alpha=0.55)
    ax.set_title("(d) MODIS-scale agreement", loc="left", fontsize=10, fontweight="bold")

    agreement_handles = [
        Patch(facecolor="#5e3c99", edgecolor="black", label="Both burned"),
        Patch(facecolor="#fdb863", edgecolor="black", label="Mapped only"),
        Patch(facecolor="#4393c3", edgecolor="black", label="MODIS only"),
        Patch(facecolor="#f3f3f3", edgecolor="black", label="Neither burned"),
    ]
    ax.legend(
        handles=agreement_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.025),
        frameon=True,
        fontsize=6.5,
        ncol=2,
        borderaxespad=0.0,
        columnspacing=0.8,
        handlelength=1.4,
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    fig.suptitle(
        "Final calibrated burned-area product over Punjab",
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )

    fig.subplots_adjust(
        left=0.03,
        right=0.985,
        top=0.93,
        bottom=0.065,
        wspace=0.08,
        hspace=0.16,
    )

    png_path = out_dir / "fig_final_burned_area_map.png"
    pdf_path = out_dir / "fig_final_burned_area_map.pdf"
    caption_path = out_dir / "fig_final_burned_area_map_caption.txt"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Final calibrated burned-area product over Punjab. "
        "(a) Calibrated burned-probability surface and (b) final 20 m burned-area "
        f"product thresholded at t = {threshold:.2f}, shown over valid Punjab "
        "cropland inference pixels; white areas include unburned cropland and "
        "pixels outside the mapped cropland/valid-inference support. "
        "(c) MODIS MCD64A1 burned area. "
        "(d) Agreement after aggregating the 20 m product to the MODIS grid, "
        "not agreement at the native 20 m resolution. This MODIS-scale comparison "
        "contextualizes the modest pixel-level F1 by showing where coarse-grid "
        "agreement and disagreement occur relative to MCD64A1."
    )
    caption_path.write_text(caption + "\n", encoding="utf-8")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {caption_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--map-dir",
        type=Path,
        default=Path("outputs/2024/phase5_maps"),
        help="Directory containing Phase 5 GeoTIFFs.",
    )
    parser.add_argument(
        "--prob-tif",
        type=Path,
        default=None,
        help="Path to final probability GeoTIFF. If omitted, searched automatically.",
    )
    parser.add_argument(
        "--modis-tif",
        type=Path,
        default=None,
        help="Path to MODIS burn GeoTIFF. If omitted, searched automatically.",
    )
    parser.add_argument(
        "--adm2-tif",
        type=Path,
        default=None,
        help="Path to ADM2 district raster GeoTIFF. If omitted, searched automatically.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Output directory for paper figure.",
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1800,
        help="Maximum display dimension for plotting panels.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    map_dir = args.map_dir

    prob_tif = args.prob_tif or find_first([
        str(map_dir / "*very_strict*_A_prob*.tif"),
        str(map_dir / "*very_strict*prob*.tif"),
        str(map_dir / "*prob*.tif"),
    ])

    modis_tif = args.modis_tif or find_first([
        str(map_dir / "*MODIS*burn*.tif"),
        str(map_dir / "*MODIS*.tif"),
    ])

    adm2_tif = args.adm2_tif or find_first([
        str(map_dir / "*adm2*.tif"),
        str(map_dir / "*ADM2*.tif"),
        str(map_dir / "*district*.tif"),
    ])

    missing = []
    if prob_tif is None:
        missing.append("probability GeoTIFF")
    if modis_tif is None:
        missing.append("MODIS GeoTIFF")
    if adm2_tif is None:
        missing.append("ADM2 district GeoTIFF")

    if missing:
        raise SystemExit(
            "Missing required input(s): "
            + ", ".join(missing)
            + "\n\nCheck available files with:\n"
            + f"  find {map_dir} -iname '*.tif'\n\n"
            + "Or pass paths explicitly, for example:\n"
            + "  python make_fig_final_burned_area_map.py "
            + "--prob-tif outputs/2024/phase5_maps/punjab_2024_phase5_very_strict_A_prob.tif "
            + "--modis-tif outputs/2024/phase5_maps/punjab_2024_phase5_MODIS_burn.tif "
            + "--adm2-tif outputs/2024/phase5_maps/punjab_2024_phase5_adm2.tif"
        )

    print("Using:")
    print(f"  Probability: {prob_tif}")
    print(f"  MODIS:       {modis_tif}")
    print(f"  ADM2:        {adm2_tif}")
    print(f"  Threshold:   {args.threshold:.2f}")

    make_figure(
        prob_tif=prob_tif,
        modis_tif=modis_tif,
        adm2_tif=adm2_tif,
        out_dir=args.out_dir,
        threshold=args.threshold,
        max_dim=args.max_dim,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
