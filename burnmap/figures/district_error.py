#!/usr/bin/env python3
"""
Create Figure B: district-level residual/error heatmap.

Expected input:
    outputs/2024/phase5_results/threshold_calibration_details/
        district_modis_comparison_threshold_0p6.csv

Outputs:
    outputs/2024/paper/figB_district_error_heatmap.png
    outputs/2024/paper/figB_district_error_heatmap.pdf
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


DISTRICT_DISPLAY_NAMES = {
    # Keep manuscript-facing district names centralized because upstream
    # validation tables use both "Nawan Shehar" and "SBS Nagar".
    "Nawan Shehar": "SBS Nagar",
}


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise KeyError(
        f"Could not find any of these columns: {candidates}\n"
        f"Available columns are: {list(df.columns)}"
    )


def load_district_comparison(path: Path, area_units: str = "km2") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find district comparison CSV at: {path}\n\n"
            "Try:\n"
            "find outputs -name 'district_modis_comparison_threshold_0p6.csv'"
        )

    df = pd.read_csv(path)

    district_col = find_col(df, ["district", "district_name", "name", "ADM2_NAME"])
    pred_col = find_col(
        df,
        [
            "ours_burn_pixels_500m",
            "our_burn_pixels_500m",
            "pred_burn_pixels_500m",
            "predicted_burn_pixels_500m",
            "ours_burned_pixels_500m",
            "predicted_burned_pixels_500m",
        ],
    )
    modis_col = find_col(
        df,
        [
            "modis_burn_pixels_500m",
            "modis_burned_pixels_500m",
            "mcd64a1_burn_pixels_500m",
            "modis_pixels",
        ],
    )

    out = df[[district_col, pred_col, modis_col]].rename(
        columns={
            district_col: "district",
            pred_col: "pred_pixels",
            modis_col: "modis_pixels",
        }
    )

    # Each 500 m pixel = 0.25 km².
    if area_units == "km2":
        out["pred_area"] = out["pred_pixels"] * 0.25
        out["modis_area"] = out["modis_pixels"] * 0.25
        value_label = "km²"
    elif area_units == "pixels":
        out["pred_area"] = out["pred_pixels"]
        out["modis_area"] = out["modis_pixels"]
        value_label = "500 m pixels"
    else:
        raise ValueError("area_units must be either 'km2' or 'pixels'.")

    out["residual"] = out["pred_area"] - out["modis_area"]

    # Add a small epsilon to avoid division by zero in sparse-burn districts.
    eps = 0.25 if area_units == "km2" else 1.0
    out["log2_ratio"] = np.log2((out["pred_area"] + eps) / (out["modis_area"] + eps))

    out = out.sort_values("modis_area", ascending=False).reset_index(drop=True)
    out["district"] = out["district"].replace(DISTRICT_DISPLAY_NAMES)

    return out, value_label


def annotate_column(ax, values, fmt, text_color="black") -> None:
    for i, v in enumerate(values):
        ax.text(
            0,
            i,
            fmt(v),
            ha="center",
            va="center",
            fontsize=7,
            color=text_color,
        )


def make_figure(
    district_csv: Path,
    out_dir: Path,
    area_units: str = "km2",
    dpi: int = 300,
) -> None:
    df, value_label = load_district_comparison(district_csv, area_units=area_units)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    n = len(df)
    fig_height = max(5.2, 0.32 * n + 1.4)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(7.6, fig_height),
        sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1.15, 1.15, 1.15]},
    )

    columns = [
        ("modis_area", f"MODIS\n({value_label})", "viridis"),
        ("pred_area", f"Mapped burned area\n({value_label})", "viridis"),
        ("residual", "Residual\nMapped - MODIS", "coolwarm"),
        ("log2_ratio", r"Log$_2$" "\n(Mapped / MODIS)", "coolwarm"),
    ]

    for j, (col, title, cmap) in enumerate(columns):
        ax = axes[j]
        values = df[col].to_numpy().reshape(-1, 1)

        if col in ["residual", "log2_ratio"]:
            vmax = np.nanmax(np.abs(df[col].to_numpy()))
            vmax = max(vmax, 1e-6)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
        else:
            im = ax.imshow(values, aspect="auto", cmap=cmap)

        ax.set_title(title, fontsize=9, pad=6)
        ax.set_xticks([])
        ax.set_yticks(np.arange(n))

        if j == 0:
            ax.set_yticklabels(df["district"], fontsize=8)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

        # Cell annotations
        if col in ["modis_area", "pred_area", "residual"]:
            annotate_column(ax, df[col].to_numpy(), lambda x: f"{x:,.0f}")
        else:
            annotate_column(ax, df[col].to_numpy(), lambda x: f"{x:+.2f}")

        # Keep borders visible.
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

    fig.suptitle(
        "District-level external-validation error at the calibrated threshold (0.60)",
        fontsize=9.5,
        fontweight="bold",
        y=0.995,
    )

    fig.subplots_adjust(
        left=0.22,
        right=0.98,
        top=0.93,
        bottom=0.19,
        wspace=0.08,
    )

    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    pos2 = axes[2].get_position()
    pos3 = axes[3].get_position()
    cbar_y = 0.075
    cbar_h = 0.018

    # Dedicated colorbar axes keep all heatmap columns the same height.
    cax1 = fig.add_axes([pos0.x0, cbar_y, pos1.x1 - pos0.x0, cbar_h])
    cax2 = fig.add_axes([pos2.x0, cbar_y, pos2.width, cbar_h])
    cax3 = fig.add_axes([pos3.x0, cbar_y, pos3.width, cbar_h])

    # Colorbars: one for area columns, one each for residual and log-ratio.
    cbar1 = fig.colorbar(
        axes[0].images[0],
        cax=cax1,
        orientation="horizontal",
    )
    cbar1.set_label(f"Burned area ({value_label})", fontsize=8)
    cbar1.ax.tick_params(labelsize=7)

    cbar2 = fig.colorbar(
        axes[2].images[0],
        cax=cax2,
        orientation="horizontal",
    )
    cbar2.set_label(f"Residual ({value_label})", fontsize=8)
    cbar2.ax.tick_params(labelsize=7)

    cbar3 = fig.colorbar(
        axes[3].images[0],
        cax=cax3,
        orientation="horizontal",
    )
    cbar3.set_label(r"Log$_2$(Mapped / MODIS)", fontsize=8)
    cbar3.ax.tick_params(labelsize=7)

    png_path = out_dir / "figB_district_error_heatmap.png"
    pdf_path = out_dir / "figB_district_error_heatmap.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--district-csv",
        type=Path,
        default=Path(
            "outputs/2024/phase5_results/threshold_calibration_details/"
            "district_modis_comparison_threshold_0p6.csv"
        ),
        help="Final-threshold district comparison CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory where figure files will be written.",
    )
    parser.add_argument(
        "--area-units",
        choices=["km2", "pixels"],
        default="km2",
        help="Plot district burned area in km2 or 500 m pixels.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    make_figure(
        district_csv=args.district_csv,
        out_dir=args.out_dir,
        area_units=args.area_units,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
