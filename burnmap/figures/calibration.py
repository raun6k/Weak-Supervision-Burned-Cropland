#!/usr/bin/env python3
"""
Create Figure A: threshold calibration sweep.

Expected input:
    outputs/2024/phase5_results/threshold_calibration_summary.csv

Outputs:
    outputs/2024/paper/figA_threshold_calibration_sweep.png
    outputs/2024/paper/figA_threshold_calibration_sweep.pdf
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise KeyError(
        f"Could not find any of these columns: {candidates}\n"
        f"Available columns are: {list(df.columns)}"
    )


def load_threshold_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find threshold summary CSV at: {path}\n\n"
            "Try:\n"
            "python make_fig_threshold_calibration.py "
            "--summary-csv outputs/2024/phase5_results/threshold_calibration_summary.csv "
            "--out-dir outputs/figures"
        )

    df = pd.read_csv(path)

    threshold_col = find_col(df, ["threshold", "map_threshold", "prob_threshold"])
    district_r2_col = find_col(df, ["district_r2", "r2", "modis_r2", "district_modis_r2"])
    ratio_col = find_col(df, ["burned_area_ratio", "area_ratio", "ratio", "our_modis_ratio"])
    spearman_col = find_col(df, ["spearman_rho", "rho", "fire_spearman", "spearman_fire"])
    pixel_f1_col = find_col(df, ["pixel_f1", "f1", "modis_pixel_f1"])
    pixel_kappa_col = find_col(df, ["pixel_kappa", "kappa", "modis_pixel_kappa"])

    out = df.rename(
        columns={
            threshold_col: "threshold",
            district_r2_col: "district_r2",
            ratio_col: "burned_area_ratio",
            spearman_col: "spearman_rho",
            pixel_f1_col: "pixel_f1",
            pixel_kappa_col: "pixel_kappa",
        }
    ).copy()

    out = out.sort_values("threshold").reset_index(drop=True)
    return out


def add_vertical_threshold_lines(ax, cv_thr: float, calibrated_thr: float) -> None:
    # Grayscale-safe distinction
    ax.axvline(
        cv_thr,
        color="0.45",
        linestyle="--",
        linewidth=1.3,
        alpha=0.95,
        zorder=1,
    )
    ax.axvline(
        calibrated_thr,
        color="0.05",
        linestyle="-",
        linewidth=1.9,
        alpha=1.0,
        zorder=1,
    )


def plot_panel(
    ax,
    df: pd.DataFrame,
    y: str,
    ylabel: str,
    panel_label: str,
    gate: float | None = None,
    gate_label: str | None = None,
    gate_label_x: float = 0.98,
    ideal: float | None = None,
    ideal_label: str | None = None,
    ideal_label_x: float = 0.98,
    cv_thr: float = 0.30,
    calibrated_thr: float = 0.60,
    title: str | None = None,
) -> None:
    ax.plot(
        df["threshold"],
        df[y],
        marker="o",
        linewidth=1.8,
        markersize=3.5,
        zorder=2,
    )

    add_vertical_threshold_lines(ax, cv_thr, calibrated_thr)

    if gate is not None:
        ax.axhline(gate, color="0.45", linestyle=":", linewidth=1.0, alpha=0.55)
        ax.text(
            gate_label_x,
            gate,
            gate_label or f"Gate: {gate:g}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left" if gate_label_x < 0.5 else "right",
            fontsize=7.2,
            color="0.45",
        )

    if ideal is not None:
        ax.axhline(ideal, color="0.45", linestyle=":", linewidth=1.0, alpha=0.55)
        ax.text(
            ideal_label_x,
            ideal,
            ideal_label or f"Ideal: {ideal:g}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left" if ideal_label_x < 0.5 else "right",
            fontsize=7.2,
            color="0.45",
        )

    ax.set_ylabel(ylabel)
    ax.set_xlim(0.08, 0.92)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, axis="y", color="0.88", linewidth=0.8)
    ax.tick_params(labelsize=8)

    if title is not None:
        ax.set_title(f"{panel_label} {title}", loc="left", fontsize=10, fontweight="bold", pad=4)


def make_figure(
    summary_csv: Path,
    out_dir: Path,
    cv_thr: float = 0.30,
    calibrated_thr: float = 0.60,
    dpi: int = 300,
) -> None:
    df = load_threshold_summary(summary_csv)

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(7.2, 5.4),
        sharex=True,
        constrained_layout=False,
    )
    axes = axes.ravel()

    # (a) District R²
    plot_panel(
        axes[0],
        df,
        y="district_r2",
        ylabel=r"$R^2$",
        panel_label="(a)",
        title="District-level MODIS fit",
        gate=0.50,
        gate_label=r"Gate: $R^2$ = 0.50",
        gate_label_x=0.02,
        cv_thr=cv_thr,
        calibrated_thr=calibrated_thr,
    )

    # (b) Burned-area ratio
    axes[1].axhspan(0.90, 1.10, alpha=0.18, zorder=0)
    plot_panel(
        axes[1],
        df,
        y="burned_area_ratio",
        ylabel="Ratio",
        panel_label="(b)",
        title="Burned-area ratio",
        ideal=1.00,
        ideal_label="Ideal ratio = 1.0",
        cv_thr=cv_thr,
        calibrated_thr=calibrated_thr,
    )
    axes[1].set_ylim(0, 6.0)

    # (c) Spearman rho
    plot_panel(
        axes[2],
        df,
        y="spearman_rho",
        ylabel=r"$\rho$",
        panel_label="(c)",
        title="Fire-count rank correlation",
        gate=0.60,
        gate_label=r"Gate: $\rho$ = 0.60",
        gate_label_x=0.02,
        cv_thr=cv_thr,
        calibrated_thr=calibrated_thr,
    )

    # (d) Pixel-level F1
    plot_panel(
        axes[3],
        df,
        y="pixel_f1",
        ylabel="F1",
        panel_label="(d)",
        title="Pixel-level MODIS overlap",
        cv_thr=cv_thr,
        calibrated_thr=calibrated_thr,
    )

    for ax in axes[2:]:
        ax.set_xlabel("Threshold")

    legend_handles = [
        Line2D([0], [0], color="C0", marker="o", linewidth=1.8, markersize=4, label="Threshold sweep"),
        Line2D([0], [0], color="0.05", linewidth=1.9, label=f"Calibrated threshold ({calibrated_thr:.2f})"),
        Line2D([0], [0], color="0.45", linestyle="--", linewidth=1.3, label=f"CV threshold ({cv_thr:.2f})"),
        Line2D([0], [0], color="0.35", linestyle=":", linewidth=1.0, label="Gate / ideal"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        handlelength=2.4,
        columnspacing=1.4,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.16, wspace=0.28, hspace=0.34)

    png_path = out_dir / "figA_threshold_calibration_sweep.png"
    pdf_path = out_dir / "figA_threshold_calibration_sweep.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/2024/phase5_results/threshold_calibration_summary.csv"),
        help="Path to threshold_calibration_summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory where figure files will be written",
    )
    parser.add_argument("--cv-threshold", type=float, default=0.30)
    parser.add_argument("--calibrated-threshold", type=float, default=0.60)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    make_figure(
        summary_csv=args.summary_csv,
        out_dir=args.out_dir,
        cv_thr=args.cv_threshold,
        calibrated_thr=args.calibrated_threshold,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
