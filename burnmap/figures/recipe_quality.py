#!/usr/bin/env python3
"""
Create Figure C: weak-label recipe quantity vs quality comparison.

Expected inputs:
    outputs/2024/paper/supplementary_recipe_summary.csv

Optional input:
    a Phase 2 label summary CSV containing recipe-level positive label counts.

Outputs:
    outputs/2024/paper/figC_recipe_quantity_quality.png
    outputs/2024/paper/figC_recipe_quantity_quality.pdf
"""

from pathlib import Path
import argparse
import warnings
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RECIPE_ORDER = ["moderate", "strict", "very_strict"]

# Fallback counts from Phase 2 project summary.
FALLBACK_POS_COUNTS = {
    "moderate": 81471,
    "strict": 73746,
    "very_strict": 59063,
}


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns are: {list(df.columns)}"
        )
    return None


def find_existing_file(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def auto_find_label_summary() -> Path | None:
    patterns = [
        "outputs/2024/**/label*summary*.csv",
        "outputs/2024/**/*label*.csv",
        "outputs/**/label*summary*.csv",
    ]

    matches = []
    for pattern in patterns:
        matches.extend(Path(".").glob(pattern))

    # Prefer files that sound recipe-level rather than per-point.
    preferred = [
        p
        for p in matches
        if "district" not in p.name.lower()
        and "point" not in p.name.lower()
        and "sample" not in p.name.lower()
    ]

    if preferred:
        return sorted(preferred, key=lambda p: len(str(p)))[0]
    if matches:
        return sorted(matches, key=lambda p: len(str(p)))[0]
    return None


def load_recipe_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find recipe summary CSV at: {path}\n\n"
            "Try:\n"
            "find outputs -name 'supplementary_recipe_summary.csv'"
        )

    df = pd.read_csv(path)

    recipe_col = find_col(df, ["recipe", "recipe_id"])
    mean_kappa_col = find_col(df, ["mean_kappa", "kappa_mean", "mean_k"])
    std_kappa_col = find_col(df, ["std_kappa", "kappa_std", "std_k"], required=False)
    mean_auc_col = find_col(df, ["mean_auc", "auc_mean"], required=False)

    out = df.rename(
        columns={
            recipe_col: "recipe",
            mean_kappa_col: "mean_kappa",
        }
    ).copy()

    if std_kappa_col is not None:
        out = out.rename(columns={std_kappa_col: "std_kappa"})
    else:
        out["std_kappa"] = 0.0

    if mean_auc_col is not None:
        out = out.rename(columns={mean_auc_col: "mean_auc"})
    else:
        out["mean_auc"] = pd.NA

    out["recipe"] = out["recipe"].astype(str)
    out = out[out["recipe"].isin(RECIPE_ORDER)].copy()
    out["recipe_order"] = out["recipe"].map({r: i for i, r in enumerate(RECIPE_ORDER)})
    out = out.sort_values("recipe_order").reset_index(drop=True)

    return out[["recipe", "mean_kappa", "std_kappa", "mean_auc"]]


def load_positive_counts(label_summary_path: Path | None) -> pd.DataFrame:
    if label_summary_path is None or not label_summary_path.exists():
        warnings.warn(
            "Could not find a label summary CSV. "
            "Using fallback Phase 2 positive-label counts."
        )
        return pd.DataFrame(
            {
                "recipe": RECIPE_ORDER,
                "positive_labels": [FALLBACK_POS_COUNTS[r] for r in RECIPE_ORDER],
            }
        )

    df = pd.read_csv(label_summary_path)

    recipe_col = find_col(df, ["recipe", "recipe_id"])
    pos_col = find_col(
        df,
        [
            "positive_labels",
            "pos_labels",
            "n_positive",
            "n_pos",
            "positive_count",
            "pos_count",
            "burned_labels",
            "n_burned",
        ],
    )

    out = df.rename(columns={recipe_col: "recipe", pos_col: "positive_labels"}).copy()
    out["recipe"] = out["recipe"].astype(str)

    # If file has multiple rows per recipe, aggregate.
    out = (
        out[out["recipe"].isin(RECIPE_ORDER)]
        .groupby("recipe", as_index=False)["positive_labels"]
        .sum()
    )

    out["recipe_order"] = out["recipe"].map({r: i for i, r in enumerate(RECIPE_ORDER)})
    out = out.sort_values("recipe_order").reset_index(drop=True)

    return out[["recipe", "positive_labels"]]


def pretty_recipe_name(recipe: str) -> str:
    if recipe == "very_strict":
        return "Very strict"
    return recipe.capitalize()


def make_figure(
    recipe_summary_csv: Path,
    label_summary_csv: Path | None,
    out_dir: Path,
    dpi: int = 300,
) -> None:
    recipe_df = load_recipe_summary(recipe_summary_csv)

    if label_summary_csv is None:
        label_summary_csv = auto_find_label_summary()

    count_df = load_positive_counts(label_summary_csv)

    df = recipe_df.merge(count_df, on="recipe", how="left")
    df["positive_labels"] = df["positive_labels"].fillna(
        df["recipe"].map(FALLBACK_POS_COUNTS)
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    labels = [pretty_recipe_name(r) for r in df["recipe"]]
    x = list(range(len(df)))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(7.0, 3.1),
        constrained_layout=False,
    )

    # Panel (a): quantity
    ax = axes[0]
    bars = ax.bar(x, df["positive_labels"], color="#4b87b8", width=0.62)
    ax.set_title(
        "(a) Statewide positive weak labels",
        fontsize=8.5,
        fontweight="bold",
        loc="left",
    )
    ax.set_ylabel("Positive weak labels", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for b, val in zip(bars, df["positive_labels"]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + df["positive_labels"].max() * 0.012,
            f"{int(val):,}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylim(0, df["positive_labels"].max() * 1.12)

    # Panel (b): downstream quality
    ax = axes[1]
    yerr = df["std_kappa"].fillna(0.0)
    ax.errorbar(
        x,
        df["mean_kappa"],
        yerr=yerr,
        fmt="o",
        color="#1f5d8a",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
        markersize=5,
        zorder=3,
    )
    ax.plot(x, df["mean_kappa"], color="#1f5d8a", linewidth=1.0, alpha=0.75, zorder=2)
    ax.set_title(
        "(b) Cross-validated performance",
        fontsize=8.5,
        fontweight="bold",
        loc="left",
    )
    ax.set_ylabel("Mean Cohen's κ (± SD)", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, max(0.46, (df["mean_kappa"] + yerr).max() + 0.06))
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for idx, (xi, kappa, err, auc) in enumerate(zip(x, df["mean_kappa"], yerr, df["mean_auc"])):
        text_x = xi
        ha = "center"
        if idx == 0:
            text_x = xi + 0.08
            ha = "left"
        elif idx == len(x) - 1:
            text_x = xi - 0.08
            ha = "right"

        ax.text(
            text_x,
            kappa + err + 0.015,
            f"κ={kappa:.3f}",
            ha=ha,
            va="bottom",
            fontsize=7,
        )
        if pd.notna(auc):
            auc_y = max(0.025, kappa - err - 0.035)
            ax.text(
                text_x,
                auc_y,
                f"AUC={float(auc):.3f}",
                ha=ha,
                va="top",
                fontsize=6.5,
                color="0.35",
            )

    fig.subplots_adjust(
        left=0.085,
        right=0.98,
        top=0.88,
        bottom=0.18,
        wspace=0.32,
    )

    png_path = out_dir / "figC_recipe_quantity_quality.png"
    pdf_path = out_dir / "figC_recipe_quantity_quality.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    if label_summary_csv is not None and label_summary_csv.exists():
        print(f"Used label summary: {label_summary_csv}")
    else:
        print("Used fallback positive-label counts.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe-summary-csv",
        type=Path,
        default=Path("outputs/2024/paper/supplementary_recipe_summary.csv"),
        help="Recipe-level performance summary CSV.",
    )
    parser.add_argument(
        "--label-summary-csv",
        type=Path,
        default=None,
        help=(
            "Optional recipe-level label summary CSV containing positive-label counts. "
            "If omitted, the script tries to find one automatically; if none is found, "
            "it uses fallback Phase 2 counts."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory where figure files will be written.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    make_figure(
        recipe_summary_csv=args.recipe_summary_csv,
        label_summary_csv=args.label_summary_csv,
        out_dir=args.out_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
