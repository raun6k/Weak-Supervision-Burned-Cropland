"""
Phase 7: Paper Tables and Figures
==================================

Generates paper tables/figures from Phase 4–6 results.

Primary outputs
---------------
Tables (CSV):
  1. Recipe parameters
  2. Main results (all classifiers × feature sets)
  3. Baseline comparison
  4. Threshold calibration summary
  5. Final external validation summary
  6. Phase 6 generalizability summary

Supplementary outputs (CSV):
  - supplementary_feature_importance.csv
  - supplementary_recipe_summary.csv

Figures (PNG via matplotlib):
  3. Multi-classifier kappa barplot
  4. Feature-set ablation
  5. Threshold calibration curves
  6. MODIS district comparison scatter
  7. Phase 6 generalizability barplot
  8. Per-fold kappa variation

Usage
-----
    python scripts/paper_tables.py --year 2024
    python scripts/paper_tables.py --year 2024 --cv_metrics path/to/all_cv_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import burnmap.config as config
from burnmap.config import recipes as recipe_mod


# Phase 5 reporting gates:
# use config values if present, otherwise fall back to the revised build-plan defaults
GATE_DISTRICT_R2_MODIS = getattr(config, "GATE_DISTRICT_R2_MODIS", 0.50)
GATE_MODIS_RATIO_MAX = getattr(config, "GATE_MODIS_RATIO_MAX", 2.0)
GATE_SPEARMAN_CAQM = getattr(config, "GATE_SPEARMAN_CAQM", 0.60)

# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 7 — Paper Tables & Figures")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--cv_metrics", type=str, default=None)
    p.add_argument("--prob_csv", type=str, default=None, help="Probability CSV for optional plots")
    p.add_argument("--phase5_dir", type=str, default=None)
    p.add_argument("--phase6_metrics", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--no_figures", action="store_true", help="Skip figure generation (tables only)")
    return p.parse_args()


# ── Generic helpers ──────────────────────────────────────────────────

def _load_csv(path: str):
    with open(path) as f:
        rows = []
        for row in csv.DictReader(f):
            clean = {}
            for k, v in row.items():
                try:
                    if v is None or v == "":
                        clean[k] = v
                    else:
                        clean[k] = float(v)
                except (ValueError, TypeError):
                    clean[k] = v
            rows.append(clean)
    return rows


def _write_csv(rows, path: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  {path} ({len(rows)} rows)")


def _mean(vals):
    vals = [float(v) for v in vals if v is not None and v != ""]
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals):
    vals = [float(v) for v in vals if v is not None and v != ""]
    if not vals:
        return 0.0
    m = _mean(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def _round(x, n=3):
    try:
        return round(float(x), n)
    except Exception:
        return x


def _find_first_existing(candidates):
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _is_classifier_row(r):
    clf = str(r.get("classifier", ""))
    return clf and not clf.startswith("baseline_") and "error" not in r


def _get_threshold_score(row):
    """
    Rank threshold rows by:
      1) number of passed gates
      2) closeness of ratio to 1
      3) district R2
      4) Spearman rho
      5) pixel F1
    """
    gates = int(row.get("gates", 0) or 0)
    ratio = float(row.get("ratio", row.get("burned_area_ratio", 999)) or 999)
    r2 = float(row.get("R2", row.get("district_r2", 0)) or 0)
    rho = float(row.get("rho", row.get("spearman_rho", 0)) or 0)
    pxf1 = float(row.get("pxF1", row.get("pixel_f1", 0)) or 0)
    return (
        gates,
        -abs(ratio - 1.0),
        r2,
        rho,
        pxf1,
    )


def _pick_best_threshold_row(rows):
    if not rows:
        return None
    return sorted(rows, key=_get_threshold_score, reverse=True)[0]


def _find_threshold_detail_file(phase5_dir: str, threshold: float, keyword: str):
    """
    Find the threshold-specific detail file written by threshold calibration.
    Supports filenames like:
      ..._threshold_0p6.csv
      ..._threshold_0p60.csv
      ..._threshold_0.6.csv
      ..._threshold_0.60.csv

    Returns None if no threshold-specific file is found.
    """
    details_dir = os.path.join(phase5_dir, "threshold_calibration_details")
    if not os.path.isdir(details_dir):
        return None

    thr_formats = [
        f"{threshold:.2f}",                  # 0.60
        f"{threshold:.1f}",                  # 0.6
        str(threshold),                      # 0.6
        f"{threshold:.2f}".replace(".", "p"),# 0p60
        f"{threshold:.1f}".replace(".", "p"),# 0p6
        str(threshold).replace(".", "p"),    # 0p6
    ]

    patterns = []
    for t in thr_formats:
        patterns.extend([
            os.path.join(details_dir, f"*{keyword}*{t}*.csv"),
            os.path.join(details_dir, f"*{t}*{keyword}*.csv"),
        ])

    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat))

    matches = sorted(set(matches))
    return matches[0] if matches else None


# ── Resolve input paths ──────────────────────────────────────────────

def resolve_paths(year: int, args):
    cv_path = args.cv_metrics or _find_first_existing([
        f"outputs/{year}/combined_results/all_cv_metrics.csv",
        f"outputs/{year}/phase4_results/combined_cv_metrics.csv",
        f"outputs/{year}/phase3_results/cv_metrics.csv",
    ])

    prob_csv = args.prob_csv or _find_first_existing([
        f"outputs/{year}/combined_results/all_test_probabilities.csv",
        f"outputs/{year}/phase4_results/phase4_test_probabilities.csv",
        f"outputs/{year}/phase3_results/test_probabilities.csv",
    ])

    phase5_dir = args.phase5_dir or f"outputs/{year}/phase5_results"

    phase6_metrics = args.phase6_metrics or _find_first_existing([
        f"outputs/{year}/phase6_results/phase6_cv_metrics.csv",
    ])

    return cv_path, prob_csv, phase5_dir, phase6_metrics


# ── Table 1: Recipe Parameters ───────────────────────────────────────

def table_recipe_params(out_dir):
    print("\n  Table 1: Recipe parameters")
    rows = []
    for r in recipe_mod.ALL_RECIPES:
        rows.append({
            "recipe": r["recipe_id"],
            "dNBR_threshold": r["burnDNBRThr"],
            "near_fire_buffer_m": r["nearFireMeters"],
            "swir_gate_dB12": r["swirGateThr"],
            "hardneg_dNBR_max": r["hardNeg_dNBR_Max"],
            "hardneg_dB12_max": r["hardNeg_dB12_Max"],
            "unburned_far_m": r["unburnedFarMeters"],
            "ndvi_decrease_gate": r.get("ndviDecreaseGate", "None"),
        })
    _write_csv(rows, os.path.join(out_dir, "table1_recipe_params.csv"))


# ── Table 2: Main Results ────────────────────────────────────────────

def table_main_results(cv_rows, out_dir):
    print("  Table 2: Main results")
    clf_rows = [r for r in cv_rows if _is_classifier_row(r)]

    groups = defaultdict(list)
    for r in clf_rows:
        key = (r["recipe_id"], r["classifier"], r.get("set_id", "A"))
        groups[key].append(r)

    table = []
    for (recipe, clf, sid), items in sorted(groups.items()):
        ks = [r.get("kappa", 0) for r in items]
        aucs = [r.get("auc", 0) for r in items]
        f1s = [r.get("f1", 0) for r in items]
        bas = [r.get("balanced_accuracy", 0) for r in items]

        table.append({
            "recipe": recipe,
            "classifier": clf,
            "feature_set": sid,
            "mean_kappa": _round(_mean(ks)),
            "std_kappa": _round(_std(ks)),
            "mean_auc": _round(_mean(aucs)),
            "mean_f1": _round(_mean(f1s)),
            "mean_bal_acc": _round(_mean(bas)),
            "n": len(items),
        })

    table.sort(key=lambda r: (-r["mean_kappa"], -r["mean_auc"]))
    _write_csv(table, os.path.join(out_dir, "table2_main_results.csv"))
    return table


# ── Table 3: Baseline Comparison ─────────────────────────────────────

def table_baseline_comparison(cv_rows, out_dir):
    print("  Table 3: Baseline comparison")

    groups = defaultdict(list)
    for r in cv_rows:
        if "error" in r:
            continue
        clf = r.get("classifier", "?")
        recipe = r.get("recipe_id", "?")
        groups[(recipe, clf)].append(r)

    table = []
    for (recipe, clf), items in sorted(groups.items()):
        ks = [r.get("kappa", 0) for r in items]
        if not ks:
            continue
        table.append({
            "recipe": recipe,
            "classifier": clf,
            "mean_kappa": _round(_mean(ks)),
            "std_kappa": _round(_std(ks)),
            "n": len(ks),
        })

    table.sort(key=lambda r: (r["recipe"], -r["mean_kappa"]))
    _write_csv(table, os.path.join(out_dir, "table3_baseline_comparison.csv"))


# ── Table 4: Threshold Calibration Summary ───────────────────────────

def table_threshold_calibration(phase5_dir, out_dir):
    print("  Table 4: Threshold calibration summary")

    path = os.path.join(phase5_dir, "threshold_calibration_summary.csv")
    if not os.path.exists(path):
        print("    SKIPPED: threshold_calibration_summary.csv not found")
        return None, None

    rows = _load_csv(path)
    if not rows:
        print("    SKIPPED: threshold calibration summary empty")
        return None, None

    table = []
    for r in rows:
        table.append({
            "threshold": _round(r.get("thr", r.get("threshold", "")), 3),
            "district_r2": _round(r.get("R2", r.get("district_r2", 0))),
            "burned_area_ratio": _round(r.get("ratio", r.get("burned_area_ratio", 0))),
            "spearman_rho": _round(r.get("rho", r.get("spearman_rho", 0))),
            "pixel_f1": _round(r.get("pxF1", r.get("pixel_f1", 0))),
            "pixel_kappa": _round(r.get("pxKappa", r.get("pixel_kappa", 0))),
            "gates": int(r.get("gates", 0) or 0),
        })

    table.sort(key=lambda r: (
        -r["gates"],
        abs(r["burned_area_ratio"] - 1.0),
        -r["district_r2"],
        -r["spearman_rho"],
    ))

    out_path = os.path.join(out_dir, "table4_threshold_calibration.csv")
    _write_csv(table, out_path)

    best_raw = _pick_best_threshold_row(rows)
    best_clean = None
    if best_raw is not None:
        best_clean = {
            "threshold": float(best_raw.get("thr", best_raw.get("threshold", 0))),
            "district_r2": float(best_raw.get("R2", best_raw.get("district_r2", 0))),
            "burned_area_ratio": float(best_raw.get("ratio", best_raw.get("burned_area_ratio", 0))),
            "spearman_rho": float(best_raw.get("rho", best_raw.get("spearman_rho", 0))),
            "pixel_f1": float(best_raw.get("pxF1", best_raw.get("pixel_f1", 0))),
            "pixel_kappa": float(best_raw.get("pxKappa", best_raw.get("pixel_kappa", 0))),
            "gates": int(best_raw.get("gates", 0) or 0),
        }

    return rows, best_clean


# ── Table 5: Final External Validation Summary ───────────────────────

def table_external_validation(best_threshold, phase5_dir, out_dir):
    print("  Table 5: Final external validation summary")

    if not best_threshold:
        print("    SKIPPED: no threshold calibration result found")
        return None

    rows = [{
        "selected_map_classifier": config.BEST_CLASSIFIER,
        "selected_recipe": config.BEST_RECIPE,
        "selected_feature_set": config.BEST_FEATURE_SET,
        "final_threshold": _round(best_threshold["threshold"], 3),
        "district_r2_vs_modis": _round(best_threshold["district_r2"]),
        "burned_area_ratio_vs_modis": _round(best_threshold["burned_area_ratio"]),
        "spearman_rho_vs_fire_counts": _round(best_threshold["spearman_rho"]),
        "pixel_f1_vs_modis": _round(best_threshold["pixel_f1"]),
        "pixel_kappa_vs_modis": _round(best_threshold["pixel_kappa"]),
        "automatic_gates_passed": int(best_threshold["gates"]),
        "district_r2_gate": "PASS" if best_threshold["district_r2"] >= GATE_DISTRICT_R2_MODIS else "FAIL",
        "area_ratio_gate": "PASS" if best_threshold["burned_area_ratio"] <= GATE_MODIS_RATIO_MAX else "FAIL",
        "fire_count_gate": "PASS" if best_threshold["spearman_rho"] >= GATE_SPEARMAN_CAQM else "FAIL",  
    }]

    out_path = os.path.join(out_dir, "table5_external_validation_summary.csv")
    _write_csv(rows, out_path)
    return rows[0]


# ── Table 6: Phase 6 Generalizability Summary ────────────────────────

def table_phase6_generalizability(phase6_metrics_path, out_dir):
    print("  Table 6: Phase 6 generalizability summary")

    if not phase6_metrics_path or not os.path.exists(phase6_metrics_path):
        print("    SKIPPED: phase6_cv_metrics.csv not found")
        return None, None

    rows = _load_csv(phase6_metrics_path)
    if not rows:
        print("    SKIPPED: phase6 metrics empty")
        return None, None

    groups = defaultdict(list)
    for r in rows:
        groups[str(r.get("experiment", "unknown"))].append(r)

    reported = {
        "haryana_zeroshot",
        "punjab_2023",
        "punjab_2024",
        "s1_fusion_A",
        "s1_fusion_E",
    }
    summary = []
    for exp_name, items in sorted(groups.items()):
        if exp_name not in reported:
            continue
        ks = [r.get("kappa", 0) for r in items]
        row = {
            "experiment": exp_name,
            "mean_kappa": _round(_mean(ks)),
            "std_kappa": _round(_std(ks)),
            "n": len(ks),
            "gate": "",
            "gate_status": "",
        }

        if exp_name == "haryana_zeroshot":
            row["gate"] = f"≥ {config.GATE_HARYANA_ZEROSHOT_KAPPA}"
            row["gate_status"] = "PASS" if row["mean_kappa"] >= config.GATE_HARYANA_ZEROSHOT_KAPPA else "FAIL"
        elif exp_name in {"s1_fusion_A", "s1_fusion_E"}:
            row["gate"] = "see S1 improvement"
            row["gate_status"] = ""
        elif exp_name.startswith("punjab_"):
            row["gate"] = "reported via cross-year std"
            row["gate_status"] = ""

        summary.append(row)

    # Add explicit S1 improvement row if both A and E are present
    exp_to_mean = {r["experiment"]: r["mean_kappa"] for r in summary}
    if "s1_fusion_A" in exp_to_mean and "s1_fusion_E" in exp_to_mean:
        diff = exp_to_mean["s1_fusion_E"] - exp_to_mean["s1_fusion_A"]
        summary.append({
            "experiment": "s1_improvement_E_minus_A",
            "mean_kappa": _round(diff),
            "std_kappa": "",
            "n": "",
            "gate": f"≥ {config.GATE_S1_IMPROVEMENT}",
            "gate_status": "PASS" if diff >= config.GATE_S1_IMPROVEMENT else "FAIL",
        })

    # Add explicit cross-year stability row if both Punjab years are present
    punjab_means = []
    for r in summary:
        if r["experiment"] in {"punjab_2023", "punjab_2024"} and r["mean_kappa"] != "":
            punjab_means.append(float(r["mean_kappa"]))
    if len(punjab_means) >= 2:
        cross_year_std = _std(punjab_means)
        summary.append({
            "experiment": "multiyear_cross_year_std",
            "mean_kappa": _round(cross_year_std),
            "std_kappa": "",
            "n": len(punjab_means),
            "gate": f"≤ {config.GATE_MULTIYEAR_KAPPA_STD}",
            "gate_status": "PASS" if cross_year_std <= config.GATE_MULTIYEAR_KAPPA_STD else "FAIL",
        })

    out_path = os.path.join(out_dir, "table6_phase6_generalizability.csv")
    _write_csv(summary, out_path)
    return rows, summary


# ── Supplementary tables ─────────────────────────────────────────────

def supplementary_feature_importance(cv_rows, out_dir):
    print("  Supplementary: Feature importance")

    rf_rows = [
        r for r in cv_rows
        if r.get("classifier") == "rf"
        and r.get("recipe_id") == config.BEST_RECIPE
        and r.get("set_id") == config.BEST_FEATURE_SET
        and "imp_1_feat" in r
    ]

    if not rf_rows:
        print("    No importance data found")
        return

    imp_sums = defaultdict(list)
    for r in rf_rows:
        for i in range(1, getattr(config, "TOPK_IMPORTANCE", 15) + 1):
            feat = r.get(f"imp_{i}_feat")
            val = r.get(f"imp_{i}_val")
            if feat and val not in ("", None):
                imp_sums[str(feat)].append(float(val))

    table = []
    for feat, vals in imp_sums.items():
        table.append({
            "feature": feat,
            "mean_importance": _round(_mean(vals), 4),
            "n_appearances": len(vals),
        })

    table.sort(key=lambda r: -r["mean_importance"])
    _write_csv(
        table[:getattr(config, "TOPK_IMPORTANCE", 15)],
        os.path.join(out_dir, "supplementary_feature_importance.csv"),
    )


def supplementary_recipe_summary(cv_rows, out_dir):
    print("  Supplementary: Recipe effect summary")

    clf_rows = [r for r in cv_rows if _is_classifier_row(r)]
    table = []
    for recipe in ["strict", "moderate", "very_strict"]:
        subset = [r for r in clf_rows if r.get("recipe_id") == recipe]
        if not subset:
            continue
        ks = [r.get("kappa", 0) for r in subset]
        aucs = [r.get("auc", 0) for r in subset]
        table.append({
            "recipe": recipe,
            "mean_kappa": _round(_mean(ks)),
            "std_kappa": _round(_std(ks)),
            "mean_auc": _round(_mean(aucs)),
            "max_kappa": _round(max(float(k) for k in ks)),
            "n": len(ks),
        })

    _write_csv(table, os.path.join(out_dir, "supplementary_recipe_summary.csv"))


# ── Figures ──────────────────────────────────────────────────────────

def generate_figures(cv_rows, phase5_dir, best_threshold, phase6_summary, prob_csv, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib/numpy not installed, skipping figures")
        print("  Install with: pip install matplotlib numpy")
        return

    clf_rows = [r for r in cv_rows if _is_classifier_row(r)]

    # Fig 3: Classifier comparison
    print("  Fig 3: Classifier comparison barplot")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    classifiers = sorted(set(str(r.get("classifier")) for r in clf_rows))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(classifiers), 1)))

    for ax, recipe in zip(axes, ["strict", "moderate", "very_strict"]):
        means, stds, labels = [], [], []
        for clf in classifiers:
            subset = [
                r for r in clf_rows
                if r.get("recipe_id") == recipe and str(r.get("classifier")) == clf
            ]
            if not subset:
                continue
            ks = [r.get("kappa", 0) for r in subset]
            means.append(_mean(ks))
            stds.append(_std(ks))
            labels.append(clf)

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=3,
               color=colors[:len(labels)], edgecolor="gray", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(recipe, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 0.75)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Cohen's Kappa", fontsize=11)
    fig.suptitle("Classifier Comparison by Recipe", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_classifier_comparison.png"), dpi=200)
    plt.close()

    # Fig 4: Feature-set ablation
    print("  Fig 4: Feature-set ablation")
    fig, ax = plt.subplots(figsize=(8, 5))
    sets = ["A", "B", "C", "D"]
    set_labels = ["A (Full 37)", "B (Delta 11)", "C (Curated 24)", "D (No-SWIR 32)"]

    top_clfs = ["rf", "xgb", "lgbm"]
    width = 0.22
    x = np.arange(len(sets))
    for i, clf in enumerate(top_clfs):
        means = []
        for sid in sets:
            subset = [
                r for r in clf_rows
                if r.get("recipe_id") == "very_strict"
                and str(r.get("classifier")) == clf
                and str(r.get("set_id")) == sid
            ]
            means.append(_mean([r.get("kappa", 0) for r in subset]) if subset else 0)

        ax.bar(x + i * width, means, width, label=clf.upper(), alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(set_labels, fontsize=10)
    ax.set_ylabel("Cohen's Kappa", fontsize=11)
    ax.set_title("Feature-Set Ablation (Very Strict Recipe)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_feature_ablation.png"), dpi=200)
    plt.close()

    # Fig 5: Threshold calibration curves
    print("  Fig 5: Threshold calibration")
    thr_path = os.path.join(phase5_dir, "threshold_calibration_summary.csv")
    if os.path.exists(thr_path):
        thr_rows = _load_csv(thr_path)
        thrs = [float(r.get("thr", r.get("threshold", 0))) for r in thr_rows]
        r2s = [float(r.get("R2", r.get("district_r2", 0))) for r in thr_rows]
        ratios = [float(r.get("ratio", r.get("burned_area_ratio", 0))) for r in thr_rows]
        rhos = [float(r.get("rho", r.get("spearman_rho", 0))) for r in thr_rows]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(thrs, r2s, marker="o", label="District R² vs MODIS")
        ax1.plot(thrs, rhos, marker="s", label="Spearman ρ vs fire counts")
        ax1.axhline(GATE_DISTRICT_R2_MODIS, linestyle="--", linewidth=1)
        ax1.axhline(GATE_SPEARMAN_CAQM, linestyle=":", linewidth=1)
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Correlation / R²")
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(thrs, ratios, marker="^", label="Burned-area ratio (ours / MODIS)")
        ax2.axhline(1.0, linestyle="-", linewidth=1)
        ax2.axhline(GATE_MODIS_RATIO_MAX, linestyle="--", linewidth=1)
        ax2.set_ylabel("Area ratio")

        if best_threshold:
            ax1.axvline(best_threshold["threshold"], linestyle="--", linewidth=1)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

        plt.title("Threshold Calibration Against External References", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig5_threshold_calibration.png"), dpi=200)
        plt.close()
    else:
        print("    SKIPPED: no threshold calibration summary found")

    # Fig 6: MODIS district scatter
    print("  Fig 6: MODIS district scatter")

    scatter_path = None
    if best_threshold:
        scatter_path = _find_threshold_detail_file(
            phase5_dir,
            best_threshold["threshold"],
            "district_modis_comparison",
        )

    if not scatter_path or not os.path.exists(scatter_path):
        print("    SKIPPED: no threshold-specific district MODIS comparison file found")
    else:
        rows = _load_csv(scatter_path)

        x_vals, y_vals, labels = [], [], []
        for r in rows:
            xv = r.get("ours_burn_pixels_500m")
            yv = r.get("modis_burn_pixels_500m")
            lab = r.get("district", "")

            if xv not in ("", None) and yv not in ("", None):
                x_vals.append(float(xv) * 0.25)
                y_vals.append(float(yv) * 0.25)
                labels.append(str(lab))

        print("    Fig6 x range:", min(x_vals), max(x_vals))
        print("    Fig6 y range:", min(y_vals), max(y_vals))

        if x_vals and y_vals:
            import matplotlib.pyplot as plt
            import numpy as np

            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            ax.scatter(x_vals, y_vals, alpha=0.8)

            mn = min(min(x_vals), min(y_vals))
            mx = max(max(x_vals), max(y_vals))
            ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1, color="gray")

            order = np.argsort(np.array(x_vals) + np.array(y_vals))[::-1][:5]
            for i in order:
                if labels[i]:
                    ax.annotate(labels[i], (x_vals[i], y_vals[i]), fontsize=8, alpha=0.8)

            ax.set_xlabel("Our burned area (km², 500 m grid)")
            ax.set_ylabel("MODIS burned area (km², 500 m grid)")
            ax.set_title(
                f"District Burned Area vs MODIS (thr={best_threshold['threshold']:.2f})",
                fontweight="bold",
            )
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig6_modis_district_scatter.png"), dpi=200)
            plt.close()
        else:
            print(f"    SKIPPED: no usable district pixel columns found in {scatter_path}")

    
    # Fig 7: Phase 6 generalizability
    print("  Fig 7: Phase 6 generalizability")
    if phase6_summary:
        plot_rows = [
            r for r in phase6_summary
            if r["experiment"] in {"punjab_2023", "punjab_2024", "haryana_zeroshot", "s1_fusion_A", "s1_fusion_E"}
        ]
        if plot_rows:
            labels = [r["experiment"] for r in plot_rows]
            means = [float(r["mean_kappa"]) for r in plot_rows]
            errs = [float(r["std_kappa"]) if r["std_kappa"] not in ("", None) else 0 for r in plot_rows]

            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=errs, capsize=3, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("Cohen's Kappa")
            ax.set_title("Phase 6 Generalizability Summary", fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig7_phase6_generalizability.png"), dpi=200)
            plt.close()
        else:
            print("    SKIPPED: no Phase 6 rows to plot")
    else:
        print("    SKIPPED: no Phase 6 summary available")

    # Fig 8: Per-fold kappa variation
    print("  Fig 8: Per-fold kappa variation")
    fig, ax = plt.subplots(figsize=(10, 5))
    recipe_data = []
    recipe_labels = []
    classifiers = sorted(set(str(r.get("classifier")) for r in clf_rows))
    for recipe in ["strict", "moderate", "very_strict"]:
        for clf in classifiers:
            subset = [
                float(r["kappa"]) for r in clf_rows
                if r.get("recipe_id") == recipe and str(r.get("classifier")) == clf
            ]
            if subset:
                recipe_data.append(subset)
                recipe_labels.append(f"{recipe[:4]}\n{clf}")

    if recipe_data:
        bp = ax.boxplot(recipe_data, labels=recipe_labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_alpha(0.7)
        ax.set_ylabel("Cohen's Kappa", fontsize=11)
        ax.set_title("Kappa Distribution Across Folds", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig8_kappa_boxplot.png"), dpi=200)
        plt.close()

    print("  Figures saved to", out_dir)


# ── Framing summary ──────────────────────────────────────────────────

def write_framing_summary(table2, best_threshold, phase6_summary, out_dir):
    if not table2:
        return

    best_trained = table2[0]

    best_rf = None
    for row in table2:
        if row["classifier"] == "rf":
            best_rf = row
            break

    phase6_by_exp = {}
    if phase6_summary:
        for r in phase6_summary:
            phase6_by_exp[r["experiment"]] = r

    lines = []
    lines.append("PHASE 7 PAPER FRAMING SUMMARY")
    lines.append("=" * 40)
    lines.append("")
    lines.append(
        f"Best trained config: {best_trained['recipe']}/{best_trained['classifier']}/{best_trained['feature_set']} "
        f"(kappa={best_trained['mean_kappa']:.3f}, auc={best_trained['mean_auc']:.3f})"
    )
    if best_rf:
        lines.append(
            f"Best deployable RF config: {best_rf['recipe']}/{best_rf['classifier']}/{best_rf['feature_set']} "
            f"(kappa={best_rf['mean_kappa']:.3f}, auc={best_rf['mean_auc']:.3f})"
        )

    if best_threshold:
        lines.append("")
        lines.append(
            f"Selected final map threshold: {best_threshold['threshold']:.2f} "
            f"(R²={best_threshold['district_r2']:.3f}, "
            f"ratio={best_threshold['burned_area_ratio']:.3f}, "
            f"rho={best_threshold['spearman_rho']:.3f})"
        )

    if phase6_by_exp:
        lines.append("")
        if "haryana_zeroshot" in phase6_by_exp:
            lines.append(
                f"Haryana zero-shot transfer: κ={phase6_by_exp['haryana_zeroshot']['mean_kappa']:.3f}"
            )
        if "multiyear_cross_year_std" in phase6_by_exp:
            lines.append(
                f"Cross-year Punjab stability std: {phase6_by_exp['multiyear_cross_year_std']['mean_kappa']:.3f}"
            )
        if "s1_improvement_E_minus_A" in phase6_by_exp:
            lines.append(
                f"S1 fusion improvement (E-A): {phase6_by_exp['s1_improvement_E_minus_A']['mean_kappa']:+.3f}"
            )

    lines.append("")
    lines.append("Recommended narrative:")
    lines.append(
        "- Weak-label recipe quality dominates classifier choice."
    )
    lines.append(
        "- MLP is the best trained model, while RF remains the most practical deployable model."
    )
    lines.append(
        "- External threshold calibration was necessary to obtain a defensible burned-area product."
    )
    lines.append(
        "- Geographic transfer to Haryana is strong."
    )
    lines.append(
        "- Temporal stability across available Punjab years is weak."
    )
    lines.append(
        "- Sentinel-1 adds no value in the current pipeline."
    )

    # Venue suggestion
    lines.append("")
    if float(best_trained["mean_kappa"]) >= 0.40:
        venue = "Tier-1 venue possible, but claims should remain moderate"
    elif float(best_trained["mean_kappa"]) >= 0.30:
        venue = "Tier-1/2 venue recommended (ISPRS JPRS / IEEE JSTARS)"
    else:
        venue = "Tier-2 venue or label-quality framing"
    lines.append(f"Venue guidance: {venue}")

    path = os.path.join(out_dir, "paper_framing_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  {path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    year = args.year
    out_dir = args.out_dir or f"outputs/{year}/paper"
    os.makedirs(out_dir, exist_ok=True)

    cv_path, prob_csv, phase5_dir, phase6_metrics = resolve_paths(year, args)

    if not cv_path or not os.path.exists(cv_path):
        print("ERROR: No CV metrics found. Run merge_phase4.py first.")
        return

    print("Phase 7 · Paper Tables & Figures")
    print(f"CV metrics:    {cv_path}")
    print(f"Prob CSV:      {prob_csv or 'not found'}")
    print(f"Phase 5 dir:   {phase5_dir}")
    print(f"Phase 6 CSV:   {phase6_metrics or 'not found'}")
    print(f"Output dir:    {out_dir}\n")

    cv_rows = _load_csv(cv_path)
    print(f"Loaded {len(cv_rows)} CV result rows\n")

    # Tables
    print("── Generating Tables ──")
    table_recipe_params(out_dir)
    table2 = table_main_results(cv_rows, out_dir)
    table_baseline_comparison(cv_rows, out_dir)
    _, best_threshold = table_threshold_calibration(phase5_dir, out_dir)
    table_external_validation(best_threshold, phase5_dir, out_dir)
    _, phase6_summary = table_phase6_generalizability(phase6_metrics, out_dir)

    # Supplementary tables
    supplementary_feature_importance(cv_rows, out_dir)
    supplementary_recipe_summary(cv_rows, out_dir)

    # Figures
    if not args.no_figures:
        print("\n── Generating Figures ──")
        generate_figures(cv_rows, phase5_dir, best_threshold, phase6_summary, prob_csv, out_dir)

    # Framing summary
    print("\n── Paper Framing ──")
    if table2:
        best = table2[0]
        print(
            f"  Best trained config: {best['recipe']}/{best['classifier']}/{best['feature_set']} "
            f"kappa={best['mean_kappa']:.3f}"
        )
        if best_threshold:
            print(
                f"  Final calibrated threshold: {best_threshold['threshold']:.2f} "
                f"(R²={best_threshold['district_r2']:.3f}, "
                f"ratio={best_threshold['burned_area_ratio']:.3f}, "
                f"rho={best_threshold['spearman_rho']:.3f})"
            )
    write_framing_summary(table2, best_threshold, phase6_summary, out_dir)

    print(f"\nPhase 7 done. All outputs in {out_dir}\n")


if __name__ == "__main__":
    main()