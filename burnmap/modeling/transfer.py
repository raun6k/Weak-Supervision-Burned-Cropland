"""
Phase 6 — Step 2: Generalizability Analysis
=============================================

Runs on exported CSVs from phase6_export:

  6.1 Multi-year: Punjab 2023 and 2024
  6.2 Haryana: zero-shot transfer, with no Haryana training or recalibration
  6.3 S1+S2 fusion: compare set E vs set A

Usage
-----
    python scripts/transfer.py --year 2024
    python scripts/transfer.py --year 2024 --experiments multiyear haryana
"""

from __future__ import annotations

import argparse
import glob
import os

import burnmap.config as config
from burnmap.modeling.cross_validation import (
    _build_band_lists,
    _load_csv,
    _assign_adm2_folds,
    write_csv,
    run_fold as run_fold_phase3,
)
from burnmap.modeling.classifiers import run_fold as run_fold_phase4


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 6 — Generalizability Analysis")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--sample_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument(
        "--experiments",
        nargs="+",
        default=["multiyear", "haryana", "s1"],
        help="Which experiments to run: multiyear, haryana, s1",
    )
    p.add_argument(
        "--classifiers",
        nargs="+",
        default=["rf"],
        help="Classifiers to test (default: rf only)",
    )
    return p.parse_args()


def _find_csvs(sample_dir: str, patterns: str | list[str]) -> list[str]:
    """Find sample CSVs. A list tries the public name, then a Drive export name."""
    if isinstance(patterns, str):
        patterns = [patterns]
    found = []
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(sample_dir, pattern))):
            if path not in seen:
                seen.add(path)
                found.append(path)
    return found


def _normalize_run_fold_result(result, label: str):
    """
    Both phase3_cv.run_fold and phase4_python.run_fold may return either:
      - (metrics_row, prob_rows, extra)
      - metrics_row dict
      - None / error
    Normalize to a metrics-row dict or None.
    """
    if result is None:
        return None

    if isinstance(result, tuple):
        metrics_row = result[0]
    elif isinstance(result, dict):
        metrics_row = result
    else:
        return None

    metrics_row["experiment"] = label
    return metrics_row


def _dispatch_run_fold(
    clf_name: str,
    train_rows: list[dict],
    test_rows: list[dict],
    band_list: list[str],
    recipe_id: str,
    set_id: str,
    seed: int,
    fold_id: int,
):
    """
    Route RF/GBT through the Phase 3 fold runner,
    and Python-only classifiers through the Phase 4 fold runner.
    """
    clf_name = str(clf_name).lower()

    if clf_name in {"rf", "gbt"}:
        return run_fold_phase3(
            clf_name,
            train_rows,
            test_rows,
            band_list,
            recipe_id,
            set_id,
            seed,
            fold_id,
        )

    return run_fold_phase4(
        clf_name,
        train_rows,
        test_rows,
        band_list,
        recipe_id,
        set_id,
        seed,
        fold_id,
    )


def _run_cv_on_samples(
    csv_paths: list[str],
    band_list: list[str],
    clf_name: str,
    recipe_id: str,
    set_id: str,
    label: str,
) -> list[dict]:
    """Run district-held-out CV on a list of full-region CSVs."""
    results = []

    for csv_path in csv_paths:
        rows = _load_csv(csv_path)
        if not rows:
            continue

        # Extract seed from filename (e.g., ..._s13_full.csv)
        fname = os.path.basename(csv_path)
        seed = config.SEEDS[0]
        for s in config.SEEDS:
            if f"_s{s}_" in fname:
                seed = s
                break

        adm2_codes = sorted(
            set(
                int(r.get("adm2", -1))
                for r in rows
                if int(r.get("adm2", -1)) >= 0
            )
        )

        if not adm2_codes:
            print(f"  WARNING: no valid adm2 codes in {fname}, skipping")
            continue

        # Fewer districts than folds → use leave-one-out
        k = min(config.K_FOLDS, len(adm2_codes))
        adm2_fold_map = _assign_adm2_folds(adm2_codes, k, seed)

        for fold_id in range(k):
            test_adm2s = {c for c, f in adm2_fold_map.items() if f == fold_id}
            train_adm2s = {c for c, f in adm2_fold_map.items() if f != fold_id}

            train_rows = [r for r in rows if int(r.get("adm2", -1)) in train_adm2s]
            test_rows = [r for r in rows if int(r.get("adm2", -1)) in test_adm2s]

            if not train_rows or not test_rows:
                continue

            raw_result = _dispatch_run_fold(
                clf_name,
                train_rows,
                test_rows,
                band_list,
                recipe_id,
                set_id,
                seed,
                fold_id,
            )
            metrics_row = _normalize_run_fold_result(raw_result, label)
            if metrics_row is not None:
                results.append(metrics_row)

    return results


# ── 6.1 Multi-year ───────────────────────────────────────────────────

def run_multiyear(sample_dir, band_list, clf_name, rid, set_id, out_dir):
    """Compare Punjab 2023 with Punjab 2024. The paper reports only these two years."""
    print("\n── 6.1 Multi-Year Replication ──")

    year_results = {}
    for year in [2023, 2024]:
        if year == 2024:
            patterns = [
                f"punjab_{year}_{rid}_{set_id}_s*.csv",
                f"*phase3_{rid}_{set_id}_s*_full.csv",
            ]
        else:
            patterns = [
                f"punjab_{year}_{rid}_{set_id}_s*.csv",
                f"*{year}_phase6_{rid}_{set_id}_s*_full.csv",
            ]

        csvs = _find_csvs(sample_dir, patterns)
        if not csvs:
            print(f"  {year}: no CSVs found (pattern: {patterns[0]})")
            continue

        print(f"  {year}: {len(csvs)} CSVs found")
        results = _run_cv_on_samples(
            csvs,
            band_list,
            clf_name,
            rid,
            set_id,
            f"punjab_{year}",
        )
        year_results[year] = results

        if results:
            ks = [r["kappa"] for r in results if "kappa" in r]
            if ks:
                mean_k = sum(ks) / len(ks)
                std_k = (sum((k - mean_k) ** 2 for k in ks) / len(ks)) ** 0.5
                print(f"    kappa: {mean_k:.3f} ±{std_k:.3f} (n={len(ks)})")

    # Stability analysis
    if len(year_results) >= 2:
        year_means = {}
        for yr, res in year_results.items():
            if res:
                ks = [r["kappa"] for r in res if "kappa" in r]
                if ks:
                    year_means[yr] = sum(ks) / len(ks)

        if len(year_means) >= 2:
            vals = list(year_means.values())
            overall_mean = sum(vals) / len(vals)
            overall_std = (sum((v - overall_mean) ** 2 for v in vals) / len(vals)) ** 0.5

            print(f"\n  Year means: {year_means}")
            print(f"  Cross-year std: {overall_std:.4f}")
            g = overall_std <= config.GATE_MULTIYEAR_KAPPA_STD
            print(
                f"  Gate (std ≤ {config.GATE_MULTIYEAR_KAPPA_STD}): "
                f"{'PASS' if g else 'FAIL'}"
            )

    all_results = []
    for res in year_results.values():
        all_results.extend(res)
    return all_results


# ── 6.2 Haryana Transfer ─────────────────────────────────────────────

def run_haryana(sample_dir, punjab_sample_dir, band_list, clf_name, rid, set_id, out_dir):
    """
    Zero-shot transfer: three Punjab-trained models, one per seed, applied
    to Haryana with the Punjab inner-validation threshold. No Haryana
    training, fine-tuning, or threshold calibration.
    """
    print("\n── 6.2 Haryana Transfer ──")

    h_patterns = [
        f"haryana_2024_{rid}_{set_id}_s*.csv",
        f"*phase6_haryana_{rid}_{set_id}_s*_full.csv",
    ]
    h_csvs = _find_csvs(sample_dir, h_patterns)

    # Punjab 2024 samples train the zero-shot model.
    p_patterns = [
        f"punjab_2024_{rid}_{set_id}_s*.csv",
        f"*phase3_{rid}_{set_id}_s*_full.csv",
    ]
    p_csvs = _find_csvs(punjab_sample_dir, p_patterns)

    results = []

    if not h_csvs:
        print(f"  No Haryana CSVs found (pattern: {h_patterns[0]})")
        return results

    print(f"  Found {len(h_csvs)} Haryana CSVs, {len(p_csvs)} Punjab CSVs")

    # a) Zero-shot transfer
    if p_csvs and h_csvs:
        print("\n  Zero-shot transfer (train Punjab → test Haryana):")
        for p_csv, h_csv in zip(p_csvs, h_csvs):
            p_rows = _load_csv(p_csv)
            h_rows = _load_csv(h_csv)

            seed = config.SEEDS[0]
            for s in config.SEEDS:
                if f"_s{s}_" in os.path.basename(p_csv):
                    seed = s
                    break

            raw_result = _dispatch_run_fold(
                clf_name,
                p_rows,
                h_rows,
                band_list,
                rid,
                set_id,
                seed,
                -1,
            )
            metrics_row = _normalize_run_fold_result(raw_result, "haryana_zeroshot")
            if metrics_row is not None:
                results.append(metrics_row)
                print(f"    s{seed}: kappa={metrics_row['kappa']:.3f}")

        ks = [r["kappa"] for r in results if r.get("experiment") == "haryana_zeroshot"]
        if ks:
            mean_k = sum(ks) / len(ks)
            g = mean_k >= config.GATE_HARYANA_ZEROSHOT_KAPPA
            print(
                f"  Zero-shot mean kappa: {mean_k:.3f} "
                f"({'PASS' if g else 'FAIL'} ≥{config.GATE_HARYANA_ZEROSHOT_KAPPA})"
            )

    return results


# ── 6.3 S1+S2 Fusion ─────────────────────────────────────────────────

def run_s1_fusion(sample_dir, band_lists, clf_name, rid, out_dir):
    """Compare set E (S1+S2) vs set A (S2 only)."""
    print("\n── 6.3 S1+S2 Fusion ──")

    e_patterns = [
        f"punjab_2024_sentinel1_{rid}_E_s*.csv",
        f"*phase6_s1_{rid}_E_s*_full.csv",
    ]
    e_csvs = _find_csvs(sample_dir, e_patterns)

    if not e_csvs:
        print(f"  No Set E CSVs found (pattern: {e_patterns[0]})")
        return []

    print(f"  Found {len(e_csvs)} Set E CSVs")

    # Run CV on set E
    e_results = _run_cv_on_samples(
        e_csvs,
        band_lists["E"],
        clf_name,
        rid,
        "E",
        "s1_fusion_E",
    )

    if e_results:
        ks_e = [r["kappa"] for r in e_results if "kappa" in r]
        if ks_e:
            mean_e = sum(ks_e) / len(ks_e)
            print(f"  Set E (S1+S2): kappa={mean_e:.3f} (n={len(ks_e)})")

            # Compare with set A from Phase 3
            a_patterns = [
                f"punjab_2024_{rid}_A_s*.csv",
                f"*phase3_{rid}_A_s*_full.csv",
            ]
            a_csvs = _find_csvs(sample_dir, a_patterns)

            if a_csvs:
                a_results = _run_cv_on_samples(
                    a_csvs,
                    band_lists["A"],
                    clf_name,
                    rid,
                    "A",
                    "s1_fusion_A",
                )
                if a_results:
                    ks_a = [r["kappa"] for r in a_results if "kappa" in r]
                    if ks_a:
                        mean_a = sum(ks_a) / len(ks_a)
                        diff = mean_e - mean_a
                        print(f"  Set A (S2 only): kappa={mean_a:.3f}")
                        print(f"  Improvement: {diff:+.4f}")
                        g = diff >= config.GATE_S1_IMPROVEMENT
                        print(
                            f"  Gate (≥{config.GATE_S1_IMPROVEMENT}): "
                            f"{'PASS (S1 helps)' if g else 'FAIL (S1 adds no value)'}"
                        )
                    e_results.extend(a_results)

    return e_results


# ── Gate checks ──────────────────────────────────────────────────────

def run_gate_checks(all_results):
    print("\n" + "=" * 60)
    print("PHASE 6 GATE CHECKS")
    print("=" * 60)

    # Group by experiment
    by_exp = {}
    for r in all_results:
        exp_name = r.get("experiment", "unknown")
        by_exp.setdefault(exp_name, []).append(r)

    for exp_name, results in sorted(by_exp.items()):
        ks = [r["kappa"] for r in results if "kappa" in r]
        if ks:
            mean_k = sum(ks) / len(ks)
            std_k = (sum((k - mean_k) ** 2 for k in ks) / len(ks)) ** 0.5
            print(f"\n  {exp_name}: kappa={mean_k:.3f} ±{std_k:.3f} (n={len(ks)})")

    print(f"\n{'=' * 60}\n")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    year = args.year
    sample_dir = args.sample_dir or "data/samples"
    out_dir = args.out_dir or f"outputs/{year}/phase6_results"
    os.makedirs(out_dir, exist_ok=True)

    rid = config.BEST_RECIPE
    set_id = config.BEST_FEATURE_SET
    clf_name = args.classifiers[0].lower() if args.classifiers else config.BEST_CLASSIFIER.lower()
    band_lists = _build_band_lists()
    bl = band_lists[set_id]

    print("Phase 6 CV · Generalizability")
    print(f"Sample dir:  {sample_dir}")
    print(f"Output dir:  {out_dir}")
    print(f"Config:      {clf_name}/{rid}/{set_id}")
    print(f"Experiments: {args.experiments}\n")

    all_results = []

    if "multiyear" in args.experiments:
        results = run_multiyear(sample_dir, bl, clf_name, rid, set_id, out_dir)
        all_results.extend(results)

    if "haryana" in args.experiments:
        results = run_haryana(sample_dir, sample_dir, bl, clf_name, rid, set_id, out_dir)
        all_results.extend(results)

    if "s1" in args.experiments:
        results = run_s1_fusion(sample_dir, band_lists, clf_name, rid, out_dir)
        all_results.extend(results)

    # Write results
    if all_results:
        print("\nWriting results …")
        write_csv(all_results, os.path.join(out_dir, "phase6_cv_metrics.csv"))

    run_gate_checks(all_results)
    print("Phase 6 done.\n")


if __name__ == "__main__":
    main()