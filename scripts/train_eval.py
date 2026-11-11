#!/usr/bin/env python3
"""District-held-out model comparison. Formerly phases 3 CV, 4, and the result merge.

Runs, in order:

1. Random forest and gradient boosting, plus dNBR and MODIS baselines.
2. Logistic regression, XGBoost, and LightGBM.
3. MLP and MLP with symmetric cross-entropy, written separately.
4. Merge of the three metric tables.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from burnmap.modeling import classifiers, cross_validation, merge_results


def main():
    p = argparse.ArgumentParser(description="Train and evaluate the paper classifiers")
    p.add_argument("--year", type=int, default=2024, choices=[2023, 2024])
    p.add_argument("--sample_dir", default="data/samples")
    p.add_argument("--recipes", nargs="+", default=["strict", "moderate", "very_strict"])
    p.add_argument("--feature_sets", nargs="+", default=["A", "B", "C", "D"])
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument("--skip_stats", action="store_true")
    args = p.parse_args()

    common = [
        "--year", str(args.year),
        "--sample_dir", args.sample_dir,
        "--recipes", *args.recipes,
        "--feature_sets", *args.feature_sets,
    ]

    sys.argv = ["cross_validation", *common]
    if args.skip_baselines:
        sys.argv.append("--skip_baselines")
    cross_validation.main()

    sys.argv = ["classifiers", *common, "--classifiers", "lr", "xgb", "lgbm"]
    if args.skip_stats:
        sys.argv.append("--skip_stats")
    classifiers.main()

    sys.argv = [
        "classifiers", *common,
        "--classifiers", "mlp", "mlp_sce",
        "--out_dir", f"outputs/{args.year}/phase4_mlp_results",
    ]
    if args.skip_stats:
        sys.argv.append("--skip_stats")
    classifiers.main()

    merge_results.main(year=args.year)


if __name__ == "__main__":
    main()
