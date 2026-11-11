"""
Merge Phase 3 + Phase 4 (LR/XGB/LGBM) + Phase 4 (MLP/MLP-SCE) results
into a single combined CSV with all 7 classifiers + baselines.

Usage:
    python merge_phase4.py
"""

from typing import Any


import csv
import os


def load_csv(path):
    with open(path) as f:
        return list[dict[str | Any, str | Any]](csv.DictReader(f))


def write_csv(rows, path):
    if not rows:
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {path} ({len(rows)} rows)")


def main(year: int = 2024):
    base = f"outputs/{year}"
    p3 = os.path.join(base, "phase3_results", "cv_metrics.csv")
    p4a = os.path.join(base, "phase4_results", "phase4_cv_metrics.csv")
    p4b = os.path.join(base, "phase4_mlp_results", "phase4_cv_metrics.csv")
    out_dir = os.path.join(base, "combined_results")
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []

    if os.path.exists(p3):
        rows = load_csv(p3)
        print(f"Phase 3: {len(rows)} rows (rf, gbt + baselines)")
        all_rows.extend(rows)
    else:
        print(f"WARNING: {p3} not found")

    if os.path.exists(p4a):
        rows = load_csv(p4a)
        print(f"Phase 4a: {len(rows)} rows (lr, xgb, lgbm)")
        all_rows.extend(rows)
    else:
        print(f"WARNING: {p4a} not found")

    if os.path.exists(p4b):
        rows = load_csv(p4b)
        print(f"Phase 4b: {len(rows)} rows (mlp, mlp_sce)")
        all_rows.extend(rows)
    else:
        print(f"WARNING: {p4b} not found")

    # Deduplicate: if Phase 3 rows were merged into Phase 4 combined files,
    # we might have duplicates. Dedupe by (recipe_id, set_id, classifier, seed, fold_id)
    seen_keys = set()
    deduped = []
    for r in all_rows:
        key = (r.get("recipe_id"), r.get("set_id"), r.get("classifier"),
               str(r.get("seed")), str(r.get("fold_id")))
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(r)

    print(f"\nAfter dedup: {len(deduped)} rows (from {len(all_rows)} total)")

    # Split into classifier results and baselines
    clf_rows = [r for r in deduped if not r.get("classifier", "").startswith("baseline_")]
    bl_rows = [r for r in deduped if r.get("classifier", "").startswith("baseline_")]

    # Summary
    classifiers = {}
    for r in clf_rows:
        c = r.get("classifier", "?")
        classifiers.setdefault(c, []).append(float(r.get("kappa", 0)))

    print(f"\nClassifier rows: {len(clf_rows)}")
    print(f"Baseline rows: {len(bl_rows)}")
    print(f"\n{'Classifier':12s} {'Mean Kappa':>10s} {'N':>5s}")
    print("-" * 30)
    for c in sorted(classifiers, key=lambda c: -sum(classifiers[c])/len(classifiers[c])):
        ks = classifiers[c]
        print(f"{c:12s} {sum(ks)/len(ks):10.3f} {len(ks):5d}")

    # Write combined
    write_csv(deduped, os.path.join(out_dir, "all_cv_metrics.csv"))

    # Also merge probability files
    prob_rows = []
    for pf in [
        os.path.join(base, "phase3_results", "test_probabilities.csv"),
        os.path.join(base, "phase4_results", "phase4_test_probabilities.csv"),
        os.path.join(base, "phase4_mlp_results", "phase4_test_probabilities.csv"),
    ]:
        if os.path.exists(pf):
            rows = load_csv(pf)
            prob_rows.extend(rows)
            print(f"  Probabilities: {len(rows)} from {pf}")

    if prob_rows:
        write_csv(prob_rows, os.path.join(out_dir, "all_test_probabilities.csv"))

    print(f"\nAll merged files in: {out_dir}")


if __name__ == "__main__":
    main()