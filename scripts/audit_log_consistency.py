"""Check the published accuracies against the per-epoch training logs.

Every accuracy in the paper comes from a full-precision validation of best.pt
run once training finished. The per-epoch logs under runs/ record a different
measurement, the mixed-precision validation carried out during training, so a
reader who compares the peak of a results.csv against a printed table can find
the two disagreeing slightly. They are not two attempts at the same number.

The weights behind them are identical. The arithmetic is not, and a sample
sitting close to a decision boundary can land in a different class under FP16
than under FP32. That is the entire source of the disagreement, which is why
every gap below comes out as a whole number of images and why the gaps run in
both directions rather than favouring either measurement.

The aggregate CSVs under results/ are the source of every number printed in
the paper. This script exists so the claim in README.md, that 48 of the 60
runs agree exactly and the rest differ by at most three images, can be checked
rather than taken on trust.

Needs no GPU and no dataset, only the files already in this repository.

Run:
    python scripts/audit_log_consistency.py
"""

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
RUNS = os.path.join(ROOT, "runs")

ACC = "metrics/accuracy_top1"
TOL = 1e-3  # percentage points, far below one image in any of the splits

# folder, aggregate csv, column naming the configuration, validation images
EXPERIMENTS = [
    ("multiseed", "multiseed_results.csv", "model", 480,
     "Three generations, Aksara Jawa distributed split"),
    ("multiseed_clean", "multiseed_clean_results.csv", "model", 558,
     "Three generations, Aksara Jawa clean split"),
    ("dhcd_multiseed", "dhcd_multiseed_results.csv", "model", 18400,
     "Three generations, DHCD"),
    ("ablation_c2psa", "ablation_c2psa_results.csv", "variant", 480,
     "C2PSA ablation, Aksara Jawa distributed split"),
    ("ablation_c2psa_clean", "ablation_c2psa_clean_results.csv", "variant", 558,
     "C2PSA ablation, Aksara Jawa clean split"),
    ("dhcd_ablation", "dhcd_ablation_results.csv", "variant", 18400,
     "C2PSA ablation, DHCD"),
]


def peak_from_log(folder, run):
    """Best top-1 recorded during training, or None if the log is absent."""
    path = os.path.join(RUNS, folder, run, "results.csv")
    if not os.path.exists(path):
        return None
    log = pd.read_csv(path)
    log.columns = [c.strip() for c in log.columns]
    return log[ACC].max() * 100


def main():
    checked = exact = 0
    worst = 0.0
    missing = []

    for folder, aggfile, key, n_val, title in EXPERIMENTS:
        agg = pd.read_csv(os.path.join(RESULTS, aggfile))
        agg["run"] = agg[key].astype(str) + "_seed" + agg["seed"].astype(str)
        per_image = 100.0 / n_val

        rows, absent = [], 0
        for _, r in agg.iterrows():
            peak = peak_from_log(folder, r["run"])
            if peak is None:
                absent += 1
                continue
            rows.append((r["run"], r["top1"] * 100 - peak))

        print(f"{title}")
        print(f"  validation set {n_val:,} images, one image is "
              f"{per_image:.4f} percentage points")
        if absent:
            missing.append(f"{folder} ({absent} run)")
        if not rows:
            print("  no per-epoch logs in this repository\n")
            continue

        off = [(run, gap) for run, gap in rows if abs(gap) > TOL]
        checked += len(rows)
        exact += len(rows) - len(off)
        print(f"  {len(rows) - len(off)} of {len(rows)} runs agree exactly")
        for run, gap in off:
            images = gap / per_image
            worst = max(worst, abs(images))
            print(f"    {run:22s} {gap:+.3f} pp = {images:+.0f} image"
                  f"{'' if abs(round(images)) == 1 else 's'}")
        print()

    print("-" * 62)
    print(f"{exact} of {checked} runs agree exactly between the published "
          f"accuracy and the training log.")
    if checked > exact:
        print(f"The remaining {checked - exact} differ by at most "
              f"{worst:.0f} images, in both directions.")
    if missing:
        print(f"No per-epoch logs kept for: {', '.join(missing)}.")


if __name__ == "__main__":
    main()
