"""Recompute every statistic reported in the paper from the per-seed results.

This is the script behind Tables 3 and 4. It reads the per-seed CSV files in
`results/` and prints the means, standard deviations, omnibus tests, pairwise
contrasts, and effect sizes exactly as the manuscript reports them, so the
numbers can be checked without rerunning any training.

Two conventions matter and are applied uniformly here:

  Standard deviation
      Sample standard deviation, n-1 in the denominator.

  Omnibus test
      One-way REPEATED-MEASURES ANOVA. Every architecture is trained on the
      same five seeds, so the design is within-subjects and the seed is the
      subject factor. A between-groups one-way ANOVA on the same data gives
      noticeably different values, for instance F = 4.449 and p = 0.036 on the
      Aksara Jawa clean split against F = 3.736 and p = 0.072 for the
      repeated-measures test, so the distinction is not cosmetic.

  Cohen's d
      Computed from the pooled standard deviation of the two conditions.

Run:
    python scripts/compute_paper_stats.py
"""
import csv
import itertools
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SEEDS = ["7", "42", "123", "999", "2024"]


def load(filename, key_column):
    path = os.path.join(RESULTS, filename)
    table = {}
    with open(path, newline="", encoding="utf8") as f:
        for row in csv.DictReader(f):
            table.setdefault(row[key_column], {})[row["seed"]] = float(row["top1"]) * 100
    return table


def repeated_measures_anova(table, groups):
    """One-way repeated-measures ANOVA, rows are subjects (seeds)."""
    X = np.array([[table[g][s] for g in groups] for s in SEEDS])
    n, k = X.shape
    grand = X.mean()
    ss_total = ((X - grand) ** 2).sum()
    ss_cond = n * ((X.mean(axis=0) - grand) ** 2).sum()
    ss_subj = k * ((X.mean(axis=1) - grand) ** 2).sum()
    ss_err = ss_total - ss_cond - ss_subj
    df_cond, df_err = k - 1, (k - 1) * (n - 1)
    F = (ss_cond / df_cond) / (ss_err / df_err)
    return F, 1 - stats.f.cdf(F, df_cond, df_err), df_cond, df_err


def pooled_cohens_d(a, b):
    va = np.array([a[s] for s in SEEDS])
    vb = np.array([b[s] for s in SEEDS])
    sp = np.sqrt((va.std(ddof=1) ** 2 + vb.std(ddof=1) ** 2) / 2)
    return (va.mean() - vb.mean()) / sp


print(__doc__.split("Run:")[0].strip())
print("\n" + "=" * 78)

COMPARISONS = [
    ("Aksara Jawa, naive split", "multiseed_results.csv"),
    ("Aksara Jawa, clean split", "multiseed_clean_results.csv"),
    ("DHCD", "dhcd_multiseed_results.csv"),
]

print("TABLE 3  Three YOLO generations")
print("=" * 78)
for label, filename in COMPARISONS:
    d = load(filename, "model")
    groups = list(d)
    print(f"\n{label}")
    for g in groups:
        v = np.array([d[g][s] for s in SEEDS])
        print(f"  {g:10} {v.mean():6.2f} +/- {v.std(ddof=1):.2f}")
    F, p, dfc, dfe = repeated_measures_anova(d, groups)
    verdict = "significant" if p < 0.05 else "not significant"
    print(f"  repeated-measures ANOVA  F({dfc}, {dfe}) = {F:.3f}, p = {p:.4f}   {verdict}")
    for a, b in itertools.combinations(groups, 2):
        t, pp = stats.ttest_rel([d[a][s] for s in SEEDS], [d[b][s] for s in SEEDS])
        diff = np.mean([d[a][s] - d[b][s] for s in SEEDS])
        print(f"    {a:9} vs {b:9} diff = {diff:+.2f} pp, p = {pp:.4f}, "
              f"d = {pooled_cohens_d(d[a], d[b]):+.2f}")

print("\n" + "=" * 78)
print("TABLE 4  C2PSA ablation")
print("=" * 78)
for label, filename in [("Aksara Jawa, clean split", "ablation_c2psa_clean_results.csv"),
                        ("DHCD", "dhcd_ablation_results.csv")]:
    d = load(filename, "variant")
    print(f"\n{label}")
    for k in ("attn", "noattn"):
        v = np.array([d[k][s] for s in SEEDS])
        print(f"  {k:7} {v.mean():6.2f} +/- {v.std(ddof=1):.2f}   "
              f"range {v.min():.2f} to {v.max():.2f}")
    t, p = stats.ttest_rel([d["attn"][s] for s in SEEDS], [d["noattn"][s] for s in SEEDS])
    diff = np.mean([d["attn"][s] - d["noattn"][s] for s in SEEDS])
    wins = sum(1 for s in SEEDS if d["noattn"][s] > d["attn"][s])
    print(f"  difference {diff:+.2f} pp, paired t-test p = {p:.4f}, "
          f"Cohen's d = {pooled_cohens_d(d['attn'], d['noattn']):+.2f}")
    print(f"  seeds where the attention-free variant wins: {wins} of {len(SEEDS)}")
print()
