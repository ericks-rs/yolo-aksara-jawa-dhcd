"""Figure 3 for BEEI #13400 Revision 1.

Answers Reviewer A: "The discussion regarding training instability would also
be stronger if supported by additional analysis such as convergence behavior
or training loss visualization."

Reads the per-epoch logs that Ultralytics wrote during the original ablation
runs on the Aksara Jawa clean split. No retraining is involved, so the curves
belong to exactly the runs that produced Table 6 in the manuscript.

Source : runs/ablation_c2psa_clean/{variant}_seed{n}/results.csv
Output : figures/fig3_convergence.pdf and .png

Run:
    python scripts/make_fig3.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "ablation_c2psa_clean")
OUT = os.path.join(ROOT, "figures")
SEEDS = [7, 42, 123, 999, 2024]
COLLAPSE_SEED = 7

ATTN_C = "#c0392b"
NOATTN_C = "#2471a3"
COLLAPSE_C = "#7d0000"


def read_curve(variant, seed):
    path = os.path.join(RUNS, f"{variant}_seed{seed}", "results.csv")
    ep, tr, va, acc = [], [], [], []
    with open(path, newline="", encoding="utf8") as f:
        for row in csv.DictReader(f):
            r = {k.strip(): v for k, v in row.items()}
            ep.append(int(float(r["epoch"])))
            tr.append(float(r["train/loss"]))
            va.append(float(r["val/loss"]))
            acc.append(float(r["metrics/accuracy_top1"]) * 100)
    return ep, tr, va, acc


data = {v: {s: read_curve(v, s) for s in SEEDS} for v in ("attn", "noattn")}

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

panels = [
    (0, 1, "Training loss", "Training loss"),
    (1, 2, "Validation loss", "Validation loss"),
    (2, 3, "Validation top-1 accuracy (%)", "Top-1 accuracy (%)"),
]

for ax_i, series_i, title, ylab in panels:
    ax = axes[ax_i]
    for variant, colour, style, label in (
        ("attn", ATTN_C, "-", "With C2PSA"),
        ("noattn", NOATTN_C, "--", "Without C2PSA"),
    ):
        labelled = False
        for s in SEEDS:
            if variant == "attn" and s == COLLAPSE_SEED:
                continue  # drawn last, on top
            ep = data[variant][s][0]
            y = data[variant][s][series_i]
            ax.plot(
                ep,
                y,
                color=colour,
                linestyle=style,
                alpha=0.45,
                linewidth=1.1,
                label=None if labelled else label,
            )
            labelled = True
    ep = data["attn"][COLLAPSE_SEED][0]
    y = data["attn"][COLLAPSE_SEED][series_i]
    ax.plot(
        ep,
        y,
        color=COLLAPSE_C,
        linewidth=2.2,
        label=f"With C2PSA, seed {COLLAPSE_SEED}",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_xlim(left=1)

# the validation-loss spike is two orders of magnitude above the working
# range, so a linear axis would flatten every other curve into the baseline
axes[1].set_yscale("log")
axes[1].set_ylabel("Validation loss (log scale)")

axes[0].legend(fontsize=7.5, loc="upper right", framealpha=0.9)

# mark the collapse itself, not just where training stopped
acc7 = data["attn"][COLLAPSE_SEED][3]
vl7 = data["attn"][COLLAPSE_SEED][2]
# anchor both annotations to the divergence epoch, which is where validation
# loss peaks. Epoch 1 also sits near chance accuracy, so taking the minimum
# accuracy would point at an untrained model instead of at the collapse.
spike = max(range(len(vl7)), key=lambda i: vl7[i])
axes[1].annotate(
    f"val loss {vl7[spike]:.1f} at epoch {spike + 1}",
    xy=(spike + 1, vl7[spike]),
    xytext=(14, -6),
    textcoords="offset points",
    fontsize=7.5,
    color=COLLAPSE_C,
    arrowprops=dict(arrowstyle="->", color=COLLAPSE_C, linewidth=0.9),
)
axes[2].annotate(
    f"accuracy falls to {acc7[spike]:.1f}%",
    xy=(spike + 1, acc7[spike]),
    xytext=(20, 34),
    textcoords="offset points",
    fontsize=7.5,
    color=COLLAPSE_C,
    arrowprops=dict(arrowstyle="->", color=COLLAPSE_C, linewidth=0.9),
)

plt.tight_layout()
for ext in ("pdf", "png"):
    p = os.path.join(OUT, f"fig3_convergence.{ext}")
    fig.savefig(p, dpi=600, bbox_inches="tight")
    print("wrote", p)

# numbers quoted in the caption and the body text, printed so they can be
# checked against what the manuscript says
print("\nper-run summary (from the same logs):")
print(f"{'run':18} {'epochs':>7} {'best top1':>10} {'final top1':>11} {'min val loss':>13} {'final val loss':>15}")
for variant in ("attn", "noattn"):
    for s in SEEDS:
        ep, tr, va, acc = data[variant][s]
        print(
            f"{variant+'_seed'+str(s):18} {len(ep):7d} {max(acc):10.2f} "
            f"{acc[-1]:11.2f} {min(va):13.4f} {va[-1]:15.4f}"
        )

for variant in ("attn", "noattn"):
    lens = [len(data[variant][s][0]) for s in SEEDS]
    print(f"\n{variant}: epochs to early stop = {sorted(lens)}, "
          f"range {min(lens)} to {max(lens)}")
