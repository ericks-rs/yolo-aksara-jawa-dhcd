"""Generate paper figures (Fig. 1, 2, 3) from experiment CSV result tables.

Reads CSVs from ../results/ (relative to this script's parent) and writes
PNG + PDF figures to ../figures/.

Output figures:
  - fig1_model_comparison.png      : YOLOv8/11/26 across 3 datasets (box plot)
  - fig2_ablation.png              : C2PSA on/off comparison (Aksara clean + DHCD)
  - fig3_training_instability.png  : per-seed bars, IN THE SUBMITTED VERSION ONLY.
                                     Dropped during revision, since the merged
                                     ablation table and the convergence plot in
                                     fig3_convergence carry the same information.
                                     Set KEEP_SUPERSEDED below to regenerate it.

Usage:
    python scripts/generate_paper_figures.py
"""
from pathlib import Path
import pandas as pd
KEEP_SUPERSEDED = False  # the paper no longer has this figure

import matplotlib.pyplot as plt

# Embed text as TrueType (42) rather than matplotlib's default Type 3.
# Type 3 text cannot be selected reliably and some publishers reject it.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
import numpy as np

# Resolve project root (parent of this script's parent)
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True, parents=True)

# Load all CSVs
df_naive   = pd.read_csv(RESULTS / 'multiseed_results.csv')
df_clean   = pd.read_csv(RESULTS / 'multiseed_clean_results.csv')
df_dhcd    = pd.read_csv(RESULTS / 'dhcd_multiseed_results.csv')
df_abl_aks = pd.read_csv(RESULTS / 'ablation_c2psa_clean_results.csv')
df_abl_dh  = pd.read_csv(RESULTS / 'dhcd_ablation_results.csv')

# Add dataset labels
df_naive['dataset'] = 'Aksara Jawa (Naive)'
df_clean['dataset'] = 'Aksara Jawa (Clean)'
df_dhcd['dataset']  = 'Devanagari (DHCD)'

# Convert top1 to percentage
for df in [df_naive, df_clean, df_dhcd, df_abl_aks, df_abl_dh]:
    df['top1_pct'] = df['top1'] * 100

# Model name pretty
MODEL_LABELS = {'yolov8n': 'YOLOv8n', 'yolo11n': 'YOLO11n', 'yolo26n': 'YOLO26n'}
MODEL_COLORS = {'yolov8n': '#2E86AB', 'yolo11n': '#A23B72', 'yolo26n': '#F18F01'}

# Fixed generator for the scatter jitter, so the figures redraw identically.
JITTER = np.random.RandomState(42)

# ========== FIGURE 1: 3-Model Comparison (Box plot) ==========
# (13, 4.2) is the aspect used in the paper: at the 11 cm print width it leaves
# room for Table 3 under the figure on the same page. (13, 5.4) did not.
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
datasets = [
    ('Aksara Jawa (Naive)', df_naive),
    ('Aksara Jawa (Clean)', df_clean),
    ('Devanagari (DHCD)', df_dhcd),
]

for ax, (title, df) in zip(axes, datasets):
    data = [df[df['model'] == m]['top1_pct'].values for m in ['yolov8n', 'yolo11n', 'yolo26n']]
    # every seed is drawn by the scatter below, so the default flier markers
    # would render the extreme runs twice and imply extra observations
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                     medianprops=dict(color='black', linewidth=1.2))
    for patch, m in zip(bp['boxes'], ['yolov8n', 'yolo11n', 'yolo26n']):
        patch.set_facecolor(MODEL_COLORS[m])
        patch.set_alpha(0.75)
    ax.set_xticklabels([MODEL_LABELS[m] for m in ['yolov8n', 'yolo11n', 'yolo26n']])
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    # Overlay scatter (per-seed dots)
    for i, m in enumerate(['yolov8n', 'yolo11n', 'yolo26n']):
        y = df[df['model'] == m]['top1_pct'].values
        x = np.full_like(y, i + 1, dtype=float) + JITTER.uniform(-0.08, 0.08, len(y))
        ax.scatter(x, y, color='black', s=18, zorder=10, alpha=0.7)

# extra horizontal padding so the three panels read as separate charts
plt.tight_layout(w_pad=3.5)
plt.savefig(OUT / 'fig1_model_comparison.png', dpi=600, bbox_inches='tight')
plt.savefig(OUT / 'fig1_model_comparison.pdf', bbox_inches='tight')
plt.close()
print('Saved fig1_model_comparison')

# ========== FIGURE 2: C2PSA Ablation Comparison ==========
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

ablation_data = [
    ('Aksara Jawa (Clean Split)', df_abl_aks),
    ('Devanagari (DHCD)', df_abl_dh),
]

for ax, (title, df) in zip(axes, ablation_data):
    attn = df[df['variant'] == 'attn']['top1_pct'].values
    noattn = df[df['variant'] == 'noattn']['top1_pct'].values

    bp = ax.boxplot([attn, noattn], patch_artist=True, widths=0.55, showfliers=False,
                     medianprops=dict(color='black', linewidth=1.2))
    bp['boxes'][0].set_facecolor('#C73E1D')  # red-ish for attn
    bp['boxes'][1].set_facecolor('#3D8B37')  # green for no-attn
    for patch in bp['boxes']:
        patch.set_alpha(0.75)
    ax.set_xticklabels(['With C2PSA', 'Without C2PSA'])
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    # Per-seed scatter
    for i, vals in enumerate([attn, noattn]):
        x = np.full_like(vals, i + 1, dtype=float) + JITTER.uniform(-0.06, 0.06, len(vals))
        ax.scatter(x, vals, color='black', s=20, zorder=10, alpha=0.75)

# extra horizontal padding so the panels read as separate charts
plt.tight_layout(w_pad=3.5)
plt.savefig(OUT / 'fig2_ablation.png', dpi=600, bbox_inches='tight')
plt.savefig(OUT / 'fig2_ablation.pdf', bbox_inches='tight')
plt.close()
print('Saved fig2_ablation')

# ========== FIGURE 3: Training Instability (per-seed bar) ==========
fig, ax = plt.subplots(figsize=(7.5, 4.5))
seeds = sorted(df_abl_aks['seed'].unique())
attn_by_seed   = [df_abl_aks[(df_abl_aks['variant']=='attn')   & (df_abl_aks['seed']==s)]['top1_pct'].values[0] for s in seeds]
noattn_by_seed = [df_abl_aks[(df_abl_aks['variant']=='noattn') & (df_abl_aks['seed']==s)]['top1_pct'].values[0] for s in seeds]

x = np.arange(len(seeds))
width = 0.35
b1 = ax.bar(x - width/2, attn_by_seed,   width, label='With C2PSA',   color='#C73E1D', alpha=0.85)
b2 = ax.bar(x + width/2, noattn_by_seed, width, label='Without C2PSA', color='#3D8B37', alpha=0.85)

# Highlight seed=7 collapse
seed7_idx = seeds.index(7)
ax.annotate('Training\ncollapse',
            xy=(seed7_idx - width/2, attn_by_seed[seed7_idx]),
            xytext=(seed7_idx + 0.35, 88),
            fontsize=9, color='darkred', ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.4))

ax.set_xticks(x)
ax.set_xticklabels([f'seed={s}' for s in seeds])
ax.set_ylabel('Top-1 Accuracy (%)')
ax.set_title('Per-Seed Accuracy on Aksara Jawa (Clean Split), C2PSA Ablation')
ax.set_ylim(80, 100)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f'{h:.1f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
if KEEP_SUPERSEDED:
    plt.savefig(OUT / 'fig3_training_instability.png', dpi=600, bbox_inches='tight')
    plt.savefig(OUT / 'fig3_training_instability.pdf', bbox_inches='tight')
    plt.close()
    print('Saved fig3_training_instability (superseded, not in the paper)')
else:
    plt.close()

print(f'\nAll figures saved to: {OUT}')
