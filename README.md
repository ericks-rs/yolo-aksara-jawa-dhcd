# Reassessing the Role of Attention Mechanisms in YOLO for Character Classification

Source code, experiment logs, and per-seed results for the paper:

> **Reassessing the Role of Attention Mechanisms in YOLO for Character Classification: Multi-Seed Empirical Evidence from Javanese and Devanagari Scripts**
>
> Ericks Rachmat Swedia¹, Cakra Trinata²
>
> ¹Department of Informatics Engineering, Faculty of Industrial Technology, Gunadarma University, Depok, Jawa Barat, Indonesia
> ²Department of Immigration, Study Program in Immigration Technology Management, Politeknik Imigrasi dan Pemasyarakatan Indonesia, Tangerang, Banten, Indonesia
>
> Under review at *Bulletin of Electrical Engineering and Informatics (BEEI)*, submission 13400

## Overview

Modern YOLO releases have added attention to the backbone, and YOLO11 and YOLO26 both carry a C2PSA spatial attention block that YOLOv8 does not have. This study asks whether that block earns its place when the task is isolated character classification, where a single glyph sits on a near-empty background.

Three YOLO generations (YOLOv8n, YOLO11n, YOLO26n) are compared on two Indic scripts, Javanese (Aksara Jawa) and Devanagari (DHCD), with five seeds per configuration, followed by a direct ablation that removes the C2PSA block and replaces it with a structurally similar C3k2 block. Across every setting the attention-equipped variants show no consistent advantage, and on the smaller dataset the attention block makes training measurably less stable.

## Key findings

- **No consistent advantage from attention.** No architecture generation differs significantly from the others in any of the three settings. Repeated-measures ANOVA returns p values from 0.072 to 0.722.
- **Removing C2PSA does not hurt, and usually helps.** Cohen's d reaches -1.13 on the Aksara Jawa clean split, a large effect in favor of the attention-free variant.
- **Attention widens the failure mode.** One with-attention run collapsed mid-training, with validation loss jumping from 0.68 to 20.76 in a single epoch and accuracy falling to 4.30%, below the 5% that random guessing gives on 20 classes. No attention-free run behaves like this. See `figures/fig4_convergence.pdf`.
- **The public Aksara Jawa split leaks.** Exactly 25.00% of its validation images are augmented variants of images that appear in the training partition, 6 per class across all 20 classes. A corrected split is provided.

Every number above is reproducible from the per-seed CSV files in `results/` without retraining:

```bash
python scripts/compute_paper_stats.py
```

The design is within-subjects, since all architectures share the same five seeds, so the omnibus test is a repeated-measures ANOVA. A between-groups ANOVA on the same data gives different values and should not be substituted for it.

## The data leakage, quantified

The Kaggle Aksara Jawa dataset ships pre-augmented. Filenames follow the pattern `<base><n>.<tag>.png`, where tags such as `.r10`, `.rm10`, `.b`, and `.meki` mark rotation, blur, and other transforms of one underlying image. In the distributed split, variants of the same base image land in both partitions.

Measured on the distributed split:

| | Naive split (as distributed) | Clean split (`clean_split_aksara.py`) |
|---|---|---|
| Train images | 2,154 | 2,076 |
| Validation images | 480 | 558 |
| Base IDs shared between train and val | 20 of 399 | **0 of 399** |
| **Validation images that leak** | **120 of 480 (25.00%)** | **0** |
| Leaked images per class | 6, uniform across all 20 classes | 0 |

The correction groups every augmented variant of one base image into a single partition, then splits at the level of base IDs rather than files, holding out roughly 20% of base IDs per class with a fixed shuffle seed of 42. Run it with:

```bash
python scripts/clean_split_aksara.py
```

Absolute accuracy is not comparable across the two splits, because regrouping changes both the size and the difficulty profile of the validation set. Only the ranking of architectures within a split is meaningful.

## Training configuration actually used

Every value below is recorded in the published `runs/<experiment>/<run>/args.yaml` files and in the training logs preserved in the notebook outputs.

| | Aksara Jawa | DHCD |
|---|---|---|
| Max epochs | 50 | 50 |
| Early stopping patience | 5 | 5 |
| Image size | 128 x 128 | 32 x 32 |
| Batch size | 64 | 128 |
| Data loader workers | 4 | 4 |
| Optimizer selection | `optimizer='auto'` | `optimizer='auto'` |
| Optimizer chosen by the framework | **AdamW**, lr 0.000417, momentum 0.9 | **MuSGD**, lr 0.01, momentum 0.9 |
| Weight decay on the decayed group | 0.0005 | 0.001 |
| Learning rate schedule | default linear decay (`cos_lr=False`) | same |
| Augmentation | Ultralytics classification default, identical for every model | same |
| Seeds | 7, 42, 123, 999, 2024 | same |

Two points are worth reading carefully.

**`args.yaml` does not show the effective optimizer.** It records `lr0: 0.01` and `momentum: 0.937`, which are the Ultralytics defaults as passed in. Because `optimizer='auto'` was used, the framework discarded those and picked its own, printing the choice to the training log. The effective values are the ones in the table above.

**Augmentation was on, not off.** The default Ultralytics classification pipeline was active (`auto_augment=randaugment`, `erasing=0.4`, `fliplr=0.5`, `scale=0.5`). It was applied identically to every architecture, so it cannot favor any of them, but it was not disabled.

The multi-version comparison used ImageNet pretrained weights. The ablation runs build from a `.yaml` config, so they start from random initialization.

## Reproducing the results

The reported numbers come from the notebooks, which are the reproduction path:

| Notebook | Produces |
|---|---|
| `notebooks/04c_compare_clean.ipynb` | Three-generation comparison, Aksara Jawa clean split |
| `notebooks/05c_ablation_clean.ipynb` | C2PSA ablation, Aksara Jawa clean split |
| `notebooks/06_dhcd_validation.ipynb` | Three-generation comparison and ablation on DHCD |
| `notebooks/01_dataset_analysis.ipynb` | Dataset exploration and the leakage inspection |

`scripts/02_train.py` is a single-run convenience wrapper. It did not produce the published results, and its defaults were aligned in August 2026 with the configuration above so that it no longer contradicts them.

Nothing in the paper requires retraining to verify. The statistics, the figures, and
the leakage measurements are all derived from files already in this repository.

```bash
python scripts/compute_paper_stats.py       # Tables 3 and 4
python scripts/audit_dataset_leakage.py     # the 25.00% and 0.0435% figures in Section 3.2
python scripts/generate_paper_figures.py    # Figures 1 and 2
python scripts/make_fig4.py                 # Figure 3, convergence traces
```

`make_fig4.py` reads the per-epoch logs under `runs/ablation_c2psa_clean/`, so it needs
no GPU. `audit_dataset_leakage.py` needs the datasets in place but does no training.

### Figure numbering

The filenames follow the numbering of the originally submitted manuscript, which had
four figures. One was removed during revision because the merged ablation table and the
convergence plot together carried its content. The mapping is therefore:

| Figure in the paper | File |
|---|---|
| Figure 1 | `fig1_model_comparison` |
| Figure 2 | `fig2_ablation` |
| Figure 3 | `fig4_convergence` |
| not in the paper | `fig3_training_instability`, per-seed bar chart from the submitted version |

`generate_paper_figures.py` still produces `fig3_training_instability`, which is kept
for reference rather than deleted.

## Repository layout

```
.
├── README.md
├── LICENSE                            MIT
├── requirements.txt
│
├── configs/
│   ├── yolo26-cls-attn.yaml           with C2PSA
│   └── yolo26-cls-noattn.yaml         C2PSA replaced by C3k2
│
├── scripts/
│   ├── clean_split_aksara.py          leakage-corrected split
│   ├── prep_dhcd.py                   DHCD preparation
│   ├── 02_train.py                    single-run wrapper
│   ├── compute_paper_stats.py         statistics behind Tables 3 and 4
│   ├── audit_dataset_leakage.py       leakage measurements in Section 3.2
│   ├── make_fig4.py                   Figure 4
│   ├── generate_paper_figures.py      Figures 1 to 3
│   ├── dump_architecture.py
│   └── export_kernels_txt.py
│
├── utils/
│   └── kernel_logger.py               per-epoch Conv and Linear weight snapshots
│
├── notebooks/                         the reproduction path, see table above
│
├── runs/                              per-run training logs, no weights
│   └── <experiment>/<run>/
│       ├── args.yaml                  exact configuration of that run
│       └── results.csv                per-epoch loss and accuracy
│
├── results/                           per-seed summary tables behind the paper tables
│
├── figures/                           Figures 1 to 4, PNG and PDF
│
└── data/README.md                     how to obtain the datasets
```

Model weights are not included because of their size. Everything needed to inspect or redraw the training behavior is in `runs/`.

## Datasets

| Dataset | Classes | Images | Source |
|---|---|---|---|
| Aksara Jawa | 20 | 2,634 | [Kaggle: phiard/aksara-jawa](https://www.kaggle.com/datasets/phiard/aksara-jawa) |
| DHCD | 46 | 92,000 | Acharya, Pant and Gyawali (2015), SKIMA, doi:10.1109/SKIMA.2015.7400041 |

Images are not redistributed here. See `data/README.md`.

## Environment

| | |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120 |
| CPU | Intel Core Ultra 9 275HX, 24 cores |
| RAM | 32 GB |
| Python | 3.10.20 (Anaconda) |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Ultralytics | 8.4.46 |

## Setup

```bash
git clone https://github.com/ericks-rs/yolo-aksara-jawa-dhcd.git
cd yolo-aksara-jawa-dhcd
conda create -n yolo-aksara python=3.10 -y
conda activate yolo-aksara
pip install -r requirements.txt
```

Then follow `data/README.md` to obtain the datasets, and run `scripts/clean_split_aksara.py` before using the Aksara Jawa notebooks.

## Citation

```bibtex
@article{swedia2026reassessing,
  title   = {Reassessing the Role of Attention Mechanisms in {YOLO} for
             Character Classification: Multi-Seed Empirical Evidence from
             {J}avanese and {D}evanagari Scripts},
  author  = {Swedia, Ericks Rachmat and Trinata, Cakra},
  journal = {Bulletin of Electrical Engineering and Informatics},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Contact

- Ericks Rachmat Swedia, ericks_rs@staff.gunadarma.ac.id ([ORCID 0000-0002-9436-0300](https://orcid.org/0000-0002-9436-0300))
- Cakra Trinata, trinatacakra@gmail.com ([ORCID 0009-0009-6039-5343](https://orcid.org/0009-0009-6039-5343))
