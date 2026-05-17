# Reassessing the Role of Attention Mechanisms in YOLO for Character Classification

This repository contains the source code, experimental scripts, and result tables for the paper:

> **Reassessing the Role of Attention Mechanisms in YOLO for Character Classification: Multi-Seed Empirical Evidence from Javanese and Devanagari Scripts**
>
> Ericks Rachmat Swedia¹, Cakra Trinata²
>
> ¹Department of Informatics Engineering, Faculty of Industrial Technology, Gunadarma University, Depok, West Java, Indonesia
> ²Immigration Technology Management Study Program, Department of Immigration, Politeknik Imigrasi (Indonesia Immigration Polytechnic), Tangerang, Banten, Indonesia
>
> Submitted to: *Bulletin of Electrical Engineering and Informatics (BEEI)*, 2026

## Overview

This study reassesses whether the C2PSA attention block in modern YOLO architectures (YOLO11, YOLO26) provides measurable benefits for character classification of low-resource Indic scripts. Three YOLO generations (YOLOv8n, YOLO11n, YOLO26n) are compared on two datasets, Javanese (Aksara Jawa) and Devanagari (DHCD), under a multi-seed evaluation protocol (5 seeds per configuration), supplemented by direct ablation of the C2PSA block. The findings indicate that attention-equipped models do not provide a statistically significant advantage over attention-free counterparts, and in some cases the attention block introduces training instability.

## Key Findings

- **No significant advantage** of attention-equipped YOLO architectures for character classification (ANOVA p = 0.107–0.673; significant only in favor of pre-attention YOLOv8, p = 0.036)
- **C2PSA ablation** shows removing attention either maintains or improves accuracy (Cohen's d up to −1.13 in favor of no-attention variant)
- **Training instability** introduced by attention on small datasets (worst-case accuracy collapse to 85%)
- **Data leakage** identified in the Kaggle Aksara Jawa dataset; a corrected splitting protocol is proposed

## Repository Structure

```
.
├── README.md                         # This file
├── LICENSE                           # MIT License
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Files excluded from git
│
├── configs/                          # YOLO model configurations
│   ├── yolo26-cls-attn.yaml          # With C2PSA (ablation: attn)
│   └── yolo26-cls-noattn.yaml        # Without C2PSA (ablation: no-attn)
│
├── scripts/                          # Experiment scripts
│   ├── 02_train.py                   # CLI training script
│   ├── dump_architecture.py          # Generate architecture summary
│   ├── export_kernels_txt.py         # Export kernel weights to .txt
│   └── generate_paper_figures.py     # Generate Fig. 1, 2, 3
│
├── utils/                            # Helper modules
│   └── kernel_logger.py              # Callback for saving Conv/Linear weights per epoch
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_dataset_analysis.ipynb     # Dataset exploration
│   ├── 02_train.ipynb                # Basic training
│   ├── 04c_compare_clean.ipynb       # 3-model comparison (clean split)
│   ├── 05c_ablation_clean.ipynb      # C2PSA ablation (clean split)
│   └── 06_dhcd_validation.ipynb      # DHCD cross-script validation
│
├── data/                             # Dataset metadata (no images)
│   └── README.md                     # Instructions to obtain datasets
│
└── results/                          # Per-seed result tables (CSV)
    ├── multiseed_results.csv
    ├── multiseed_clean_results.csv
    ├── dhcd_multiseed_results.csv
    ├── ablation_c2psa_results.csv
    ├── ablation_c2psa_clean_results.csv
    └── dhcd_ablation_results.csv
```

## Environment

- **OS**: Windows 11
- **GPU**: NVIDIA RTX 5080 Laptop (16 GB VRAM, sm_120 / Blackwell)
- **Python**: 3.10.20 (Anaconda)
- **CUDA**: 12.8
- **PyTorch**: 2.11.0+cu128
- **Ultralytics**: 8.4.46

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/yolo-aksara-jawa.git
cd yolo-aksara-jawa
```

### 2. Create conda environment

```bash
conda create -n yolo-aksara python=3.10 -y
conda activate yolo-aksara
pip install -r requirements.txt
```

### 3. Obtain datasets

The datasets are NOT included in this repository. See `data/README.md` for instructions on downloading from Kaggle (Aksara Jawa) and the DHCD source (Devanagari).

## Reproduce the Experiments

### Multi-version comparison (YOLOv8 vs YOLO11 vs YOLO26)

```bash
python scripts/02_train.py --model yolov8n-cls.pt --data <path_to_dataset> --seed 42
python scripts/02_train.py --model yolo11n-cls.pt --data <path_to_dataset> --seed 42
python scripts/02_train.py --model yolo26n-cls.pt --data <path_to_dataset> --seed 42
```

Repeat for seeds `7`, `123`, `999`, `2024`.

### C2PSA ablation

```bash
python scripts/02_train.py --model configs/yolo26-cls-attn.yaml --data <path_to_dataset> --seed 42
python scripts/02_train.py --model configs/yolo26-cls-noattn.yaml --data <path_to_dataset> --seed 42
```

### Generate paper figures

```bash
python scripts/generate_paper_figures.py
```

Output figures will be saved to `figures/`.

## Datasets

| Dataset | Classes | Total Images | Source |
|---------|---------|--------------|--------|
| Aksara Jawa | 20 | ~2,634 | [Kaggle: phiard/aksara-jawa](https://www.kaggle.com/datasets/phiard/aksara-jawa) |
| DHCD | 46 | 92,000 | Acharya et al. (2015), SKIMA |

**Note**: The Aksara Jawa Kaggle dataset has a known data leakage issue in its original train/validation split. A corrected splitting protocol is described in the paper (Section 3.3) and implemented in the relevant notebooks.

## Citation

If you use this code or our findings, please cite:

```bibtex
@article{swedia2026reassessing,
  title   = {Reassessing the Role of Attention Mechanisms in YOLO for Character Classification: Multi-Seed Empirical Evidence from Javanese and Devanagari Scripts},
  author  = {Swedia, Ericks Rachmat and Trinata, Cakra},
  journal = {Bulletin of Electrical Engineering and Informatics},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions about the paper or code:

- **Ericks Rachmat Swedia** — ericks_rs@staff.gunadarma.ac.id
- **Cakra Trinata** — trinatacakra@gmail.com
