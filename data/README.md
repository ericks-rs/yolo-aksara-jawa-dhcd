# Datasets

The datasets used in this study are NOT included in this repository due to their size and licensing terms. Please download them from the original sources as described below.

## 1. Aksara Jawa Dataset

- **Source**: Kaggle, contributed by [phiard](https://www.kaggle.com/phiard)
- **URL**: https://www.kaggle.com/datasets/phiard/aksara-jawa
- **Classes**: 20 (ba, ca, da, dha, ga, ha, ja, ka, la, ma, na, nga, nya, pa, ra, sa, ta, tha, wa, ya)
- **Total images**: ~2,634

### Download

1. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```

2. Configure your Kaggle API token (see [Kaggle docs](https://www.kaggle.com/docs/api))

3. Download:
   ```bash
   kaggle datasets download -d phiard/aksara-jawa
   unzip aksara-jawa.zip -d data/aksara-jawa
   ```

### IMPORTANT: Data Leakage Issue

The original train/validation split provided in this dataset contains data leakage. Augmented variants of the same base image (denoted by filename suffixes such as `.r10`, `.rm10`, `.b`, and `.meki`) appear in both training and validation sets.

Measured on the distributed split, exactly 120 of the 480 validation images (25.00%) are augmented variants of an image that also appears in training, spread uniformly at 6 images per class across all 20 classes.

The corrected split is implemented in `scripts/clean_split_aksara.py`, which groups every variant of one base image into a single partition and then splits at the level of base IDs. Run it before using the Aksara Jawa notebooks. We strongly recommend the corrected split for any comparative model evaluation.

## 2. Devanagari Handwritten Character Dataset (DHCD)

- **Source**: Acharya et al. (2015), SKIMA Conference
- **Paper**: S. Acharya, A. K. Pant, and P. K. Gyawali, "Deep learning based large scale handwritten Devanagari character recognition," in *Proc. 9th Int. Conf. Softw., Knowl., Inf. Manage. Appl. (SKIMA)*, 2015, doi: 10.1109/SKIMA.2015.7400041
- **Classes**: 46 (36 consonants + 10 numerals)
- **Total images**: 92,000 (2,000 per class)
- **Train/Validation split**: pre-defined 85:15, no leakage

### Download

The DHCD is available from several mirrors:

- [UCI Machine Learning Repository (DevanagariHandwrittenCharacterDataset)](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset)
- [Kaggle mirrors](https://www.kaggle.com/datasets/) (search "Devanagari Handwritten Character Dataset")

After downloading, place the dataset under `data/dhcd/` with the standard `train/` and `val/` subdirectories.

## Directory Structure (after download)

```
data/
├── README.md                     # This file
├── aksara-jawa/                  # Aksara Jawa dataset
│   ├── train/
│   │   ├── ba/
│   │   ├── ca/
│   │   └── ...
│   └── val/
│       ├── ba/
│       └── ...
└── dhcd/                         # Devanagari dataset
    ├── train/
    │   ├── character_01_ka/
    │   ├── character_02_kha/
    │   └── ...
    └── val/
        ├── character_01_ka/
        └── ...
```

## Citation

Please cite the original dataset authors when using these datasets:

For DHCD:
```bibtex
@inproceedings{acharya2015dhcd,
  title     = {Deep learning based large scale handwritten Devanagari character recognition},
  author    = {Acharya, Shailesh and Pant, Ashok Kumar and Gyawali, Prashnna Kumar},
  booktitle = {Proc. 9th Int. Conf. Softw., Knowl., Inf. Manage. Appl. (SKIMA)},
  pages     = {121--126},
  year      = {2015},
  doi       = {10.1109/SKIMA.2015.7400041}
}
```
