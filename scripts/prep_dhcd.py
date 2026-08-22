"""Prepare DHCD dataset untuk Ultralytics classification training.

DHCD source structure:
  DatasetDHCD/Images/Images/character_NN_xxx/{2000 png files}
  DatasetDHCD/Images/Images/digit_N/{2000 png files}

Output Ultralytics-compatible:
  DatasetDHCD_split/
    ├── train/{class}/  (80%)
    └── val/{class}/    (20%)

Run:
    python scripts/prep_dhcd.py
"""
import shutil
import random
from pathlib import Path

random.seed(42)  # reproducible split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'DatasetDHCD' / 'Images' / 'Images'
DST = ROOT / 'DatasetDHCD_split'

TRAIN_RATIO = 0.8

if not SRC.exists():
    raise FileNotFoundError(f'DHCD source not found: {SRC}')

classes = sorted([p.name for p in SRC.iterdir() if p.is_dir()])
print(f'Found {len(classes)} classes in DHCD')
print(f'Sample: {classes[:3]} ... {classes[-3:]}\n')

# Create split directories
for split in ['train', 'val']:
    for cls in classes:
        (DST / split / cls).mkdir(parents=True, exist_ok=True)

total_train = 0
total_val = 0
for cls in classes:
    files = sorted((SRC / cls).glob('*.png'))
    random.shuffle(files)
    n_train = int(len(files) * TRAIN_RATIO)
    train_files = files[:n_train]
    val_files = files[n_train:]

    for f in train_files:
        shutil.copy2(f, DST / 'train' / cls / f.name)
    for f in val_files:
        shutil.copy2(f, DST / 'val' / cls / f.name)

    total_train += len(train_files)
    total_val += len(val_files)
    print(f'  {cls:30s}: train={len(train_files):5d}  val={len(val_files):5d}')

print(f'\nTOTAL: train={total_train}  val={total_val}  classes={len(classes)}')
print(f'Output: {DST}')
