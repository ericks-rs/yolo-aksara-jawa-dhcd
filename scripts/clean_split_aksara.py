"""Re-split Aksara Jawa dataset by base ID untuk hindari data leak.

Original problem:
  Existing split di Dataset/{train,val} mengandung augmentasi dari image yg sama
  tersebar di train DAN val. Contoh:
    train/ha/ha0.meki.r10.png
    val/ha/ha0.meki.b.png
  Model "ngapal" augmentasi varian, akurasi inflated.

Solution:
  Group by base ID (`ha0`, `ha1`, ...). Split base ID ke train/val.
  SEMUA augmentasi 1 base ID hanya muncul di salah satu split.

Output: DatasetPhiard/{train,val}/{class}/

Run:
    python scripts/clean_split_aksara.py
"""
import shutil
import random
import re
from pathlib import Path
from collections import defaultdict

random.seed(42)  # reproducible

ROOT = Path(__file__).resolve().parents[1]
SRC_TRAIN = ROOT / 'Dataset' / 'train'
SRC_VAL   = ROOT / 'Dataset' / 'val'
DST = ROOT / 'DatasetPhiard'

TRAIN_RATIO = 0.8

if not SRC_TRAIN.exists():
    raise FileNotFoundError(f'Source not found: {SRC_TRAIN}')


def base_id(filename: str) -> str:
    """Extract base ID dari filename, contoh: 'ha0.meki.r10' -> 'ha0'."""
    stem = Path(filename).stem
    return stem.split('.')[0]


def group_by_base(folder: Path) -> dict[str, list[Path]]:
    """Group files in folder by base ID."""
    groups = defaultdict(list)
    for f in folder.glob('*.png'):
        groups[base_id(f.name)].append(f)
    return groups


classes = sorted([p.name for p in SRC_TRAIN.iterdir() if p.is_dir()])
print(f'Found {len(classes)} classes: {classes}\n')

# Create output directories
for split in ['train', 'val']:
    for cls in classes:
        (DST / split / cls).mkdir(parents=True, exist_ok=True)

stats = []
for cls in classes:
    # Combine all files from existing train + val per class (sebelum di-split ulang)
    combined_groups = defaultdict(list)
    for src_folder in [SRC_TRAIN / cls, SRC_VAL / cls]:
        if src_folder.exists():
            for base, files in group_by_base(src_folder).items():
                combined_groups[base].extend(files)

    # Get unique base IDs and split them
    base_ids = sorted(combined_groups.keys())
    random.shuffle(base_ids)
    n_train_bases = max(1, int(len(base_ids) * TRAIN_RATIO))
    train_bases = set(base_ids[:n_train_bases])
    val_bases   = set(base_ids[n_train_bases:])

    train_count = 0
    val_count = 0
    for base in train_bases:
        for f in combined_groups[base]:
            shutil.copy2(f, DST / 'train' / cls / f.name)
            train_count += 1
    for base in val_bases:
        for f in combined_groups[base]:
            shutil.copy2(f, DST / 'val' / cls / f.name)
            val_count += 1

    stats.append({
        'class': cls,
        'unique_bases': len(base_ids),
        'train_bases': len(train_bases),
        'val_bases': len(val_bases),
        'train_files': train_count,
        'val_files': val_count,
    })
    print(f'  {cls:5s}: {len(base_ids):3d} unique IDs -> '
          f'train: {len(train_bases):3d} bases ({train_count:4d} files)  |  '
          f'val: {len(val_bases):3d} bases ({val_count:4d} files)')

print()
total_train = sum(s['train_files'] for s in stats)
total_val   = sum(s['val_files'] for s in stats)
total_bases = sum(s['unique_bases'] for s in stats)
print(f'TOTAL: {len(classes)} classes, {total_bases} unique base IDs')
print(f'       train: {total_train} files  |  val: {total_val} files')
print(f'\nOutput: {DST}')
print(f'\nNo data leak guarantee: SEMUA augmentasi dari 1 base ID hanya muncul di SALAH SATU split.')
