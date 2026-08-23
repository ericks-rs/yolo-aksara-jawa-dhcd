"""Measure the data leakage the paper reports, on both datasets.

Section 3.2 of the paper states two quantities that a reader should be able
to check without training anything:

  Aksara Jawa, distributed split
      120 of 480 validation images (25 percent) are augmented variants of
      an image that also appears in training, 6 in each of the 20 classes,
      arising from 20 of the 399 base identifiers being split across both
      partitions. The corrected split brings this to zero.

  DHCD
      8 of 18,400 validation images (0.0435 percent) are pixel-identical
      copies of a training image, all inside the class character_15_adna and
      all inherited from the source collection rather than introduced by our
      split. They are left in place. The rate is roughly 575 times lower
      than the one corrected in Aksara Jawa, but very small accuracy
      differences on DHCD should still be interpreted with caution.

The Aksara Jawa check groups files by the base identifier encoded in the
filename. The DHCD check compares MD5 hashes of the image bytes, because
DHCD filenames carry no variant structure to group by.

Expects the datasets in place as described in data/README.md:
    <ROOT>/Dataset/            the split as distributed on Kaggle
    <ROOT>/DatasetPhiard/      output of clean_split_aksara.py
    <ROOT>/DatasetDHCD_split/  output of prep_dhcd.py

Run:
    python scripts/audit_dataset_leakage.py
"""
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def base_id(name):
    """'ha0.meki.r10.png' -> 'ha0'."""
    return Path(name).stem.split(".")[0]


def audit_aksara(root, label):
    if not (root / "train").exists():
        print(f"  {label}: not found at {root}, skipped")
        return
    total_val = leaked_val = 0
    all_bases = shared = 0
    per_class = []
    for cls_dir in sorted((root / "train").iterdir()):
        if not cls_dir.is_dir():
            continue
        train_bases = {base_id(f.name) for f in cls_dir.glob("*.png")}
        val_dir = root / "val" / cls_dir.name
        val_files = list(val_dir.glob("*.png")) if val_dir.exists() else []
        val_bases = {base_id(f.name) for f in val_files}
        leaked = [f for f in val_files if base_id(f.name) in train_bases]
        total_val += len(val_files)
        leaked_val += len(leaked)
        all_bases += len(train_bases | val_bases)
        shared += len(train_bases & val_bases)
        per_class.append((cls_dir.name, len(val_files), len(leaked)))

    pct = leaked_val / total_val * 100 if total_val else 0
    print(f"\n  {label}")
    print(f"    validation images        : {total_val}")
    print(f"    base identifiers, total  : {all_bases}")
    print(f"    base identifiers, shared : {shared}")
    print(f"    leaked validation images : {leaked_val}  ({pct:.2f}%)")
    counts = {n for _, _, n in per_class}
    if len(counts) == 1:
        print(f"    per class                : {counts.pop()}, uniform across "
              f"{len(per_class)} classes")
    else:
        print(f"    per class                : varies, {sorted(counts)}")


def audit_dhcd(root):
    if not (root / "train").exists():
        print(f"\n  DHCD: not found at {root}, skipped")
        return
    digests = defaultdict(list)
    n = 0
    for split in ("train", "val"):
        for cls_dir in sorted((root / split).iterdir()):
            if not cls_dir.is_dir():
                continue
            for f in cls_dir.glob("*.png"):
                n += 1
                digests[hashlib.md5(f.read_bytes()).hexdigest()].append(
                    (split, cls_dir.name, f.name))
                if n % 20000 == 0:
                    print(f"    hashed {n} images...", flush=True)

    n_val = sum(1 for v in digests.values() for s, _, _ in v if s == "val")
    affected, classes = set(), defaultdict(int)
    for group in digests.values():
        if len({s for s, _, _ in group}) > 1:
            for split, cls, fn in group:
                if split == "val":
                    affected.add((cls, fn))
                    classes[cls] += 1
    pct = len(affected) / n_val * 100 if n_val else 0
    print(f"\n  DHCD")
    print(f"    images hashed            : {n}")
    print(f"    distinct image hashes    : {len(digests)}")
    print(f"    validation images        : {n_val}")
    print(f"    duplicated across splits : {len(affected)}  ({pct:.4f}%)")
    if classes:
        print(f"    classes affected         : {dict(classes)}")


print(__doc__.split("Expects")[0].strip())
print("\n" + "=" * 74)
print("MEASURED")
print("=" * 74)
audit_aksara(ROOT / "Dataset", "Aksara Jawa, split as distributed")
audit_aksara(ROOT / "DatasetPhiard", "Aksara Jawa, corrected split")
audit_dhcd(ROOT / "DatasetDHCD_split")
print()
