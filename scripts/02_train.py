"""Single-run training helper for a YOLO classification model on Aksara Jawa.

WHAT THIS SCRIPT IS, AND WHAT IT IS NOT
---------------------------------------
This is a convenience wrapper for training one model once. The multi-seed
results reported in the paper were NOT produced by this script. They come
from the notebooks, which are the reproduction path:

    notebooks/04c_compare_clean.ipynb   Table 4, three YOLO generations
    notebooks/05c_ablation_clean.ipynb  Table 6, C2PSA ablation
    notebooks/06_dhcd_validation.ipynb  Tables 5 and 7, DHCD

The defaults below were aligned in August 2026 with the configuration
actually used in those runs, which is recorded in the published
`runs/<experiment>/<run>/args.yaml` files: 50 epochs, early stopping at
patience 5, 128x128 images for Aksara Jawa, and batch 64. DHCD used 32x32
images with batch 128, so pass --imgsz 32 --batch 128 for that dataset.

Optimizer selection is left to the Ultralytics `optimizer='auto'` default,
which is what the reported runs used. The framework chose AdamW with an
initial learning rate of 0.000417 for Aksara Jawa and an SGD-family
optimizer with a learning rate of 0.01 for DHCD. Do not hard-code an
optimizer here if you intend to reproduce the published numbers.

This script uses paths relative to the project root (parent of `scripts/`).
By default it expects:
    <ROOT>/Dataset/                  # train/ and val/ subfolders with class folders
    <ROOT>/weights/yolo26n-cls.pt    # pretrained weights

Override these with CLI flags --data and --weights.

Run:
    conda activate yolo-aksara
    python scripts/02_train.py
    python scripts/02_train.py --data path/to/DatasetPhiard --seed 42
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from utils.kernel_logger import KernelLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO26n-cls on Aksara Jawa')
    parser.add_argument('--data', type=str, default=str(ROOT / 'Dataset'),
                        help='Path to dataset directory (with train/ and val/ subfolders)')
    parser.add_argument('--weights', type=str, default=str(ROOT / 'weights' / 'yolo26n-cls.pt'),
                        help='Path to pretrained weights .pt file or model config .yaml')
    parser.add_argument('--epochs', type=int, default=50, help='Max training epochs (early stopping usually ends sooner)')
    parser.add_argument('--imgsz', type=int, default=128, help='Image size (square). 128 for Aksara Jawa, 32 for DHCD')
    parser.add_argument('--batch', type=int, default=64, help='Batch size. 64 for Aksara Jawa, 128 for DHCD')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='0', help='CUDA device id or "cpu"')
    parser.add_argument('--name', type=str, default='aksara_yolo26n', help='Run name')
    parser.add_argument('--project', type=str, default=str(ROOT / 'runs'), help='Output project directory')
    parser.add_argument('--snapshot-dir', type=str, default=str(ROOT / 'kernel_snapshots'),
                        help='Directory for kernel weight snapshots (per epoch)')
    parser.add_argument('--workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--no-snapshot', action='store_true',
                        help='Disable kernel logger callback (faster, less disk usage)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Layers to monitor (kernel snapshots per epoch)
    target_layers = [
        'model.0.conv',   # 16 filter RGB 3x3 (most visualizable)
        'model.1.conv',   # 32 filter 16ch 3x3
        'model.3.conv',   # 64 filter 3x3 deeper features
    ]

    model = YOLO(args.weights)

    if not args.no_snapshot:
        logger = KernelLogger(out_dir=args.snapshot_dir, layers=target_layers)
        logger.attach(model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        plots=True,
        seed=args.seed,
        workers=args.workers,
    )
    print(f'\nTraining selesai. Best weights: {results.save_dir}')


if __name__ == '__main__':
    main()
