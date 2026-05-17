"""Training script for YOLO26n-cls classification model on Aksara Jawa.

This script uses paths relative to the project root (parent of `scripts/`).
By default it expects:
    <ROOT>/Dataset/                  # train/ and val/ subfolders with class folders
    <ROOT>/weights/yolo26n-cls.pt    # pretrained weights

Override these with CLI flags --data and --weights.

Run:
    conda activate yolo-aksara
    python scripts/02_train.py
    python scripts/02_train.py --data path/to/Dataset --seed 42 --epochs 100
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
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=64, help='Image size (square)')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='0', help='CUDA device id or "cpu"')
    parser.add_argument('--name', type=str, default='aksara_yolo26n', help='Run name')
    parser.add_argument('--project', type=str, default=str(ROOT / 'runs'), help='Output project directory')
    parser.add_argument('--snapshot-dir', type=str, default=str(ROOT / 'kernel_snapshots'),
                        help='Directory for kernel weight snapshots (per epoch)')
    parser.add_argument('--workers', type=int, default=4, help='Data loader workers')
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
        patience=20,
        plots=True,
        seed=args.seed,
        workers=args.workers,
    )
    print(f'\nTraining selesai. Best weights: {results.save_dir}')


if __name__ == '__main__':
    main()
