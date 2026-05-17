"""Callback untuk save bobot kernel konvolusi tiap epoch selama training YOLO26."""
from pathlib import Path
import torch
import torch.nn as nn


class KernelLogger:
    """Save weights of selected Conv2d layers each epoch to .pt files.

    Usage:
        logger = KernelLogger(out_dir='kernel_snapshots',
                              layers=['model.0.conv', 'model.1.conv', 'model.3.conv'])
        model = YOLO('yolo26n-cls.pt')
        logger.attach(model)
        model.train(...)
    """

    def __init__(self, out_dir: str, layers: list[str] | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.layers = layers
        self.epoch_count = 0

    def _snapshot(self, torch_model: nn.Module, tag: str) -> None:
        snap = {}
        for name, mod in torch_model.named_modules():
            if not isinstance(mod, (nn.Conv2d, nn.Linear)):
                continue
            if self.layers is not None and name not in self.layers:
                continue
            snap[name] = mod.weight.detach().cpu().clone()
        path = self.out_dir / f'kernels_{tag}.pt'
        torch.save(snap, path)
        print(f'[KernelLogger] saved {len(snap)} layers (conv+fc) -> {path.name}')

    def attach(self, yolo_model) -> None:
        """Register callbacks on the Ultralytics YOLO model."""

        def on_train_start(trainer):
            self._snapshot(trainer.model, 'epoch_000_init')

        def on_train_epoch_end(trainer):
            self.epoch_count = trainer.epoch + 1
            self._snapshot(trainer.model, f'epoch_{self.epoch_count:03d}')

        def on_train_end(trainer):
            self._snapshot(trainer.model, 'epoch_final')

        yolo_model.add_callback('on_train_start', on_train_start)
        yolo_model.add_callback('on_train_epoch_end', on_train_epoch_end)
        yolo_model.add_callback('on_train_end', on_train_end)
