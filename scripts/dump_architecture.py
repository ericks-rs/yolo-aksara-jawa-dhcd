"""Dump arsitektur lengkap YOLO26n-cls dengan tanda Backbone / Head + role tiap layer."""
from pathlib import Path
import torch.nn as nn
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'weights' / 'yolo26n-cls.pt'
OUT = ROOT / 'kernel_snapshots' / 'architecture.txt'
OUT.parent.mkdir(parents=True, exist_ok=True)

model = YOLO(str(WEIGHTS)).model

# Section mapping (model.N -> role)
# YOLO26-cls = backbone + head (classification, ga ada neck FPN/PAN)
ROLE = {
    '0':  ('BACKBONE', 'Stem-1     ', 'Conv 3x3 s=2 — RGB→16ch (downsample 2x)'),
    '1':  ('BACKBONE', 'Stem-2     ', 'Conv 3x3 s=2 — 16→32ch (downsample 2x)'),
    '2':  ('BACKBONE', 'Stage-1    ', 'C3k2 block — proses fitur 32→64ch'),
    '3':  ('BACKBONE', 'Down-1     ', 'Conv 3x3 s=2 — 64→64ch (downsample)'),
    '4':  ('BACKBONE', 'Stage-2    ', 'C3k2 block — proses fitur 64→128ch'),
    '5':  ('BACKBONE', 'Down-2     ', 'Conv 3x3 s=2 — 128→128ch (downsample)'),
    '6':  ('BACKBONE', 'Stage-3    ', 'C3k2 block (deep, 2 bottleneck) — 128ch'),
    '7':  ('BACKBONE', 'Down-3     ', 'Conv 3x3 s=2 — 128→256ch (downsample)'),
    '8':  ('BACKBONE', 'Stage-4    ', 'C3k2 block (deep, 2 bottleneck) — 256ch'),
    '9':  ('BACKBONE', 'Attention  ', 'C2PSA — Position-Sensitive Spatial Attention'),
    '10': ('HEAD',     'Classifier ', 'Classify — pool + conv 1x1 + FC → 20 kelas'),
}

lines = []
hdr = lambda c: '=' * 92 if c == '=' else '-' * 92
lines.append(hdr('='))
lines.append('YOLO26n-cls — ARSITEKTUR LENGKAP (Backbone / Head)')
lines.append(hdr('='))
lines.append('')
lines.append('Catatan: YOLO classification = BACKBONE + HEAD saja.')
lines.append('Tidak ada NECK (FPN/PAN) — neck hanya dipakai di YOLO *detection* untuk fuse')
lines.append('multi-scale feature. Di classification, output backbone langsung dipool & diklasifikasi.')
lines.append('')

# 1. Top-level modules dengan section
lines.append('## TOP-LEVEL MODULES + ROLE')
lines.append(hdr('-'))
lines.append(f'{"section":<12}{"sub-role":<14}{"idx":<5}{"name":<10}{"type":<10}{"#params":>11}  description')
lines.append(hdr('-'))

last_section = None
section_params = {}
backbone_idx = []
for sub_name, sub_mod in model.model.named_children():
    n_params = sum(p.numel() for p in sub_mod.parameters())
    section, role, desc = ROLE.get(sub_name, ('?', '?', '?'))
    section_params[section] = section_params.get(section, 0) + n_params
    if section == 'BACKBONE':
        backbone_idx.append(sub_name)
    if section != last_section:
        if last_section is not None:
            lines.append('')
        lines.append(f'>>> [{section}] <<<')
        last_section = section
    lines.append(f'{section:<12}{role:<14}{sub_name:<5}'
                 f'{"model."+sub_name:<10}{type(sub_mod).__name__:<10}'
                 f'{n_params:>11,}  {desc}')

lines.append('')
lines.append(hdr('-'))
lines.append('SUMMARY PER SECTION:')
total = sum(section_params.values())
for sec, p in section_params.items():
    pct = 100 * p / total
    lines.append(f'  {sec:<12} {p:>12,} params  ({pct:5.1f}%)')
lines.append(f'  {"TOTAL":<12} {total:>12,} params')
lines.append('')

# 2. Semua Conv2d, dikelompokkan per section
lines.append('## SEMUA 39 CONV2D — DIKELOMPOKKAN BACKBONE / HEAD')
lines.append(hdr('-'))
lines.append(f'{"#":<4}{"section":<11}{"layer":<43}{"shape":<24}{"#params":>10}')
lines.append(hdr('-'))

count = 0
total_conv = 0
last_top = None
for full_name, m in model.named_modules():
    if not isinstance(m, nn.Conv2d):
        continue
    parts = full_name.split('.')
    # full_name e.g. "model.6.m.0.cv1.conv" -> top idx = "6"
    top_idx = parts[1] if parts[0] == 'model' else '?'
    section = ROLE.get(top_idx, ('?',))[0]
    shape = tuple(m.weight.shape)
    n = m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
    count += 1
    total_conv += n
    if top_idx != last_top:
        lines.append(f'  --- model.{top_idx}  [{section}] ---')
        last_top = top_idx
    lines.append(f'{count:<4}{section:<11}{full_name:<43}{str(shape):<24}{n:>10,}')

lines.append(hdr('-'))
lines.append(f'TOTAL: {count} Conv2d, {total_conv:,} params (cuma conv weights, belum termasuk BN/bias luar conv)')
lines.append('')

# 3. Spatial flow (assuming input 128x128)
lines.append('## SPATIAL FLOW (input 128x128)')
lines.append(hdr('-'))
flow = [
    ('input',     '3 x 128 x 128', '—'),
    ('model.0',   '16 x 64 x 64',  'stride 2'),
    ('model.1',   '32 x 32 x 32',  'stride 2'),
    ('model.2',   '64 x 32 x 32',  'C3k2 (no downsample)'),
    ('model.3',   '64 x 16 x 16',  'stride 2'),
    ('model.4',   '128 x 16 x 16', 'C3k2'),
    ('model.5',   '128 x 8 x 8',   'stride 2'),
    ('model.6',   '128 x 8 x 8',   'C3k2 deep'),
    ('model.7',   '256 x 4 x 4',   'stride 2'),
    ('model.8',   '256 x 4 x 4',   'C3k2 deep'),
    ('model.9',   '256 x 4 x 4',   'C2PSA attention'),
    ('model.10',  '20 logits',     'Classify head: GAP + 1x1 conv + FC'),
]
lines.append(f'{"layer":<12}{"output shape":<20}{"keterangan":<30}')
lines.append(hdr('-'))
for name, shape, desc in flow:
    lines.append(f'{name:<12}{shape:<20}{desc:<30}')
lines.append('')

# 4. Full architecture (PyTorch print)
lines.append('## FULL MODULE TREE (torch repr)')
lines.append(hdr('-'))
lines.append(str(model))

OUT.write_text('\n'.join(lines), encoding='utf-8')
print(f'Saved -> {OUT}  ({OUT.stat().st_size/1024:.1f} KB)')
print(f'Total Conv2d: {count},  total conv params: {total_conv:,}')
