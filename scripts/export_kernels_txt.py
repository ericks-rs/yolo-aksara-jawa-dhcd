"""Export bobot kernel konvolusi tiap epoch ke file .txt readable.

Usage:
    python scripts/export_kernels_txt.py                              # default: kernel_snapshots/
    python scripts/export_kernels_txt.py --snap-dir kernel_snapshots_full
    python scripts/export_kernels_txt.py --snap-dir kernel_snapshots_full --layer model.10.conv.conv
"""
import argparse
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


# Mapping layer -> (section, sub-role)
# Berdasarkan arsitektur YOLO26-cls (lihat kernel_snapshots/architecture.txt)
ROLE_MAP = {
    'model.0':  ('BACKBONE', 'Stem-1     (RGB→16ch, downsample 2x)'),
    'model.1':  ('BACKBONE', 'Stem-2     (16→32ch, downsample 2x)'),
    'model.2':  ('BACKBONE', 'Stage-1    (C3k2 block, 32→64ch)'),
    'model.3':  ('BACKBONE', 'Down-1     (64→64ch, downsample 2x)'),
    'model.4':  ('BACKBONE', 'Stage-2    (C3k2 block, 64→128ch)'),
    'model.5':  ('BACKBONE', 'Down-2     (128→128ch, downsample 2x)'),
    'model.6':  ('BACKBONE', 'Stage-3    (C3k2 deep, 128ch)'),
    'model.7':  ('BACKBONE', 'Down-3     (128→256ch, downsample 2x)'),
    'model.8':  ('BACKBONE', 'Stage-4    (C3k2 deep, 256ch)'),
    'model.9':  ('BACKBONE', 'Attention  (C2PSA block, 256ch)'),
    'model.10': ('HEAD',     'Classifier (1x1 conv → pool → FC, output 20 kelas)'),
}


def role_of(layer_name: str):
    # extract top-level idx, e.g. "model.10.conv.conv" -> "model.10"
    parts = layer_name.split('.')
    if len(parts) >= 2 and parts[0] == 'model':
        top = f'{parts[0]}.{parts[1]}'
        return ROLE_MAP.get(top, ('?', '?'))
    return ('?', '?')


def fmt_matrix(mat, prec=6, width=12):
    return '\n'.join('  ' + '  '.join(f'{v:>{width}.{prec}f}' for v in row) for row in mat)


def export_one(pt_path: Path, txt_path: Path, only_layers=None) -> None:
    snap = torch.load(pt_path, map_location='cpu', weights_only=True)
    lines = []
    epoch_tag = pt_path.stem.replace('kernels_', '')
    lines += ['=' * 78, f'KERNEL SNAPSHOT — {epoch_tag}',
              f'Source: {pt_path.name}', '=' * 78, '']

    # Index of contents (per section)
    lines += ['CONTENTS:']
    last_sec = None
    for ln_name in snap.keys():
        if only_layers and ln_name not in only_layers:
            continue
        sec, role = role_of(ln_name)
        if sec != last_sec:
            lines.append(f'  [{sec}]')
            last_sec = sec
        lines.append(f'    - {ln_name:<35s}  {role}')
    lines += ['', '']

    for layer_name, w in snap.items():
        if only_layers and layer_name not in only_layers:
            continue
        section, role = role_of(layer_name)
        w_np = w.numpy()
        bar = '#' * 78
        section_bar = ('=' * 78) if section == 'HEAD' else ('-' * 78)

        if w_np.ndim == 4:
            # Conv2d: (out, in, kH, kW)
            out_c, in_c, kh, kw = w_np.shape
            kind = 'CONV2D'
            shape_desc = (f'({out_c}, {in_c}, {kh}, {kw}) — '
                          f'{out_c} output filter, {in_c} input channel, '
                          f'kernel {kh}x{kw}')
        elif w_np.ndim == 2:
            # Linear / FC: (out, in)
            out_c, in_c = w_np.shape
            kind = 'LINEAR (FC)'
            shape_desc = (f'({out_c}, {in_c}) — fully-connected, '
                          f'{out_c} output neuron × {in_c} input neuron')
        else:
            kind = f'?-D ({w_np.ndim}D)'
            shape_desc = str(w_np.shape)

        lines += ['',
                  section_bar,
                  f'>>> SECTION: [{section}]   ROLE: {role}',
                  section_bar,
                  bar,
                  f'# LAYER: {layer_name}  [{kind}]',
                  f'# Shape: {shape_desc}',
                  f'# Stats: min={w_np.min():.6f}  max={w_np.max():.6f}  '
                  f'mean={w_np.mean():.6f}  std={w_np.std():.6f}',
                  bar, '']

        if w_np.ndim == 4:
            for o in range(out_c):
                lines.append(f'-- filter[{o}] (out channel {o}) --')
                for i in range(in_c):
                    lines.append(f'  in_channel[{i}]:')
                    lines.append(fmt_matrix(w_np[o, i]))
                    lines.append('')
                lines.append('')
        elif w_np.ndim == 2:
            # Linear: print full weight matrix
            lines.append(f'-- FC weight matrix (out × in) = ({out_c} × {in_c}) --')
            lines.append('-- baris[o] = bobot output neuron ke-o terhadap semua input --')
            lines.append('')
            for o in range(out_c):
                row = w_np[o]
                # 1 output neuron = 1 baris (full vector, semua input bobot di-line yang sama)
                lines.append(f'output_neuron[{o}]: ' +
                             '  '.join(f'{v:>11.6f}' for v in row))
                lines.append('')
        lines.append('')

    txt_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snap-dir', default='kernel_snapshots',
                    help='Folder snapshot relatif ke project root')
    ap.add_argument('--layer', action='append', default=None,
                    help='Filter layer tertentu (boleh dipakai berkali-kali)')
    args = ap.parse_args()

    snap_dir = ROOT / args.snap_dir
    out_dir = snap_dir / 'txt'
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(snap_dir.glob('kernels_epoch_*.pt'))
    if not files:
        print(f'[WARN] Tidak ada file kernels_epoch_*.pt di {snap_dir}')
        return
    print(f'Mengekspor {len(files)} snapshot dari {snap_dir}')
    if args.layer:
        print(f'  filter layer: {args.layer}')
    for pt in files:
        txt = out_dir / (pt.stem + '.txt')
        export_one(pt, txt, only_layers=args.layer)
        print(f'  OK  {pt.name}  ->  {txt.name}  ({txt.stat().st_size/1024:.1f} KB)')
    print(f'\nSelesai. Output di {out_dir}')


if __name__ == '__main__':
    main()
