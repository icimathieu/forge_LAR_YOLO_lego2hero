#!/usr/bin/env python3
"""Forge multiple mosaic samples by varying random seeds (and optionally inputs).

For 50 mosaics with a single input image, this produces 50 distinct
fragmentations of the same target. To grow real diversity, feed different
canvas_mosaic.png files via --inputs.
"""

import argparse
from pathlib import Path

from forge_dataset import forge_one
from visualize import visualize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', default=['exp1/canvas_mosaic.png'],
                   help='One or more canvas_mosaic.png paths')
    p.add_argument('--out', default='dataset',
                   help='Root output directory (one subfolder per sample)')
    p.add_argument('--n-samples', type=int, default=5,
                   help='Total number of mosaic samples to generate')
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--n-sides-min', type=int, default=16)
    p.add_argument('--n-sides-max', type=int, default=24)
    p.add_argument('--n-frag-min', type=int, default=10)
    p.add_argument('--n-frag-max', type=int, default=15)
    p.add_argument('--canvas-w', type=int, default=3500)
    p.add_argument('--canvas-h', type=int, default=3500)
    p.add_argument('--debug', action='store_true',
                   help='écrit les fichiers DEBUG (pieces.json + source_yolo_viz.png) ; OFF par défaut')
    p.add_argument('--name-from-input', action='store_true',
                   help="Nomme le dossier mosaic_<id> d'après le PNG d'entrée "
                        "(canvas_mosaic_<id>.png) au lieu de mosaic_000 séquentiel "
                        "→ noms uniques (uuid) préservés en sortie")
    p.add_argument('--no-rotation', action='store_true',
                   help='Curriculum L1 : fragments éclatés SANS rotation (translation seule)')
    p.add_argument('--placement', choices=['scatter', 'explode'], default='scatter',
                   help='explode = curriculum L0 : vue éclatée (positions relatives gardées)')
    p.add_argument('--frag-distribution', choices=['balanced', 'compact'], default='balanced',
                   help='balanced (défaut) ou compact (~rectangulaire, moins de sommets)')
    p.add_argument('--erode-px-min', type=int, default=0)
    p.add_argument('--erode-px-max', type=int, default=0)
    p.add_argument('--holes-min', type=int, default=0)
    p.add_argument('--holes-max', type=int, default=0)
    p.add_argument('--missing-min', type=int, default=0)
    p.add_argument('--missing-max', type=int, default=0)
    args = p.parse_args()
    degrade = {
        'erode_px': (args.erode_px_min, args.erode_px_max),
        'holes': (args.holes_min, args.holes_max),
        'missing': (args.missing_min, args.missing_max),
    }

    out_root = Path(args.out)
    for i in range(args.n_samples):
        inp = args.inputs[i % len(args.inputs)]
        seed = args.seed_start + i
        if args.name_from_input:
            stem = Path(inp).stem
            uid = stem[len('canvas_mosaic_'):] if stem.startswith('canvas_mosaic_') else stem
            sample_dir = out_root / f'mosaic_{uid}'
        else:
            sample_dir = out_root / f'mosaic_{i:03d}'
        print(f"\n=== sample {i:03d} (seed={seed}, input={inp}) ===")
        forge_one(
            target_path=inp,
            out_dir=sample_dir,
            n_sides_range=(args.n_sides_min, args.n_sides_max),
            n_frag_range=(args.n_frag_min, args.n_frag_max),
            canvas_size=(args.canvas_w, args.canvas_h),
            stud_size=None,
            seed=seed,
            degrade=degrade,
            rotate=not args.no_rotation,
            placement=args.placement,
            frag_distribution=args.frag_distribution,
            debug=args.debug,
        )
        if args.debug:
            visualize(sample_dir)


if __name__ == '__main__':
    main()
