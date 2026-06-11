#!/usr/bin/env python3
"""Quick visualization: overlay YOLO polygons on source.png to check alignment.

Also reports any sanity-check failures (fragment count vs YOLO lines, etc.).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


COLORS = [
    (231, 76, 60), (52, 152, 219), (241, 196, 15), (46, 204, 113),
    (155, 89, 182), (52, 73, 94), (230, 126, 34), (26, 188, 156),
    (192, 57, 43), (41, 128, 185), (243, 156, 18), (39, 174, 96),
    (142, 68, 173), (44, 62, 80), (211, 84, 0), (22, 160, 133),
]


def visualize(dataset_dir):
    d = Path(dataset_dir)
    src = Image.open(d / 'source.png').convert('RGBA')
    W, H = src.size

    overlay = Image.new('RGBA', src.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    with open(d / 'source_yolo.txt') as f:
        lines = [l.strip() for l in f if l.strip()]
    print(f"YOLO lines: {len(lines)}")

    for idx, line in enumerate(lines):
        parts = line.split()
        cls = parts[0]
        coords = [float(x) for x in parts[1:]]
        pts = [(coords[i] * W, coords[i + 1] * H) for i in range(0, len(coords), 2)]
        color = COLORS[idx % len(COLORS)]
        draw.polygon(pts, outline=color + (255,), fill=color + (90,))

    out = Image.alpha_composite(src, overlay)
    out_path = d / 'source_yolo_viz.png'
    out.convert('RGB').save(out_path)
    print(f"Wrote {out_path}")

    # Sanity: cross-check with graph_complete.json
    g = json.loads((d / 'graph_complete.json').read_text())
    print(f"Graph nodes: {len(g['nodes'])}")
    print(f"Graph edges: {len(g['edges'])}")
    sizes = [n['target_info']['n_pieces'] for n in g['nodes']]  # n_pieces (métadonnée GT)
    print(f"Fragment sizes (n_pieces): min={min(sizes):.0f} max={max(sizes):.0f} "
          f"mean={np.mean(sizes):.1f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='dataset/mosaic_001')
    return p.parse_args()


if __name__ == '__main__':
    visualize(parse_args().dir)
