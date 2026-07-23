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


def visualize_recob(dataset_dir):
    """Viz des DEUX pertes d'aire, en repère target (depuis `gt_layout.json`).

    Par fragment, trois couches :
      couleur pâle  = `polygon_raw` (empreinte PARFAITE)
      hachures GRISES fines       = perte de DÉGRADATION (raw − raw_degraded :
                                    érosion, trous — voulue, c'est l'input L3/L4)
      hachures COULEUR marquées   = perte d'ENCODAGE reco-B (raw_degraded −
                                    polygon_n : budget --max-area-loss, ~ε)
      trait noir    = `polygon_n` (ce que voit le GNN)
    NB : `source.png` n'est PAS affecté par la perte d'encodage (vrais pixels
    découpés par le masque dégradé) — elle n'existe que dans polygon_n.
    """
    import cv2

    d = Path(dataset_dir)
    gl = json.loads((d / 'gt_layout.json').read_text())
    W, H = gl['target_size']
    img = np.full((H, W, 3), 255, dtype=np.uint8)

    hatch = np.zeros((H, W), dtype=np.uint8)          # trame diagonale //
    for c in range(-H, W, 14):
        cv2.line(hatch, (c, 0), (c + H, H), 1, 2)
    hatch2 = np.zeros((H, W), dtype=np.uint8)         # trame croisée \\ serrée
    for c in range(-H, W, 8):
        cv2.line(hatch2, (c + H, 0), (c, H), 1, 1)

    deg_total = enc_total = raw_total = 0.0
    for idx, f in enumerate(gl['fragments']):
        raw = np.array(f['polygon_raw'], dtype=np.int32)
        deg = np.array(f['polygon_raw_degraded'], dtype=np.int32)
        enc = np.array(f['polygon_n'], dtype=np.int32)
        color = COLORS[idx % len(COLORS)]
        pale = tuple(int(v + (255 - v) * 0.72) for v in color)
        m = {}
        for k, poly in (('raw', raw), ('deg', deg), ('enc', enc)):
            m[k] = np.zeros((H, W), dtype=np.uint8)
            if len(poly) >= 3:
                cv2.fillPoly(m[k], [poly], 1)
        lost_deg = (m['raw'] > 0) & (m['deg'] == 0)   # dégradation (voulue)
        lost_enc = (m['deg'] > 0) & (m['enc'] == 0)   # encodage reco-B (ε)
        img[m['raw'] > 0] = pale
        img[lost_deg & (hatch2 > 0)] = (150, 150, 150)
        img[lost_enc & (hatch > 0)] = color
        if len(enc) >= 3:
            cv2.polylines(img, [enc], True, (0, 0, 0), 2)
        deg_total += float(lost_deg.sum())
        enc_total += float(lost_enc.sum())
        raw_total += float((m['raw'] > 0).sum())

    out_path = d / 'recob_viz.png'
    Image.fromarray(img).save(out_path)
    print(f"Wrote {out_path}  (perte dégradation [hachures grises] : "
          f"{100 * deg_total / max(raw_total, 1):.2f} % ; perte encodage reco-B "
          f"[hachures couleur] : {100 * enc_total / max(raw_total, 1):.2f} %)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='dataset/mosaic_001')
    p.add_argument('--recob', action='store_true',
                   help="écrit aussi recob_viz.png (perte d'encodage reco-B hachurée)")
    return p.parse_args()


if __name__ == '__main__':
    a = parse_args()
    visualize(a.dir)
    if a.recob:
        visualize_recob(a.dir)
