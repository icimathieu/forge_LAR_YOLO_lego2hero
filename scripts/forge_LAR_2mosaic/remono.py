#!/usr/bin/env python3
"""Re-forge une mosaïque existante en mode MONO (tuiles 1×1) — pour `ultracompact`.

Entrée = le `piece_grid_<uuid>.json` d'une mosaïque déjà forgée (n'importe quel
mode) : il porte la couleur EXACTE de chaque cellule → on reconstruit la même
image de mosaïque, mais où chaque cellule est une pièce 1×1 (joints partout).
Aucune re-quantification (repasser le canvas rendu dans la forge moyennerait les
joints gris dans les cellules) ; l'uuid est conservé → l'appariement des
curriculums (même mosaïque de base) reste valable entre variable-tile et mono.

Usage :
    python3 scripts/forge_LAR_2mosaic/remono.py \
        --piece-grids 'output/wikiart_inputs/piece_grid_*.json' \
        --out-dir output/wikiart_inputs_mono [--limit 100]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from mosaic import render_mosaic, MONO_DIMS, JOINT_COLOR  # noqa: E402


def mono_from_piece_grid(pg):
    """piece_grid dict (mode quelconque) -> (PIL.Image mono, piece_grid mono)."""
    W, H = pg["grid_width"], pg["grid_height"]
    stud, joint = pg["stud_size_px"], pg["joint_width_px"]
    # 1) grille de couleurs exacte depuis les pièces (couverture 1× garantie)
    grid = np.zeros((H, W, 3), dtype=np.uint8)
    for p in pg["pieces"]:
        c, r, w, h = p["col"], p["row"], p["width"], p["height"]
        grid[r:r + h, c:c + w] = p["color_rgb"]
    # 2) palette locale = couleurs distinctes de CETTE mosaïque (rendu exact)
    colors = np.unique(grid.reshape(-1, 3), axis=0)
    lut = {tuple(c): i for i, c in enumerate(colors)}
    unit_pid = MONO_DIMS[(1, 1)]
    pieces = [(c, r, 1, 1, lut[tuple(grid[r, c])], unit_pid)
              for r in range(H) for c in range(W)]
    img = render_mosaic(pieces, W, H, stud, joint, palette=colors)
    piece_grid = {
        **{k: pg[k] for k in ("grid_width", "grid_height", "stud_size_px",
                              "joint_width_px", "source_image")},
        "joint_color_rgb": list(JOINT_COLOR),
        "mode": "mono", "max_piece": [1, 1],
        "remono_from_mode": pg.get("mode"),
        "n_pieces": len(pieces), "n_cells": W * H,
        "pieces": [{"col": c, "row": r, "width": 1, "height": 1, "part_id": pid,
                    "color_rgb": [int(v) for v in colors[ci]]}
                   for c, r, _w, _h, ci, pid in pieces],
    }
    return img, piece_grid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--piece-grids", required=True,
                    help="glob des piece_grid_*.json source")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="ne traiter que les N premiers (ordre trié)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(a.piece_grids))[: a.limit]
    for i, jp in enumerate(paths, 1):
        pg = json.load(open(jp))
        uid = Path(jp).stem[len("piece_grid_"):]
        png = out / f"canvas_mosaic_{uid}.png"
        if png.exists():                                   # resume
            continue
        img, mono = mono_from_piece_grid(pg)
        img.save(png)
        json.dump(mono, open(out / f"piece_grid_{uid}.json", "w"))
        if i % 20 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] {uid}")
    print(f"done → {out}")


if __name__ == "__main__":
    main()
