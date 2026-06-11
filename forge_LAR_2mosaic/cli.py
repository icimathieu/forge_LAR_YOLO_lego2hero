#!/usr/bin/env python3
"""Forge locale LAR — une image -> canvas_mosaic.png + piece_grid.json.

Exemple :
    python3 forge_LAR_2mosaic/cli.py --input photo.jpg --out-png out/canvas_mosaic.png \
        --out-json out/piece_grid.json --grid 96 --stud-size 40
"""
import argparse
from pathlib import Path

from mosaic import forge_to_files


def parse_args():
    p = argparse.ArgumentParser(description="Image -> mosaïque LEGO (tile/plate/brick) + GT.")
    p.add_argument("--input", required=True, help="image source (jpg/png/…)")
    p.add_argument("--out-png", help="chemin du canvas_mosaic.png (défaut : à côté de --input)")
    p.add_argument("--out-json", help="chemin du piece_grid.json")
    p.add_argument("--grid", type=int, default=96, help="studs par côté (48/64/96/100/128)")
    p.add_argument("--stud-size", type=int, default=40, help="px par stud (≥40)")
    p.add_argument("--joint-px", type=int, default=None, help="largeur du joint (défaut ~3)")
    p.add_argument("--mode", choices=["tile", "plate", "brick"], default="tile",
                   help="jeu de pièces LEGO (tile par défaut, max 2×6 ; plate = jusqu'à 4×4/4×10)")
    p.add_argument("--big-plates", action="store_true",
                   help="réactive 4×8/4×10 (décochées par défaut sur le site)")
    return p.parse_args()


def main():
    a = parse_args()
    out_png = a.out_png or str(Path(a.input).with_name("canvas_mosaic.png"))
    out_json = a.out_json or str(Path(out_png).with_name("piece_grid.json"))
    grid = forge_to_files(
        a.input, out_png, out_json,
        grid_w=a.grid, grid_h=a.grid, stud_size=a.stud_size, joint_px=a.joint_px,
        mode=a.mode, big_plates=a.big_plates,
    )
    print(f"{out_png}  ({grid['grid_width']}×{grid['grid_height']} studs, mode={grid['mode']}, "
          f"max {grid['max_piece'][0]}×{grid['max_piece'][1]}, "
          f"{grid['n_pieces']} pièces, joint {grid['joint_width_px']}px)")
    print(f"{out_json}")


if __name__ == "__main__":
    main()
