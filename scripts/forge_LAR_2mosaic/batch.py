#!/usr/bin/env python3
"""Forge locale LAR — dossier d'images -> dataset_inputs/canvas_mosaic_XXX.png
+ piece_grid_XXX.json (numérotés, prêts pour `mosaic2fragments/batch.py`).

Exemple :
    python3 forge_LAR_2mosaic/batch.py --input-dir mes_images --out-dir dataset_inputs \
        --grid 96 --stud-size 40

Puis la pipeline aval consomme tout :
    python3 mosaic2fragments/batch.py --inputs dataset_inputs/canvas_mosaic_*.png --out dataset
"""
import argparse
from pathlib import Path

from mosaic import forge_to_files

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def parse_args():
    p = argparse.ArgumentParser(description="Dossier d'images -> mosaïques LEGO + GT.")
    p.add_argument("--input-dir", required=True, help="dossier d'images sources")
    p.add_argument("--out-dir", default="dataset_inputs", help="dossier de sortie")
    p.add_argument("--grid", type=int, default=96, help="studs par côté")
    p.add_argument("--stud-size", type=int, default=40, help="px par stud (≥40)")
    p.add_argument("--joint-px", type=int, default=None, help="largeur du joint (défaut ~3)")
    p.add_argument("--mode", choices=["tile", "plate", "brick", "mono"], default="tile",
                   help="jeu de pièces LEGO (tile par défaut, max 2×6 ; plate = jusqu'à 4×4/4×10)")
    p.add_argument("--big-plates", action="store_true",
                   help="réactive 4×8/4×10 (décochées par défaut sur le site)")
    p.add_argument("--start-index", type=int, default=0, help="indice de départ du nom")
    p.add_argument("--max-dominant-frac", type=float, default=None,
                   help="garde-fou : rejeter une image si une couleur LEGO couvre > cette "
                        "fraction des cellules (aplats monochromes). None = désactivé")
    return p.parse_args()


def main():
    a = parse_args()
    imgs = sorted(p for p in Path(a.input_dir).iterdir()
                  if p.suffix.lower() in EXTS)
    if not imgs:
        raise SystemExit(f"Aucune image dans {a.input_dir} (extensions {sorted(EXTS)})")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(imgs)} image(s) -> {out}/")
    made = rejected = 0
    for k, src in enumerate(imgs, start=a.start_index):
        png = out / f"canvas_mosaic_{k:03d}.png"
        js = out / f"piece_grid_{k:03d}.json"
        g = forge_to_files(str(src), str(png), str(js),
                           grid_w=a.grid, grid_h=a.grid,
                           stud_size=a.stud_size, joint_px=a.joint_px,
                           mode=a.mode, big_plates=a.big_plates,
                           max_dominant_frac=a.max_dominant_frac)
        if g.get("rejected"):                     # garde-fou : aplat monochrome
            rejected += 1
            print(f"  [{k:03d}] {src.name:30s} -> REJETÉ (couleur dominante {g['dominant_frac']:.0%})")
            continue
        made += 1
        print(f"  [{k:03d}] {src.name:30s} -> {png.name}  ({g['n_pieces']} pièces)")
    print(f"Fait : {made} mosaïques (rejetées par garde-fou : {rejected}).")
    print(f"Ensuite : python3 scripts/mosaic2fragments/batch.py --inputs {out}/canvas_mosaic_*.png --out dataset")


if __name__ == "__main__":
    main()
