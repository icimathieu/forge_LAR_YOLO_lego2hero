#!/usr/bin/env python3
"""Stream wikiart (HuggingFace) → mosaïques LEGO, SANS télécharger le dataset.

Tire N images de `huggan/wikiart` à la volée et les passe dans la forge locale →
`dataset_inputs/canvas_mosaic_XXX.png` + `piece_grid_XXX.json` (prêts pour
`mosaic2fragments`). Aucun téléchargement complet (~1-2 Mo/image en flux).

Exemple :
    python3 forge_LAR_2mosaic/wikiart.py --n 200 --out-dir dataset_inputs --grid 96 --mode plate
    python3 mosaic2fragments/batch.py --inputs dataset_inputs/canvas_mosaic_*.png --out dataset_wikiart

Dépendance : `pip install datasets` (HuggingFace). Réseau requis.
"""
import argparse
import json
import uuid
from itertools import islice
from pathlib import Path

from mosaic import forge_mosaic


def parse_args():
    p = argparse.ArgumentParser(description="wikiart (streaming) -> mosaïques LEGO + GT.")
    p.add_argument("--n", type=int, default=200, help="nb de mosaïques à générer")
    p.add_argument("--skip", type=int, default=0, help="sauter les K premières images du flux")
    p.add_argument("--out-dir", default="dataset_inputs", help="dossier de sortie")
    p.add_argument("--grid", type=int, default=96, help="studs par côté")
    p.add_argument("--stud-size", type=int, default=40, help="px par stud (≥40)")
    p.add_argument("--joint-px", type=int, default=None, help="largeur du joint (défaut ~3)")
    p.add_argument("--mode", choices=["tile", "plate", "brick", "mono"], default="tile",
                   help="mono = tuiles 1×1 seules (pour le curriculum ultracompact)")
    p.add_argument("--big-plates", action="store_true", help="réactive 4×8/4×10")
    p.add_argument("--max-dominant-frac", type=float, default=0.55,
                   help="garde-fou : rejeter une image si une seule couleur LEGO couvre "
                        "> cette fraction des cellules (impressionnistes → aplats monochromes). "
                        "1.0 ou plus = désactivé")
    p.add_argument("--dataset", default="huggan/wikiart",
                   help="dataset HF en streaming (image attendue dans ex['image'])")
    return p.parse_args()


def main():
    a = parse_args()
    from datasets import load_dataset            # import tardif : message clair si absent
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"streaming {a.dataset} (train) → {a.n} mosaïques "
          f"(mode={a.mode}, grid={a.grid}, skip={a.skip})…")
    ds = load_dataset(a.dataset, split="train", streaming=True)

    made = 0
    rejected = 0                                   # garde-fou couleur dominante
    for ex in islice(ds, a.skip, None):
        if made >= a.n:
            break
        try:
            img = ex["image"].convert("RGB")
        except Exception as e:                   # image illisible / champ manquant
            print(f"  skip (image illisible): {e}")
            continue
        if min(img.size) < a.grid:               # trop petite pour la grille demandée
            continue
        uid = str(uuid.uuid4())                   # nom unique, uuid4 canonique (wikiart n'a pas d'ID)
        try:
            mos, grid = forge_mosaic(
                img, grid_w=a.grid, grid_h=a.grid, stud_size=a.stud_size,
                joint_px=a.joint_px, mode=a.mode, big_plates=a.big_plates,
                source_name=uid, max_dominant_frac=a.max_dominant_frac)
        except Exception as e:                   # forge échoue → on saute l'image
            print(f"  skip (forge: {e})")
            continue
        if mos is None:                          # garde-fou : aplat monochrome → rejeté
            rejected += 1
            continue
        mos.save(out / f"canvas_mosaic_{uid}.png")
        (out / f"piece_grid_{uid}.json").write_text(json.dumps(grid))
        made += 1
        if made % 20 == 0:
            print(f"  {made}/{a.n}… (rejetées par garde-fou : {rejected})")
    print(f"Fait : {made} mosaïques dans {out}/  (rejetées par garde-fou couleur : {rejected})")
    print(f"Ensuite : python3 scripts/mosaic2fragments/batch.py "
          f"--inputs {out}/canvas_mosaic_*.png --out dataset_wikiart")


if __name__ == "__main__":
    main()
