#!/usr/bin/env python3
"""Curriculum de dégradation — génère les 5 paliers L1→L5 dans des sous-dossiers
`<out>/L1_..` … `<out>/L5_..`, depuis UN MÊME jeu de mosaïques de base.

Design **apparié** : même mosaïque + même seed à tous les paliers → la GT
(fragmentation + arêtes, fixée avant tout tirage de dégradation) est IDENTIQUE
d'un palier à l'autre ; seule la difficulté de l'INPUT change. → la courbe
score(difficulté) du GNN se lit à variance faible (cf. CLAUDE.md « curriculum »).

Paliers (ladder monotone) :
  L1 translation seule (sans rotation), aucune dégradation   ← le + facile
  L2 + rotation, aucune dégradation                          (= clean actuel)
  L3 + rotation, dégradation LÉGÈRE
  L4 + rotation, dégradation FORTE                           (= degraded actuel : erode[2,6] holes[1,3] missing[0,2])
  L5 + rotation, dégradation LOURDE                          ← knobs poussés ; ⚠ DAFNE B (distribution de découpe) + D (parasites) PAS encore branchés

Inputs = mosaïques rendues : soit des `canvas_mosaic_*.png`, soit les `target.png`
d'un dataset clean existant (target.png = copie de la mosaïque d'entrée).

Exemples :
    # apparié 1:1 sur les 100 mosaïques wikiart déjà rendues
    python3 scripts/mosaic2fragments/curriculum.py \
        --inputs output/100mosaic4mathias/mosaic_*/target.png --out output

    # augmenter : 5 fragmentations (seeds) par mosaïque → 500/palier à partir de 100 images
    python3 scripts/mosaic2fragments/curriculum.py \
        --inputs output/100mosaic4mathias/mosaic_*/target.png --out output --n-per-input 5

Puis collate par palier :
    for L in output/L?_*; do python3 scripts/features/collate.py --dataset "$L"; done
"""
import argparse
import glob
from pathlib import Path

from forge_dataset import forge_one
from visualize import visualize

# (placement, rotate, érosion px, trous, fragments manquants) par palier.
#   placement explode = vue éclatée (positions relatives gardées, juste espacées)
#   placement scatter = placement aléatoire (peut échanger les positions)
LEVELS = {
    "L0_explode":     dict(placement="explode", rotate=False, erode=(0, 0), holes=(0, 0), missing=(0, 0)),
    "L1_translation": dict(placement="scatter", rotate=False, erode=(0, 0), holes=(0, 0), missing=(0, 0)),
    "L2_rotation":    dict(placement="scatter", rotate=True,  erode=(0, 0), holes=(0, 0), missing=(0, 0)),
    "L3_light":       dict(placement="scatter", rotate=True,  erode=(1, 2), holes=(0, 1), missing=(0, 0)),
    "L4_strong":      dict(placement="scatter", rotate=True,  erode=(2, 6), holes=(1, 3), missing=(0, 2)),  # = degraded actuel
}
# L5 « knobs poussés » RETIRÉ (saturait : 0.092 vs 0.088 pour L4). Le vrai durcissement
# passe par l'axe ORTHOGONAL --frag-distribution {voronoi, clusters} (DAFNE B), pas par
# plus d'érosion. Structure de sortie = output/<frag-distribution>/<palier>/mosaic_*.

FRAG_DISTRIBUTIONS = ["balanced", "compact", "ultracompact", "voronoi", "clusters"]


def input_uid(path):
    """Identifiant stable d'une mosaïque de base (pour apparier les paliers)."""
    p = Path(path)
    if p.stem == "target":                       # .../mosaic_<uuid>/target.png
        name = p.parent.name
        return name[len("mosaic_"):] if name.startswith("mosaic_") else name
    if p.stem.startswith("canvas_mosaic_"):       # canvas_mosaic_<id>.png
        return p.stem[len("canvas_mosaic_"):]
    return p.stem


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", default=[],
                   help="mosaïques de base (canvas_mosaic_*.png ou target.png d'un dataset clean)")
    p.add_argument("--inputs-glob", default=None,
                   help="motif glob résolu en interne (robuste pour des milliers d'entrées, "
                        "évite ARG_MAX) ; ex. 'output/wikiart_inputs/canvas_mosaic_*.png'")
    p.add_argument("--out", default="output", help="racine → output/<frag-distribution>/<palier>/")
    p.add_argument("--frag-distribution", choices=FRAG_DISTRIBUTIONS, default="balanced",
                   help="motif de découpe (axe orthogonal aux paliers). Seul 'balanced' "
                        "est implémenté ; voronoi/clusters (DAFNE B) + compact = à venir")
    p.add_argument("--n-per-input", type=int, default=1,
                   help="nb de fragmentations (seeds) par mosaïque de base (augmentation)")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--n-sides-min", type=int, default=16)
    p.add_argument("--n-sides-max", type=int, default=24)
    p.add_argument("--n-frag-min", type=int, default=10)
    p.add_argument("--n-frag-max", type=int, default=15)
    p.add_argument("--canvas-w", type=int, default=4096)
    p.add_argument("--canvas-h", type=int, default=4096)
    p.add_argument("--levels", nargs="+", default=list(LEVELS),
                   help=f"sous-ensemble de paliers (défaut : tous → {list(LEVELS)})")
    p.add_argument("--jobs", type=int, default=1,
                   help="processus parallèles (1 = série). Forge = CPU-bound, embarrassingly "
                        "parallel. Sur machine partagée : plafonner (ex. 12 sur 24), pas tous les cœurs")
    p.add_argument("--debug", action="store_true",
                   help="écrit les fichiers DEBUG (pieces.json + source_yolo_viz.png), "
                        "lus nulle part par l'entraînement ; OFF par défaut")
    return p.parse_args()


def _run_task(t):
    """Worker (1 instance). Top-level → picklable par multiprocessing (spawn macOS).

    ROBUSTESSE (grosses générations) :
    - **resume** : si l'instance est déjà complète (`degradation.md` = dernier fichier
      écrit), on la saute → un relancement reprend où ça s'est arrêté.
    - **tolérance** : toute exception d'une instance est CAPTURÉE et renvoyée comme
      `FAIL …` au lieu de planter tout le batch (1 mosaïque pourrie ≠ 25 000 perdues).
    """
    if (Path(t["dir"]) / "degradation.md").exists():
        return f"skip  {t['label']}"
    try:
        forge_one(
            target_path=t["inp"], out_dir=t["dir"],
            n_sides_range=t["n_sides"], n_frag_range=t["n_frag"], canvas_size=t["canvas"],
            stud_size=None, seed=t["seed"], degrade=t["degrade"],
            rotate=t["rotate"], placement=t["placement"], frag_distribution=t["frag_distribution"],
            debug=t["debug"],
        )
        if t["viz"]:
            visualize(t["dir"])
        return t["label"]
    except Exception as e:
        return f"FAIL  {t['label']}: {type(e).__name__}: {e}"


def main():
    a = parse_args()
    if a.frag_distribution in ("voronoi", "clusters"):
        raise SystemExit(f"--frag-distribution {a.frag_distribution} : pas encore "
                         "implémenté (DAFNE B à venir ; dispo : balanced, compact)")
    inputs = list(a.inputs) + (sorted(glob.glob(a.inputs_glob)) if a.inputs_glob else [])
    if not inputs:
        raise SystemExit("aucune entrée : fournir --inputs ... ou --inputs-glob 'motif'")
    out_root = Path(a.out) / a.frag_distribution
    tasks = []
    for inp in inputs:
        uid = input_uid(inp)
        for s in range(a.n_per_input):
            seed = a.seed_start + s
            # même seed sur tous les paliers → mosaïque/fragmentation appariée
            suffix = f"_s{seed}" if a.n_per_input > 1 else ""
            for level in a.levels:
                cfg = LEVELS[level]
                tasks.append(dict(
                    inp=inp, dir=str(out_root / level / f"mosaic_{uid}{suffix}"),
                    n_sides=(a.n_sides_min, a.n_sides_max),
                    n_frag=(a.n_frag_min, a.n_frag_max),
                    canvas=(a.canvas_w, a.canvas_h), seed=seed,
                    degrade={"erode_px": cfg["erode"], "holes": cfg["holes"], "missing": cfg["missing"]},
                    rotate=cfg["rotate"], placement=cfg["placement"],
                    frag_distribution=a.frag_distribution, viz=a.debug, debug=a.debug,
                    label=f"{level} mosaic_{uid}{suffix}"))
    total = len(tasks)
    print(f"curriculum [{a.frag_distribution}] : {total} instances → {out_root}/ (jobs={a.jobs})")
    fails = skips = 0

    def _tally(lab, i):
        nonlocal fails, skips
        if lab.startswith("FAIL"):
            fails += 1
        elif lab.startswith("skip"):
            skips += 1
        if lab.startswith("FAIL") or i % 100 == 0 or i == total:
            print(f"[{i}/{total}] {lab}")

    if a.jobs <= 1:
        for i, t in enumerate(tasks, 1):
            _tally(_run_task(t), i)
    else:
        import multiprocessing as mp
        with mp.Pool(a.jobs) as pool:
            for i, lab in enumerate(pool.imap_unordered(_run_task, tasks), 1):
                _tally(lab, i)
    print(f"\nFait : {total - fails - skips} générées, {skips} sautées (resume), {fails} échecs.")
    print(f"Collate par palier :\n"
          f"  for L in {out_root}/L*_*; do python3 scripts/features/collate.py --dataset \"$L\"; done")


if __name__ == "__main__":
    main()
