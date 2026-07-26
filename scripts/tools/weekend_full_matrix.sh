#!/bin/zsh
# ================================================================
# RUN WEEK-END — matrice COMPLÈTE {poly,mono} × k={2,3,5,10-15}
#   × 5000 images × 5 paliers = 8 datasets × 25 000 = 200 000 instances,
#   collate (pad+masque ET ragged dans chaque npz), upload HF, nettoyage.
#
# Pipeline STRICTEMENT SÉQUENTIEL (disque ~80 Go) :
#   générer D → collate D → tar+upload D (palier par palier) →
#   vérifier sur le repo → SUPPRIMER D en local → dataset suivant.
#   Pic disque ≈ 1 dataset (~43 Go) + 1 tar de palier (~9 Go).
#   Dernière étape : renommage HF `balanced/` → `LEGACY_DO_NOT_USE/` (pas de
#   suppression — décision Mathieu).
#
# PRÉREQUIS (à faire pendant que tu es encore là vendredi matin) :
#   1. hf auth login    (dans TON terminal — le script vérifie et s'arrête sinon)
#   2. couvercle OUVERT + secteur
#   3. lancement :
#      caffeinate -i zsh scripts/tools/weekend_full_matrix.sh > output/weekend.log 2>&1 &
#      tail -f output/weekend.log        # pour surveiller les premières heures
#
# RESUME TOTAL : chaque étage est idempotent (forge saute les instances
# complètes, l'upload saute les tars déjà sur le repo, la suppression ne se
# fait qu'après vérification). Relancer la même commande reprend où c'était.
# ================================================================
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
REPO=icimathieu/lego2hero-100mosaics
export HF_HUB_ENABLE_HF_TRANSFER=1
MIN_FREE_GB=70

# --- ordre : du plus précieux au moins précieux (si panne, l'essentiel est fait)
#     nom_local:frag_distribution:kmin:kmax:inputs_glob
# Hiérarchie (HF **et** local) : <grille_croissance>/<k>/<palier>
#   poly_balanced/  = tuiles variables + BFS équilibré   (la voie principale)
#   mono_compact/   = tuiles 1×1 + croissance compacte   (= ultracompact)
#   × k02 · k03 · k05 · k10-15   × L0…L4 (tar.gz + gnn_meta.json par palier)
DATASETS=(
  "poly_balanced/k10-15:balanced:10:15:output/wikiart_inputs_poly/canvas_mosaic_*.png"
  "poly_balanced/k05:balanced:5:5:output/wikiart_inputs_poly/canvas_mosaic_*.png"
  "poly_balanced/k03:balanced:3:3:output/wikiart_inputs_poly/canvas_mosaic_*.png"
  "poly_balanced/k02:balanced:2:2:output/wikiart_inputs_poly/canvas_mosaic_*.png"
  "mono_compact/k10-15:ultracompact:10:15:output/wikiart_inputs_mono/canvas_mosaic_*.png"
  "mono_compact/k05:ultracompact:5:5:output/wikiart_inputs_mono/canvas_mosaic_*.png"
  "mono_compact/k03:ultracompact:3:3:output/wikiart_inputs_mono/canvas_mosaic_*.png"
  "mono_compact/k02:ultracompact:2:2:output/wikiart_inputs_mono/canvas_mosaic_*.png"
)

free_gb() { df -g / | tail -1 | awk '{print $4}' }

echo "=== [0] prérequis ($(date '+%d/%m %H:%M'))"
WHO=$($PY -c 'from huggingface_hub import HfApi
try: print(HfApi().whoami()["name"])
except Exception as e: print(f"ERREUR:{e}")')
if [[ "$WHO" != "icimathieu" ]]; then
  echo "ABORT : HF non connecté ('$WHO'). Fais 'hf auth login' puis relance."; exit 1
fi
echo "HF ok ($WHO) · disque libre : $(free_gb) Go"

echo "=== [1] remono 5000 ($(date '+%H:%M'))"
$PY scripts/forge_LAR_2mosaic/remono.py \
    --piece-grids 'output/wikiart_inputs_poly/piece_grid_*.json' \
    --out-dir output/wikiart_inputs_mono

# --- carte du repo (matrice complète)
cat > output/hf_card.md <<'CARD'
# lego2hero — mosaïques LEGO fragmentées (matrice complète, reco-B ε=1 %)

8 datasets appariés (mêmes 5000 peintures wikiart, mêmes seeds), hiérarchie
`<grille_croissance>/<k>/<palier>` :

```
poly_balanced/   tuiles variables + BFS équilibré (voie principale)
mono_compact/    tuiles 1×1 + croissance compacte (« ultracompact », peu de sommets)
  └─ k02 | k03 | k05 | k10-15      nombre de fragments (DAFNE A)
       └─ L0_explode … L4_strong   1 tar.gz + 1 gnn_meta.json par palier
LEGACY_DO_NOT_USE/                 ancien `balanced/` pré-correctif — NE PAS UTILISER
```

Paliers : L0 vue éclatée · L1 translation · L2 +rotation · L3 dégradation légère ·
L4 forte. `missing` plafonné à ~15 % de k (DAFNE C).

Chaque palier = `<dataset>/L*.tar.gz` (+ `L*.gnn_meta.json` browsable).
Par mosaïque : `target.png`, `source.png` + `source_yolo.txt` (GT YOLO-Seg),
`graph_fragments.json` (entrée GNN) / `graph_complete.json` (cible),
`gt_layout.json` (GT de pose), `fragments/*.png` (VLM), `gnn_ready.npz`
(**2 contrats** : dense pad-n_max+masque ET ragged `verts_flat`+`node_offsets`).

Encodage géométrique : reco-B corrigé du 23/07/2026 — budget de **perte d'aire**
ε=1 % (Visvalingam-Whyatt restreint aux convexes), invariant
`polygon_n ⊆ polygon_raw`. ⚠️ **`LEGACY_DO_NOT_USE/`** (ex-`balanced/`) = données
d'avant le correctif (perte d'aire médiane 21 %, non bornée) — conservées pour
archive, **ne pas entraîner dessus** ; le successeur corrigé est
`poly_balanced/k10-15/`. ⚠️ `mono_compact/k02` et `k03` : configs volontairement
dégénérées (fragments ~rectangulaires) — incluses pour l'étude d'ablation.

Généré par https://github.com/icimathieu/forge_LAR_YOLO_lego2hero (MIT).
CARD

FIRST_EXPORT=1
for SPEC in "${DATASETS[@]}"; do
  IFS=':' read -r NAME DIST KMIN KMAX GLOB <<< "$SPEC"
  OUT=output/wk/$NAME
  SRC=$OUT/$DIST
  echo "=== [$NAME] début ($(date '+%d/%m %H:%M')) · libre $(free_gb) Go"
  # RESUME inter-datasets (fix 26/07 00:xx) : un dataset dont les 5 tars sont
  # déjà sur le repo est SAUTÉ (sinon toute relance re-forgerait les datasets
  # finis dont le local a été nettoyé).
  DEJA=$($PY -c 'import sys
from huggingface_hub import HfApi
api = HfApi(); name = sys.argv[1]
ls = ["L0_explode","L1_translation","L2_rotation","L3_light","L4_strong"]
print("OK" if all(api.file_exists("icimathieu/lego2hero-100mosaics",
      f"{name}/{L}.tar.gz", repo_type="dataset") for L in ls) else "NON")' "$NAME")
  if [[ "$DEJA" == "OK" ]]; then
    echo "=== [$NAME] déjà intégralement sur HF — SAUTÉ"; continue
  fi
  # Garde-fou disque : seulement pour un dataset qui démarre À NEUF (dossier
  # absent = ~50 Go à créer). Un dataset en cours (dossier présent) passe :
  # sa donnée est déjà sur le disque, l'export n'a besoin que d'~1 tar.
  if [[ ! -d $OUT ]] && (( $(free_gb) < MIN_FREE_GB )); then
    echo "ABORT : moins de $MIN_FREE_GB Go libres pour un dataset neuf — nettoyer avant de relancer."; exit 1
  fi

  echo "--- forge (25 000 instances, resume)"
  $PY scripts/mosaic2fragments/curriculum.py \
      --inputs-glob "$GLOB" --out $OUT --frag-distribution $DIST \
      --n-frag-min $KMIN --n-frag-max $KMAX --max-area-loss 0.01 --jobs 6

  echo "--- collate (pad + ragged)"
  for L in $SRC/L*_*; do
    $PY scripts/features/collate.py --dataset "$L" || { echo "collate FAIL $L"; exit 1; }
  done

  echo "--- export HF → $NAME/ (tar/upload par palier, resume)"
  CARDARG=""
  if (( FIRST_EXPORT )); then CARDARG="--card output/hf_card.md"; FIRST_EXPORT=0; fi
  $PY scripts/tools/hf_export_tars.py --src $SRC --repo-id $REPO \
      --path-in-repo $NAME ${=CARDARG}

  echo "--- vérification repo puis NETTOYAGE local"
  OK=$($PY -c 'import sys
from huggingface_hub import HfApi
api = HfApi(); name = sys.argv[1]
ls = ["L0_explode","L1_translation","L2_rotation","L3_light","L4_strong"]
print("OK" if all(api.file_exists("icimathieu/lego2hero-100mosaics",
      f"{name}/{L}.tar.gz", repo_type="dataset") for L in ls) else "MANQUE")' "$NAME")
  if [[ "$OK" == "OK" ]]; then
    rm -rf $OUT
    echo "--- $NAME : uploadé + supprimé en local ✓"
  else
    echo "ABORT : tars manquants sur le repo pour $NAME — rien supprimé."; exit 1
  fi
done

# --- legacy 'balanced/' (données du bug reco-B) : RENOMMÉ en LEGACY_DO_NOT_USE/
#     (décision Mathieu : conserver mais rendre le statut évident). Copie
#     serveur-side pour les tars LFS (pas de re-upload), download/re-upload pour
#     les petits json, puis suppression de l'ancien dossier. Non fatal si échec.
$PY -c '
from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationDelete, hf_hub_download
repo = "icimathieu/lego2hero-100mosaics"
api = HfApi()
try:
    files = [f for f in api.list_repo_files(repo, repo_type="dataset")
             if f.startswith("balanced/")]
    if not files:
        print("[legacy] balanced/ absent du repo — rien a renommer")
    else:
        tars  = [f for f in files if f.endswith(".tar.gz")]        # LFS → copie serveur
        small = [f for f in files if not f.endswith(".tar.gz")]    # json → re-upload
        ops = [CommitOperationCopy(src_path_in_repo=f,
                                   path_in_repo="LEGACY_DO_NOT_USE/" + f.split("/", 1)[1])
               for f in tars]
        api.create_commit(repo_id=repo, repo_type="dataset", operations=ops,
                          commit_message="copie legacy balanced -> LEGACY_DO_NOT_USE (bug reco-B, perte aire 21 pct)")
        for f in small:
            local = hf_hub_download(repo, f, repo_type="dataset")
            api.upload_file(path_or_fileobj=local,
                            path_in_repo="LEGACY_DO_NOT_USE/" + f.split("/", 1)[1],
                            repo_id=repo, repo_type="dataset",
                            commit_message=f"copie legacy {f}")
        api.create_commit(repo_id=repo, repo_type="dataset",
                          operations=[CommitOperationDelete(path_in_repo="balanced/", is_folder=True)],
                          commit_message="retrait ancien chemin balanced/ (renomme LEGACY_DO_NOT_USE)")
        print(f"[legacy] balanced/ -> LEGACY_DO_NOT_USE/ ({len(tars)} tars copies serveur, {len(small)} petits fichiers)")
except Exception as e:
    print(f"[legacy] renommage ECHOUE (non fatal, a refaire a la main) : {e}")
'

echo "=== WEEK-END TERMINÉ ($(date '+%d/%m %H:%M')) → https://huggingface.co/datasets/$REPO"
