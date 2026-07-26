# lego2hero

## Mise à jour — juillet 2026 (ce qui a changé depuis la v1)

La v1 (juin 2026) publiait **un seul dataset** (`balanced/`, 5 000 mosaïques ×
5 paliers = 25 000 instances). Elle est **dépréciée** : un bug d'encodage
géométrique (ci-dessous) y a faussé tout le signal de forme. La v2 la remplace
par une **matrice complète de 8 datasets appariés** (200 000 instances).

| # | changement | avant (v1) | après (v2) |
|---|---|---|---|
| 1 | **Budget de simplification du contour** | ~~budget en **nombre de sommets** : on garde d'abord **tous les sommets reflex**, puis on complète par des convexes jusqu'à `n_target` (16-24)~~ ❌ **abandonné** — `#reflex` (méd. 58) ≥ `n_target` dans **100 %** des fragments ⇒ **aucun convexe n'était jamais gardé**, **21 % d'aire perdue en médiane** | budget sur la **perte d'aire** (`--max-area-loss`, **ε = 1 %**), `n` en **sortie** ; Visvalingam-Whyatt restreint aux convexes ⇒ perte **0,94 % médiane / 1,00 % max** |
| 2 | **Nombre de fragments** (DAFNE A) | k = 10-15 uniquement | axe ouvert : **k = 2 · 3 · 5 · 10-15** (`--n-frag-min/max`) — la difficulté est dans les **arêtes** (1,0 → 27,1 par mosaïque) |
| 3 | **Motif de découpe** | `balanced` seul | `balanced` · `compact` · **`ultracompact`** (mosaïque forgée en tuiles 1×1 via `remono.py` + croissance compacte) |
| 4 | **Contrat de données GNN** | dense seul (pad `n_max` + masque) | **2 contrats dans le même `.npz`** : dense **et** ragged (`verts_flat` + `node_offsets`, sans `n_max`) → encodeur agnostique à la longueur possible (PointNet / Deep Sets) |
| 5 | **`n_max` cross-datasets** | par palier, non comparable | `collate.py --n-max` (n_max **joint**) ; **304** couvre LEGO **et** RePAIR réel |
| 6 | **Fragments manquants** (DAFNE C) | compte absolu 0-2 (à k=2, L4 ne laissait qu'**un** fragment) | plafond **relatif** automatique `round(0.15·k)`, surchargeable par `--missing-max` |
| 7 | **Dataset publié** | `balanced/` | hiérarchie `<grille_croissance>/<k>/<palier>` — cf. section suivante ; l'ancien devient **`LEGACY_DO_NOT_USE/`** |

⚠️ **Tous les `n_max` publiés avant le 23/07/2026 sont caducs** (mesurés sur des
polygones faux). La hiérarchie entre configs tient, les valeurs absolues sont
re-mesurées en v2.

## Datasets publiés — [🤗 icimathieu/lego2hero-100mosaics](https://huggingface.co/datasets/icimathieu/lego2hero-100mosaics)

**8 datasets appariés** : mêmes 5 000 peintures wikiart, mêmes seeds, seule la
config change → les courbes score(difficulté) sont comparables à variance faible.

```
poly_balanced/     tuiles variables + BFS équilibré      ← la voie principale
mono_compact/      tuiles 1×1 + croissance compacte      ← « ultracompact » (peu de sommets)
  └─ k02 · k03 · k05 · k10-15        nombre de fragments (DAFNE A)
       └─ L0_explode … L4_strong     1 tar.gz + 1 gnn_meta.json par palier

LEGACY_DO_NOT_USE/   ancien `balanced/` (v1) — ⛔ NE PAS ENTRAÎNER DESSUS
```

- **`LEGACY_DO_NOT_USE/`** = le dataset de juin, **conservé pour archive
  uniquement**. Il a été produit avec l'encodage bugué (perte d'aire **21 %
  médiane, p95 85 %, non bornée**) : `polygon_n_canonical` et `side_features` y
  décrivent des formes fausses. Son successeur corrigé est
  **`poly_balanced/k10-15/`**. Seuls les 7 scalaires `gnn_input` y restent valides.
- **`mono_compact/k02` et `k03`** sont volontairement **dégénérés** (fragments
  quasi rectangulaires, signal d'appariement géométrique mort) : ils sont inclus
  comme **bornes basses d'ablation**, pas comme configs d'entraînement.
- À **k=2** la prédiction de lien est dégénérée par construction (1 seule arête
  possible, toujours présente ⇒ F1 = 1,0 pour n'importe quel modèle) : seule la
  **pose** (Q_pos) y est mesurable.

### Prise en main (récupérer un palier et lire les tenseurs)

```python
from huggingface_hub import hf_hub_download
p = hf_hub_download("icimathieu/lego2hero-100mosaics",
                    "poly_balanced/k10-15/L2_rotation.tar.gz", repo_type="dataset")
# tar xzf <p> -C data/     → data/L2_rotation/mosaic_<id>/…
```

```python
import numpy as np
z = np.load("data/L2_rotation/mosaic_<id>/gnn_ready.npz")

# (A) contrat DENSE — tenseurs à dimension fixe
z["gnn_input"]            # (N, 7)        area, perimeter, R, G, B, bbox_w, bbox_h
z["polygon_n_canonical"]  # (N, n_max, 2) paddé à 0 au-delà de n_sides
z["side_features"]        # (N, n_max, 5) [length, angle, R, G, B] par côté
z["valid_mask"]           # (N, n_max)    1 = vrai sommet, 0 = padding

# (B) contrat RAGGED — mêmes données, sans n_max ni padding
off = z["node_offsets"]                    # (N+1,)
verts_i = z["verts_flat"][off[i]:off[i+1]] # (n_i, 2)  polygone du nœud i
sides_i = z["sides_flat"][off[i]:off[i+1]] # (n_i, 5)
```

`node_id` (`(N,)`) fait le lien avec `graph_fragments.json` (entrée) et
`graph_complete.json` (cible : arêtes + poses). **Ne jamais donner
`graph_complete.json` ni `gt_layout.json` en entrée** — ce sont les cibles.

## Pipeline

```
                         huggan/wikiart  (streaming)
                                  │
              ┌─────────────  forge_LAR_2mosaic  ─────────────┐
              │  image → quantif palette LEGO → packing tuiles │
              │         → rendu (joints gris)                  │
              └───────────────────────┬────────────────────────┘
                                      ▼
                target.png  (mosaïque LEGO)   +  piece_grid.json (GT)
                                      │
              ┌──────────────  mosaic2fragments  ──────────────┐
              │  joints → pièces → FRAGMENTATION → reco-B (ε=1%) │
              │                    → placement → dégradation     │
              │                                                  │
              │  --n-frag-min/max  (DAFNE A) : k = 2 · 3 · 5 · 10-15 │
              │                                                  │
              │  --frag-distribution  (motif de découpe) :       │
              │    balanced  compact  ultracompact¹  voronoi│clusters │
              │    (défaut)  (~rect.)  (forge mono 1×1) (DAFNE B, à venir) │
              │                                                  │
              │  curriculum (placement × dégradation) :          │
              │    L0 éclaté · L1 transl. · L2 +rot · L3 · L4     │
              └───────────┬──────────────────────────┬───────────┘
                          ▼                            ▼
              source.png + source_yolo.txt     graph_complete.json (cible GNN)
              (fragments éclatés + labels)      graph_fragments.json (entrée GNN)
                          │                      gt_layout.json (GT reconstruction)
                          ▼                                 │
              ┌───────  YOLO-Seg  ───────┐                  │
              │  TRAIN sur nos labels     │                  │
              │  exacts (synthétique) ;   │                  │
              │  INFER sur fresques réelles│                 │
              └────────────┬──────────────┘                  │
                           ▼                                  │
                 masques de fragments  (synth. = GT,          │
                  réel = détections YOLO)                     │
                           │                                  │
                           ▼                                  ▼
   ┌──────────  features  (post-YOLO / pré-GNN, PARTAGÉ synth.↔réel) ──────────┐
   │  masque → reco-B (budget ε de perte d'aire) + PCA canonical + gnn_input    │
   │  → collate : 2 contrats dans gnn_ready.npz                                 │
   │      (A) dense  = pad n_max + masque                                       │
   │      (B) ragged = verts_flat + node_offsets  (sans n_max)                  │
   └────────────────────────────────────┬──────────────────────────────────────┘
                                         ▼
              encodeur de nœud (côté GNN, entraîné avec lui) :
              (A) → tenseurs denses   (B) → PointNet/Deep Sets (agnostique)
                                         ▼
                                GNN  (réassemblage)
```

<sub>¹ `ultracompact` = `compact` sur une mosaïque forgée en `--mode mono` (tuiles
1×1). Les mono se régénèrent **exactement** depuis les `piece_grid.json`
existants via `scripts/forge_LAR_2mosaic/remono.py` (mêmes couleurs, mêmes uuid →
appariement conservé).</sub>

`features` est **partagé** : sur le synthétique il consomme la GT, sur le réel il
consomme les masques détectés par YOLO — même code.

### Paliers de dégradation (difficulté croissante)

Driver `scripts/mosaic2fragments/curriculum.py` → `output/<frag-distribution>/<palier>/`.
**Mêmes mosaïques de base à tous les paliers** (design apparié : seule la difficulté change).

| palier | placement | rotation | dégradation | n_max* |
|---|---|---|---|---:|
| **L0_explode** | vue éclatée (positions relatives gardées) | non | aucune | 156 |
| **L1_translation** | scatter aléatoire | non | aucune | 156 |
| **L2_rotation** | scatter | oui | aucune | 156 |
| **L3_light** | scatter | oui | légère (érosion 1-2 px, 0-1 trou) | 114 |
| **L4_strong** | scatter | oui | forte (érosion 2-6 px, trous, manquants) | 185 |

<sub>*n_max mesuré sur le set `balanced` 5000/palier (21/06). L0=L1=L2 identiques : le placement ne touche pas la géométrie.*</sub>

> ⚠️ Ces `n_max` (ainsi que le « −15 % » de `compact` et le « 156→60 » d'
> `ultracompact`) ont été mesurés **avant le correctif de reco-B du 23/07/2026**
> et sont donc **caducs** — le budget de simplification a changé de nature. La
> hiérarchie entre paliers et entre modes devrait tenir ; les valeurs absolues
> sont à re-mesurer. Cf. « Encodage géométrique — reco-B ».

### Les 5 paramètres de difficulté DAFNE

**DAFNE** (*Digital Anastylosis of Frescoes challeNgE*, Univ. Pavie, PRL 2020) est
un benchmark **synthétique** de réassemblage de fresques 2D. Il ne définit **pas**
une échelle ordonnée mais **5 curseurs indépendants** — c'est notre gabarit de
dégradation. Nos paliers L0-L4 ne sont donc **pas** « les paliers DAFNE » : ils
combinent un axe **pose** (notre ajout, absent de DAFNE) et deux curseurs DAFNE.

| DAFNE | ce que c'est | chez nous | statut |
|---|---|---|---|
| **A** — nb de fragments | plus de fragments = plus dur | `--n-frag-min/max` — historiquement figé à **10-15** ; configs **k=2/3/5** ajoutées (cf. tableau suivant) | ✅ |
| **B** — distribution de découpe | tailles/formes des coupes | l'axe `--frag-distribution` : `balanced`/`compact`/`ultracompact` ✅ — `voronoi`/`clusters` (le vrai durcissement DAFNE B) ❌ | 🚧 partiel |
| **C** — % de fragments manquants | pièces absentes de l'entrée | `--missing-min/max`, palier **L4** = 0-2. ⚠️ **compte absolu** chez nous, **pourcentage** chez DAFNE → voir `--missing-max` ci-dessous | ✅ |
| **D** — fragments parasites | pièces d'un autre objet à rejeter | — | ❌ |
| **E** — érosion | bords rongés | `--erode-px-min/max`, **L3** = 1-2 px, **L4** = 2-6 px | ✅ |

En plus des curseurs DAFNE, on ajoute les **trous internes** (`--holes-min/max`,
aire totale plafonnée à 10 % du fragment) et l'axe **pose** (L0→L2).

DAFNE C étant un **pourcentage** et notre `missing` un **compte absolu**,
`curriculum.py` plafonne automatiquement le tirage à **~15 % de k**
(`round(0.15 · n_frag_max)` ; surcharge possible via `--missing-max`). À k=10-15
ça donne 2 — le réglage historique exact, le run 25k reste reproductible.

### Nombre de fragments (DAFNE A)

Axe **orthogonal** aux paliers et aux modes de découpe (`--n-frag-min/max`).
Réduire k rend le **graphe** beaucoup plus facile sans rendre la **pose** triviale.

Mesuré sur le pilote du 23/07 — **100 mosaïques wikiart identiques** pour les 4
configs, `balanced`, reco-B corrigé (ε=1 %), 2 000 instances, 0 échec :

| k | nœuds/mosaïque | **arêtes**/mosaïque | `n` médian | `n_max` (L0-L2) | sortie |
|---|---:|---:|---:|---:|---|
| **2** | 2 | **1,0** | 75 | 140 | `output/pilot100_poly_balanced/k02/` |
| **3** | 3 | **2,9** | 92 | 254 | `output/pilot100_poly_balanced/k03/` |
| **5** | 5 | **7,5** | 84 | 214 | `output/pilot100_poly_balanced/k05/` |
| **10-15** (défaut) | 13,0 | **27,1** | 75 | 184 | `output/pilot100_poly_balanced/k10-15/` |

**Ce que ça dit.** Le levier de difficulté est le nombre d'**arêtes** (27× plus à
k=10-15 qu'à k=2), pas le nombre de sommets. Contre-intuitivement, `n` **n'est pas
monotone en k** : il culmine à k=3 (92) et redescend à 75 aux deux extrêmes. À
k=2, une grande part du contour d'un fragment est le **bord extérieur droit** de
la mosaïque (peu de sommets) ; à k=10-15 les fragments sont petits donc leur
périmètre absolu est court ; c'est entre les deux que la part de bord **interne
zigzaguant** est maximale.

⚠️ `n_max` est un **maximum**, il croît avec la taille de l'échantillon : ces
valeurs sont sur **100** mosaïques et monteront sur 5 000. Elles ne sont pas
comparables telles quelles aux `n_max` du run 25k.

⚠️ À **k=2 la prédiction de lien est dégénérée** : une seule arête possible,
toujours présente → F1 sur le mating graph = 1,0 par construction, y compris pour
un prédicteur constant. Ce qui reste mesurable est la **pose** (Q_pos). k=2 est
donc un **diagnostic** (« le modèle sait-il aligner deux pièces le long d'un bord
partagé ? »), pas une validation de la pipeline.

Le plafond `missing` (~15 % de k) est automatique — cf. section DAFNE.

### Modes de fragmentation (`--frag-distribution`)

Axe **orthogonal** aux paliers de dégradation (sortie `output/<mode>/<palier>/`).

| mode | règle de découpe | forme des fragments | statut |
|---|---|---|---|
| **balanced** | seeds farthest-point + BFS à priorité (le plus petit fragment grandit d'abord) | blobs d'aire ~égale | ✅ défaut |
| **compact** | croissance à **périmètre minimal** : chaque fragment absorbe la pièce-frontière qui maximise le remplissage de sa bbox | bords + lisses, médiane sommets −15 % | ✅ |
| **ultracompact** | ⚠️ **pas un algo différent** : c'est `compact` appliqué à une mosaïque forgée en **`--mode mono`** (tuiles 1×1). « L'ultra » vient de l'ENTRÉE, pas de la coupe | rectangulaires, **peu de sommets** (coupes droites sur grille régulière) | ✅ |
| **voronoi** | k graines dispersées, pièce → graine la + proche | tailles **inégales**, bords irréguliers (**DAFNE B**) | 🚧 |
| **clusters** | graines agglutinées | zones très/peu fragmentées (**DAFNE B**) | 🚧 |

**Le nombre de sommets vient du TILING × la CROISSANCE — mesuré le 23/07**
(reco-B corrigé ε=1 %, mêmes 100 mosaïques, k=10-15, L0) :

| combinaison (grille × croissance) | `n` médian | p95 | `n_max` | arêtes/mos. | sortie |
|---|---:|---:|---:|---:|---|
| poly-tuiles + `balanced` (défaut) | 75 | 122 | 184 | 27,1 | `output/pilot100_poly_balanced/` |
| poly-tuiles + `compact` | **60** | 107 | 183 | 27,1 | `output/probe_poly_compact/` |
| **mono 1×1 + `balanced`** | **169** | 262 | 262 | 27,0 | `output/probe_mono_balanced/` |
| mono 1×1 + `compact` = **`ultracompact`** | **33** | 63 | **63** | 20,0 | `output/pilot100_ultra_mono_compact/` |

Aucun des deux leviers ne marche **seul** : la grille fine sans croissance
compacte est la **pire** combinaison (les frontières irrégulières du BFS font des
escaliers d'1 stud très nombreux, que le budget ε=1 % ne peut pas se payer de
lisser : ~7 % d'aire). C'est la **combinaison** grille fine × coupes droites qui
divise `n` par ~2,3 et `n_max` par ~3. `voronoi`/`clusters` restent l'axe de
**durcissement** (DAFNE B).

⚠️ **`ultracompact` + petit k est à PROSCRIRE** : à k=2 il produit littéralement
deux rectangles (5 sommets médian ET max sur 100 mosaïques ; k=3 → 7 ; k=5 → 11).
Tout bord droit s'emboîte avec tout bord droit → le signal d'appariement
géométrique est mort, il ne reste que la couleur. `ultracompact` n'a de sens
qu'à k≥10, où les marches subsistent (n méd 33).

### Encodage géométrique — reco-B (`polygon_n`)

Le contour brut d'un fragment (`polygon_raw`, sorti de `cv2.findContours`) est
simplifié en `polygon_n`, qui est ce que voit le GNN. Deux garanties :

- **`polygon_n ⊆ polygon_raw`** — on ne *gagne* jamais d'aire, on en perd (« comme
  si les coins étaient ébréchés »). Les fragments réassemblés ne se chevauchent
  donc **jamais** : ils laissent des jours. C'est le régime physiquement
  réalisable, et ça protège la métrique Q_pos.
- **Budget sur la PERTE D'AIRE** (`--max-area-loss`, défaut **ε = 1 %**), le nombre
  de sommets `n` étant la **sortie** et non l'entrée.

Vérifié sur le pilote (2 000 instances) : sur les paliers **non dégradés**
(L0-L2), perte d'aire **0,93-0,97 % médiane** et **1,00 % maximum** — le budget
est saturé sans jamais être dépassé — et **aucune aire gagnée** sur aucun
fragment. Sur L3/L4 la perte mesurée vs le contour *parfait* est plus élevée
(jusqu'à 7,9 % de médiane à L4) : c'est **la dégradation elle-même** (érosion,
trous), pas l'encodage — reco-B travaille sur le contour déjà dégradé et y
respecte le même ε de 1 %.

L'algorithme est **Visvalingam-Whyatt** (1993) *restreint aux sommets convexes* :
on retire itérativement le sommet dont le triangle (préc., lui, suiv.) a la plus
petite aire, tant que la perte cumulée reste ≤ ε. Seuls les **convexes** sont
retirables — retirer un sommet **reflex** pontifierait une concavité et
**re-gagnerait** de l'aire. Les sommets ~colinéaires ont un triangle d'aire nulle
et sont donc retirés **gratuitement** (c'est le gros du bruit de contour LEGO).
Le plancher dur de `n` reste **#reflex**, atteint automatiquement.

> ⚠️ **Correctif du 23/07/2026 — invalide les `n_max` publiés avant cette date.**
> ~~L'implémentation précédente budgétait le **nombre de sommets**
> (`--n-sides-min/max`, 16-24) en amorçant l'ensemble à garder avec tous les
> sommets reflex, puis en complétant par les convexes jusqu'au quota.~~ ❌ **Étape
> supprimée.** Comme `#reflex`
> (médiane **58** sur LEGO) ≥ `n_target` dans **100 %** des fragments, la boucle
> d'ajout des convexes sortait immédiatement : **aucun sommet convexe n'était
> jamais gardé**. L'invariant ⊆ raw tenait (pas d'aire fantôme) mais **l'ampleur
> de la perte n'était ni bornée ni mesurée** — en pratique **21,4 % d'aire perdue
> en médiane** (p95 85,8 %, un quart des fragments perdant >50 %) sur la config
> k=10-15, et jusqu'à **99 % à k=2**. `polygon_n_canonical` et `side_features` —
> tout le signal de forme donné au GNN — décrivaient donc des formes fausses. Les
> 7 scalaires `gnn_input` n'étaient **pas** touchés (calculés depuis le masque).
> `--n-sides-min/max` est conservé pour info mais **n'est plus le budget**.

## Arborescence

```
lego2hero/
├── scripts/
│   ├── forge_LAR_2mosaic/      # image → mosaïque LEGO (canvas_mosaic.png + piece_grid.json)
│   │   ├── mosaic.py           # quantif Lab → packing glouton → rendu tuiles + joints
│   │   ├── cli.py · batch.py   # 1 image / dossier d'images
│   │   ├── wikiart.py          # source d'images : huggan/wikiart en streaming
│   │   ├── remono.py           # re-forge exacte en tuiles 1×1 depuis piece_grid.json (uuid conservés)
│   │   └── palette.py          # 82 couleurs LEGO (bricklink, gris filtrés)
│   ├── mosaic2fragments/       # mosaïque → fragments → YOLO-Seg + graphes GNN
│   │   ├── forge_dataset.py    # pipeline principal (joints → pièces → fragments → GT)
│   │   ├── curriculum.py       # driver des paliers L0–L4 × --frag-distribution → output/<dist>/<palier>/
│   │   └── batch.py · visualize.py
│   ├── features/               # post-YOLO / pré-GNN — PARTAGÉ synthétique↔réel
│   │   ├── fragment_features.py   # reco-B + PCA canonical + side_features + gnn_input
│   │   └── collate.py             # → gnn_ready.npz : dense (pad n_max + masque) ET ragged (verts_flat + node_offsets)
│   └── tools/
│       ├── hf_upload.py        # upload d'un dossier vers le HuggingFace Hub (dataset)
│       ├── hf_export_tars.py   # export HF en tar.gz par palier (resume)
│       └── weekend_full_matrix.sh · weekend_babysitter.sh   # run de la matrice 8 datasets (séquentiel, resume, nettoyage disque)
├── todo.md                 # statut concis (historique + à faire)
├── LICENSE                 # MIT
└── output/                 # ⚠️ GÉNÉRÉ — gitignoré, hors du dépôt (données lourdes)
    └── <grille_croissance>/<k>/<palier>/mosaic_<id>/   ← structure détaillée ci-dessous
```

## Structure d'un sous-dossier de mosaïque

Une mosaïque = un dossier `output/<dataset>/mosaic_<id>/`. C'est l'**entrée** des
étapes aval (YOLO / GNN / VLM).

| Fichier | Rôle |
|---|---|
| `target.png` | mosaïque **complète** = cible de reconstruction / contexte VLM |
| `source.png` | fragments **éclatés** sur fond blanc = **entrée YOLO** (et VLM) |
| `source_yolo.txt` | **vérité terrain YOLO-Seg** (1 polygone/fragment, `classe x1 y1 … xn yn` normalisés `[0,1]`) — on entraîne YOLO sur la paire (`source.png`, `source_yolo.txt`) |
| `source_yolo_viz.png` | overlay debug des labels (inspection humaine) — *absent par défaut* (opt-in `--debug`) |
| `pieces.json` | debug pur (pièces LEGO détectées) ; **n'entraîne PAS le YOLO**, lu nulle part — *absent par défaut* (opt-in `--debug`) |
| **`graph_fragments.json`** | **ENTRÉE GNN** : nœuds (features/fragment), zéro arête, sans `target_info` |
| **`graph_complete.json`** | **CIBLE GNN** : mêmes nœuds + `target_info` (leak) + arêtes (mating graph) |
| `gt_layout.json` | **GT de RECONSTRUCTION** (réponse finale) : footprint exact + pose `(x,y,rot)` de chaque fragment dans `target.png`. Sert à **scorer l'assemblage** (IoU/Q_pos) APRÈS la tête de pose du GNN / le VLM qui place les fragments |
| `gnn_ready.npz` | entrée GNN à **dimension fixe** (pad `n_max` + masque de validité) |
| `degradation.md` | rapport de dégradation (tout à 0 = clean) |
| `fragments/frag_XX.png` | crop alpha par fragment = **entrée VLM** |

Au niveau dataset : **`gnn_meta.json`** (`n_max`, noms de features).

### Schéma d'un nœud (`graph_fragments.json`)

| Bloc | Contenu | Taille |
|---|---|---|
| `node_id` | identifiant stable (référencé par les arêtes via `src`/`dst`) | — |
| `gnn_input` | `[area, perimeter, R, G, B, bbox_w, bbox_h]` — domaine-agnostique | 7 |
| `polygon_n_canonical` | contour en repère **PCA canonique** (invariant en rotation) | `n_sides`×2 |
| `side_features` | par côté `[length, angle, R, G, B]` | `n_sides`×5 |

`n_sides` est **variable** par nœud. Ce que livre `collate.py` (23/07) :
**deux contrats côte à côte dans le même `gnn_ready.npz`** —

- **(A) dense** : `polygon_n_canonical`/`side_features` paddés à `n_max` +
  `valid_mask` ;
- **(B) ragged** : `verts_flat` (Σn×2), `sides_flat` (Σn×5), `node_offsets`
  (N+1) — le nœud *i* = tranche `[offsets[i]:offsets[i+1]]`, **sans aucun
  `n_max`**. Entrée naturelle d'un encodeur agnostique (MLP partagé par sommet +
  scatter-pooling par segment, ex. `torch_scatter.scatter_max(h, seg_ids)`).

⚠️ **Le pad à `n_max` n'est PAS une obligation.** La dimension fixe est exigée sur
l'**embedding de nœud**, pas sur le polygone brut. L'encodeur (quelle que soit la
voie) est **appris avec le GNN** — ce n'est pas une étape de prétraitement des
données, ses poids n'existent qu'à l'entraînement. Trois voies possibles, à
trancher côté GNN :

| voie | dimension fixe obtenue par | `n_max` | transférable au réel (RePAIR) |
|---|---|---|---|
| **pad + masque** (livré) | remplissage à `n_max` + masque | oui, global | ⚠️ possible depuis le correctif reco-B : les contours réels (bruts 211-1 329) **descendent à 132-291** après reco-B ε=1 % → un `n_max` **joint** ≈300 suffit. Mais n_max reste par-dataset (re-collate à chaque ajout) |
| **k keypoints fixes** (ReassembleNet, k=20) | sélection de k points par pièce | disparaît | ✅ mais **casse `polygon_n ⊆ polygon_raw`** |
| **encodeur agnostique** (PointNet / Deep Sets / Set Transformer) | MLP partagé par sommet + pooling max/somme, **placé avant le message-passing** et entraîné avec le GNN | disparaît | ✅ conserve tous les invariants |


## Arêtes (cible) — `graph_complete.json`
`edges` : `src`/`dst` (node_id) + `features` `[shared_length, mean_angle, n_segments]`
+ `src_side_idx`/`dst_side_idx`.

## État de l'art — méthodes de réassemblage 2D

Comment chaque méthode représente un fragment et ce qu'elle rapporte :

| méthode | représentation du fragment | architecture | résultats rapportés |
|---|---|---|---|
| **ReassembleNet** (ICCV 2025, [arXiv:2505.21117](https://arxiv.org/abs/2505.21117)) | **k=20 keypoints FIXES** sur le contour (init Harris, top-k **appris** par GNN-pooling ; FPS en baseline) + features texture. ⚠️ les 20 points sont un **nuage descriptif**, PAS un polygone-empreinte : aire/inscription non définies, la faisabilité physique n'est pas garantie par la représentation | attention inter/intra-pièce + **diffusion** qui débruite les poses | SOTA fresque 2D : **−57 % RMSE rotation, −87 % RMSE translation** vs méthodes antérieures ; pré-entraîné sur son dataset semi-synth (5 000 mosaïques / 45 834 fragments), évalué sur RePAIR |
| **DiffAssemble** (CVPR 2024, [arXiv:2402.19302](https://arxiv.org/abs/2402.19302)) | éléments d'un ensemble = nœuds d'un graphe spatial | GNN + **diffusion** unifiée 2D/3D (positions + rotations) | SOTA sur la plupart des tâches 2D/3D ; puzzles jusqu'à **900 éléments** ; **11× plus rapide** que le meilleur solveur par optimisation |
| **PairingNet** (2024, [arXiv:2312.08704](https://arxiv.org/abs/2312.08704)) | **contour DENSE complet** (longueur variable), **zero-paddé à 2 900** / tronqué à 1 408 + texture le long du bord | GNN (ResGCN) contour+texture, transformer linéaire + loss contrastive (pair-searching), fusion pondérée (pair-matching) | son propre dataset : 390 images Pexels → 8 196 fragments / 14 951 paires (+ set réel 34 images / 320 fragments). Comparable à notre étage **link-prediction** |
| **Nash Meets Wertheimer** (2024, [arXiv:2410.16857](https://arxiv.org/abs/2410.16857)) | uniquement les **motifs linéaires** traversant les bords (bonne continuation gestaltiste) — ignore couleur et forme | théorie des jeux (équilibre) | testé sur fresques archéo réelles ; montre que « continuité des motifs » est un signal exploitable seul |
| **VLHSA** (2025, [arXiv:2509.25202](https://arxiv.org/abs/2509.25202)) | patchs image + sémantique | VLM + alignement sémantique hiérarchique | puzzles à **gaps érodés** (cas archéo) ; pertinent pour la voie VLM |

**Les deux extrêmes de représentation** — et où on se place :
PairingNet paie un `n_max` géant (2 900) pour ne rien perdre ; ReassembleNet
compresse à 20 points en **renonçant à la sémantique de surface** (GT de pose
issue de l'objet intact, pas de garantie no-overlap dans la représentation).
Notre reco-B (ε=1 %) est un **milieu adaptatif avec garantie** : contour complet
simplifié au minimum de sommets qui préserve 99 % de l'aire et l'invariant
`polygon_n ⊆ polygon_raw` (faisabilité physique). Mesuré : LEGO `n` médian
75-92 ; **fragments RePAIR réels : brut 211-1 329 sommets → 132-291 après
reco-B ε=1 %** (médiane 178, perte 1,00 %) — **même ordre de grandeur que le
LEGO**, donc le schéma d'encodage est transférable tel quel.

## Datasets réels & benchmarks (réassemblage vs style)

| dataset | nature | contenu | rôle pour nous |
|---|---|---|---|
| **RePAIR** (NeurIPS 2024, [site](https://repairproject.github.io/RePAIR_dataset/), [Zenodo](https://zenodo.org/records/13993089)) | **RÉEL** — fresques de Pompéi bombardées en 39-45 | **16 000 fragments** au total ; GT (pose annotée par des archéologues, des années de terrain) sur **~1 000 pièces** ; benchmark 2D : 121 objets / 957 fragments (97 train / 24 test) ; 2D **+ 3D** | **notre cible de transfert + source de métriques** (Q_pos, RMSE rot/trans, F1 mating graph). Fragments 2D déjà en local |
| **DAFNE** (PRL 2020, [dataset](https://vision.unipv.it/DAFchallenge/DAFNE_dataset/)) | SYNTHÉTIQUE 2D | 62 fresques × 18 configs ; 5 paramètres de difficulté A-E | notre **gabarit de dégradation** (cf. tableau DAFNE) |
| **ReassembleNet semi-synth** ([Drive](https://drive.google.com/drive/folders/1tflCUoct63Zhzt8dWs37vtfnphOqgReB?usp=sharing)) | semi-synthétique | 5 000 mosaïques / 45 834 fragments (80/20) | calibre notre échelle (25 000 instances ✓) |
| **PairingNet data** | synthétique + petit set réel | 390 images → 8 196 fragments ; réel : 34 images / 320 fragments | baseline link-prediction |
| **POMPAAF** ([arXiv:2501.00836](https://arxiv.org/abs/2501.00836)) · **CLEOPATRA** (JAIHC 2023) | réels | fragments de fresques pompéiennes | ⚠️ **classification de STYLE, PAS réassemblage** — utiles au VLM, pas au GNN. Ne pas les citer comme benchmarks de reconstruction |

**Pourquoi le manuel ne passe pas à l'échelle** (motivation, cf. sources dans le
journal de bord) : le casque de Sutton Hoo = 18 mois d'un conservateur pour ~500
fragments ; Pompéi = ~10 000 fragments en entrepôt, la GT RePAIR (~1 000 pièces)
a demandé des années ; Akrotiri = des pans entiers non assemblés 50 ans après le
début des fouilles.


hhhhhh