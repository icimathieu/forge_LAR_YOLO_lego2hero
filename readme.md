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
| 5 | **Fragments manquants** (DAFNE C) | compte absolu 0-2 (à k=2, L4 ne laissait qu'**un** fragment) | plafond **relatif** automatique `round(0.15·k)`, surchargeable par `--missing-max` |
| 6 | **Dataset publié** | `balanced/` | hiérarchie `<grille_croissance>/<k>/<palier>` — cf. section suivante ; l'ancien devient **`LEGACY_DO_NOT_USE/`** |

## Datasets publiés — [🤗 icimathieu/lego2hero-100mosaics](https://huggingface.co/datasets/icimathieu/lego2hero-100mosaics)

**8 datasets appariés** : mêmes 5 000 peintures wikiart, mêmes seeds, seule la
config change.

```
poly_balanced/     tuiles variables + BFS équilibré      ← la voie principale
mono_compact/      tuiles 1×1 + croissance compacte      ← « ultracompact » (peu de sommets)
  └─ k02 · k03 · k05 · k10-15        nombre de fragments (DAFNE A)
       └─ L0_explode … L4_strong     1 tar.gz + 1 gnn_meta.json par palier

LEGACY_DO_NOT_USE/   ancien `balanced/` (v1) — ⛔ NE PAS ENTRAÎNER DESSUS
```

- **`LEGACY_DO_NOT_USE/`** = le dataset de juin, **conservé pour archive
  uniquement**. Il a été produit avec l'encodage bugué (perte d'aire **21 %
  médiane**) : `polygon_n_canonical` et `side_features` y décrivent des formes
  fausses. Son successeur corrigé est **`poly_balanced/k10-15/`**.
- **`mono_compact/k02` et `k03`** sont volontairement **dégénérés** (fragments
  quasi rectangulaires, signal d'appariement géométrique mort) : ils sont inclus
  comme **bornes basses d'ablation**, pas comme configs d'entraînement.
- À **k=2** la prédiction de lien est dégénérée par construction : une seule arête
  possible, toujours présente.

### `n_max` re-mesurés (v2, 5 000 mosaïques par case)

| dataset | L0 | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|---:|
| `poly_balanced/k10-15` | 224 | 224 | 224 | 169 | 223 |
| `poly_balanced/k05` | 331 | 331 | 331 | 227 | 279 |
| `poly_balanced/k03` | **344** | **344** | **344** | 249 | 279 |
| `poly_balanced/k02` | 208 | 208 | 208 | 157 | 206 |
| `mono_compact/k10-15` | 63 | 63 | 63 | 54 | 78 |
| `mono_compact/k05` | 48 | 48 | 48 | 40 | 54 |
| `mono_compact/k03` | 9 | 9 | 9 | 40 | 30 |
| `mono_compact/k02` | **5** | **5** | **5** | 33 | 38 |

**Pour un `n_max` joint** (contrat dense transférable) : **344** couvre les 8
datasets, et les fragments **RePAIR réels** plafonnent à 291 après reco-B ε=1 % —
donc `collate.py --n-max 344` suffit pour LEGO **et** réel.

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

| palier | placement | rotation | dégradation |
|---|---|---|---|
| **L0_explode** | vue éclatée (positions relatives gardées) | non | aucune |
| **L1_translation** | scatter aléatoire | non | aucune |
| **L2_rotation** | scatter | oui | aucune |
| **L3_light** | scatter | oui | légère (érosion 1-2 px, 0-1 trou) |
| **L4_strong** | scatter | oui | forte (érosion 2-6 px, trous, manquants) |

<sub>`n_max` par palier : cf. le tableau « `n_max` re-mesurés » plus haut.</sub>

### Les 5 paramètres de difficulté DAFNE

**DAFNE** (*Digital Anastylosis of Frescoes challeNgE*, Univ. Pavie, PRL 2020) est
un benchmark **synthétique** de réassemblage de fresques 2D.

| DAFNE | ce que c'est | chez nous | statut |
|---|---|---|---|
| **A** — nb de fragments | plus de fragments = plus dur | `--n-frag-min/max` — historiquement figé à **10-15** ; configs **k=2/3/5** ajoutées (cf. tableau suivant) | ✅ |
| **B** — distribution de découpe | tailles/formes des coupes | l'axe `--frag-distribution` : `balanced`/`compact`/`ultracompact` ✅ — `voronoi`/`clusters` (le vrai durcissement DAFNE B) ❌ | 🚧 partiel |
| **C** — % de fragments manquants | pièces absentes de l'entrée | `--missing-min/max`, palier **L4**. Plafond automatique à **~15 % de k** (`round(0.15 · n_frag_max)`, surcharge par `--missing-max`) : notre `missing` est un compte absolu, DAFNE C un pourcentage | ✅ |
| **D** — fragments parasites | pièces d'un autre objet à rejeter | — | ❌ |
| **E** — érosion | bords rongés | `--erode-px-min/max`, **L3** = 1-2 px, **L4** = 2-6 px | ✅ |

En plus des curseurs DAFNE, on ajoute les **trous internes** (`--holes-min/max`,
aire totale plafonnée à 10 % du fragment) et l'axe **pose** (L0→L2).

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

### Modes de fragmentation (`--frag-distribution`)

Axe **orthogonal** aux paliers de dégradation (sortie `output/<mode>/<palier>/`).

| mode | règle de découpe | forme des fragments | statut |
|---|---|---|---|
| **balanced** | seeds farthest-point + BFS à priorité (le plus petit fragment grandit d'abord) | blobs d'aire ~égale | ✅ défaut |
| **compact** | croissance à **périmètre minimal** : chaque fragment absorbe la pièce-frontière qui maximise le remplissage de sa bbox | bords + lisses, médiane sommets −15 % | ✅ |
| **ultracompact** | ⚠️ **pas un algo différent** : c'est `compact` appliqué à une mosaïque forgée en **`--mode mono`** (tuiles 1×1). « L'ultra » vient de l'ENTRÉE, pas de la coupe | rectangulaires, **peu de sommets** (coupes droites sur grille régulière) | ✅ |
| **voronoi** | k graines dispersées, pièce → graine la + proche | tailles **inégales**, bords irréguliers (**DAFNE B**) | 🚧 |
| **clusters** | graines agglutinées | zones très/peu fragmentées (**DAFNE B**) | 🚧 |

### Encodage géométrique — reco-B (`polygon_n`)

Le contour brut d'un fragment (`polygon_raw`, sorti de `cv2.findContours`) est
simplifié en `polygon_n`, qui est ce que voit le GNN. Deux garanties :

- **`polygon_n ⊆ polygon_raw`** — on ne *gagne* jamais d'aire, on en perd (« comme
  si les coins étaient ébréchés »). Les fragments réassemblés ne se chevauchent
  donc **jamais** : ils laissent des jours. C'est le régime physiquement
  réalisable.
- **Budget sur la PERTE D'AIRE** (`--max-area-loss`, défaut **ε = 1 %**).

L'algorithme est **Visvalingam-Whyatt** (1993) *restreint aux sommets convexes* :
on retire itérativement le sommet dont le triangle (préc., lui, suiv.) a la plus
petite aire, tant que la perte cumulée reste ≤ ε. Seuls les **convexes** sont
retirables — retirer un sommet **reflex** pontifierait une concavité et
**re-gagnerait** de l'aire. Les sommets ~colinéaires ont un triangle d'aire nulle
et sont donc retirés **gratuitement** (c'est le gros du bruit de contour LEGO).
Le plancher dur de `n` reste **#reflex**, atteint automatiquement.

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

## Datasets réels & benchmarks (réassemblage vs style)

| dataset | nature | contenu | rôle pour nous |
|---|---|---|---|
| **RePAIR** (NeurIPS 2024, [site](https://repairproject.github.io/RePAIR_dataset/), [Zenodo](https://zenodo.org/records/13993089)) | **RÉEL** — fresques de Pompéi bombardées en 39-45 | **16 000 fragments** au total ; GT (pose annotée par des archéologues, des années de terrain) sur **~1 000 pièces** ; benchmark 2D : 121 objets / 957 fragments (97 train / 24 test) ; 2D **+ 3D** | **notre cible de transfert + source de métriques** (Q_pos, RMSE rot/trans, F1 mating graph). Fragments 2D déjà en local |
| **DAFNE** (PRL 2020, [dataset](https://vision.unipv.it/DAFchallenge/DAFNE_dataset/)) | SYNTHÉTIQUE 2D | 62 fresques × 18 configs ; 5 paramètres de difficulté A-E | notre **gabarit de dégradation** (cf. tableau DAFNE) |
| **ReassembleNet semi-synth** ([Drive](https://drive.google.com/drive/folders/1tflCUoct63Zhzt8dWs37vtfnphOqgReB?usp=sharing)) | semi-synthétique | 5 000 mosaïques / 45 834 fragments (80/20) | calibre notre échelle (25 000 instances ✓) |
| **PairingNet data** | synthétique + petit set réel | 390 images → 8 196 fragments ; réel : 34 images / 320 fragments | baseline link-prediction |
| **POMPAAF** ([arXiv:2501.00836](https://arxiv.org/abs/2501.00836)) · **CLEOPATRA** (JAIHC 2023) | réels | fragments de fresques pompéiennes | ⚠️ **classification de STYLE, PAS réassemblage** — utiles au VLM, pas au GNN. Ne pas les citer comme benchmarks de reconstruction |

