# lego2hero

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
              │  joints → pièces → FRAGMENTATION → reco-B       │
              │                    → placement → dégradation     │
              │                                                  │
              │  --frag-distribution  (motif de découpe) :       │
              │    balanced   compact   voronoi │ clusters       │
              │    (actuel)   (~rect.)   ( DAFNE B, à venir )     │
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
   │  masque → reco-B + PCA canonical + gnn_input → collate (pad n_max+masque) │
   └────────────────────────────────────┬──────────────────────────────────────┘
                                         ▼
                                GNN  (réassemblage)
```

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

### Modes de fragmentation (`--frag-distribution`)

Axe **orthogonal** aux paliers de dégradation (sortie `output/<mode>/<palier>/`).

| mode | règle de découpe | forme des fragments | statut |
|---|---|---|---|
| **balanced** | BFS équilibré (le plus petit fragment grandit d'abord) | blobs de taille ~égale | ✅ défaut |
| **compact** | croissance à **périmètre minimal** (remplit les concavités) | bords + lisses, médiane sommets −15 % (n_max inchangé) | ✅ |
| **voronoi** | k graines dispersées, pièce → graine la + proche | tailles **inégales**, bords irréguliers (**DAFNE B**) | 🚧 |
| **clusters** | graines agglutinées | zones très/peu fragmentées (**DAFNE B**) | 🚧 |
| **ultracompact** | forge **tuiles 1×1** (`--mode mono`) **+** compact | rectangulaires, **peu de sommets** (coupes droites sur grille régulière) | 🚧 plan |

`balanced`↔`compact` = facile/régulier ; `voronoi`/`clusters` = durcissement ;
`ultracompact` = le levier réel sur le nombre de sommets (il faut des mosaïques
1×1, donc il touche aussi la forge).

## Arborescence

```
lego2hero/
├── scripts/
│   ├── forge_LAR_2mosaic/      # image → mosaïque LEGO (canvas_mosaic.png + piece_grid.json)
│   │   ├── mosaic.py           # quantif Lab → packing glouton → rendu tuiles + joints
│   │   ├── cli.py · batch.py   # 1 image / dossier d'images
│   │   ├── wikiart.py          # source d'images : huggan/wikiart en streaming
│   │   └── palette.py          # 82 couleurs LEGO (bricklink, gris filtrés)
│   ├── mosaic2fragments/       # mosaïque → fragments → YOLO-Seg + graphes GNN
│   │   ├── forge_dataset.py    # pipeline principal (joints → pièces → fragments → GT)
│   │   ├── curriculum.py       # driver des paliers L0–L4 × --frag-distribution → output/<dist>/<palier>/
│   │   └── batch.py · visualize.py
│   ├── features/               # post-YOLO / pré-GNN — PARTAGÉ synthétique↔réel
│   │   ├── fragment_features.py   # reco-B + PCA canonical + side_features + gnn_input
│   │   └── collate.py             # pad n_max + masque → gnn_ready.npz
│   └── tools/
│       └── hf_upload.py        # upload d'un dossier vers le HuggingFace Hub (dataset)
├── todo.md                 # statut concis (historique + à faire)
├── LICENSE                 # MIT
└── output/                 # ⚠️ GÉNÉRÉ — gitignoré, hors du dépôt (données lourdes)
    └── <frag-distribution>/<palier>/mosaic_<id>/   ← structure détaillée ci-dessous
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

`n_sides` est **variable** par nœud → côté GNN : pad à `n_max` + masque (cf. `gnn_ready.npz`). Nécessaire pour GNN (dimension fixe == nombre de sommets fixe) SAUF à rendre GNN sommets-agnostique...

## Arêtes (cible) — `graph_complete.json`
`edges` : `src`/`dst` (node_id) + `features` `[shared_length, mean_angle, n_segments]`
+ `src_side_idx`/`dst_side_idx`.
