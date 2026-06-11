# lego2hero

Chaîne de génération de données pour la **reconstitution d'objets fragmentés** :
`forge_LAR_2mosaic` (image → mosaïque LEGO) → `mosaic2fragments` (mosaïque →
fragments + graphes) → `features` (post-YOLO / pré-GNN, partagé synthétique↔réel).

## Arborescence

```
lego2hero/
├── forge_LAR_2mosaic/      # image → mosaïque LEGO (canvas_mosaic.png + piece_grid.json)
│   ├── mosaic.py           # quantif Lab → packing glouton → rendu tuiles + joints
│   ├── cli.py · batch.py   # 1 image / dossier d'images
│   ├── wikiart.py          # source d'images : huggan/wikiart en streaming
│   └── palette.py          # 82 couleurs LEGO (bricklink, gris filtrés)
├── mosaic2fragments/       # mosaïque → fragments → YOLO-Seg + graphes GNN
│   ├── forge_dataset.py    # pipeline principal (joints → pièces → fragments → GT)
│   └── batch.py · visualize.py
├── features/               # post-YOLO / pré-GNN — PARTAGÉ synthétique↔réel
│   ├── fragment_features.py   # reco-B + PCA canonical + side_features + gnn_input
│   └── collate.py             # pad n_max + masque → gnn_ready.npz
├── tools/
│   └── hf_upload.py        # upload d'un dossier vers le HuggingFace Hub (dataset)
├── LICENSE                 # MIT
└── output/                 # ⚠️ GÉNÉRÉ — gitignoré, hors du dépôt (données lourdes)
    └── <dataset>/mosaic_<id>/   ← structure détaillée ci-dessous
```

## Structure d'un sous-dossier de mosaïque

Une mosaïque = un dossier `output/<dataset>/mosaic_<id>/`. C'est l'**entrée** des
étapes aval (YOLO / GNN / VLM).

| Fichier | Rôle |
|---|---|
| `target.png` | la mosaïque **complète** (objet reconstitué) |
| `source.png` | les fragments **éclatés** sur fond blanc (≈ photo d'entrée) |
| `source_yolo.txt` | labels **YOLO-Seg** (`classe x1 y1 … xn yn`, normalisés `[0,1]`) |
| `source_yolo_viz.png` | overlay debug des labels sur `source.png` |
| `pieces.json` | debug : pièces LEGO détectées |
| **`graph_fragments.json`** | **ENTRÉE GNN** : nœuds (features/fragment), zéro arête, sans `target_info` |
| **`graph_complete.json`** | **CIBLE GNN** : mêmes nœuds + `target_info` (leak) + arêtes (mating graph) |
| `gt_layout.json` | GT de reconstruction (polygones parfait/dégradé + pose) |
| `gnn_ready.npz` | entrée GNN à **dimension fixe** (pad `n_max` + masque de validité) |
| `degradation.md` | rapport de dégradation (tout à 0 = clean) |
| `fragments/frag_XX.png` | crop alpha par fragment |

Au niveau dataset : **`gnn_meta.json`** (`n_max`, noms de features).

### Schéma d'un nœud (`graph_fragments.json`)

| Bloc | Contenu | Taille |
|---|---|---|
| `node_id` | identifiant stable (référencé par les arêtes via `src`/`dst`) | — |
| `gnn_input` | `[area, perimeter, R, G, B, bbox_w, bbox_h]` — domaine-agnostique | 7 |
| `polygon_n_canonical` | contour en repère **PCA canonique** (invariant en rotation) | `n_sides`×2 |
| `side_features` | par côté `[length, angle, R, G, B]` | `n_sides`×5 |

`n_sides` est **variable** par nœud → côté GNN : pad à `n_max` + masque (cf. `gnn_ready.npz`).

⚠️ **`target_info`** (centroid, polygon absolu, polygon_raw, pose ; dans `graph_complete`)
est de la **vérité terrain** → ne **jamais** la donner en entrée au GNN.

## Arêtes (cible) — `graph_complete.json`
`edges` : `src`/`dst` (node_id) + `features` `[shared_length, mean_angle, n_segments]`
+ `src_side_idx`/`dst_side_idx`. **Tâche première = link prediction** (prédire ces
arêtes depuis `graph_fragments`).
