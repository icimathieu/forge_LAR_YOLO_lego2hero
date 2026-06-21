"""Préparation des entrées GNN à dimension fixe (post-YOLO / pré-GNN).

`polygon_n_canonical` et `side_features` ont une longueur VARIABLE par nœud
(`n_sides`). Un encodeur de nœud GNN exige une dimension F fixe. On résout par
**pad à n_max + masque de validité** (option A) : on garde la forme honnête et on
complète par des slots hors-bande, marqués invalides → le GNN les ignore.

`n_max` est calculé sur l'ENSEMBLE du dataset (tous les graphes), pas par
mosaïque. Numpy pur, sans dépendance torch (le wrap torch_geometric se fait côté
GNN, ex. : `x = concat([gnn_input, poly.flatten(), sides.flatten()])`, masque
fourni séparément).

Usage :
    nodes_per_graph = [json.load(open(p))['nodes'] for p in graph_fragments_paths]
    n_max = compute_n_max(nodes_per_graph)
    padded = [[pad_node(n, n_max) for n in nodes] for nodes in nodes_per_graph]
"""
import argparse
import glob
import json
import os

import numpy as np


def compute_n_max(nodes_per_graph):
    """max(n_sides) sur tous les nœuds de tous les graphes (cible de padding)."""
    return max((len(n["polygon_n_canonical"])
                for nodes in nodes_per_graph for n in nodes), default=0)


def pad_node(node, n_max):
    """Renvoie les features du nœud à dimension fixe + masque de validité.

    - gnn_input           : (7,)        inchangé (déjà fixe)
    - polygon_n_canonical : (n_max, 2)  paddé à 0 au-delà de n_sides
    - side_features       : (n_max, 5)  paddé à 0 au-delà de n_sides
    - valid_mask          : (n_max,)    1.0 pour les vrais sommets, 0.0 pour le padding
    """
    poly = np.asarray(node["polygon_n_canonical"], dtype=np.float32)
    sides = np.asarray(node["side_features"], dtype=np.float32)
    n = len(poly)
    if n > n_max:
        raise ValueError(f"n_sides={n} > n_max={n_max} : recalculer n_max sur tout le dataset")
    poly_p = np.zeros((n_max, 2), dtype=np.float32)
    poly_p[:n] = poly
    sides_p = np.zeros((n_max, 5), dtype=np.float32)
    sides_p[:n] = sides
    mask = np.zeros(n_max, dtype=np.float32)
    mask[:n] = 1.0
    return {
        "gnn_input": np.asarray(node["gnn_input"], dtype=np.float32),
        "polygon_n_canonical": poly_p,
        "side_features": sides_p,
        "valid_mask": mask,
    }


def write_dataset_readme(dataset_dir, n_max, n_graphs):
    """README d'arborescence à la racine du dataset (auto-généré par collate)."""
    name = os.path.basename(os.path.normpath(dataset_dir))
    txt = f"""# `{name}/` — mosaïques LEGO fragmentées

## Fichiers par mosaïque (`mosaic_<id>/`)
| Fichier | Rôle |
|---|---|
| `target.png` | mosaïque **complète** = cible de reconstruction / contexte VLM |
| `source.png` | fragments **éclatés** sur fond blanc = **entrée YOLO** (et entrée VLM) |
| `source_yolo.txt` | **vérité terrain YOLO-Seg** : 1 polygone/fragment, normalisé `[0,1]` (entraîne YOLO avec `source.png`) |
| `source_yolo_viz.png` | overlay debug des labels (inspection humaine) — *absent par défaut* (opt-in `--debug`) |
| `pieces.json` | debug pur (pièces LEGO détectées) ; **n'entraîne PAS le YOLO** (c'est `source_yolo.txt`), lu nulle part — *absent par défaut* (opt-in `--debug`) |
| `graph_fragments.json` | **ENTRÉE GNN** : nœuds (features/fragment), zéro arête, sans `target_info` |
| `graph_complete.json` | **CIBLE GNN** : mêmes nœuds + `target_info` (leak) + arêtes (mating graph) |
| `gt_layout.json` | **GT de RECONSTRUCTION** (réponse finale) : footprint exact + pose `(x,y,rot)` de chaque fragment dans `target.png`. Sert à **scorer l'assemblage** (IoU/Q_pos) APRÈS la tête de pose du GNN / le VLM qui place les fragments |
| `gnn_ready.npz` | entrée GNN à **dimension fixe** (pad `n_max={n_max}` + masque) |
| `degradation.md` | rapport de dégradation (0 si clean) |
| `fragments/frag_XX.png` | crop alpha par fragment = **entrée VLM** |

Au niveau dataset : **`gnn_meta.json`** (`n_max={n_max}`, noms de features).

## Schéma d'un nœud (`graph_fragments.json`)
| Bloc | Contenu | Taille |
|---|---|---|
| `node_id` | identifiant stable (référencé par les arêtes) | — |
| `gnn_input` | `[area, perimeter, R, G, B, bbox_w, bbox_h]` — domaine-agnostique | 7 |
| `polygon_n_canonical` | contour en repère **PCA canonique** (invariant rotation) | `n_sides`×2 |
| `side_features` | par côté `[length, angle, R, G, B]` | `n_sides`×5 |

`n_sides` **variable** par nœud → paddé à `n_max={n_max}` + masque dans `gnn_ready.npz`.
⚠️ **`target_info`** (dans `graph_complete`) = vérité terrain → ne **jamais** donner en entrée.

_Régénéré par `scripts/mosaic2fragments/{{batch,curriculum}}.py` + `scripts/features/collate.py`._
"""
    open(os.path.join(dataset_dir, "README.md"), "w", encoding="utf-8").write(txt)


def build_dataset(dataset_dir, out_name="gnn_ready.npz"):
    """Étape COLLATE (post-YOLO / pré-GNN) : scanne tout le dataset, calcule n_max,
    pad+masque chaque nœud, et écrit par mosaïque un `gnn_ready.npz` (entrée GNN à
    dimension fixe) + un `gnn_meta.json` (n_max + schéma). Numpy pur : les amis
    GNN chargent et wrappent en torch_geometric.Data eux-mêmes (les arêtes/cibles
    restent dans graph_complete.json)."""
    mosaics = sorted(glob.glob(os.path.join(dataset_dir, "mosaic_*")))
    nodes_per = [json.load(open(os.path.join(m, "graph_fragments.json")))["nodes"]
                 for m in mosaics]
    n_max = compute_n_max(nodes_per)
    for m, nodes in zip(mosaics, nodes_per):
        if not nodes:
            continue
        padded = [pad_node(n, n_max) for n in nodes]
        np.savez(
            os.path.join(m, out_name),
            gnn_input=np.stack([p["gnn_input"] for p in padded]),               # (N,7)
            polygon_n_canonical=np.stack([p["polygon_n_canonical"] for p in padded]),  # (N,n_max,2)
            side_features=np.stack([p["side_features"] for p in padded]),       # (N,n_max,5)
            valid_mask=np.stack([p["valid_mask"] for p in padded]),             # (N,n_max)
            node_id=np.array([n["node_id"] for n in nodes], dtype=np.int64),    # (N,)
        )
    sample = json.load(open(os.path.join(mosaics[0], "graph_fragments.json")))
    meta = {
        "n_max": n_max,
        "n_graphs": len(mosaics),
        "gnn_input_dim": len(sample["gnn_input_feature_names"]),
        "gnn_input_feature_names": sample["gnn_input_feature_names"],
        "side_feature_names": sample["side_feature_names"],
        "note": ("Entrées GNN à dimension fixe : gnn_ready.npz par mosaïque "
                 "(gnn_input + polygon_n_canonical/side_features paddés à n_max + valid_mask). "
                 "Cibles/arêtes : graph_complete.json. Régénérable via "
                 "`python3 features/collate.py --dataset <dir>`."),
    }
    json.dump(meta, open(os.path.join(dataset_dir, "gnn_meta.json"), "w"),
              indent=2, ensure_ascii=False)
    write_dataset_readme(dataset_dir, n_max, len(mosaics))
    return n_max


def main():
    p = argparse.ArgumentParser(description="Collate post-YOLO → tenseurs GNN à dim fixe")
    p.add_argument("--dataset", required=True, help="dossier dataset (contient mosaic_*/)")
    p.add_argument("--out-name", default="gnn_ready.npz")
    args = p.parse_args()
    n_max = build_dataset(args.dataset, args.out_name)
    print(f"[collate] {args.dataset} : n_max={n_max} → {args.out_name} par mosaïque + gnn_meta.json")


if __name__ == "__main__":
    main()
