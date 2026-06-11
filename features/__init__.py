"""Module post-YOLO / pré-GNN — PARTAGÉ entre forge synthétique et chaîne réelle.

- fragment_features : (masque, image) → polygon_n (reco B + PCA), side_features, gnn_input.
- collate          : dataset → n_max → padding + masque de validité → entrée GNN à dim fixe.
"""
from .fragment_features import (
    extract_polygon,
    resample_reflex_aware,
    pca_canonical_rotation,
    compute_fragment_features,
)

__all__ = [
    "extract_polygon",
    "resample_reflex_aware",
    "pca_canonical_rotation",
    "compute_fragment_features",
]
