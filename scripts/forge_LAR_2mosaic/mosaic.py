"""Forge de mosaïques LEGO « variable-tile/plate/brick » — clean-room, Python pur.

Remplace lego-art-remix.com pour notre usage : image quelconque → mosaïque LEGO
à pièces plates de tailles variables (1×1 … 4×10) avec joints gris, PLUS la
vérité-terrain exacte des pièces (`piece_grid.json`).

On NE copie aucun code de l'outil source (GPL-3.0) : on réimplémente la logique
publique (quantif vers palette + packing glouton du plus grand rectangle). Les
JEUX DE DIMENSIONS et n° de pièce BrickLink par mode sont des FAITS extraits de
`algo.js` (TILE/PLATE/BRICK_DIMENSIONS_TO_PART_ID). Sortie au FORMAT attendu par
`mosaic2fragments/forge_dataset.py` (carré, ≥40 px/stud, joints ~RGB(136), palette sans gris).

Pipeline :
  1. crop carré + downsample area-average à grid×grid (1 couleur/cellule)
  2. quantif nearest-color en espace Lab vers la palette LEGO (sans gris)
  3. packing glouton : plus grand rectangle MONOCHROME d'abord (aire ↓, comme le
     site) ; reste comblé en 1×1
  4. rendu : pièces plates pleines + joints gris ; export PNG + piece_grid.json

Déterministe (aucun aléa) : même image + mêmes params → même sortie.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from palette import LEGO_PALETTE, JOINT_COLOR, hex_of


# ============================================================
# Jeux de dimensions LEGO par mode (studs) + n° de pièce BrickLink.
# Faits extraits de lego-art-remix/app/js/algo.js (clean-room).
# ============================================================

TILE_DIMS = {                                       # "Variable Tile" (plat, lisse)
    (1, 1): "3070b", (1, 2): "3069b", (1, 3): "63864", (1, 4): "2431",
    (1, 6): "6636", (1, 8): "4162", (2, 2): "3068b", (2, 3): "26603",
    (2, 4): "87079", (2, 6): "69729",
}
PLATE_DIMS = {                                      # "Variable Plate" (le + de tailles)
    (1, 1): "3024", (1, 2): "3023", (1, 3): "3623", (1, 4): "3710",
    (1, 6): "3666", (1, 8): "3460", (2, 2): "3022", (2, 3): "3021",
    (2, 4): "3020", (2, 6): "3795", (2, 8): "3034", (4, 4): "3031",
    (4, 8): "3035", (4, 10): "3030",
}
BRICK_DIMS = {                                      # "Variable Brick"
    (1, 1): "3005", (1, 2): "3004", (1, 3): "3622", (1, 4): "3010",
    (1, 6): "3009", (1, 8): "3008", (2, 2): "3003", (2, 3): "3002",
    (2, 4): "3001", (2, 6): "2456", (2, 8): "3007",
}
MODE_DIMS = {"tile": TILE_DIMS, "plate": PLATE_DIMS, "brick": BRICK_DIMS}
# Le site DÉCOCHE par défaut les plus grosses plates (UI : DEFAULT_DISABLED_DEPTH_PLATES).
DEFAULT_DISABLED = {(4, 8), (4, 10)}


def build_allowed_parts(mode="tile", big_plates=False):
    """Liste (w, h, part_id) pour le mode donné, 2 orientations, triée comme le
    site : aire décroissante puis plus petite 1re dimension d'abord (déterministe).
    `big_plates=True` réactive 4×8/4×10 (décochées par défaut sur le site)."""
    dims = dict(MODE_DIMS[mode])
    if not big_plates:
        for d in DEFAULT_DISABLED:
            dims.pop(d, None)
    seen, parts = set(), []
    for (a, b), pid in dims.items():
        for w, h in {(a, b), (b, a)}:
            if (w, h) not in seen:
                seen.add((w, h))
                parts.append((w, h, pid))
    parts.sort(key=lambda p: (p[0] * p[1], -p[0]), reverse=True)
    return parts


# ============================================================
# Espace couleur : sRGB [0,255] -> CIE Lab (D65)
# ============================================================

def srgb_to_lab(rgb):
    """rgb : (..., 3) en [0,255] -> Lab (..., 3). Pur numpy (pas de dépendance)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    x /= 0.95047; z /= 1.08883                      # point blanc D65 (Yn = 1)
    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)
    fx, fy, fz = f(x), f(y), f(z)
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


# ============================================================
# Étapes 1-2 : image -> grille d'indices de palette
# ============================================================

def image_to_color_grid(image, grid_w, grid_h):
    """Crop carré centré + downsample area-average -> (grid_h, grid_w, 3) uint8.
    `image` : chemin OU objet PIL.Image (utile pour le streaming wikiart)."""
    im = (image if isinstance(image, Image.Image) else Image.open(image)).convert("RGB")
    w, h = im.size
    s = min(w, h)                                   # crop carré centré
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((grid_w, grid_h), Image.BOX)     # moyenne par cellule
    return np.asarray(im, dtype=np.uint8)


def quantize_to_palette(color_grid, palette=LEGO_PALETTE):
    """(H,W,3) -> (H,W) indices de la palette (nearest en Lab)."""
    pal = np.array(palette, dtype=np.float64)
    pal_lab = srgb_to_lab(pal)                       # (P,3)
    cells_lab = srgb_to_lab(color_grid.reshape(-1, 3))  # (H*W,3)
    d = ((cells_lab[:, None, :] - pal_lab[None, :, :]) ** 2).sum(-1)  # (H*W,P)
    idx = d.argmin(1).reshape(color_grid.shape[:2])
    return idx.astype(np.int32)


# ============================================================
# Étape 3 : packing glouton du plus grand rectangle monochrome
# ============================================================

def pack_variable_tiles(idx_grid, allowed_parts):
    """Glouton : pose le plus grand rectangle MONOCHROME possible, sans
    chevauchement, en balayant en row-major ; reste comblé par des 1×1.

    `allowed_parts` : liste (w, h, part_id) déjà triée (aire ↓). Retourne la
    liste des pièces (col, row, width, height, color_idx, part_id). Chaque pièce
    est mono-couleur (une pièce LEGO = une couleur) ; deux pièces voisines de même
    couleur restent séparées par un joint -> deux pièces distinctes en aval.
    """
    H, W = idx_grid.shape
    occupied = np.zeros((H, W), dtype=bool)
    pieces = []
    for pw, ph, pid in allowed_parts:
        if pw > W or ph > H:
            continue
        for r in range(H - ph + 1):
            for c in range(W - pw + 1):
                if occupied[r:r + ph, c:c + pw].any():
                    continue
                block = idx_grid[r:r + ph, c:c + pw]
                if (block == block[0, 0]).all():
                    occupied[r:r + ph, c:c + pw] = True
                    pieces.append((c, r, pw, ph, int(block[0, 0]), pid))
    # garde-fou : toute cellule restante -> 1×1 (ne devrait pas arriver)
    unit_pid = next((pid for w, h, pid in allowed_parts if (w, h) == (1, 1)), None)
    for r, c in np.argwhere(~occupied):
        pieces.append((int(c), int(r), 1, 1, int(idx_grid[r, c]), unit_pid))
    return pieces


# ============================================================
# Étape 4 : rendu PNG (pièces plates + joints gris)
# ============================================================

def render_mosaic(pieces, grid_w, grid_h, stud_size, joint_px, palette=LEGO_PALETTE):
    """Fond gris (joints) + chaque pièce peinte en retrait de joint_px/2 sur
    chaque bord -> joint de joint_px px entre pièces voisines. Aucun anti-alias :
    pixels de pièce = couleur palette exacte, pixels de joint = RGB(136) exact."""
    H_px, W_px = grid_h * stud_size, grid_w * stud_size
    img = np.empty((H_px, W_px, 3), dtype=np.uint8)
    img[:] = JOINT_COLOR
    li, ri = joint_px - joint_px // 2, joint_px // 2   # ceil / floor -> joint exact
    for c, r, w, h, ci, _pid in pieces:
        x0, y0 = c * stud_size + li, r * stud_size + li
        x1, y1 = (c + w) * stud_size - ri, (r + h) * stud_size - ri
        img[y0:y1, x0:x1] = palette[ci]
    return Image.fromarray(img, "RGB")


# ============================================================
# Orchestration
# ============================================================

def forge_mosaic(image, grid_w=96, grid_h=96, stud_size=40, joint_px=None,
                 mode="tile", big_plates=False, palette=LEGO_PALETTE, source_name=None):
    """image (chemin ou PIL.Image) -> (PIL.Image mosaïque, dict piece_grid). mode: tile|plate|brick."""
    if joint_px is None:
        joint_px = max(2, round(stud_size * 0.075))    # ~3 px à 40 px/stud
    allowed = build_allowed_parts(mode, big_plates)
    color_grid = image_to_color_grid(image, grid_w, grid_h)
    idx_grid = quantize_to_palette(color_grid, palette)
    pieces = pack_variable_tiles(idx_grid, allowed)
    img = render_mosaic(pieces, grid_w, grid_h, stud_size, joint_px, palette)
    piece_grid = {
        "grid_width": grid_w,
        "grid_height": grid_h,
        "stud_size_px": stud_size,
        "joint_width_px": joint_px,
        "joint_color_rgb": list(JOINT_COLOR),
        "mode": mode,
        "max_piece": list(max((p[:2] for p in allowed), key=lambda d: d[0] * d[1])),
        "source_image": source_name or (Path(image).name if isinstance(image, (str, Path)) else "image"),
        "n_pieces": len(pieces),
        "n_cells": grid_w * grid_h,
        "pieces": [
            {"col": c, "row": r, "width": w, "height": h,
             "part_id": pid, "color_rgb": list(palette[ci]),
             "color_hex": hex_of(palette[ci])}
            for c, r, w, h, ci, pid in pieces
        ],
    }
    return img, piece_grid


def forge_to_files(image_path, out_png, out_json, **kw):
    img, grid = forge_mosaic(image_path, **kw)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    Path(out_json).write_text(json.dumps(grid, indent=2))
    return grid
