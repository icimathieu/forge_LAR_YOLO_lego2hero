#!/usr/bin/env python3
"""Forge a LEGO mosaic dataset for YOLO + GNN training.

Inputs: a canvas_mosaic.png produced by lego-art-remix (variable-tile option).
Outputs (per mosaic, in <out_dir>/mosaic_XXX/):
    target.png             — copy of the input mosaic
    source.png             — fragments exploded on a white canvas (YOLO-Seg input)
    source_yolo.txt        — YOLO-Seg polygons, one line per fragment
    pieces.json            — detected LEGO pieces (grid bbox + sampled color)
    graph_complete.json    — N nodes (fragment features) + adjacency edges (target)
    graph_fragments.json   — same N nodes, zero edges (GNN input)
    fragments/frag_XX.png  — per-fragment alpha crop in target frame

Pipeline:
    1. Joint detection on the rendered mosaic → LEGO piece layout
    2. Union-find on cells → pieces (rectangles on the stud grid)
    3. Piece adjacency graph → region-growing fragmentation (k ∈ [10,15])
    4. Per-fragment polygon → resampled to n=16 equidistant points
    5. Per-fragment features (global) + per-side features
    6. Fragment adjacency = boundary segments shared between fragments
    7. Random placement of rotated fragments on a white canvas (axes-aligned + ±5°)
    8. Polygon export in YOLO-Seg format (normalized)
"""

import argparse
import collections
import heapq
import json
import os
import random
import sys
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from PIL import Image

# Extraction de features = module PARTAGÉ post-YOLO/pré-GNN (features/), importé
# depuis la racine du repo → forge synthétique et future chaîne YOLO utilisent le
# MÊME code (alignement de schéma garanti).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.fragment_features import (  # noqa: E402
    _polygon_signed_area, extract_polygon, resample_reflex_aware,
    pca_canonical_rotation, compute_fragment_features,
)


# ============================================================
# 1. Joint detection
# ============================================================

def is_joint_pixel(px, target=136, tol=35):
    r, g, b = int(px[0]), int(px[1]), int(px[2])
    if abs(r - target) > tol or abs(g - target) > tol or abs(b - target) > tol:
        return False
    if max(r, g, b) - min(r, g, b) > 25:
        return False
    return True


def _cell_is_gray(img, i, j, stud_size):
    """True if the interior of cell (i, j) is itself joint-gray (= a gray piece)."""
    cy = min(j * stud_size + stud_size // 2, img.shape[0] - 1)
    cx = min(i * stud_size + stud_size // 2, img.shape[1] - 1)
    return is_joint_pixel(img[cy, cx])


def detect_joints(img, stud_size, tolerate_gray=True):
    """Return joint_v[i, j] (vertical joints) and joint_h[i, j] (horizontal joints).

    joint_v[i, j] = True iff there is a joint between cell (i, j) and (i+1, j).
    joint_h[i, j] = True iff there is a joint between cell (i, j) and (i, j+1).
    Convention: i = column index (x-axis), j = row index (y-axis).

    With tolerate_gray=True, a gray boundary is NOT a joint when BOTH adjacent
    cell interiors are themselves gray — we are inside a gray LEGO piece, not on
    a real joint. This lets gray pieces stay in the palette. The only
    unresolvable case is two *different* gray pieces sharing a gray joint (rare,
    accepted). For mosaics generated without gray pieces this is a strict no-op.
    """
    H, W = img.shape[:2]
    n_cols = W // stud_size
    n_rows = H // stud_size

    joint_v = np.zeros((n_cols - 1, n_rows), dtype=bool)
    joint_h = np.zeros((n_cols, n_rows - 1), dtype=bool)

    for i in range(n_cols - 1):
        x = (i + 1) * stud_size
        for j in range(n_rows):
            y_mid = j * stud_size + stud_size // 2
            samples = [
                is_joint_pixel(img[j * stud_size + stud_size // 4, x]),
                is_joint_pixel(img[y_mid, x]),
                is_joint_pixel(img[j * stud_size + 3 * stud_size // 4, x]),
            ]
            is_joint = sum(samples) >= 2
            if (is_joint and tolerate_gray
                    and _cell_is_gray(img, i, j, stud_size)
                    and _cell_is_gray(img, i + 1, j, stud_size)):
                is_joint = False  # interior of a gray piece, not a joint
            joint_v[i, j] = is_joint

    for j in range(n_rows - 1):
        y = (j + 1) * stud_size
        for i in range(n_cols):
            x_mid = i * stud_size + stud_size // 2
            samples = [
                is_joint_pixel(img[y, i * stud_size + stud_size // 4]),
                is_joint_pixel(img[y, x_mid]),
                is_joint_pixel(img[y, i * stud_size + 3 * stud_size // 4]),
            ]
            is_joint = sum(samples) >= 2
            if (is_joint and tolerate_gray
                    and _cell_is_gray(img, i, j, stud_size)
                    and _cell_is_gray(img, i, j + 1, stud_size)):
                is_joint = False
            joint_h[i, j] = is_joint

    return joint_v, joint_h


# ============================================================
# 2. Piece detection via union-find on cells
# ============================================================

def detect_pieces(img, stud_size):
    joint_v, joint_h = detect_joints(img, stud_size)
    n_cols = joint_v.shape[0] + 1
    n_rows = joint_h.shape[1] + 1

    parent = {(i, j): (i, j) for i in range(n_cols) for j in range(n_rows)}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n_cols - 1):
        for j in range(n_rows):
            if not joint_v[i, j]:
                union((i, j), (i + 1, j))
    for i in range(n_cols):
        for j in range(n_rows - 1):
            if not joint_h[i, j]:
                union((i, j), (i, j + 1))

    groups = collections.defaultdict(list)
    for c in parent:
        groups[find(c)].append(c)

    pieces = []
    for cells in groups.values():
        cells = sorted(cells)
        i_min = min(c[0] for c in cells)
        i_max = max(c[0] for c in cells)
        j_min = min(c[1] for c in cells)
        j_max = max(c[1] for c in cells)
        # Sample interior color from the center of an interior cell
        ci, cj = cells[len(cells) // 2]
        cx = ci * stud_size + stud_size // 2
        cy = cj * stud_size + stud_size // 2
        color = img[cy, cx][:3].tolist()
        pieces.append({
            'cells': cells,
            'bbox_grid': [i_min, j_min, i_max - i_min + 1, j_max - j_min + 1],
            'color': [int(c) for c in color],
        })
    return pieces


# ============================================================
# 2b. Sanity checks on detected pieces
# ============================================================

def sanity_check_pieces(pieces, n_cols, n_rows):
    """Return a list of warning strings if detected pieces look suspicious.

    Checks:
        - all grid cells are covered exactly once
        - every piece is rectangular (its cells fill its bbox)
        - share of 1x1 pieces is not abnormally high (a tell-tale sign of
          gray-tile false-positive joints, since gray pieces get internally
          split into 1x1s if joint detection is over-aggressive)
    """
    warnings = []
    total = sum(len(p['cells']) for p in pieces)
    expected = n_cols * n_rows
    if total != expected:
        warnings.append(f"cell coverage: {total} from pieces, expected {expected}")

    seen = set()
    dup = 0
    for p in pieces:
        for c in p['cells']:
            if c in seen:
                dup += 1
            seen.add(c)
    if dup:
        warnings.append(f"{dup} cells assigned to multiple pieces")

    non_rect = 0
    for p in pieces:
        cells = p['cells']
        i_min = min(c[0] for c in cells)
        i_max = max(c[0] for c in cells)
        j_min = min(c[1] for c in cells)
        j_max = max(c[1] for c in cells)
        if len(cells) != (i_max - i_min + 1) * (j_max - j_min + 1):
            non_rect += 1
    if non_rect:
        warnings.append(f"{non_rect}/{len(pieces)} pieces are non-rectangular "
                        "(joint detection inconsistency)")

    n_1x1 = sum(1 for p in pieces if len(p['cells']) == 1)
    pct_1x1 = 100.0 * n_1x1 / max(len(pieces), 1)
    if pct_1x1 > 60.0:
        warnings.append(
            f"{pct_1x1:.0f}% of pieces are 1x1 ({n_1x1}/{len(pieces)}). "
            "Anything above ~60% suggests joint over-detection."
        )

    # Targeted gray-tile check: a large cluster of 1x1 with a similar near-gray
    # color is the smoking gun for Light/Dark Bluish Gray false positives.
    gray_1x1 = []
    for p in pieces:
        if len(p['cells']) != 1:
            continue
        r, g, b = p['color']
        if (max(r, g, b) - min(r, g, b) < 25
                and 80 <= (r + g + b) / 3 <= 200):
            gray_1x1.append(p)
    if len(gray_1x1) > 30:
        warnings.append(
            f"{len(gray_1x1)} gray-ish 1x1 pieces detected — likely "
            "Light Bluish Gray / Dark Bluish Gray / Flat Silver / "
            "Pearl Light Gray / Metallic Silver tiles being mistaken "
            "for joints. Disable those colors in the palette."
        )
    return warnings


# ============================================================
# 3. Piece adjacency graph
# ============================================================

def build_piece_graph(pieces):
    cell_to_pid = {}
    for pid, p in enumerate(pieces):
        for cell in p['cells']:
            cell_to_pid[cell] = pid

    g = nx.Graph()
    for pid in range(len(pieces)):
        g.add_node(pid)
    for cell, pid in cell_to_pid.items():
        i, j = cell
        for di, dj in [(1, 0), (0, 1)]:
            neighbor = (i + di, j + dj)
            if neighbor in cell_to_pid:
                pid2 = cell_to_pid[neighbor]
                if pid != pid2:
                    g.add_edge(pid, pid2)
    return g


# ============================================================
# 4. Fragmentation by balanced region growing
# ============================================================

def _bfs_distances(piece_graph, source, dist=None):
    """Unweighted shortest-path distances from `source` over `piece_graph`.
    If `dist` is provided, distances are updated only where shorter."""
    if dist is None:
        dist = {n: float('inf') for n in piece_graph.nodes}
    dist[source] = 0
    queue = collections.deque([source])
    while queue:
        u = queue.popleft()
        for v in piece_graph.neighbors(u):
            if dist[u] + 1 < dist[v]:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def select_seeds_farthest(piece_graph, k):
    """Farthest-point sampling: each new seed maximises distance to the nearest existing seed."""
    nodes = list(piece_graph.nodes)
    first = random.choice(nodes)
    seeds = [first]
    dist = _bfs_distances(piece_graph, first)
    while len(seeds) < k:
        # Argmax over nodes; ties broken randomly to avoid bias.
        max_d = max(dist.values())
        candidates = [n for n, d in dist.items() if d == max_d]
        nxt = random.choice(candidates)
        seeds.append(nxt)
        # Update distance-to-nearest-seed by running BFS from new seed.
        _bfs_distances(piece_graph, nxt, dist)
    return seeds


def fragment_pieces(piece_graph, k):
    """Balanced multi-source region growing.

    Seeds are picked via farthest-point sampling (k-center). At each step the
    currently smallest fragment expands by one queued node — ensuring balanced
    fragment sizes regardless of local connectivity differences.
    """
    nodes = list(piece_graph.nodes)
    if k >= len(nodes):
        return {n: i for i, n in enumerate(nodes)}

    seeds = select_seeds_farthest(piece_graph, k)
    assigned = {s: i for i, s in enumerate(seeds)}
    frontier = [collections.deque([s]) for s in seeds]
    sizes = [1] * k

    # Min-heap keyed on (current size, fragment_id) → smallest grows first.
    # Tie-breaker is fragment_id which is stable.
    heap = [(1, i) for i in range(k)]
    heapq.heapify(heap)

    while heap:
        _, fid = heapq.heappop(heap)
        if not frontier[fid]:
            continue
        current = frontier[fid].popleft()
        for neighbor in piece_graph.neighbors(current):
            if neighbor not in assigned:
                assigned[neighbor] = fid
                frontier[fid].append(neighbor)
                sizes[fid] += 1
        if frontier[fid]:
            heapq.heappush(heap, (sizes[fid], fid))
    return assigned


# ============================================================
# 5. Fragment polygon and resampling
# ============================================================

def fragment_mask(fragment_cells, img_shape, stud_size):
    H, W = img_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    for (i, j) in fragment_cells:
        x0, y0 = i * stud_size, j * stud_size
        mask[y0:y0 + stud_size, x0:x0 + stud_size] = 1
    return mask


# extract_polygon, resample_reflex_aware (+ helpers PCA), compute_fragment_features
# → déplacés dans features/fragment_features.py (module partagé), importés en tête.
# (resample_polygon équidistant supprimé : remplacé par resample_reflex_aware.)


# ============================================================
# 7. Fragment adjacencies (segments shared between fragments)
# ============================================================

def compute_fragment_adjacencies(pieces, piece_to_frag, stud_size):
    cell_to_frag = {}
    for pid, p in enumerate(pieces):
        fid = piece_to_frag[pid]
        for cell in p['cells']:
            cell_to_frag[cell] = fid

    adj = collections.defaultdict(list)
    for cell, fid in cell_to_frag.items():
        i, j = cell
        right = (i + 1, j)
        if right in cell_to_frag and cell_to_frag[right] != fid:
            other = cell_to_frag[right]
            a, b = min(fid, other), max(fid, other)
            x = (i + 1) * stud_size
            adj[(a, b)].append((x, j * stud_size, x, (j + 1) * stud_size))
        bottom = (i, j + 1)
        if bottom in cell_to_frag and cell_to_frag[bottom] != fid:
            other = cell_to_frag[bottom]
            a, b = min(fid, other), max(fid, other)
            y = (j + 1) * stud_size
            adj[(a, b)].append((i * stud_size, y, (i + 1) * stud_size, y))
    return adj, cell_to_frag


def find_outer_fragments_per_side(polygon_n, centroid, frag_id, cell_to_frag,
                                   stud_size, offset=6, n_samples=7):
    """For each side of polygon_n, vote over multiple samples along the side
    to find the most common adjacent fragment (or None if image edge)."""
    out = []
    n = len(polygon_n)
    for i in range(n):
        p0, p1 = polygon_n[i], polygon_n[(i + 1) % n]
        votes = collections.Counter()
        for t in np.linspace(0.15, 0.85, n_samples):
            sample = (1 - t) * p0 + t * p1
            outward = sample - centroid
            norm = float(np.linalg.norm(outward))
            if norm < 1e-6:
                continue
            outer = sample + outward / norm * offset
            ci, cj = int(outer[0] // stud_size), int(outer[1] // stud_size)
            if (ci, cj) in cell_to_frag:
                o = cell_to_frag[(ci, cj)]
                if o != frag_id:
                    votes[o] += 1
        if votes:
            best, _ = votes.most_common(1)[0]
            out.append(int(best))
        else:
            out.append(None)
    return out


# ============================================================
# 8. Exploded image: cut + rotate + place
# ============================================================

def _disk(radius):
    d = 2 * int(radius) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def degrade_mask(mask, erode_px=0, n_holes=0, stud_size=40, max_hole_area_frac=0.10):
    """Degraded copy of a fragment alpha mask — only ever REMOVES material, so
    degraded ⊆ perfect (keeps the "area never gained" invariant).

    erode_px           : morphological erosion radius in px (boundary wear).
    n_holes            : interior holes punched (missing tesselae *inside*).
    max_hole_area_frac : hard cap on TOTAL hole area (default 10% of the
                         post-erosion fragment area). The n_holes disks share
                         this budget, so holes never remove more than the cap
                         (overlaps / boundary clipping only reduce it further).
    Uses the module RNG (seeded in forge_one) for reproducibility.
    """
    m = (mask > 0).astype(np.uint8)
    if erode_px > 0:
        m = cv2.erode(m, _disk(erode_px))
    if n_holes > 0:
        ys, xs = np.where(m > 0)
        area = len(xs)
        if area:
            budget = max_hole_area_frac * area          # ≤10% of fragment area
            r = max(1, int(np.sqrt(budget / (n_holes * np.pi))))
            for _ in range(n_holes):
                k = random.randint(0, area - 1)
                cv2.circle(m, (int(xs[k]), int(ys[k])), r, 0, -1)
    return m


def make_fragment_rgba(fragment_cells, target_img, stud_size, mask=None):
    if mask is None:
        mask = fragment_mask(fragment_cells, target_img.shape, stud_size)
    mask = (mask > 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    crop = target_img[y_min:y_max + 1, x_min:x_max + 1]
    crop_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
    rgba = np.dstack([crop, crop_mask * 255]).astype(np.uint8)
    return rgba, (x_min, y_min, x_max, y_max)


def rotate_rgba(rgba, angle):
    pil = Image.fromarray(rgba, 'RGBA')
    rot = pil.rotate(angle, resample=Image.BILINEAR, expand=True)
    return np.array(rot)


def _collides(x, y, alpha, occupied):
    """True if fragment alpha placed at (x, y) overlaps any occupied mask."""
    h, w = alpha.shape
    for (ox, oy, omask) in occupied:
        oh, ow = omask.shape
        ix1, iy1 = max(x, ox), max(y, oy)
        ix2, iy2 = min(x + w, ox + ow), min(y + h, oy + oh)
        if ix1 >= ix2 or iy1 >= iy2:
            continue
        a = alpha[iy1 - y:iy2 - y, ix1 - x:ix2 - x]
        b = omask[iy1 - oy:iy2 - oy, ix1 - ox:ix2 - ox]
        if np.logical_and(a, b).any():
            return True
    return False


def _scan_free_position(alpha, occupied, cw, ch, padding, step):
    """Deterministic raster scan for the first non-overlapping top-left position.
    Returns None only if no free spot exists anywhere on the current canvas."""
    h, w = alpha.shape
    y = padding
    while y + h + padding <= ch:
        x = padding
        while x + w + padding <= cw:
            if not _collides(x, y, alpha, occupied):
                return x, y
            x += step
        y += step
    return None


def place_fragments(fragments_data, canvas_size, max_tries=200, padding=4):
    """Scatter fragments on a white canvas WITHOUT overlap.

    Per fragment (largest first): (1) random search, to keep the scattered look;
    (2) deterministic raster scan — guaranteed to find a free spot if one exists;
    (3) last resort, grow the canvas downward. Steps 2-3 replace the old
    "place at random, may overlap" fallback that caused the mosaic_004 overlaps.
    Returns the (possibly grown) canvas size so the caller normalises YOLO
    coordinates against the right dimensions.
    """
    cw, ch = canvas_size
    placements_by_id = {}
    occupied = []  # list of (x, y, alpha_mask)

    # Place larger fragments first to ease packing
    order = sorted(range(len(fragments_data)),
                   key=lambda k: -fragments_data[k]['rgba_rotated'].shape[0] *
                                  fragments_data[k]['rgba_rotated'].shape[1])

    for k in order:
        rgba = fragments_data[k]['rgba_rotated']
        h, w = rgba.shape[:2]
        alpha = rgba[..., 3] > 0

        # Fragment larger than the canvas → grow so it (and the random range) fit.
        while w + 2 * padding > cw:
            cw += w
        while h + 2 * padding > ch:
            ch += h

        pos = None
        for _ in range(max_tries):
            x = random.randint(padding, cw - w - padding)
            y = random.randint(padding, ch - h - padding)
            if not _collides(x, y, alpha, occupied):
                pos = (x, y)
                break

        if pos is None:
            # Random search exhausted: deterministic scan is guaranteed to find a
            # gap if the canvas still has room (mosaic_004 fix).
            step = max(16, min(h, w) // 2)
            pos = _scan_free_position(alpha, occupied, cw, ch, padding, step)

        if pos is None:
            # Canvas genuinely full: add a strip at the bottom and drop it there.
            pos = (padding, ch + padding)
            ch += h + 2 * padding

        x, y = pos
        placements_by_id[k] = {'x': x, 'y': y, 'w': w, 'h': h}
        occupied.append((x, y, alpha))

    canvas = Image.new('RGB', (cw, ch), (255, 255, 255))
    placements = []
    # Paste in original index order (deterministic z-order).
    for idx, fdata in enumerate(fragments_data):
        place = placements_by_id[idx]
        pil = Image.fromarray(fdata['rgba_rotated'], 'RGBA')
        canvas.paste(pil, (place['x'], place['y']), pil)
        placements.append(place)
    return canvas, placements, (cw, ch)


# ============================================================
# 9. YOLO polygon export from rotated alpha
# ============================================================

def extract_yolo_polygon(rgba_rotated, place_x, place_y, canvas_size, simplify_epsilon=1.5):
    alpha = (rgba_rotated[..., 3] > 0).astype(np.uint8)
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if simplify_epsilon > 0:
        cnt = cv2.approxPolyDP(cnt, simplify_epsilon, True)
    cnt = cnt.reshape(-1, 2).astype(np.float32)
    cnt[:, 0] = (cnt[:, 0] + place_x) / canvas_size[0]
    cnt[:, 1] = (cnt[:, 1] + place_y) / canvas_size[1]
    return cnt


# ============================================================
# 10. Top-level orchestration
# ============================================================

def forge_one(target_path, out_dir, n_sides_range, n_frag_range, canvas_size,
              stud_size, seed, degrade=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    degrade = degrade or {}
    erode_rng = tuple(degrade.get('erode_px', (0, 0)))
    holes_rng = tuple(degrade.get('holes', (0, 0)))
    missing_rng = tuple(degrade.get('missing', (0, 0)))  # absolute count [min, max]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'fragments').mkdir(exist_ok=True)

    pil = Image.open(target_path).convert('RGB')
    img = np.array(pil)
    H, W = img.shape[:2]
    if stud_size is None:
        # Grille la PLUS FINE valide : plus grand g divisant l'image avec un
        # stud résultant ≥ 40 px (contrainte projet : ≥40 px/stud). Désambiguïse
        # les images divisibles par plusieurs tailles (ex. 3840 = 48·80 = 96·40
        # → grille 96, stud 40) au lieu de retomber sur la grille la plus
        # grossière. Les mosaïques 1920 restent en grille 48 (stud 40).
        candidates = [g for g in (48, 50, 64, 96, 100, 128)
                      if W % g == 0 and H % g == 0 and W // g >= 40]
        stud_size = W // max(candidates) if candidates else 40
    print(f"[forge] image {W}x{H}, stud_size={stud_size} → grid {W // stud_size}x{H // stud_size}")

    print("[forge] detecting LEGO pieces…")
    pieces = detect_pieces(img, stud_size)
    print(f"[forge]   {len(pieces)} pieces detected")

    n_cols, n_rows = W // stud_size, H // stud_size
    warnings = sanity_check_pieces(pieces, n_cols, n_rows)
    for w in warnings:
        print(f"[warn] {w}")

    piece_graph = build_piece_graph(pieces)

    k = random.randint(n_frag_range[0], n_frag_range[1])
    print(f"[forge] fragmenting into {k} fragments…")
    piece_to_frag = fragment_pieces(piece_graph, k)
    frag_pieces = collections.defaultdict(list)
    for pid, fid in piece_to_frag.items():
        frag_pieces[fid].append(pid)
    actual_k = len(frag_pieces)
    print(f"[forge]   {actual_k} non-empty fragments")

    adjacencies, cell_to_frag = compute_fragment_adjacencies(pieces, piece_to_frag, stud_size)
    print(f"[forge]   {len(adjacencies)} fragment-fragment adjacencies")

    print("[forge] extracting polygons & features per fragment…")
    fragment_data = []
    sorted_fids = sorted(frag_pieces.keys())
    fid_to_idx = {fid: idx for idx, fid in enumerate(sorted_fids)}

    for fid in sorted_fids:
        all_cells = []
        weighted_color = np.zeros(3, dtype=np.float64)
        total_cells = 0
        for pid in frag_pieces[fid]:
            all_cells.extend(pieces[pid]['cells'])
            n_cells = len(pieces[pid]['cells'])
            weighted_color += np.array(pieces[pid]['color'], dtype=np.float64) * n_cells
            total_cells += n_cells
        mean_color = (weighted_color / max(total_cells, 1)).tolist()

        mask = fragment_mask(all_cells, img.shape, stud_size)
        polygon_raw = extract_polygon(mask)          # PERFECT footprint → GT
        # --- degradation: input only (the GT polygon_raw stays perfect) ---
        erode_px = random.randint(*erode_rng) if erode_rng[1] > 0 else 0
        n_holes = random.randint(*holes_rng) if holes_rng[1] > 0 else 0
        if erode_px or n_holes:
            mask_in = degrade_mask(mask, erode_px, n_holes, stud_size)
            polygon_raw_deg = extract_polygon(mask_in)
            if len(polygon_raw_deg) >= 4:
                # erosion/holes add rasterisation noise (1px stair-steps) that
                # would explode #reflex → simplify so reco B keeps n reasonable.
                eps = max(1.5, stud_size * 0.06)
                cnt = polygon_raw_deg.reshape(-1, 1, 2).astype(np.float32)
                polygon_raw_deg = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
            if len(polygon_raw_deg) < 4:             # degradation wiped it → revert
                mask_in = (mask > 0).astype(np.uint8)
                polygon_raw_deg = polygon_raw
        else:
            mask_in = (mask > 0).astype(np.uint8)
            polygon_raw_deg = polygon_raw

        # Degradation severity = area lost by the degraded footprint vs the
        # perfect one (DERIVED metric, not an input knob — cf. reco/06/06).
        perf_area = abs(_polygon_signed_area(polygon_raw)) if len(polygon_raw) >= 3 else 0.0
        deg_area = abs(_polygon_signed_area(np.asarray(polygon_raw_deg, dtype=np.float64))) \
            if len(polygon_raw_deg) >= 3 else 0.0
        lost_frac = float(1.0 - deg_area / perf_area) if perf_area > 0 else 0.0

        n_target = random.randint(n_sides_range[0], n_sides_range[1])
        polygon_n = resample_reflex_aware(polygon_raw_deg, n_target)  # INPUT, ⊆ perfect
        centroid = polygon_n.mean(axis=0)
        # rotation-invariance : repère canonique PCA (sur le masque observé)
        R = pca_canonical_rotation(mask_in)
        polygon_canon = ((polygon_n - centroid) @ R.T).astype(np.float32)
        pca_angle = float(np.degrees(np.arctan2(R[0, 1], R[0, 0])))  # axe majeur (repère image)

        global_feat, side_feat = compute_fragment_features(
            polygon_n, float(len(all_cells) * stud_size * stud_size),
            mean_color, img, polygon_geom=polygon_canon
        )
        outer_per_side = find_outer_fragments_per_side(
            polygon_n, centroid, fid, cell_to_frag, stud_size
        )
        # Translate fragment_id → contiguous node index
        outer_per_side = [fid_to_idx[o] if o is not None and o in fid_to_idx else None
                          for o in outer_per_side]

        rgba, bbox = make_fragment_rgba(all_cells, img, stud_size, mask=mask_in)
        angle = random.choice([0, 90, 180, 270]) + random.uniform(-5.0, 5.0)
        rgba_rot = rotate_rgba(rgba, angle)

        fragment_data.append({
            'node_id': fid_to_idx[fid],
            'cells': all_cells,
            'piece_ids': frag_pieces[fid],
            'polygon_raw': polygon_raw.tolist(),
            'polygon_raw_degraded': np.asarray(polygon_raw_deg).tolist(),
            'degradation_lost_frac': lost_frac,
            'erode_px': int(erode_px),
            'n_holes': int(n_holes),
            'polygon_n': polygon_n.tolist(),
            'polygon_n_canonical': polygon_canon.tolist(),  # repère PCA (invariant rotation)
            'pca_angle_deg': pca_angle,
            'global_features': global_feat,
            'side_features': side_feat,
            'mean_color': [float(c) for c in mean_color],
            'rgba': rgba,
            'rgba_rotated': rgba_rot,
            'rotation_angle': float(angle),
            'bbox_in_target': bbox,
            'outer_per_side': outer_per_side,
        })

    # --- missing fragments: removed from the exploded INPUT, kept in the GT ---
    missing_ids = set()
    if missing_rng[1] > 0 and len(fragment_data) > 1:
        n_missing = min(random.randint(missing_rng[0], missing_rng[1]),
                        len(fragment_data) - 1)  # keep at least one fragment
        if n_missing > 0:
            missing_ids = set(random.sample(
                [f['node_id'] for f in fragment_data], n_missing))
            print(f"[forge]   {len(missing_ids)} fragment(s) marked missing (input only)")
    present = [f for f in fragment_data if f['node_id'] not in missing_ids]

    print(f"[forge] placing fragments on {canvas_size[0]}x{canvas_size[1]} white canvas…")
    requested_size = canvas_size
    canvas, placements, canvas_size = place_fragments(present, canvas_size)
    if canvas_size != requested_size:
        print(f"[forge]   canvas grown to {canvas_size[0]}x{canvas_size[1]} to fit all fragments")
    for f, place in zip(present, placements):
        f['placement'] = place

    print("[forge] extracting YOLO polygons…")
    yolo_lines = []
    for fdata, place in zip(present, placements):
        poly = extract_yolo_polygon(
            fdata['rgba_rotated'], place['x'], place['y'], canvas_size
        )
        if poly is None:
            continue
        coords = ' '.join(f'{v:.6f}' for v in poly.flatten())
        yolo_lines.append(f'0 {coords}')

    print("[forge] writing outputs…")
    pil.save(out_dir / 'target.png')
    canvas.save(out_dir / 'source.png')
    (out_dir / 'source_yolo.txt').write_text('\n'.join(yolo_lines) + '\n')

    for fdata in present:
        Image.fromarray(fdata['rgba'], 'RGBA').save(
            out_dir / 'fragments' / f"frag_{fdata['node_id']:02d}.png"
        )

    pieces_json = [{
        'cells': p['cells'],
        'bbox_grid': p['bbox_grid'],
        'color': p['color'],
    } for p in pieces]
    (out_dir / 'pieces.json').write_text(json.dumps(pieces_json, indent=2))

    # Build two node representations:
    #   - 'gnn_input': features the GNN may legitimately consume (no target-frame
    #     position leak). polygon_n_canonical = polygon centred at the origin.
    #   - 'target_info': ground-truth position/orientation, used as supervision
    #     and for viz. NOT to be fed to the GNN as input.
    def split_node(f):
        gf = f['global_features']  # [area, perim, cx, cy, R, G, B, bbox_w, bbox_h]
        # gnn_input = DOMAINE-AGNOSTIQUE (transférable LEGO→fresque) — aucune feature LEGO-only.
        gnn_input = [gf[0], gf[1], gf[4], gf[5], gf[6], gf[7], gf[8]]  # area, perim, R, G, B, bbox_w, bbox_h
        return {
            'gnn_input': gnn_input,
            'n_sides': len(f['polygon_n_canonical']),
            'polygon_n_canonical': f['polygon_n_canonical'],   # repère PCA (invariant rotation)
            'side_features': f['side_features'],
            'target_info': {
                'centroid': [gf[2], gf[3]],
                'pca_angle_deg': f['pca_angle_deg'],           # orientation retirée de l'input → GT
                'polygon_n_absolute': f['polygon_n'],
                'polygon_raw': f['polygon_raw'],
                'mean_color': f['mean_color'],
                'n_pieces': len(f['piece_ids']),
            },
        }

    mean_lost = (float(np.mean([f['degradation_lost_frac'] for f in fragment_data]))
                 if fragment_data else 0.0)
    nodes_full = [{'node_id': f['node_id'], 'missing': f['node_id'] in missing_ids,
                   **split_node(f)} for f in fragment_data]
    # graph_fragments = INPUT: missing fragments are not observed → excluded.
    nodes_input_only = [
        {k: v for k, v in n.items() if k not in ('target_info', 'missing')}
        for n in nodes_full if not n['missing']
    ]

    edges_json = []
    for (fa, fb), segments in adjacencies.items():
        a = fid_to_idx[fa]
        b = fid_to_idx[fb]
        total_length = float(sum(
            np.sqrt((s[2] - s[0]) ** 2 + (s[3] - s[1]) ** 2) for s in segments
        ))
        angles = [float(np.arctan2(s[3] - s[1], s[2] - s[0])) for s in segments]
        mean_angle = float(np.mean(angles))
        a_sides = [i for i, o in enumerate(fragment_data[a]['outer_per_side']) if o == b]
        b_sides = [i for i, o in enumerate(fragment_data[b]['outer_per_side']) if o == a]
        edges_json.append({
            'src': a, 'dst': b,
            'features': [total_length, mean_angle, float(len(segments))],
            'src_side_idx': a_sides,
            'dst_side_idx': b_sides,
            'polyline_raw': [list(s) for s in segments],
        })

    common_meta = {
        'n_sides_range': list(n_sides_range),
        'n_sides_note': ("polygon_n_canonical / side_features ont une longueur "
                         "VARIABLE par nœud (cf. champ 'n_sides' de chaque nœud). "
                         "Reco B reflex-aware : polygon_n ⊆ polygon_raw (l'aire "
                         "n'est jamais gagnée). Plancher dur = nb de sommets "
                         "concaves. Pour un tenseur fixe côté GNN : padder au max "
                         "des 'n_sides' du dataset + masque de validité."),
        'gnn_input_feature_names': ['area', 'perimeter', 'R', 'G', 'B', 'bbox_w', 'bbox_h'],
        'feature_convention_note': (
            "gnn_input = features 100 % DOMAINE-AGNOSTIQUES (transférables LEGO→fresque ; "
            "aucune feature LEGO-only). polygon_n_canonical / side_features ont une longueur "
            "VARIABLE (n_sides) → côté GNN : padder à n_max + masque (cf. features/collate.py), "
            "ou set-encoder. area/perimeter/bbox sont en PIXELS (non normalisés) ; "
            "normaliser avant transfert cross-échelle."),
        'side_feature_names': ['length', 'angle', 'R', 'G', 'B'],
        'edge_feature_names': ['shared_length', 'mean_angle', 'n_segments'],
        'target_info_note': "centroid/polygon_n_absolute/polygon_raw are in the "
                            "target frame and MUST NOT be fed to the GNN as "
                            "input — they are supervision targets / metadata.",
        'degradation': {
            'erode_px': list(erode_rng),
            'holes': list(holes_rng),
            'missing': list(missing_rng),
            'n_missing': len(missing_ids),
            'mean_lost_frac': round(mean_lost, 4),   # sévérité moyenne (métrique dérivée)
            'note': "Dégradation appliquée à l'INPUT seulement ; la GT (arêtes + "
                    "target_info.polygon_raw perfect) reste sur la partition intacte. "
                    "mean_lost_frac = aire moyenne perdue (parfait→dégradé), métrique "
                    "post-hoc de sévérité, pas un paramètre d'entrée.",
        },
    }

    graph_complete = dict(common_meta, nodes=nodes_full, edges=edges_json)
    (out_dir / 'graph_complete.json').write_text(json.dumps(graph_complete, indent=2))

    graph_fragments = dict(common_meta, nodes=nodes_input_only, edges=[])
    (out_dir / 'graph_fragments.json').write_text(json.dumps(graph_fragments, indent=2))

    # gt_layout.json — ground truth for the RECONSTRUCTION metric (IoU / Q_pos).
    # Per fragment: its exact footprint in target.png (polygon_raw) + the reco-B
    # polygon, both in target coords, plus where it sits in the exploded source.
    # The GNN+pose / VLM prediction is scored against these placed polygons.
    gt_fragments = []
    for f in fragment_data:
        gf = f['global_features']
        place = f.get('placement')
        gt_fragments.append({
            'node_id': f['node_id'],
            'missing': f['node_id'] in missing_ids,
            'polygon_raw': f['polygon_raw'],              # PERFECT footprint, target coords
            'polygon_raw_degraded': f['polygon_raw_degraded'],  # what the model sees (eroded/holed)
            'degradation_lost_frac': round(f['degradation_lost_frac'], 4),
            'polygon_n': f['polygon_n'],                  # reco-B polygon, target coords
            'centroid': [gf[2], gf[3]],
            'bbox_target': list(f['bbox_in_target']),
            'area': gf[0],
            'source_placement': (                         # location in source.png (exploded)
                {'x': place['x'], 'y': place['y'], 'w': place['w'], 'h': place['h'],
                 'rotation_deg': f['rotation_angle']}
                if place is not None else None            # None ⟺ missing fragment
            ),
        })
    gt_layout = {
        'mosaic': out_dir.name,
        'target_size': [W, H],
        'canvas_size': [canvas_size[0], canvas_size[1]],
        'mean_degradation_lost_frac': round(mean_lost, 4),
        'note': ("Vérité terrain pour la métrique de RECONSTRUCTION (IoU / Q_pos). "
                 "polygon_raw = footprint exact du fragment dans target.png ; on y "
                 "compare la prédiction GNN+pose / VLM (fragment replacé). "
                 "source_placement = pose du fragment dans source.png (éclaté)."),
        'fragments': gt_fragments,
    }
    (out_dir / 'gt_layout.json').write_text(json.dumps(gt_layout, indent=2))

    # degradation.md — rapport lisible : paramètres tirés + aire perdue par fragment
    report = [
        f"# Dégradation — {out_dir.name}", "",
        "## Configuration (intervalles ; tirage aléatoire par fragment)",
        f"- Érosion morpho : `{list(erode_rng)}` px",
        f"- Trous internes : `{list(holes_rng)}` (aire totale plafonnée à 10 % du fragment)",
        f"- Fragments manquants : `{list(missing_rng)}` → **{len(missing_ids)}** tiré(s)",
        f"- Sommets (reco B) : `{list(n_sides_range)}` (plancher dur = #reflex)", "",
        "## Par fragment",
        "| node_id | manquant | érosion px | trous | n sommets | aire perdue % |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]
    for f in fragment_data:
        report.append(
            f"| {f['node_id']} | {'oui' if f['node_id'] in missing_ids else '—'} "
            f"| {f['erode_px']} | {f['n_holes']} | {len(f['polygon_n'])} "
            f"| {100 * f['degradation_lost_frac']:.1f} |")
    report += [
        "", "## Résumé",
        f"- **Aire moyenne perdue par fragment : {100 * mean_lost:.2f} %**",
        f"- Fragments manquants : {sorted(missing_ids) if missing_ids else 'aucun'}",
        f"- Total fragments : {len(fragment_data)} (présents dans l'input : {len(present)})",
        "",
        "_Dégradation appliquée à l'INPUT seulement ; la GT (arêtes + polygon_raw parfait) "
        "reste sur la partition intacte. L'aire perdue est une métrique dérivée, "
        "pas un paramètre._", "",
    ]
    (out_dir / 'degradation.md').write_text("\n".join(report))

    print(f"[forge] done → {out_dir}")
    print(f"        pieces={len(pieces)}  fragments={len(fragment_data)}  "
          f"adjacencies={len(edges_json)}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='exp1/canvas_mosaic.png',
                   help='Path to canvas_mosaic.png (target)')
    p.add_argument('--out', default='dataset/mosaic_001',
                   help='Output directory')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-sides-min', type=int, default=16,
                   help='Min target vertex count per fragment (reco B, variable n)')
    p.add_argument('--n-sides-max', type=int, default=24,
                   help='Max target vertex count per fragment (floor = #reflex)')
    p.add_argument('--n-frag-min', type=int, default=10)
    p.add_argument('--n-frag-max', type=int, default=15)
    p.add_argument('--canvas-w', type=int, default=3500)
    p.add_argument('--canvas-h', type=int, default=3500)
    p.add_argument('--stud-size', type=int, default=None,
                   help='Pixels per stud (auto-detected from image size if omitted)')
    # --- degradation knobs (curriculum) ; all default OFF → clean reco-B ---
    p.add_argument('--erode-px-min', type=int, default=0,
                   help='Min boundary-erosion radius in px (DAFNE E)')
    p.add_argument('--erode-px-max', type=int, default=0,
                   help='Max boundary-erosion radius in px (0 = no erosion)')
    p.add_argument('--holes-min', type=int, default=0,
                   help='Min interior holes per fragment')
    p.add_argument('--holes-max', type=int, default=0,
                   help='Max interior holes per fragment (0 = no holes)')
    p.add_argument('--missing-min', type=int, default=0,
                   help='Min number of fragments removed from the input (DAFNE C)')
    p.add_argument('--missing-max', type=int, default=0,
                   help='Max number of fragments removed from the input (0 = none)')
    return p.parse_args()


def degrade_from_args(args):
    return {
        'erode_px': (args.erode_px_min, args.erode_px_max),
        'holes': (args.holes_min, args.holes_max),
        'missing': (args.missing_min, args.missing_max),
    }


def main():
    args = parse_args()
    forge_one(
        target_path=args.input,
        out_dir=args.out,
        n_sides_range=(args.n_sides_min, args.n_sides_max),
        n_frag_range=(args.n_frag_min, args.n_frag_max),
        canvas_size=(args.canvas_w, args.canvas_h),
        stud_size=args.stud_size,
        seed=args.seed,
        degrade=degrade_from_args(args),
    )


if __name__ == '__main__':
    main()
