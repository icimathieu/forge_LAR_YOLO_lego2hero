"""Extraction de features par fragment — module PARTAGÉ (post-YOLO / pré-GNN).

Domaine-agnostique : marche sur n'importe quel masque de fragment (+ image),
qu'il vienne de la forge synthétique LEGO OU de YOLO sur une mosaïque réelle.
C'est la garantie d'alignement de schéma entre synthétique et réel : le même
code produit les mêmes features dans les deux chaînes.

Contenu : contour, resample reflex-aware (reco B, polygon_n ⊆ polygon_raw),
canonicalisation PCA (invariance en rotation), et features par fragment
(géométrie + couleur de bord). Aucune dépendance LEGO ici (pas de studs, pas de
grille) — aucune notion LEGO ici. Les bits LEGO/GT (adjacence grille, n_pieces en
métadonnée) restent dans la forge.
"""
import numpy as np
import cv2


# ============================================================
# Contour
# ============================================================

def extract_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)
    cnt = max(contours, key=cv2.contourArea)
    return cnt.reshape(-1, 2).astype(np.float32)


# ============================================================
# Resample reflex-aware (reco B) : polygon_n ⊆ polygon_raw
# ============================================================

def _polygon_signed_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _classify_vertices(P):
    """Per vertex: +1 convex, -1 reflex (concave), 0 ~collinear."""
    M = len(P)
    o = np.sign(_polygon_signed_area(P)) or 1.0
    kind = np.zeros(M, dtype=int)
    for i in range(M):
        a, b, c = P[i - 1], P[i], P[(i + 1) % M]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        v = cross * o
        kind[i] = 0 if abs(v) < 1e-6 else (1 if v > 0 else -1)
    return kind


def _clip_to_polygon(poly_enc, poly_raw):
    """Safety clamp: intersect the encoded polygon with the real piece so that
    polygon_n ⊆ polygon_raw is GUARANTEED even on a pathological contour.
    No-op if shapely is unavailable (reflex-aware alone already gives ~0 gain)."""
    try:
        from shapely.geometry import Polygon
    except Exception:
        return poly_enc
    C = Polygon(poly_enc).buffer(0).intersection(Polygon(poly_raw).buffer(0))
    if C.geom_type == 'Polygon':
        polys = [C]
    elif C.geom_type in ('MultiPolygon', 'GeometryCollection'):
        polys = [g for g in C.geoms if g.geom_type == 'Polygon' and not g.is_empty]
    else:
        polys = []
    if not polys:
        return poly_enc
    best = max(polys, key=lambda g: g.area)
    return np.array(best.exterior.coords[:-1], dtype=np.float32)


MAX_AREA_LOSS_DEFAULT = 0.01      # ε : perte d'aire tolérée pour polygon_n (1 %)


def resample_reflex_aware(polygon_raw, n_min=0, clip=True,
                          max_area_loss=MAX_AREA_LOSS_DEFAULT):
    """Reco B — simplifie le contour en **budgétant la PERTE D'AIRE**, `n` en sortie.

    Garantit `polygon_n ⊆ polygon_raw` : on ne *gagne* jamais d'aire, on en perd
    (« comme si les coins étaient ébréchés ») → les fragments réassemblés ne se
    chevauchent jamais, ils laissent des jours (régime physiquement réalisable).
    Renvoie un polygone de longueur VARIABLE.

    Algorithme = **Visvalingam-Whyatt** (1993) *restreint aux sommets convexes* :
    on retire itérativement le sommet dont le triangle (préc., lui, suiv.) a la
    plus petite aire, tant que la perte CUMULÉE reste ≤ `max_area_loss` (fraction
    de l'aire du polygone). Deux invariants :
      - **seuls les convexes sont retirables** — retirer un sommet reflex
        pontifierait une concavité et **re-gagnerait** de l'aire (viole ⊆ raw) ;
      - les sommets ~colinéaires ont un triangle d'aire nulle → retirés en
        premier, **gratuitement** (c'est le gros du bruit de contour LEGO).

    ⚠️ **Corrigé le 23/07/2026.** L'implémentation précédente budgétait le NOMBRE
    de sommets (`n_target` ∈ [16,24]) en amorçant `keep` avec tous les reflex :
    comme `#reflex` (médiane **58** sur LEGO) ≥ `n_target` dans **100 %** des
    fragments, la boucle sortait immédiatement et **aucun convexe n'était jamais
    gardé**. Résultat : perte d'aire médiane **21,4 %** (p95 85,8 %, un quart des
    fragments perdant >50 % de leur aire) sur la config k=10-15 du dataset 25k,
    et jusqu'à 99 % à k=2. L'invariant ⊆ raw tenait (pas d'aire fantôme) mais
    l'AMPLEUR de la perte n'était ni bornée ni mesurée. On budgète désormais la
    grandeur qui compte (la fidélité), et `n` en découle.

    `n_min` : plancher optionnel sur le nombre de sommets (0 = pas de plancher).
    Le plancher réel reste **#reflex**, atteint automatiquement.
    """
    import heapq

    P = np.asarray(polygon_raw, dtype=np.float32)
    M = len(P)
    if M < 4:
        return P.copy()

    total_area = abs(_polygon_signed_area(P))
    if total_area <= 0:
        return P.copy()
    budget = max_area_loss * total_area

    orient = np.sign(_polygon_signed_area(P)) or 1.0
    prev = [(i - 1) % M for i in range(M)]
    nxt = [(i + 1) % M for i in range(M)]
    alive = [True] * M
    n_alive = M

    def tri(i):
        """(aire du triangle retiré, convexe ?) pour le sommet i dans l'état courant."""
        a, b, c = P[prev[i]], P[i], P[nxt[i]]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        return abs(cross) / 2.0, (cross * orient) >= 0      # ≥ 0 : convexe ou colinéaire

    heap = []
    for i in range(M):
        a, cvx = tri(i)
        if cvx:
            heapq.heappush(heap, (a, i))

    lost = 0.0
    while heap and n_alive > 3:
        a, i = heapq.heappop(heap)
        if not alive[i]:
            continue
        a_now, cvx_now = tri(i)                             # ré-évaluation paresseuse
        if not cvx_now:                                     # devenu reflex → intouchable
            continue
        if a_now > a + 1e-9:                                # coût périmé → re-pousser
            heapq.heappush(heap, (a_now, i))
            continue
        if lost + a_now > budget:                           # budget épuisé
            break
        if n_min and n_alive <= n_min:
            break
        alive[i] = False
        n_alive -= 1
        lost += a_now
        p, n = prev[i], nxt[i]
        nxt[p], prev[n] = n, p
        for j in (p, n):                                    # coûts des voisins modifiés
            aj, cj = tri(j)
            if cj and alive[j]:
                heapq.heappush(heap, (aj, j))

    poly = P[[i for i in range(M) if alive[i]]]
    if clip:
        poly = _clip_to_polygon(poly, P)                    # filet de sécurité ⊆ raw
    return np.asarray(poly, dtype=np.float32)


# ============================================================
# Canonicalisation PCA (invariance en rotation)
# ============================================================

def pca_canonical_rotation(mask):
    """Rotation 2×2 amenant le fragment dans son repère PCA canonique (axe
    principal → x), signe déterministe (skewness du 3e moment sur l'axe majeur).

    Rend la forme INVARIANTE EN ROTATION : un même fragment, à n'importe quelle
    orientation, donne le même descripteur → plus de fuite d'orientation, et
    repère-target ≡ repère-source (ce que produirait YOLO). Rotation PURE
    (det=+1, pas de miroir : la chiralité compte pour le matching). Ambigu pour
    les formes ~symétriques / ~carrées (axe instable) — limitation acceptée.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return np.eye(2, dtype=np.float64)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(pts.T @ pts)
    u = evecs[:, int(np.argmax(evals))]            # axe principal (unitaire)
    if float(((pts @ u) ** 3).sum()) < 0:          # désambiguïse le flip 180°
        u = -u
    return np.array([[u[0], u[1]], [-u[1], u[0]]], dtype=np.float64)  # u → axe x, det=+1


# ============================================================
# Features par fragment (géométrie canonique + couleur de bord)
# ============================================================

def _is_joint_pixel(px, target=136, tol=30):
    r, g, b = int(px[0]), int(px[1]), int(px[2])
    return (abs(r - target) < tol and abs(g - target) < tol and abs(b - target) < tol
            and max(r, g, b) - min(r, g, b) < 25)


def _sample_inward(p0, p1, centroid, img, offset=6):
    """Sample piece interior color along a polygon side.

    The naive midpoint + small offset can land on a LEGO joint (gray) — we
    therefore try a few samples along the side and reject pixels that look like
    joints, falling back to the median of all samples if everything is gray.
    (Sur fresque réelle : à enrichir vers un profil de couleur le long du bord.)
    """
    samples = []
    for t in (0.3, 0.5, 0.7):
        pt = (1 - t) * p0 + t * p1
        inward = centroid - pt
        norm = float(np.linalg.norm(inward))
        if norm < 1e-6:
            continue
        inward = inward / norm * offset
        s = pt + inward
        sx = int(np.clip(s[0], 0, img.shape[1] - 1))
        sy = int(np.clip(s[1], 0, img.shape[0] - 1))
        px = img[sy, sx][:3]
        samples.append(px)
    if not samples:
        cx = int(np.clip(centroid[0], 0, img.shape[1] - 1))
        cy = int(np.clip(centroid[1], 0, img.shape[0] - 1))
        return img[cy, cx][:3].tolist()
    non_joint = [s for s in samples if not _is_joint_pixel(s)]
    pool = non_joint if non_joint else samples
    return np.median(np.array(pool), axis=0).astype(int).tolist()


def compute_fragment_features(polygon_n, area, mean_color, img, polygon_geom=None):
    """GÉOMÉTRIE (perimeter, bbox, length, angle) calculée sur `polygon_geom`
    (repère canonique PCA → invariant en rotation, si fourni) ; COULEUR
    échantillonnée sur `polygon_n` (repère image, pour indexer les pixels).

    `area` est passé en scalaire (la forge : len(cells)·stud² ; un chemin post-YOLO
    réel : masque.sum()) → fonction 100 % DOMAINE-AGNOSTIQUE (aucune notion LEGO).
    cx,cy = centroïde image (GT de position, pas un input)."""
    geom = polygon_n if polygon_geom is None else np.asarray(polygon_geom, dtype=np.float64)
    centroid_img = polygon_n.mean(axis=0)
    cx, cy = float(centroid_img[0]), float(centroid_img[1])
    closed = np.vstack([geom, geom[0:1]])
    perimeter = float(np.sqrt(((np.diff(closed, axis=0)) ** 2).sum(axis=1)).sum())
    bbox_w = float(geom[:, 0].max() - geom[:, 0].min())   # repère canonique
    bbox_h = float(geom[:, 1].max() - geom[:, 1].min())

    side_features = []
    n = len(polygon_n)
    for i in range(n):
        g0, g1 = geom[i], geom[(i + 1) % n]
        length = float(np.linalg.norm(g1 - g0))
        angle = float(np.arctan2(g1[1] - g0[1], g1[0] - g0[0]))   # angle canonique
        c = _sample_inward(polygon_n[i], polygon_n[(i + 1) % n], centroid_img, img)
        side_features.append([length, angle, float(c[0]), float(c[1]), float(c[2])])

    global_features = [
        float(area), perimeter, cx, cy,
        float(mean_color[0]), float(mean_color[1]), float(mean_color[2]),
        bbox_w, bbox_h,
    ]
    return global_features, side_features
