"""Palette LEGO — couleurs solides BrickLink, SANS les gris/argentés.

Les valeurs hex sont des FAITS (couleurs LEGO officielles, non protégeables) ;
elles ont été extraites de `bricklink-colors.js` (lego-art-remix, GPL-3.0) puis
filtrées hors-ligne. AUCUN code de l'outil source n'est importé ici (clean-room).

Filtre appliqué : on retire toute couleur « confusable joint », i.e. quasi-grise
(max-min des canaux < 25) ET d'intensité moyenne (70 ≤ moyenne ≤ 200). C'est la
fenêtre où le détecteur de joints de `mosaic2fragments/` (gris ~RGB 136 ± 30) ferait un
faux positif. Blanc (255) et Noir (33) sont conservés (loin de 136). Voir
CLAUDE.md « Palette LEGO sans couleurs gris/argent ».
"""

# (R, G, B) — 82 couleurs LEGO solides, gris/argent retirés.
LEGO_PALETTE = [
    (255, 255, 255),  # White
    (232, 232, 232),  # Very Light Gray
    (228, 232, 232),  # Very Light Bluish Gray
    ( 33,  33,  33),  # Black
    (106,  14,  21),  # Dark Red
    (179,   0,   6),  # Red
    (181,  44,  32),  # Rust
    (248, 131, 121),  # Coral
    (244,  92,  64),  # Salmon
    (255, 222, 220),  # Light Salmon
    (140, 107, 107),  # Sand Red
    (137,  53,  29),  # Reddish Brown
    ( 83,  33,  21),  # Brown
    ( 51,   0,   0),  # Dark Brown
    (144, 116,  80),  # Dark Tan
    (222, 198, 156),  # Tan
    (254, 204, 176),  # Light Nougat
    (255, 175, 125),  # Nougat
    (227, 160,  91),  # Medium Nougat
    (231, 139,  62),  # Dark Nougat
    (161, 108,  66),  # Medium Brown
    (179, 105,  78),  # Fabuland Brown
    (239, 145,  33),  # Fabuland Orange
    (230, 136,  29),  # Earth Orange
    (179,  84,   8),  # Dark Orange
    (250,  89,  71),  # Neon Orange
    (255, 126,  20),  # Orange
    (255, 165,  49),  # Medium Orange
    (247, 186,  48),  # Bright Light Orange
    (247, 173,  99),  # Light Orange
    (230, 192,  93),  # Very Light Orange
    (221, 152,  46),  # Dark Yellow
    (247, 209,  23),  # Yellow
    (243, 224,  85),  # Bright Light Yellow
    (255, 227, 131),  # Light Yellow
    (235, 238, 143),  # Light Lime
    (223, 238, 165),  # Yellowish Green
    (188, 239, 102),  # Neon Green
    (189, 198,  24),  # Medium Lime
    (166, 202,  85),  # Lime
    (124, 144,  81),  # Olive Green
    ( 46,  85,  67),  # Dark Green
    (  0, 100,  46),  # Green
    ( 16, 203,  49),  # Bright Green
    ( 98, 245, 142),  # Medium Green
    (165, 219, 181),  # Light Green
    (118, 162, 144),  # Sand Green
    (  0, 138, 128),  # Dark Turquoise
    ( 49, 181, 202),  # Light Turquoise
    (181, 211, 214),  # Aqua
    (204, 255, 255),  # Light Aqua
    ( 20,  48,  68),  # Dark Blue
    (  0,  87, 166),  # Blue
    ( 51, 153, 255),  # Dark Azure
    ( 66, 192, 251),  # Medium Azure
    ( 97, 175, 255),  # Medium Blue
    (107, 173, 214),  # Maersk Blue
    (159, 195, 233),  # Bright Light Blue
    (180, 210, 227),  # Light Blue
    (125, 191, 221),  # Sky Blue
    ( 90, 113, 132),  # Sand Blue
    ( 80, 108, 239),  # Blue-Violet
    ( 32,  50, 176),  # Dark Blue-Violet
    ( 52,  72, 164),  # Violet
    (147, 145, 228),  # Medium Violet
    (201, 202, 226),  # Light Violet
    ( 95,  38, 131),  # Dark Purple
    (165,  73, 156),  # Purple
    (218, 112, 214),  # Light Purple
    (136,  94, 158),  # Medium Lavender
    (224, 170, 217),  # Clikits Lavender
    (177, 140, 191),  # Lavender
    (181, 125, 165),  # Sand Purple
    (181,  41,  82),  # Magenta
    (200, 112, 128),  # Dark Pink
    (247, 133, 177),  # Medium Dark Pink
    (255, 187, 255),  # Bright Pink
    (255, 192, 203),  # Pink
    (255, 225, 255),  # Light Pink
    (255, 252,   0),  # Neon Yellow
    (231, 149,   0),  # Pearl Gold
    (217, 197, 148),  # Medium Tan
]

JOINT_COLOR = (136, 136, 136)  # gris des joints — RGB(136) ≈ rendu lego-art-remix


def hex_of(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)
