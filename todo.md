# lego2hero — TODO

> Statut concis (1 ligne par item). Détails : `readme.md`.
> Légende : ✅ fait · 🚧 à implémenter · ⏸ différé.

## Historique (fait)

**≤ 09/06** — forge & features
- ✅ Forge amont LAR (`scripts/forge_LAR_2mosaic/`) : image → mosaïque LEGO clean-room (modes tile/plate/brick, palette 82 sans gris) + `piece_grid.json` (GT exacte)
- ✅ Source images : wikiart en streaming, noms uuid4
- ✅ Forge fragments (`scripts/mosaic2fragments/`) : joints→union-find→fragmentation→reco-B→YOLO-Seg + 2 graphes GNN + `gt_layout.json`
- ✅ Reco-B reflex-aware (`polygon_n ⊆ polygon_raw`, n variable, plancher = #reflex) ; dégradations CLI input-only (érosion/trous/manquants) ; features partagé (gnn_input 7 floats + PCA canonical + collate)
- ✅ Datasets 100 mosaïques + publication HF `icimathieu/lego2hero-100mosaics` ; repo public MIT

**20/06** — réorg & curriculum
- ✅ Scripts regroupés sous `scripts/` ; `todo.md` racine
- ✅ Curriculum : driver `curriculum.py` (paliers appariés par seed, `--jobs`, resume, try/except) ; paliers **L0** éclaté · **L1** translation · **L2** +rotation · **L3** léger · **L4** fort (L5 « knobs poussés » retiré : saturait)
- ✅ Axe `--frag-distribution` : **balanced** (BFS) + **compact** (min-périmètre, −12 % médiane, n_max ~inchangé : l'escalier du tiling domine)

**21/06** — gros run, ultracompact, HF
- ✅ Garde-fou wikiart `--max-dominant-frac 0.55` (rejette aplats monochromes ; 781/5781 rejetées sur le run)
- ✅ **25 000 instances générées** : 5000 wikiart fraîches × 5 paliers, balanced → `output/balanced/L0–L4` (0 échec). n_max L0/L1/L2=156, L3=114, L4=185 ; sévérité 0/0/0/0.031/0.071
- ✅ **Export HF en tar.gz/palier** (`scripts/tools/hf_export_tars.py` + resume) : carte + 5 `gnn_meta.json` + 5 tars `balanced/L*.tar.gz` ; legacy `clean/`+`degraded/` supprimés
- ✅ **Ultracompact** (`--mode mono` 1×1 + `--frag-distribution ultracompact`) : validé sur 50 mosaïques → **n_max 156 → 60**, médiane 26 (pas dégénéré)
- ✅ `pieces.json` **et** `source_yolo_viz.png` = debug → **OFF par défaut**, opt-in `--debug` (ex-`--lean`/`--no-viz` retirés). GT YOLO = `source.png` + `source_yolo.txt`. Docs MAJ

## À faire

### Forge (NOUS)
- 🚧 **DAFNE B** : `--frag-distribution {voronoi, clusters}` (découpe à tailles inégales = durcissement)
- 🚧 **DAFNE D** : parasites/distracteurs (fragments d'autres mosaïques à rejeter) — plus tard
- ⏳ Re-collate local `output/balanced/L*` (le générateur de README a changé)

### YOLO (NOUS)
- 🚧 Entraînement **YOLO-Seg** sur `source.png` + `source_yolo.txt` — pas commencé
- 🚧 Split train/val (`yolo/`) + jitter d'augmentation

### GNN (équipe : Mathias)
- 🚧 **5 modèles, un par palier** (n_max différent par palier → pas de fine-tuning cross-palier avec la repré pad+masque actuelle) + scoring comparé L0→L4 (la métrique doit décroître avec la difficulté)
- 🚧 Choix représentation : pad n_max+masque (actuel) vs **encodeur n_max-agnostique** (set-encoder/PointNet, ou k keypoints fixes type ReassembleNet) — lèverait la contrainte n_max + nécessaire pour le réel

### Transfert / fine-tuning — datasets existants (cibles aval)
- **RePAIR** (NeurIPS 2024) — fresques réelles, 121 objets / 957 fragments. **2D déjà en local** (`python_avance_CVG/FromLegoToHero/2fetchFORGE/2D_Fragments/`). Notre benchmark. <https://repairproject.github.io/RePAIR_dataset/>
- **ReassembleNet** (ICCV 2025) — semi-synth, 5000 mosaïques / 45 834 fragments. [arXiv:2505.21117](https://arxiv.org/abs/2505.21117) · [GitHub](https://github.com/adeela-islam/ReassembleNet) · [Drive](https://drive.google.com/drive/folders/1tflCUoct63Zhzt8dWs37vtfnphOqgReB?usp=sharing)
- **PairingNet** (ECCV 2024) — fragments d'images (390 Pexels → 8196 fragments). [arXiv:2312.08704](https://arxiv.org/abs/2312.08704)
- **DiffAssemble** (CVPR 2024) — graph-diffusion 2D/3D. [GitHub](https://github.com/IIT-PAVIS/DiffAssemble)
- 🚧 Expé transfert : `features/` sur les fragments RePAIR isolés (skip YOLO) + scoring vs GT de pose

### Différé
- ⏸ Réduire n_max au niveau **forge/tiling** (le levier réel ; `compact` ne suffit pas) — si n_max bloque
- ⏸ Normalisation d'échelle adimensionnelle (area/aire_image, perimeter/√area) — avant transfert cross-échelle
