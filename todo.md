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

**23/07** — correctif reco-B & axe DAFNE A
- ⚠️ **BUG MAJEUR trouvé dans reco-B** (`features/fragment_features.py`) : le budget portait sur le **nombre de sommets** (`n_target` 16-24) en amorçant l'ensemble gardé avec tous les reflex ; comme `#reflex` (médiane 58) ≥ `n_target` dans **100 %** des fragments, **aucun convexe n'était jamais gardé**. Perte d'aire **21,4 % médiane** (p95 85,8 %, 25 % des fragments >50 %) à k=10-15, **99 % à k=2**. L'invariant ⊆ raw tenait, mais l'ampleur de la perte n'était ni bornée ni mesurée. → **le dataset 25k publié sur HF est affecté** (`polygon_n_canonical` + `side_features` faux ; `gnn_input` intact car issu du masque)
- ✅ **Correctif** : reco-B budgète désormais la **PERTE D'AIRE** (`--max-area-loss`, défaut ε=1 %), `n` en sortie. Algo = **Visvalingam-Whyatt restreint aux convexes** (retirer un reflex re-gagnerait de l'aire ; les colinéaires partent gratuitement). Vérifié : perte 0,94 % médiane / 0,99 % max, zéro aire gagnée
- ✅ `curriculum.py --missing-max N` : plafonne DAFNE C (notre `missing` est un **compte absolu**, DAFNE C un **pourcentage** → à k=2, L4 ne laissait qu'un seul fragment). Défaut `None` = historique inchangé
- ✅ Axe **DAFNE A** ouvert : configs **k=2 / 3 / 5** en plus de 10-15. **Pilote 100 images identiques × 4 configs × 5 paliers = 2000 instances, 0 échec** → `output/pilot100_poly_balanced/k{02,03,05,10-15}/`. Arêtes/mosaïque 1,0 / 2,9 / 7,5 / 27,1 (c'est **là** qu'est la difficulté) ; `n` médian 75/92/84/75 — **non monotone en k**, culmine à k=3 ; `n_max` (L0-L2) 140/254/214/184 sur 100 mosaïques (croît avec l'échantillon, non comparable au run 25k). Tableau dans `readme.md`

**23/07 (suite)** — ragged, remono, pilote ultracompact
- ✅ `collate.py` : **2e contrat « ragged »** dans `gnn_ready.npz` (`verts_flat`/`sides_flat` + `node_offsets`, SANS n_max) à côté du dense pad+masque → entrée directe d'un encodeur agnostique (PointNet/Deep Sets, scatter-pooling). Cohérence dense↔ragged vérifiée
- ✅ `forge_LAR_2mosaic/remono.py` : re-forge **mono 1×1 EXACTE** depuis les `piece_grid.json` existants (aucune re-quantification, mêmes couleurs vérifiées 100 %, mêmes uuid → appariement conservé). Les 100 mono du pilote → `output/wikiart_inputs_mono/`
- ✅ Pilote **ultracompact** : mêmes 100 images (mono) × k={2,3,5,10-15} × 5 paliers = **2000 instances, 0 échec** → `output/pilot100_ultra_mono_compact/`. k=10-15 : n méd **33**, n_max **63** (vs 75/184 balanced). ⚠️ **ultracompact + petit k À PROSCRIRE** : k=2 → 5 sommets (rectangles purs, signal d'appariement mort). Sonde **mono+balanced = la PIRE combinaison** (n méd 169, max 262) : le levier est la COMBINAISON grille fine × croissance compacte, ni l'un ni l'autre seul
- ✅ **Mesure RePAIR réel avec NOTRE chaîne** (53 fragments isolés locaux) : contour brut méd 704 (211-1329) → **reco-B ε=1 % : méd 178 (132-291)**, perte 1,00 %. **Même ordre de grandeur que le LEGO** → un n_max joint ≈300 rend même le pad+masque transférable ; l'encodeur agnostique reste la voie la plus propre
- ✅ `readme.md` : sections **État de l'art** (tableau méthodes : représentation/archi/résultats) + **Datasets réels & benchmarks** ajoutées
- ✅ **Comparaison grille × croissance** (mêmes 100 images, k=10-15, L0, ε=1 %) :

  | combinaison | n méd | p95 | n_max | arêtes |
  |---|---:|---:|---:|---:|
  | poly-tuiles + balanced (défaut) | 75 | 122 | 184 | 27,1 |
  | poly-tuiles + compact | 60 | 107 | 183 | 27,1 |
  | mono + balanced (**pire**) | 169 | 262 | 262 | 27,0 |
  | mono + compact = ultracompact | **33** | 63 | **63** | 20,0 |

  `compact` seul : −20 %% de médiane mais **queue inchangée** (n_max 183≈184). Le levier = la **combinaison**. Ultracompact OK à k≥10 seulement (k≤5 → rectangles purs)
- ✅ **`--missing-max` pérennisé** : défaut = `round(0.15·n_frag_max)` (≡ réglage historique 0-2 à k=10-15, le 25k reste reproductible) ; flag = override
- ✅ `collate.py --n-max N` : **n_max JOINT imposé** cross-paliers/datasets/réel (erreur si < max observé ; `gnn_meta.json` garde `n_max_observed`). Repère : 304 couvre LEGO + RePAIR réel
- ✅ `visualize.py --recob` : **viz des 2 pertes d'aire** (`recob_viz.png`) — hachures grises = dégradation (érosion/trous, voulue), hachures couleur = encodage reco-B (ε), trait noir = `polygon_n`. `source.png` n'est PAS affecté par la perte d'encodage (vrais pixels découpés par le masque dégradé)

## À faire

### Forge (ICI)
- 🟡 **RUN WEEK-END — EN COURS (7/8 en ligne au 26/07 18:45)** : `poly_balanced/{k10-15,k05,k03,k02}` ✅ · `mono_compact/{k10-15,k05,k03}` ✅ · `mono_compact/k02` en forge (~68 %). Restent ensuite le collate+upload de k02 puis le renommage HF `balanced/` → `LEGACY_DO_NOT_USE/`. Résilience 3 étages (resume du script + `weekend_babysitter.sh` détaché + watchdog Claude) : 2 kills de process et 1 faille de resume rattrapés, **0 perte de donnée**. Détail initial ↓
- 🔴 **RUN WEEK-END (vendredi matin → lundi/mardi)** : matrice COMPLÈTE = **8 datasets / 200 000 instances**, hiérarchie **`<grille_croissance>/<k>/<palier>`** (`poly_balanced/` + `mono_compact/` × `k02·k03·k05·k10-15` × `L0…L4`, idem sur HF), pipeline séquentiel générer→collate→upload→nettoyage local, resume total, garde-fou 70 Go. Script prêt et testé bout-en-bout : `scripts/tools/weekend_full_matrix.sh`. ✅ **HF connecté** (token write `4datasetLEGO2HERO`). Lancement vendredi : `caffeinate -i zsh scripts/tools/weekend_full_matrix.sh > output/weekend.log 2>&1 &` (couvercle ouvert + secteur). Inclut la régénération du 25k (= `poly_balanced/k10-15`) et le **renommage** du legacy HF `balanced/` → `LEGACY_DO_NOT_USE/` (conservé, pas supprimé). Résout aussi la re-mesure des `n_max`
- 🚧 **DAFNE B** : `--frag-distribution {voronoi, clusters}` (découpe à tailles inégales = durcissement)
- 🚧 **DAFNE D** : parasites/distracteurs (fragments d'autres mosaïques à rejeter) — plus tard
- ⏳ Re-collate local `output/balanced/L*` (le générateur de README a changé)

### YOLO (ICI)
- 🚧 Entraînement **YOLO-Seg** sur `source.png` + `source_yolo.txt` — pas commencé
- 🚧 Split train/val (`yolo/`) + jitter d'augmentation

### GNN (Mathias)
- 🚧 **5 modèles, un par palier** (n_max différent par palier → pas de fine-tuning cross-palier avec la repré pad+masque actuelle) + scoring comparé L0→L4 (la métrique doit décroître avec la difficulté)
- 🚧 Choix représentation : pad n_max+masque (actuel) vs **encodeur n_max-agnostique** (PointNet/Deep Sets/Set Transformer : MLP partagé par sommet + pooling, **placé avant le message-passing**, entraîné avec le GNN) vs **k keypoints fixes** (ReassembleNet k=20 ; casse `polygon_n ⊆ polygon_raw`). Les deux dernières suppriment `n_max` et sont les seules transférables à RePAIR (contours réels 403-1457 sommets)
- 🔴 Impact du correctif reco-B : `n_max` monte (~1,5×) et les tenseurs changent → à intégrer avant tout gros entraînement

### VLM (Charles & Manon)
- jsp trop ce que vous allez faire, il y a moins de contraintes...


### Transfert / fine-tuning — datasets existants (cibles aval)

**Tour d'horizon (23/07) — RePAIR est le SEUL dataset public de fragments RÉELS
avec GT de réassemblage.** C'est pour ça que tout le monde pré-entraîne sur du
synthétique (et que notre forge existe). Personne ne publie de stats de sommets
(chacun les fait disparaître dans sa représentation) — notre mesure RePAIR est
donc un petit résultat original :

| dataset | fragments réels ? | GT réassemblage ? | sommets/fragment |
|---|---|---|---|
| **RePAIR** | ✅ Pompéi | ✅ (~1000 pièces annotées) | **mesuré par NOUS** : brut méd 704 (211-1329) → **méd 178 (132-291)** après reco-B ε=1 % |
| DAFNE | ❌ synth (fresques réelles découpées virtuellement) | ✅ | non publié — téléchargeable, mesurable avec notre chaîne |
| ReassembleNet semi-synth | ❌ (coupes synthétiques) | ✅ | non publié (tout est réduit à k=20) |
| PairingNet | ❌ (Pexels déchirées ; petit set « réel » 34 img/320 frags) | ✅ | non publié — pad 2900 / troncature 1408 borne leurs contours denses |
| POMPAAF / CLEOPATRA | ✅ réels | ❌ **style seulement** | non publié, inutilisable pour nous |
| GARF / Breaking Bad | ✅/❌ | ✅ | **3D**, hors périmètre |

- **RePAIR** (NeurIPS 2024) — fresques réelles, 121 objets / 957 fragments. **2D déjà en local** (`python_avance_CVG/FromLegoToHero/2fetchFORGE/2D_Fragments/`). Notre benchmark. <https://repairproject.github.io/RePAIR_dataset/>
- **ReassembleNet** (ICCV 2025) — semi-synth, 5000 mosaïques / 45 834 fragments. [arXiv:2505.21117](https://arxiv.org/abs/2505.21117) · [GitHub](https://github.com/adeela-islam/ReassembleNet) · [Drive](https://drive.google.com/drive/folders/1tflCUoct63Zhzt8dWs37vtfnphOqgReB?usp=sharing)
- **PairingNet** (ECCV 2024) — fragments d'images (390 Pexels → 8196 fragments). [arXiv:2312.08704](https://arxiv.org/abs/2312.08704)
- **DiffAssemble** (CVPR 2024) — graph-diffusion 2D/3D. [GitHub](https://github.com/IIT-PAVIS/DiffAssemble)
- 🚧 Expé transfert : `features/` sur les fragments RePAIR isolés (skip YOLO) + scoring vs GT de pose

### Différé
- ⏸ Réduire n_max au niveau **forge/tiling** (le levier réel ; `compact` ne suffit pas) — si n_max bloque
- ⏸ Normalisation d'échelle adimensionnelle (area/aire_image, perimeter/√area) — avant transfert cross-échelle
