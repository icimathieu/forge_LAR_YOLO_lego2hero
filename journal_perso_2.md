# état de l'art collectif et définitif :

États de l’art déjà dispos : 
A Survey on Computational Solutions for Reconstructing Complete Objects by Reassembling Their Fractured Parts (arXiv:2410.14770, mise à jour 2025). La référence de synthèse : couvre algorithmes, datasets, packages open-source, applications.
Computational techniques for virtual reconstruction of fragmented archaeological textiles (npj Heritage Science 2023). Classe les méthodes par type d'input et type de fragmentation.
Thématique : 

Use case & autres méthodes : 
ReassembleNet (Islam et al., arXiv:2505.21117, mai 2025) — Learnable Keypoints and Diffusion for 2D Fresco Reconstruction. La référence la plus directement pertinente. Chaque pièce est représentée par un ensemble de keypoints de contour, et le modèle apprend à sélectionner les plus informatifs via des techniques de GNN pooling. Combine géométrie + texture dans un bloc d'attention inter- et intra-pièce, le tout dans un processus de diffusion qui dénoise itérativement les poses. Dataset semi-synthétique de fresques. Résout les trois limitations identifiées : scalabilité, multimodalité, applicabilité réelle (formes irrégulières, érosion complexe).


Nash Meets Wertheimer (Tsesmelis et al., arXiv:2410.16857, 2024) — Using Good Continuation in Jigsaw Puzzles. Exploite la loi gestaltiste de bonne continuation (les lignes et courbes qui traversent les bords des fragments doivent continuer). Ignore volontairement couleur et forme, garde uniquement les patterns géométriques linéaires. Testé sur fresques archéologiques réelles. Important pour le projet : une contrainte "continuité des motifs" du type discuté dans la formalisation mathématique du projet est implémentable comme feature appris.
Corpus synthétique : 
https://github.com/debkbanerji/lego-art-remix 
qui utilise le modèle : 
Ranftl, René, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. "Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer." (2020). IEEE Transactions on Pattern Analysis and Machine Intelligence

appli de joachim gassen

datasets de production de fragments archéologiques à partir de données propres OU reconstruction à partir de fragments (objet final complet ou non) — **statut vérifié 08/06/2026** :
- **RePAIR** (NeurIPS 2024, [site](https://repairproject.github.io/RePAIR_dataset/)) — RÉEL, 121 fresques de Pompéi, 957 fragments avec GT (positions + orientations) annotée par archéologues. **Seul dataset à fournir une suite de métriques officielles complète et transférable à notre link-prediction** (Q_pos, RMSE rotation/translation, précision/rappel/F1 sur le *mating graph*). C'est notre cible de transfert ET notre source de métriques.
- **DAFNE** = *Digital Anastylosis of Frescoes challeNgE* (Dondi, Lombardi, Setti — Univ. Pavie/CVMLab ; *Pattern Recognition Letters* 2020, vol.138 pp.631-637 ; [dataset](https://vision.unipv.it/DAFchallenge/DAFNE_dataset/)). SYNTHÉTIQUE, réassemblage 2D : 62 fresques × 18 configurations, tessellation par plan aléatoire + érosion. ⚠️ Il n'existe **pas** de « DAFNE 1 & 2 » : DB1/DB2 sont les splits *train/test* (le « 2 » vient de l'URL d'un special issue). **Ses 5 paramètres de difficulté = gabarit direct pour notre phase de dégradation** : A=nb fragments, B=type de distribution, C=% fragments manquants, D=% fragments parasites (distracteurs), E=ratio d'érosion. Cf. section *dégradation*.
- **CLEOPATRA** (BIPlab, Univ. Salerne ; Cascone et al. 2023, *JAIHC* 14(4):4087-4097 ; [page](http://www.biplab.unisa.it/home/cleopatra/)) — ⚠️ **classification de STYLE pictural de fragments, PAS réassemblage** (11 styles/époques, test ~80 pièces). Tâche amont — pertinent pour le VLM de Charles, pas pour le GNN de placement. Aucune métrique de placement.
- **POMPAAF** = *Pompeii Archive Artistic-styles Fragments* (Elkin et al., Ben-Gurion Univ. ; [arXiv:2501.00836](https://arxiv.org/abs/2501.00836)) — ⚠️ idem : **classification de style** (4 styles pompéiens), PAS réassemblage. 311 images de base, fragmentées en 12/40/84/160 pièces ; évaluation = accuracy/F1 de classif.

YOLO : masques & lissage + encodage des points 
GroundCount (arXiv:2603.10978, 2026). Grounding Vision-Language Models with Object Detection for Mitigating Counting Hallucinations. Le titre dit tout. Le paper note que les VLM autorégressifs hallucinent sur les tâches de comptage ("combien de bols ?") — même en raisonnant pas à pas. Les modèles de détection d'objets (YOLO, DETR) font ça trivialement et fournissent des sorties structurées (bounding boxes, scores, classes). La contribution : injecter les détections YOLOv13x en prompt du VLM — avec un encodage spatial (grille 3×3, ordonnancement gauche-droite / bas-haut) — réduit les hallucinations et améliore le comptage. Exactement le pipeline envisagé dans le projet.




VLM : 
VLMs seuls sur tâches équivalentes sont nuls : 
Jigsaw-Puzzles (Wang et al., arXiv:2505.20728, mai 2025). Benchmark de 1100 images avec haute complexité spatiale, cinq tâches pour évaluer la perception spatiale, la compréhension structurelle et le raisonnement. Évaluation sur 24 VLM state-of-the-art. Le meilleur modèle, Gemini-2.5-Pro, atteint 77.14% en accuracy globale et 30% seulement sur la tâche d'Order Generation (reconstruire l'ordre correct des pièces), face à >90% pour des humains.


AGILE (Zeng et al., arXiv:2510.01304, oct. 2025 — v3 fév. 2026). Constate que sur des puzzles simples 2×2, les VLM existants performent près du hasard (~9.5%). Propose un framework agentique où le VLM génère du code Python pour interagir avec le puzzle (swap de tuiles, crop, zoom), avec reward sur la résolution. Performance passe de 9.5% à 82.8% en 2×2, et amélioration moyenne de 3.1% sur 9 benchmarks de vision générale (indique que résoudre des jigsaws transfère à d'autres tâches visuelles).


LEGO Co-builder (arXiv:2507.05515, juillet 2025). Benchmark pour évaluer la capacité des VLM à suivre des instructions d'assemblage LEGO étape par étape. Évalue 9 VLMs (GPT-4o, Gemini, Qwen-VL, LLaVA, BLIP-2…). Résultat brutal : GPT-4o atteint seulement 40.54% F1 sur la détection d'état (détecter qu'une pièce est mal placée). Confirme que les VLM grand public ne sont pas fiables pour l'assemblage fin.



GNN : 
A Gentle Introduction to Graph Neural Networks (distill.pub). Graph representation learning (cs.mcgill.ca). Graph Neural Networks: A Review of Methods and Applications (arXiv:1812.08434). Généralistes. Bonus : Enhancing Graph-based Learning through Sheaf Theory (Tesi_dottorato_Cassarà.pdf). Surtout intéressant pour le non-linear sheaf laplacian (que je connaissais pas et qui minimise la distance en variation totale (intéressant pour nous); cf.3.2.2. & thm A.1.3 ?
Recurrent Relational Networks (arXiv:1711.08028v4). Préhistoire. La nature des puzzles est assez différente de la notre (ils utilisent des sudoku). 
Thompson et al. 2020 — Building LEGO Using Deep Generative Models of Graphs (arXiv:2012.11543, NeurIPS 2020 workshop). Modélise un set LEGO comme un graphe dont les nœuds sont les briques et les arêtes encodent les contraintes de support. Entraîne un modèle génératif par réseau de graphes pour produire des constructions plausibles. C'est la référence canonique pour la formulation LEGO = graphe.
Using Graph Neural Networks to Reconstruct Ancient Documents (arXiv:2011.07048). On s’est fait devancer!
Graph Neural Networks: Self-supervised Learning (ssl_for_gnns.pdf). Juste pour la Figure 5.
Distributional Neural Networks for Automatic Resolution of Crossword Puzzles (P15-2033.pdf). Ouverture aux mots croisés (techniques existantes).
Solving Jigsaw Puzzles By The Graph Connection Laplacian (arXiv:1811.03188v5). OH MERDE! Mais… est-ce que ce ne serait pas exactement ce que je cherche depuis plusieurs mois + ça pourrait fitter parfaitement pour le projet ? Problème: les hypothèses sont trop contraignantes (découpage en carrés)! Faudrait pousser le bins et voir ce que l’on peut en faire!
Dirichlet Energy Constrained Learning for Deep Graph Neural Networks (arXiv:2107.02392v1). Pour minimisation de fonctionnelle. À voir plus tard comment l’utiliser et ce qui a été fait depuis. (Bonus.)

PUZZLES: A Benchmark for Neural Algorithmic Reasoning (openreview.net). Pas exactement pour la partie GNN mais a le mérite d’exister.




GLM !? 

3D…
Li, S., Jiang, Z., Chen, G., Xu, C., Tan, S., Wang, X., ... & Zhang, J. (2025). Garf: Learning generalizable 3d reassembly for real-world fractures. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 5711-5721). 
eux ils ont m^ fait leur propre dataset…

Exemples d’institutions qui l’ont fait a la mano miskin :
BOOM preuve que le projet est super utile : la reconstruction manuelle du patrimoine fragmenté : état des lieux 
La reconstruction d'objets patrimoniaux fragmentés constitue l'un des défis les plus chronophages de la recherche archéologique contemporaine. Avant d'envisager toute approche computationnelle, il convient de mesurer précisément ce que représente la reconstruction manuelle en termes de coût humain, d'échelle et de limites structurelles.
(en gros j’ai mis 3 cas : un objet, un site et un chantier en cours)
Le coût humain à l'échelle d'un objet : le casque de Sutton Hoo
Le cas le plus documenté est celui du casque de Sutton Hoo, conservé au British Museum. Le conservateur Nigel Williams a passé dix-huit mois; dont une année pleine de travail effectif,à réarranger les plus de 500 fragments de cet objet entre 1970 et 1971 (j’espère vraiment pour lui qu’il était autiste), dans des conditions particulièrement défavorables : aucune photographie des fragments in situ n'avait été prise lors de la fouille originale de 1939, et leurs positions relatives n'avaient pas été enregistrées. Ce même conservateur a par ailleurs reconstitué les près de 31 000 fragments de vases grecs retrouvés dans l'épave du HMS Colossus (1787). AKA même pour des corpus bien délimités, confiés à un spécialiste de haut niveau, la reconstruction manuelle mobilise des ressources humaines considérables pour des résultats unitaires.
L'impossibilité à l'échelle d'un site : les fresques de Pompéi et le projet RePAIR 
Le passage à l'échelle d'un site archéologique majeur rend la reconstruction manuelle structurellement intenable. Les entrepôts du Parc Archéologique de Pompéii contiennent environ 10 000 fragments de fresques en attente de réassemblage, et ce n'est, selon la conservatrice Elena Gravina, "qu'une petite partie" de l'ensemble disponible. Depuis 2018, une équipe de l'Université de Lausanne dirigée par le professeur Michel E. Fuchs travaille manuellement sur un seul ensemble, la Maison des Peintres au Travail, sans avoir pu couvrir l'intégralité des fragments disponibles.
Le projet européen RePAIR a tenté de quantifier précisément ce goulot d'étranglement. Le dataset constitué dans ce cadre contient 16 000 fragments ; la vérité terrain a été générée après plusieurs années de travail de terrain continu : fouille, nettoyage, puis résolution manuelle du puzzle par des archéologues,  sur un sous-ensemble d'environ 1 000 pièces seulement, soit 6% du total. Les 15 000 fragments restants sont considérés comme hors de portée du travail humain dans tout horizon temporel raisonnable. Le coordinateur du projet, le professeur Marcello Pelillo, résume l'enjeu ainsi : l'objectif est d'éliminer l'une des activités les plus laborieuses et frustrantes de la recherche archéologique, permettant ainsi de canaliser énergie et expertise vers des activités plus strictement scientifiques et créatives 
La durée structurelle d'un chantier de reconstruction : les fresques d'Akrotiri
Le site minoen d'Akrotiri (Santorin, ~1627 av. J.-C.), fouillé depuis 1967, offre un troisième éclairage sur la temporalité de la reconstruction manuelle. Les fragments de fresques, de quelques centimètres à quelques dizaines de centimètres, doivent être triés, catalogués et appariés manuellement, un processus prenant typiquement plusieurs années par ensemble. La fresque des Singes Bleus, aujourd'hui partiellement restaurée au Musée National Archéologique d'Athènes, illustre l'état habituel de ces collections : plus de cinquante ans après le début des fouilles, des pans entiers restent non assemblés. C'est précisément ce constat qui a motivé le développement d'outils computationnels d'assistance à la reconstruction dès les années 2000 (Bronstein, Bronstein & Kimmel, Princeton Graphics Group).
DOOONC 
La reconstruction manuelle n'est pas seulement lente, elle est non-reproductible, non-répétable et non-scalable. À l'échelle d'un fragment isolé, elle mobilise des années d'expertise ; à l'échelle d'un site, elle laisse structurellement la majorité des collections inaccessibles.

# journal de bord collectif : 

## 07/04/2026 (ékipafond): 
CR :  
Je mets les notes du début ici pour pas être en emploi fictif : 

Notes réunion  : RECONSTITUTION 3D VS MISSILES
Pas de compet’ :
7 T d’images sur le dataset turc.
Accès au cloud List (cloud européen pour la recherche) è Permet le gros transfert de fichier
Créer un google drive, mais max 16 gb.
 
Truc utile pour les mémoires de diplomatie ?
 Classification ouverte, reconnaissance è faire des labels è automatisable facilement.
R/ on paye au million de token, coût plutôt faible.
 
Reconstruction 3D automatique à partir d’image bruitées
Ex : dans une pièce, 4 photos d’une pièce avec un peu de flou, automatiquement générer un modèle 3D de la pièce.
ð  HuggingFace
Question du jeu de données ; lesquels sont communicables ?
DGSE, images déclassifiées ?
 
Propositon mathias : algo de reconstitution de construction de legos è possibilité de l’adapter à des débris d’obus.
AKA : mapper le lego pour qu’on puisse envoyer le script sur des parpins puis l’appliquer direct à des images nouvelles jamais entrainées pour avant.
ð  Logique de reconstruction et de transfert intéressante, à étudier.
DONC è reconstruire n’importe quoi à partir de débris.
 
Simulation d’environnement physique où on fait exploser qlq chose è applicable catastrophe environnementales.
è Demander éventuellement extension jusqu’à fin juin.
Marisol : chercher applications HN/ ONU pour motiver le projet.
Réduxtion de la complexité
+ : tâches pour tout le monde (dont du scrapping).
ð  « forcer à ce que ça puisse s’emboiter grâce à l’albumentation » // ou sur Blender.
o   Finetuning ?
ð  Solides déformables ?
 
Caler un truc utile à chacun pour le projet.
Ébauche mail CVG : 
Objet :  From Lego to… Artifacts? HERO ? 
 
Monsieur, 
 
La compétition qui portait sur la classification d’images aéronautiques a été annulée. Cependant, en discutant nous avons eu une idée : faire de la reconstruction d’objets à partir de débris. On a pensé que cela pourrait être utile notamment dans les cas de reconstitution d’objets archéologiques (dinosaures, vases, pièces de monnaies, monuments etc..).
 
L’idée serait d’entrainer un algorithme assembleur de sets Lego[1]. On a pensé à la 3D mais… c’est trop gourmand et… on n’a pas d’argent. Une question demeure, est-ce qu’on utilise l’algorithme pour constituer un jeu d’entraînement pour notre modèle ou est-ce qu’on le fine tune ? 
Pour la généralisation, on voudrait déformer les pièces. Si la généralisation ne marche pas, nous serons obligés de reconstituer un autre set d’entraînement. 
 
Tâche de reconstitution pure : Faire de l’optimisation sous contrainte sur des graphes (contraintes typiques qu’on a normalement sur des pièces).  
 
Utilisation d’un GNN
___________________________________________________________________ 
 Méthodologie : 

on commencerait par un mur lego dont le graphe d’assemblage serait aisément reproductible à partir d’une photo 2D → use case : mosaique/fresque éclatée
puis construction lego “avec de la profondeur” : 2 murs perpendiculaires ? 




Pipeline :
 
 
- Détection YOLO des objets, chacun appartenant à une classe (LEGO : un nb de classes restreints, kapla : une seule classe) + détection des contiguïtés 
-  Méthode minimisation de fonctionnelles / génération de graphes qui représentent la méthode d’assemblage des objets
- Entraînement du GNN
- (Modèle de décision)
- Visualisation de l’objet


[1] Cf : Article « from bricks to bot »

360er0/awesome-lego-machine-learning: A curated list of resources dedicated to Machine Learning applications to LEGO bricks
Réponse CVG : 

## 15/04/2026 (Manon & Charles) : 
Présents: Charles, Manon
Ordre du jour: Détermination d’un plan d’action, planning prévisionnel pour son application 

Tâches identifiées: 
Faire un big état de l’art: Modèles de reconstruction
YOLO: entraînement pour détection des morceaux et annotation briques? (On se questionne sur la pertinence de l’annotation étant donné qu’il n’y aura pas de classes prédéfinies dans la généralisation)
Test reconstruction du set; test méthodes
Augmentation: en sortie pièces altérées (+ target)
Re test reconstruction avec altérations
MoE sur les différents modèles 
Pipeline proposée (en utilisation, pas entraînement):
Yolo pour détection des pièces
Modèle de reconstruction: VLM / GNN / autre à identifier dans l’état de l’art
MoE 

Calendrier prévisionnel proposé:

1er mai: état de l’art done
D’ici le 1er mai: définir date d’un date lego shop pour set yolo
8 mai: yolo modèle done
29 mai: modèles testés et fonctionnels
12 juin: augmentations faites
26 juin: premiers résultats sur modèles avec augmentations
10 juillet: Premier jet compte-rendu de nos recherches
Pour la question du VLM:
On propose d’essayer d’appeler un VLM sur la target et les pièces en lui demandant de reconstruire la target avec les pièces. Voir si ça peut donner des résultats.
À défaut, pour construire la target à terme: on lui feed des exemples du genre d’objet qu’on attend (par exemple plein de mosaïques) et les pièces. L’idée serait qu’il génère une target approximative pour les modèles de reconstitution.
Sachant que le VLM en lui-même pourrait être un modèle de reconstitution. 
Pipeline : 
On a au départ toutes les pièces (peut être à ordonner selon leur unicité / taux d’information au départ).
P est l’ensemble des pièces qu’on a à assembler.
On a en départ un ensemble C = P de pièces  célibataires, B = ø l’ensemble des blocs de pièces. 
Un bloc b_i est un ensemble de la forme {p_i, p_j, …, p_n}. 
Pour chaque pièce libre c_i ∈ C, est-ce qu’elle semble être voisine d’un bloc existant b_i ∈ B / est-ce qu’elle est voisine d’une des autres pièces c_j ∈ C? 
Si oui, alors c_i ∈ b_i / c_i, c_j = b_j ∈ B
Sinon, on passe à la suivante. On fait un tour par pièce. On peut retenter une fois ou deux par pièce restante, si jamais elles ne sont pas classées quoiqu’il, on augmente un peu la température 
À la fin, C = ø et P ⊂ B 
## 23/04/2026 (ékipafond): 
Mathias propose VLM pour détection de contour et contextuel. Il sert pour les informations contextuelles, et le graphe pour la reconstitution.

refonte propre de la pipeline et distribution des rôles / du planning : 
2 pipelines : une VLM seule 
une VLM-GNN (VLM fournit une table de vérité)
toujours classifieur YOLO (masques) avt VLM pour identifier les pièces et leur nombre
pbm LLM soulevé par Mathias (et que son petit notebook confirme cf PetitsMotsADN_From_LEGO_to_HERO.ipynb)
production du corpus lego : couper mosaïques aléatoirement (et casser les agrégats/aplats de m^ couleur de pièces). environ 10-15 fragments par image

annotation : 
pour YOLO (masques)
graphes pour GNN

bonus : 
explorer GLM : intérêt Language Model de spécifier un objet-visé (“rassemble moi ces pièces qui forment un vase”)
comment lissera-t-on nos fragments pour faire le graphe ? 
graphe : chaque noeud un fragment & arêtes pour fragments contingus & toutes les informations du fragment sont dans le noeuf !? OU chaque noeud un côté du fragment DONC plusieurs graphes en entrée (chaque fragment) pour former un graphe final

# notes personnelles : 
## 04/2026 : 
Pipeline : 
CoT VLM avec reconstruction proche en proche / classification hiérarchique
VLM d’un coup avec MoE à partir des pièces éclatées doit reconstruire
GNN à partir d’un Yolo qui détecte et classe
Toujours un yolo au départ du détecte (et classe ? charles dit que c useless mais nous permet de commencer par cas simple…)
Forcément une étape de dégradation de nos images (albumentations)
Faut forcément avoir un dataset annoté prêt pour entraîner nos modèles
Plusieurs étapes en fait : 
0 - modèle YOLO qui à partir d’une image de toutes les pièces en fait une liste. Détection + classification
Pour le non-lego peut être détection + mask comme ça on a qd même des objets avec forme, etc. pour le VLM. éventuellement à partir des masques on peut refaire du puzzle ? 

1 - Créer un modèle VLM qui sait à partir d’un set de plate/briques donnés (et taille de la mosaique finale), renconstruire une mosaique avec un prompt qui lui décrit à quoi doit ressembler la reconstruction. 

2 - à partir de ce modèle, l’entraîner pour considérer qu’il puisse y avoir des trous dans sa mosaïque (on donne toujours taille mosaique finale + peut calculer taille sommée de ses plate/briques) 
Pour créer les fragments de mosaiques suffit de prendre l’image, la découper en fragments, éventuellement en supprimer qqs uns puis on envoie au VLM les fragments détectés par YOLO + prompt-goal

3 - Puis faire plein de sous-modèles fine-tunés : 
Un pour les fresques
Un pour les mosaiques
Un pour les vases
Etc. 
Qui a toujours du prompt + connaît déjà son objet type. 
→ pour cet entraînement il faudra lui donner des images-target + un prompt descriptif + une liste d’objets détectés par YOLO

4 - à partir de tous ces sous-modèles faire une mixture of experts capables d’arbitrer entre ces sous-modèles ? 

Sur l’incomplétude : on peut demander à partir d’un set incomplet de produire une mosaique/un vase complet AVEC les fragments donnés + du généré ?
Dataset d’entraînement YOLO : 
https://lego-art-remix.com/ (LAR)
Ou https://github.com/joachim-gassen/legoartmosaic 
Pour générer des mosaiques lego de plate à partir d’images. Et donne aussi les pièces nécessaires. 
À partir de cette liste de pièce il faudrait obtenir automatiquement une image de toutes les pièces les unes à côté des autres 
https://www.bricklink.com/catalogList.asp?v=0&pg=1&catString=26&catType=P 
→ ici les 44 plate existantes. À multiplier par notre nb de couleurs 
Lego-art-remix liste les couleurs disponibles. 

On récupère aussi sur bricklink une image pour chaque pièce-couleur → on pourra générer des images avec toutes les pièces. Cette image passe dans détecteur YOLO. 

On utilise la mosaïque lego de l’image comme target d’entraînement pour le VLM. 
Datasets use-case : 
https://www.kaggle.com/datasets/gisemos/classification-data-of-mosaics-and-frescos?resource=download
Mosaiques et fresques très faciles à passer dans la moulinette LAR (à partir de ce dataset qu’on créé nos mosaiques-lego d’entraînement ?)

PROBLÈME : va apprendre à générer des trous… les supprimer pour qu’ils sachent compléter ? 

PROBLÈME2 : on va être obligé de faire une pipeline par type d’objet ? si on ne lui donne que des pièces ? 
Ou aussi intégrer une partie texte descriptive, exp : “ce sont des fragments de vase. Reconstitue le vase sachant qu’il y a des pièces manquantes, dont des parties du vase qui ne sont pas là.”
→ comment font les puzzle-solver ? “Generating Physically Stable and Buildable Brick Structures from Text” font de la next-brick probability…
Production dataset lego + graphes : 
Bon l’appli conseillé par claude ct à chier bref il faut que je découpe mes mosaïques legos. 
En y réfléchissant ça ne va pas fonctionner car je vais sectionner mon image en plusieurs fragments de plusieurs aplats… 
BREF même si je récupère le graphe des aplats (pas compliqué avec une grille 100*100 et les couleurs je pense) il faut toujours que je sois capable de faire le m^ sectionnage entre le graphe et l’image…

## 11/05/2026 : 

**Objectif de la journée** : trouver et valider la technique pour forger le dataset LEGO end-to-end. L'industrialisation (50+ mosaïques) viendra après.

### Décisions techniques

1. **Récupération des positions des pièces LEGO** : détection de joints sur l'image (CV pure) plutôt que de détourner le JS de lego-art-remix. Choix motivé par la réutilisabilité — le même détecteur servira pour les images HDA dégradées plus tard, et il n'y a pas de dépendance externe au site déployé.

2. **Pas de coupage des pièces LEGO** lors de la fragmentation. Les frontières des fragments suivent uniquement les joints existants entre pièces. La fragmentation ne dépend donc que du tiling LEGO, pas du contenu visuel.

3. **10 à 15 fragments par mosaïque**, tirés aléatoirement.

4. **Représentation géométrique des fragments** : chaque fragment est rééchantillonné en **n=16 sommets équidistants** le long de son périmètre. Ce choix résout trois contraintes simultanément :
    - permet d'encoder *tous les côtés et points* du fragment dans un vecteur de features de **taille fixe** (PyG-compatible), donc la "logique puzzle" est conservée (chaque côté a sa signature)
    - s'aligne avec la future simplification des fragments HDA (qui tendra aussi à donner *n* côtés)
    - le polygone brut (taille variable) est conservé en métadonnée pour pouvoir changer d'approche plus tard sans rerunner la pipeline
    - ⚠️ **Révisé le 08/06/2026** : on abandonne le `n` fixe (et le padding collinéaire option b) au profit d'un `n` **variable** par fragment (toujours < nb de sommets réels) + invariant `polygon_n ⊆ polygon_raw` + resample reflex-aware. Voir la section *« 08/06/2026 — Décisions géométrie + métrique »*.

5. **Deux graphes sauvegardés par mosaïque** :
    - `graph_complete.json` : N nœuds (features géom du fragment) + arêtes (adjacences avec géom du joint partagé) → **target du GNN**
    - `graph_fragments.json` : mêmes N nœuds, **zéro arête** → **input du GNN**

    Ce découpage formule explicitement la tâche : "à partir de graph_fragments, prédire graph_complete". Le choix de l'architecture GNN (link prediction, edge classification sur graphe dense, etc.) est laissé aux entraînements futurs.

6. **Annotation YOLO automatique** : on place nous-mêmes les fragments sur le canvas blanc, donc on connaît au pixel près où ils sont. Les polygones YOLO-Seg sont exportés directement par la pipeline — **vérité terrain synthétique exacte, aucune annotation manuelle**. (Vérification visuelle via `source_yolo_viz.png`.) L'annotation manuelle (type Label Studio) ne serait utile que pour un futur volet HDA sur photos réelles.

7. **Rotation des fragments dans l'image éclatée** : axes-aligned (0/90/180/270) ± 5° aléatoire. Compromis entre simplicité YOLO et variabilité réaliste.

8. **Palette LEGO** : 5 couleurs gris/argent sont des faux-positifs du détecteur de joints gris — **liste complète** : Light Bluish Gray ~(160,165,169), Dark Bluish Gray ~(108,110,117), Flat Silver ~(137,135,136), Pearl Light Gray ~(157,158,154), Metallic Silver ~(166,168,166). ✅ **Depuis le 08/06/2026 le détecteur tolère les pièces grises** (un joint gris n'en est pas un si les 2 cellules adjacentes sont elles-mêmes grises), donc cette exclusion est devenue une **recommandation douce** plutôt qu'une contrainte dure (seul cas non résolu : deux pièces grises *différentes* séparées par un joint gris). Liste de référence répétée en section *Scraping*.

### Pipeline livrée

```
lego2hero/
├── forge_LAR_2mosaic/              # AMONT : image → mosaïque LEGO + piece_grid.json
│   ├── mosaic.py · cli.py · batch.py · palette.py
│   └── README_forge_LAR.md
├── mosaic2fragments/               # mosaïque → fragments → YOLO + graphes GNN
│   ├── forge_dataset.py            # pipeline principal end-to-end
│   ├── batch.py                    # driver multi-mosaïques
│   └── visualize.py                # overlay YOLO debug + sanity checks
├── exp1/
│   ├── canvas_mosaic.png           # image de référence (48×48, 40 px/stud)
│   ├── canvas_pixels.png
│   └── Enfant Apprendre à dessiner un arbre (11).jpg
└── dataset/
    └── mosaic_XXX/
        ├── target.png              # mosaïque complète (= canvas_mosaic.png)
        ├── source.png              # fragments éclatés sur fond blanc
        ├── source_yolo.txt         # polygones YOLO-Seg normalisés [0,1]
        ├── source_yolo_viz.png     # overlay debug
        ├── pieces.json             # debug : pièces LEGO détectées
        ├── graph_complete.json     # target GNN
        ├── graph_fragments.json    # input GNN
        └── fragments/frag_XX.png   # crops alpha par fragment
```

### Étapes du pipeline (forge_dataset.py)

1. **Détection de joints** par échantillonnage RGB ~136 sur la grille (3 samples par joint pour robustesse)
2. **Union-find** sur les cellules de la grille → liste des pièces LEGO (bbox grille + couleur)
3. **Sanity checks** : couverture totale, rectangularité des pièces, détection ciblée de sur-détection sur tuiles grises
4. **Construction du graphe d'adjacence pièce-à-pièce** (networkx)
5. **Sélection des seeds par farthest-point sampling** + **BFS à file de priorité** (le plus petit fragment grandit en premier) → fragments équilibrés en taille
6. **Extraction du contour** de chaque fragment via `cv2.findContours` sur un masque rasterisé, puis **rééchantillonnage** à n=16 points équidistants
7. **Features par fragment, séparées en deux blocs pour éviter le leak de position** :
    - `gnn_input` (9 floats, dans `graph_fragments.json` ET `graph_complete.json`) : area, perimeter, R, G, B, n_pieces, n_cells, bbox_w, bbox_h
    - `polygon_n_canonical` : 16 points centrés au centroïde (pas de leak)
    - `side_features` : par côté, length, angle, RGB échantillonné à l'intérieur
    - `target_info` (UNIQUEMENT dans `graph_complete.json`) : centroid (cx, cy), polygon_n_absolute, polygon_raw, mean_color, n_pieces — supervision et viz, à NE PAS donner au GNN en input
8. **Adjacences entre fragments** : segments partagés sur la grille + mapping côté-à-côté (vote sur multi-échantillonnage)
9. **Placement aléatoire** axes-aligned ± 5° rotation, sans chevauchement, sur canvas 3500×3500
10. **Export YOLO** via contour du masque alpha rotaté → coords normalisées

### Résultats validation

5 mosaïques générées avec seeds différents sur la même `canvas_mosaic.png` :

| sample | fragments | adjacences | pieces | balance n_pieces/fragment (min/max/mean) |
|---|---|---|---|---|
| 000 | 11 | 22 | 609 | 31 / 50 / 40.6 |
| 001 | 14 | 28 | 609 | similaire |
| 002 | 11 | 22 | 609 | similaire |
| 003 | 15 | 34 | 609 | similaire |
| 004 | 10 | 20 | 609 | similaire |

Pipeline robuste, fragments bien équilibrés, masques YOLO précis (vérifiés visuellement), graphes cohérents.

### Limitations connues

- **Side-mapping incomplet** : ~3% des arêtes n'ont pas de `src_side_idx`/`dst_side_idx` quand la frontière partagée est très courte (~1 stud) et que l'échantillonnage à n=16 ne la capture pas. Tolérable pour V1, fixable plus tard via le polygone brut.
- **Lissage des coins** : le rééchantillonnage équidistant n'aligne pas les sommets sur les coins rectilignes naturels. C'est cohérent avec la simplification HDA, et le polygone brut reste accessible.
- ✅ **Robustesse aux pièces grises (08/06/2026)** : `detect_joints(tolerate_gray=True)` ignore un joint gris quand les 2 cellules adjacentes sont elles-mêmes grises (= intérieur d'une pièce grise). L'exclusion de palette (Light/Dark Bluish Gray, Flat Silver, Pearl Light Gray, Metallic Silver) devient une recommandation douce. Reste non résolu : deux pièces grises *différentes* contiguës séparées par un joint gris (cas rare, accepté).

### 08/06/2026 — Décisions géométrie + métrique (post-feedback Chahan)

**Contexte** : digestion du mail de Chahan (`2026-05-17_remarques_CVG`). Décisions actées :

1. **Invariant géométrique fondamental — `polygon_n ⊆ polygon_raw` partout (érosion monotone).** Le polygone encodé (« pièce virtuelle ») ne **gagne jamais** d'aire vs la pièce réelle ; au pire il en perd (« comme si ses coins étaient cassés »). C'est l'invariant qui rend la cible **physiquement réalisable** : des polygones tous inscrits sont 2-à-2 disjoints → ils laissent des trous, **jamais d'overlap impossible** (= Régime-2 « eroded gaps », pas un cas insoluble). Protège aussi toute métrique d'overlap (Q_pos). Origine du bug évité : le resample équidistant actuel ([forge_dataset.py:334](mosaic2fragments/forge_dataset.py:334)) *gagne* de l'aire aux coins **concaves (reflex)** non échantillonnés (la corde enjambe l'encoche).

2. **`n` (nb de sommets du polygone encodé) : variable, plus fixe.** Abandon de `n=16` littéral et du padding collinéaire (b). À la place : `n` tiré dans un intervalle, **toujours < nb de sommets réels** du fragment (~50 % en exemple). Le « padding fixe » qu'on garde n'est PAS un `n` littéral commun mais le **même schéma d'encodage cross-domaine** (LEGO ↔ fresque). ⚠️ conséquence GNN : `n` variable ⇒ `polygon_n_canonical` n'est plus un vecteur de taille fixe → prévoir padding à `n_max` + **masque de validité**, ou un mini-encodeur d'ensemble par nœud (à trancher à l'implémentation). Garde-fou : ne pas descendre `n` trop bas, sinon on détruit les indices de coin que le matching exploite.

3. **Méthode d'encodage retenue — « reco B » : resample reflex-aware.** Garder **obligatoirement tous les sommets concaves (reflex)** du contour ; ne **couper que les sommets convexes** (corde inscrite → perte d'aire only). Garantit l'invariant (1) par construction, donne un `n` variable ≤ réel (2), et **préserve l'info d'emboîtement** (les coins concaves portent le matching). Déterministe, sans dépendance externe. *Option paranoïa* : clipper le résultat de B sur `polygon_raw` (= méthode A, intersection Shapely) en filet de sécurité — quasi inutile mais garantit ⊆ même cas pathologique. **On code ça ce soir.**
   - ✅ **Prototype validé** (`scratch/reco_b_prototype.py`, sur les 61 fragments des 5 mosaïques) : équidistant actuel `n=16` → **4,83 % d'aire fantôme, 61/61 fragments débordent** (jusqu'à 12,8 %). Reco B reflex-aware → **0,13 %, 1/61** (1 contour pathologique). Reco B **+ clip** → **0,000 %, 0/61** garanti.
   - ⚠️ **Contrainte plancher découverte** : `n ≥ #reflex` (sommets concaves, moy ~22 ici) — **impossible de simplifier sous le nombre de coins concaves** sans re-gagner de l'aire. Donc l'item-2 « `n` < 50 % des coins » n'est atteignable que si peu de coins sont concaves ; sur ces fragments LEGO, le `n` variable retenu vaut ~57 % des coins réels (moy 20,7 vs 36,4).
   - Pourquoi pas la **méthode A seule** (intersection `resample ∩ polygon_raw`) : elle garantit (1) maintenant que `n` est libre, mais elle va à l'**inverse** de (2) — l'intersection *ajoute* des sommets au lieu de simplifier — et nécessite un clipping polygonal robuste (risque de slivers). B domine ; A reste utile seulement comme garde-fou.

4. **La dégradation ne corrompt PAS la vérité terrain — point central du curriculum.** La GT du GNN est la **relation d'assemblage** (ensemble d'arêtes pour le link-prediction ; pose R,T pour la régression plus tard), définie sur la **partition grille intacte** et **invariante** sous toute dégradation géométrique. L'érosion/simplification ne touche QUE l'**input** (géométrie des nœuds) ; le label reste figé. La métrique compare prédiction-de-relation vs GT-de-relation, **jamais** géométrie-cassée vs géométrie-intacte → **pas de baisse automatique de métrique**. (C'est le fonctionnement de RePAIR/ReassembleNet/DiffAssemble : fragments érodés en entrée, GT de pose issue de l'objet intact.) Donc `graph_complete.json` n'a **pas besoin** des bouts cassés : il porte le *label*, pas la géométrie matchée. Seul risque résiduel : sur-éroder au point de détruire le signal géométrique nécessaire au matching — ce n'est pas un problème de GT mais de *difficulté*, contrôlé par le paramètre E (DAFNE).
   - **Règle dure associée** : l'adjacence reste un **label issu de la grille** ([compute_fragment_adjacencies, forge_dataset.py:428](mosaic2fragments/forge_dataset.py:428)) ; **ne jamais re-dériver l'adjacence depuis la géométrie érodée** (sinon des fragments usés « ne se touchent plus » → on perdrait de vraies arêtes). Et après changement de polygone (reco B), **recalculer** le mapping `src_side_idx`/`dst_side_idx` ([forge_dataset.py:453](mosaic2fragments/forge_dataset.py:453)) sur le nouveau polygone (les indices de côté sont relatifs au polygone).

5. **Pipeline VLM — abandon du CoT O(N²) + « augmenter la température ».** Le raisonnement pas-à-paire (une requête API par paire de fragments) est ingouvernable pour N grand, et « monter la température » pour débloquer est une fausse intuition (Renze & Guven 2024, [arXiv:2402.05201](https://arxiv.org/abs/2402.05201)). Design correct si VLM utilisé : **une passe → matrice de compatibilité N×N → matching de graphe (Hongrois)**, façon VLHSA. VLM = baseline légitime-mais-faible, pas le cœur.

6. **Métrique & baselines.** On s'aligne sur **RePAIR** (seule suite officielle complète et transférable) : F1 sur le *mating graph* pour le link-prediction ; Q_pos + RMSE(R)/RMSE(T) quand on ajoutera une tête de pose. Baselines triviales : voir *Restant à faire*. ⚠️ **link-prediction (prédire les arêtes) ≠ pose-regression (prédire R,T par fragment)** : seul **PairingNet** est comparable à notre étage link-prediction ; DiffAssemble/ReassembleNet font de la pose et ne sont comparables sur Q_pos qu'une fois une tête de pose ajoutée. POMPAAF/CLEOPATRA = classification de style (PAS réassemblage), reclassées dans la liste datasets en tête de readme.
   - **Pondération du F1 (item 4)** : pondérer chaque arête par la **longueur de contact** (RePAIR le fait déjà). Pondérer par le contact *survivant* (post-dégradation) = « on ne score que le récupérable » ; pondérer par le contact *original* = « récupérer la topologie complète » (position archéo). On choisit et on documente — **le set d'arêtes (label) reste l'original intact, on ne change que les poids**. Possibilité de **stratifier le F1 par fraction de contact survivante** (buckets) pour un diagnostic fin.

7. **Étage de pose = problème de synchronisation (item 5).** `link-prediction → tête de pose` est une chaîne légitime et interprétable (artefact intermédiaire scorable = le graphe d'adjacence). Passer de « qui touche qui + quels côtés » à des coordonnées **absolues globalement cohérentes** = composer les transforms relatives par arête → poses absolues. **Sur LEGO clean c'est quasi-trivial** : contacts exacts → transforms relatives exactes → composition par BFS depuis une ancre, sans dérive, les cycles ferment ; `src_side_idx`/`dst_side_idx` donnent la transform relative presque directement (= le pont adjacence→pose). **Ça ne devient « du vrai travail » qu'en régime dégradé/fresque** : contacts érodés → transforms bruitées → dérive + incohérence sur les cycles → besoin d'optimisation globale (rotation averaging / pose-graph, type fermeture de boucle SLAM). Ce durcissement EST une facette du curriculum.

### Restant à faire
- "area/perimeter/coords sont en pixels → dépendent de la résolution. Une mosaïque LEGO à 40 px/stud (~232 000 px² par fragment) ≠ un tesson de fresque photographié à une autre résolution. Si le GNN apprend « aire ≈ 232 000 ⇒ … » sur LEGO, c'est vide de sens sur du réel. Normaliser = passer à des ratios adimensionnels (area/aire_image, perimeter/√area, coords/échelle) → features comparables entre domaines/résolutions. D'accord : on garde ça (+ yolo/, jitter) pour APRÈS — les datasets sont maintenant propres et prêts pour tes collègues."

- Conversion `graph_complete.json` → `torch_geometric.data.Data` (~5 lignes) au moment de brancher le GNN.

- **Baselines triviales pour la métrique (≈1h ; Mathieu s'en occupe pour l'équipe)** — sans elles, impossible de démontrer que le GNN apporte quelque chose (remarque directe de Chahan) :
    - *random adjacency* : graphe d'adjacence aléatoire, **moyenné sur 1000 seeds**, reporter **moyenne ± écart-type** (tend vers ~0 quand N grandit) ;
    - *matching par couleur dominante* : relier les fragments de couleur la plus proche — les `side_features` RGB sont déjà là → **ablation gratuite** qui isole l'apport de la géométrie vs la couleur seule.

- **Chevauchements vs trous (fragments virtuels vs fragments réels)** : quand on padde le polygone vers n_max (ou qu'on simplifie à n=16), on crée des fragments "virtuels" géométriquement différents des fragments réels matérialisés à partir de YOLO. Risque : le GNN prédit qu'un fragment virtuel A est adjacent au virtuel B à un endroit donné, mais quand on rematérialise avec les vrais polygones, A et B se chevauchent physiquement (impossible). Pistes classées par ratio effort/résultat :
    - **(à implémenter en priorité, V2/V3)** Path 3 — *post-processing au décodage* : placement glouton triant les paires par score d'adjacence prédit décroissant, on rejette toute paire qui causerait un chevauchement → l'assemblage est toujours valide géométriquement, certaines adjacences prédites sont sacrifiées. ~1 jour de code, déterministe, pas de modif du GNN.
    hyper important pour le use-case où il doit avoir m^ comportement
    - **(plus tard, raffinement)** Path 2 — *loss "no-overlap" pendant l'entraînement* : terme de pénalité dans le loss, mais ça demande une version soft différentiable (approximation par disques distance-centroïdes ≥ rayons combinés), gain marginal sur LEGO rectiligne, plus pertinent pour HDA aux formes courbes.

- **Compléter l'état de l'art** (à intégrer dans la section principale en haut) :
    - **VLHSA: Vision-Language Hierarchical Semantic Alignment for Jigsaw Puzzle with Eroded Gaps** ([arXiv:2509.25202](https://arxiv.org/html/2509.25202), sept 2025) — VLM combiné avec alignement sémantique structurel pour puzzles avec gaps érodés (cas archéo). Pertinent pour notre pipeline hybride VLM+GNN.
    - **DiffAssemble: A Unified Graph-Diffusion Model for 2D and 3D Reassembly** ([arXiv:2402.19302](https://arxiv.org/abs/2402.19302), CVPR 2024) — graph-diffusion unifié, attention-based GNN qui débruite itérativement positions/rotations. Scale jusqu'à 900 éléments. État de l'art reassembly généraliste.
    - **PuzLM: Solving Jigsaw Puzzles with Sequence-to-Sequence Language Models** ([arXiv:2511.06315](https://arxiv.org/html/2511.06315v2), nov 2025) — tokenisation discrète des pièces, Seq2Seq pour reconstruire. Approche LM-pure, alternative au GNN.
    - **PairingNet: Learning-based Pair-searching and -matching Network for Image Fragments** ([arXiv:2312.08704](https://arxiv.org/html/2312.08704v2), ECCV 2024) — GNN (ResGCN) pour matching de paires de fragments. Dataset synthétique de 390 images Pexels → 8 196 fragments → 14 951 paires. Permet de calibrer notre échelle.
    - **Solving Jigsaw Puzzles in the Wild: Human-Guided Reconstruction of Cultural Heritage Fragments** ([arXiv:2603.06389](https://arxiv.org/html/2603.06389)) — human-in-the-loop pour la reconstitution archéo, à signaler côté limitations actuelles du fully-automatic.
    - **Graph Language Models (Plenz & Frank, ACL 2024)** ([aclanthology](https://aclanthology.org/2024.acl-long.245.pdf)) — la référence GLM canonique, à citer si on évoque la voie GLM en discussion.

- **Tailles de datasets de référence** (à comparer à notre échelle cible pour calibrer) :
    - ReassembleNet pretraining : **5 000 mosaïques** semi-synthétiques × ~9 fragments = **45 834 fragments**. Split 80/20 = 4 000 train / 1 000 test.
    - RePAIR benchmark (réel, fresques) : 121 puzzles, **957 fragments** (97 train / 24 test).
    - PairingNet synthétique : 390 images, **8 196 fragments**, **14 951 paires** (4 098 fragments train).
    - PairingNet réel : 34 images, 320 fragments, 202 paires.
    - DiffAssemble : scale jusqu'à 900 éléments par instance, 30×30 puzzles à 95% accuracy.
    - **Notre cible actuelle** : 50 mosaïques × ~12 fragments = **600 fragments**. C'est **76× plus petit** que le pretraining de ReassembleNet, **7× plus petit** que PairingNet synthetic. À tenir en tête : pour une comparaison crédible, viser 500–1 000+ mosaïques LEGO (= 5 000–10 000 fragments) avant de revendiquer un transfer learning utile.

## Contrat de données GNN (articulation forge → GNN)

### Fixe vs variable (le point qui dissipe toutes les confusions)
**Un GNN gère un nombre de nœuds/arêtes VARIABLE** — c'est sa raison d'être : poids **partagés**,
réutilisés sur chaque nœud/arête, donc message-passing sur n'importe quel graphe. (Un MLP/CNN
classique exige une entrée de taille fixe ; **pas** un GNN.)

| Quantité | Statut | Pourquoi |
|---|---|---|
| **N** = nombre de fragments (nœuds) | **VARIABLE** ✅ | géré nativement par le GNN |
| **E** = nombre d'arêtes | **VARIABLE** ✅ | idem |
| **F** = dimension du vecteur de features **par nœud** | **FIXE** (obligatoire) | l'encodeur de nœud a une entrée fixe |
| **D** = dimension des features **par arête** | **FIXE** (obligatoire) | idem côté arête |
| **Paramètres entraînables** (poids) | **FIXE & FINI** | partagés sur tous les nœuds/arêtes → indépendants de N, E |

→ Donc « le seul paramètre fini » n'est pas F : **les poids du modèle** sont l'ensemble fini &
fixe (au sens ML de « paramètres »). F et D sont des **dimensions** d'architecture à fixer. N et E
sont libres.

### Les 3 blocs de features par nœud
| Bloc | Quoi | Taille |
|---|---|---|
| `gnn_input` | 7 nombres décrivant le fragment **globalement** : aire, périmètre, couleur moyenne (R,G,B), largeur/hauteur de bbox. **100 % domaine-agnostique** (aucune feature LEGO-only). | 7 (fixe) |
| `polygon_n_canonical` | **le contour** = liste de points (x,y) en repère PCA canonique (invariant rotation) | `n_sides`×2 |
| `side_features` | par **côté** du contour : longueur, angle (canonique), couleur (R,G,B) adjacente | `n_sides`×5 |

**Le seul terme qui rend F variable = `n_sides`** (gnn_input est fixe). Donc F devient fixe
dès qu'on traite `n_sides` (ci-dessous). Input GNN = `graph_fragments.json` → `Data(x ∈ ℝ^{N×F},
edge_index vide)` ; **cible** = `graph_complete.json` (arêtes du mating graph + edge features
shared_length/mean_angle/n_segments). Tâche d'abord = **link prediction**.

### Rendre F fixe malgré `n_sides` variable (côté loader GNN, PAS la forge)
- **(A) pad à `n_max` + masque de validité** : on garde le polygone honnête reco-B (ses vrais
  sommets) et on **ajoute des slots placeholder hors-bande** (ex. (0,0)) jusqu'à `n_max`, marqués
  invalides par un **masque binaire** (1 = vrai sommet, 0 = remplissage). Le GNN **ignore** les
  slots masqués → la forme réelle est intacte. **Lossless, pas de gain d'aire, pas d'homogénéisation.**
  `n_max` = **max des `n_sides` sur tout le dataset** (pas par mosaïque). Les points de padding NE
  sont PAS sur le polygone — ce ne sont pas des points collinéaires « inutiles », juste du
  remplissage de tableau ignoré.
- **(B) mini-encodeur par nœud** (DeepSets / PointNet / 1D-CNN à poids partagés sur les sommets)
  → embedding fixe, gère `n` variable nativement, sans padding.

**Pourquoi pas un `n=16` fixe par resample ?** Incompatible avec l'invariant `polygon_n ⊆
polygon_raw` : le plancher dur est `#reflex` > 16 pour les fragments complexes → forcer 16 =
dropper des coins concaves (regagner de l'aire, interdit) **ou** homogénéiser tous les fragments
en 16-gones (perte du signal de forme). Le pad-à-`n_max`+masque (A) donne F fixe **sans** ça.

### ✅ Invariance en rotation (résout l'ancienne fuite d'orientation) — implémenté 09/06
**Problème (résolu)** : décrire la forme dans le repère TARGET (orientation *résolue*) faisait voir
au GNN des fragments **déjà bien orientés** → tâche réduite à de la translation (orientation
**leakée**) → ne transfère pas aux fragments réels (YOLO à orientation arbitraire).
**Fix** : **canonicalisation PCA** (`pca_canonical_rotation`, dans `features/`). Chaque fragment est
décrit dans **son propre repère intrinsèque** (axe principal → x, signe déterministe par skewness)
→ descripteur **invariant en rotation** : le même fragment donne le même `polygon_n_canonical`
quelle que soit son orientation. **⚠️ Ce n'est PAS « orienter toutes les pièces pareil »** : chaque
fragment garde son orientation-cible propre, qui devient une **sortie** (pose à prédire), **différente
par fragment**. L'orientation absolue est stockée en GT (`target_info.pca_angle_deg`). Validé
invariant à ~1e-14 ; repère-target ≡ repère-source ⟹ alignement synthétique↔réel garanti. **Caveat** :
fragments ~symétriques/carrés = axe instable (accepté ; descripteur de courbure plus robuste si besoin, 🚧).

### Conventions de transfert (LEGO→fresque réelle)
1. **n_pieces/n_cells** (n'existent pas sur un tesson) : ✅ **RETIRÉS** des features (plus produits ;
   `gnn_input` est 100 % agnostique). n_pieces reste seulement en métadonnée GT (`target_info`).
2. **Normaliser l'échelle** : area/perimeter/bbox sont en px → adimensionner (area/aire_image,
   perimeter/√area, coords/échelle) sinon valeurs LEGO ≠ valeurs fresque. 🚧 à décider.
3. **Rotation invariante / canonicalisée** (cf. problème connu ci-dessus). 🚧 à décider.
4. **Couleur par côté** : un seul RGB est grossier (et pas unique : un fragment multi-pièces a des
   couleurs différentes selon le côté — c'est voulu, signal *local* de matching). Pour la fresque,
   enrichir vers un **profil** de pixels le long du bord (façon PairingNet/RePAIR). 🚧.

## Stratégies de dégradation (axe curriculum)

Catalogue des dégradations à appliquer pour faire passer le dataset du **Régime 1**
(LEGO clean, contacts parfaits, signal = couleur/gradient) au **Régime 2** (fresque/vase
usé, contacts érodés, signal = forme + sémantique). Chaque dégradation est un **bouton
paramétré** (intensité 0 = clean) → permet un curriculum gradué et mesurable. On reprend
les **5 paramètres DAFNE** (A–E) comme épine dorsale citable, complétés par des leviers
photométriques.

**Invariant à respecter pour TOUTES** : la dégradation ne touche que l'**input** (géométrie/
couleur des nœuds) ; la **GT (arêtes / pose) reste figée sur la partition intacte** (cf.
décisions, point 4). On ne re-dérive jamais l'adjacence de la géométrie dégradée.

Légende : ✅ implémenté dans `forge_dataset.py` · 🚧 à venir. **Chaque dégradation implémentée
écrit ses paramètres dans `graph_*.json → degradation` (traçabilité curriculum).**

### Géométriques (forme du fragment)
**Intervalles recommandés par paramètre** (tirés aléatoirement par fragment dans [min,max]) :

| Dégradation | Flags CLI | Intervalle conseillé | Borne dure |
|---|---|---|---|
| Réduction de sommets | `--n-sides-min --n-sides-max` | `16` … `24` | plancher = #reflex (jamais en dessous) |
| Érosion morpho (E) | `--erode-px-min --erode-px-max` | `1` … `8` px (léger 2–4, fort 6–10) | — |
| Trous internes | `--holes-min --holes-max` | `0` … `3` trous | **aire totale des trous ≤ 10 %** (cap dur dans `degrade_mask`) |
| Fragments manquants (C) | `--missing-min --missing-max` | `0` … `2` fragments | garde toujours ≥ 1 fragment |
| Nombre de fragments (A) | `--n-frag-min --n-frag-max` | `10` … `15` | — |

Pas de paramètre « aire perdue » : c'est une **métrique post-hoc** de sévérité, calculée et
écrite par la forge — `degradation_lost_frac` par fragment + `mean_lost_frac`/
`mean_degradation_lost_frac` au niveau mosaïque (cf. décisions, raisonnement aire perdue).

- ✅ **Coins cassés + réduction de sommets** (reco B reflex-aware + clip) : `polygon_n ⊆
  polygon_raw`, `n` variable. Toujours actif. Première marche du Régime 2.
- ✅ **Érosion morphologique** (= DAFNE **E**) : `cv2.erode` du masque, rayon px tiré par
  fragment. **Bouton Régime-2 principal.** 🚧 variante *asymétrique* (bords de contact) à venir.
- ✅ **Trous internes** (tesselles manquantes *dans* le fragment) : `cv2.circle` à 0, **aire
  totale plafonnée à 10 %** de l'aire du fragment (budget réparti sur les `n_holes` disques).
- 🚧 **Jitter de sommets**, **arrondi des coins**, **micro-ébréchures** dirigées.

### Topologiques (ensemble des fragments)
- ✅ **Fragments manquants** (= DAFNE **C**) : retirés de l'INPUT (`source.png` +
  `graph_fragments`), gardés en GT (`graph_complete` avec `missing:True` + arêtes intactes).
  Compte absolu tiré dans `[--missing-min, --missing-max]`. Supporte les 2 variantes
  (supervisée via le flag, ou non-supervisée en ignorant le flag).
- 🚧 **Fragments parasites / distracteurs** (= DAFNE **D**) : injecter des fragments d'**autres**
  mosaïques → à rejeter. *(plus tard)*
- 🚧 **Sur-fragmentation** : recasser un fragment en sous-fragments. *(plus tard)*

### Photométriques (couleur / texture) — 🚧 toutes reportées
- Décoloration/fading, bruit chromatique, illumination, flou/résolution, vieillissement
  indépendant par fragment. *(reportées : on ne fait pas la photométrie pour l'instant.)*

### Configurationnelles (difficulté globale)
- ✅ **Nombre de fragments** (= DAFNE **A**) : `--n-frag-min --n-frag-max` (défaut 10/15).
- 🚧 **Type de distribution de découpe** (= DAFNE **B**) : BFS équilibré actuel vs clusters vs
  Voronoï.
- 🚧 **Gaps de placement** ; 🚧 **rotation continue** `[0,360°)` (régime fresque, pose = synchro).

### Lignes de commande (à tenir à jour)
```bash
# mosaïque CLEAN (reco B, n variable, aucune dégradation) — niveau 0 du curriculum
python3 mosaic2fragments/forge_dataset.py --input IMG.png --out dataset/mosaic_XXX --seed S

# DÉGRADÉE : érosion 2–6 px + 1–3 trous internes (≤10% aire) + 0–2 fragments manquants
python3 mosaic2fragments/forge_dataset.py --input IMG.png --out OUT --seed S \
    --erode-px-min 2 --erode-px-max 6 --holes-min 1 --holes-max 3 --missing-min 0 --missing-max 2

# batch (mêmes flags de dégradation)
python3 mosaic2fragments/batch.py --inputs dataset_inputs/*.png --n-samples 50 --out dataset \
    --erode-px-min 2 --erode-px-max 6 --missing-min 0 --missing-max 2

# réduction de sommets plus agressive (n plus petit, borné par #reflex)
python3 mosaic2fragments/forge_dataset.py --input IMG.png --out OUT --seed S --n-sides-min 12 --n-sides-max 18
```

**Ordre de priorité restant** : asymétrie d'érosion → jitter → (plus tard) parasites D,
sur-fragmentation, distribution B, rotation continue. Photométrie reportée.

## Scraping dataset lego — contraintes à respecter

Production des 50 `canvas_mosaic.png` à partir de 50 images variées (dessins, photos, peintures, mosaïques antiques) via lego-art-remix.com.

**Paramètres lego-art-remix** :
- **Option "variable tile" ON** (pas l'option 1×1-only)
- **Résolution recommandée** : 96×96 ou 100×100 studs (48×48 est trop grossier pour des use-case fresque/mosaïque réalistes ; 64×64 acceptable mais limite). La pipeline auto-détecte 48/64/96/100/128.
- **Format de sortie** : PNG, carré (H == W), résolution ~1920×1920 ou plus (40 px/stud minimum pour que les joints gris fassent ≥ 2 px)

**Palette de couleurs — exclusions strictes** :

Toutes les couleurs LEGO dont RGB est ~(R, G, B) avec R≈G≈B et intensité 100-180 sont des faux-positifs garantis pour le détecteur de joints. Désactiver dans la palette :
- **Light Bluish Gray** ~(160, 165, 169)
- **Dark Bluish Gray** ~(108, 110, 117)
- **Flat Silver** ~(137, 135, 136)
- **Pearl Light Gray** ~(157, 158, 154)
- **Metallic Silver** ~(166, 168, 166)

Toutes les autres couleurs LEGO sont safe (suffisamment saturées ou suffisamment éloignées du gris milieu).

**Paramètres à NE PAS varier (consistency)** :
- Couleur des joints (gris ~RGB(136), 2-3 px de large) — c'est le rendu par défaut, ne rien modifier
- Format carré
- Format de sortie PNG (pas JPG, l'anti-aliasing des joints doit être propre)
- Option variable-tile

**Paramètres à varier (diversité)** :
- Images d'entrée : dessins simples / photos contrastées / œuvres d'art / motifs reconnaissables
- Seed de génération de l'outil (si exposé)
- Idéalement même grille pour les 50 (96 ou 100), variable possible mais reste dans la liste auto-détectée

**Sortie attendue** : `dataset_inputs/canvas_mosaic_XXX.png` (50 fichiers numérotés).

**Lancement de la pipeline une fois les 50 PNGs prêts** :
```
python3 mosaic2fragments/batch.py --inputs dataset_inputs/*.png --n-samples 50 --out dataset
```

**Sanity-check intégré** : la pipeline détecte automatiquement la sur-détection liée à une palette grise mal nettoyée (warning `N gray-ish 1x1 pieces detected`). Si ce warning apparaît, vérifier la palette de couleurs.


