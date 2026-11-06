# Veille : étape 4, optimiseur local par rapprochement à l'image (2026-11-06)

Agent : Veille modèles. Périmètre : **étape 4** du pipeline envisagé (générateur de mesh → optimiseur de
mesh → optimiseur de texture → **optimiseur local par rapprochement à l'image**). Les étapes 1 à 3 sont
traitées par d'autres veilles.

Rien n'a été installé ni testé. Les chiffres de VRAM et de temps marqués « estimation » sont des calculs
de l'agent, pas des mesures. Toutes les licences ci-dessous restent à auditer par `/gardien-licences`
avant toute intégration ; aucune ligne de `LICENSES.md` n'a été modifiée.

Légende de la colonne « Source » :
- **LICENSE lu** : l'agent a lu le fichier de licence (ou l'en-tête du fichier source) dans le dépôt le
  2026-11-06.
- **déclaré** : licence affichée sur une fiche HuggingFace, un README ou une étiquette GitHub, fichier non lu.

## 1. Besoin et limites de l'étape

Objectif : déformer localement le mesh (et éventuellement sa texture) pour que son rendu, vu sous
l'angle de l'image de concept, colle à l'image : silhouette, contours, visage, accessoires.

Principe commun à toutes les pistes : une boucle d'optimisation par gradient
`paramètres du mesh → rendu différentiable (masque, profondeur, normales, couleur) → pertes contre
l'image et ses cartes estimées → régularisation → mise à jour`.

### Ce que l'étape peut corriger (défauts de `docs/experiments.md`)

| Défaut connu | Réponse possible | Condition |
|---|---|---|
| Silhouette qui s'écarte du masque | Perte de silhouette (IoU ou L1 sur masque doux) | Pose de caméra correcte (voir 1.3) |
| Accessoire mal placé (arme inventée ailleurs) | Retrait de la partie hors masque et ajout dans le masque : **nécessite un changement de topologie**, donc une représentation implicite (FlexiCubes) ; une simple déformation de sommets ne déplace pas proprement un composant détaché | Seulement si l'accessoire est **dans le masque**. Sur `3dfab…` la lame a un alpha 0 : l'étape 4 ne peut pas la recréer, il faut d'abord corriger le détourage |
| Visage peu détaillé (tête d'environ 40 px) | Passe « zoom » : rendu et pertes sur un recadrage de la tête, normales estimées sur le recadrage agrandi | L'information doit exister dans l'image de concept. Si le visage y fait 40 px, aucune optimisation n'inventera de détail fiable ; il faudrait l'image source en haute résolution |
| Signature d'artiste devenue excroissance | Indirectement : si la signature est retirée du masque, la perte de silhouette tire la géométrie vers l'intérieur | Masque nettoyé en amont |

### Ce que l'étape ne peut pas corriger

- **Faces cachées** (dos, dessous, intérieur des bras) : aucune supervision depuis une vue unique. Il
  faut les **geler** (masque de visibilité par sommet ou par voxel) ou les régulariser fortement, sinon
  l'optimisation les déforme sans contrôle.
- **Effets semi-transparents** (magie, fumée) : pas de surface à faire correspondre.
- **Ambiguïté de profondeur** : une vue unique contraint mal l'épaisseur ; les estimateurs de profondeur
  sont relatifs (échelle et décalage inconnus) et peu fiables sur un concept art peint et stylisé.
- **Éclairage peint** : une perte photométrique sur la couleur confond ombres peintes et géométrie ;
  à éviter pour la géométrie, ou à faible poids.

### Prérequis non trivial : la caméra

Le concept art n'a pas de caméra connue. Avant de raffiner, il faut estimer la pose et la focale (ou
supposer une projection orthographique de face, comme la plupart des générateurs image → 3D) puis
**optimiser d'abord la caméra seule** (échelle, translation, lacet) sur la perte de silhouette, mesh figé.
Sans cette étape, la perte de silhouette déforme le mesh pour compenser une erreur de cadrage.

## 2. Rendu différentiable

Point critique demandé : **nvdiffrast est confirmé non commercial** (fichier `LICENSE.txt` lu le
2026-11-06) : « Nvidia Source Code License (1-Way Commercial) », section 3.3 : « The Work and any
derivative works thereof only may be used or intended for use non-commercially », où « non-commercially »
signifie « for research or evaluation purposes only ». Seuls NVIDIA et ses affiliés ont un droit
commercial. Une licence commerciale passe par NVIDIA Research Licensing (formulaire). **nvdiffrast est
donc interdit pour le produit**, et avec lui tout code qui l'importe sans alternative.

| Brique | Licence | Source | Windows / Py 3.12 / torch 2.6 cu124 | Gradients de silhouette | Remarques |
|---|---|---|---|---|---|
| **Kaolin 0.18.0** (`kaolin.render.mesh.rasterize`, backend `cuda`, `dibr_rasterization`, `dibr_soft_mask`) | Apache-2.0 (sauf `kaolin/non_commercial/`, voir plus bas) | LICENSE lu | **Roue officielle `kaolin-0.18.0-cp312-cp312-win_amd64.whl` pour torch-2.6.0_cu124** (index NVIDIA vérifié) | Oui, masque doux DIB-R | Backend par défaut `cuda` = opérateurs Kaolin, **nvdiffrast optionnel** (ne pas installer les backends `nvdiffrast` et `nvdiffrast_fwd`). Fournit aussi FlexiCubes (section 3). Dépendances : usd-core, warp-lang (Apache-2.0, LICENSE lu), pygltflib, plyfile, scipy, etc. |
| **Mitsuba 3.9.1 + Dr.Jit** | BSD-3 | LICENSE lu | Roues `cp312 win_amd64` sur PyPI (pilote NVIDIA ≥ 535) | Oui, intégrateurs projectifs (`direct_projective`) | Le plus rigoureux (lancer de rayons, visibilité correcte). Tutoriel officiel d'optimisation de forme avec Large Steps + remaillage gpytoolbox. Plus lent, interop PyTorch via `dr.wrap` ; mal adapté à une topologie qui change à chaque itération (FlexiCubes). |
| **PyTorch3D** | BSD-3 | LICENSE lu | Pas de roue Windows ; compilation depuis les sources sous Visual Studio, versions de torch officiellement listées jusqu'à 2.4.1 | Oui, `SoftRasterizer` | Mûr et bien documenté, mais **risque d'installation élevé** sur notre machine. |
| **DRTK** (Meta) | MIT | LICENSE lu | Pas de binaire, compilation (≥ 30 min annoncées) ; Windows non documenté | Oui, module `edge_grad` | Prometteur (remplaçant direct de nvdiffrast côté API), maturité et support Windows à tester. |
| Rasteriseur maison (PyTorch pur) | Notre code | — | Oui | À écrire (masque doux type SoftRas) | Possible pour une silhouette sur un mesh modeste, coûteux en mémoire au-delà de quelques dizaines de milliers de faces en 512². En dernier recours. |
| nvdiffrast | NVIDIA Source Code License, non commercial | LICENSE lu | — | — | **Interdit.** Aussi tiré par threestudio, Unique3D, les exemples de FlexiCubes et de Large Steps. |
| diff-gaussian-rasterization (Inria) | Licence Gaussian-Splatting Inria, non commerciale | déclaré | — | — | **Interdit** (et hors sujet mesh). gsplat (Apache-2.0 déclaré) existe si un jour on passe par des gaussiennes. |

**Attention Kaolin** : le dépôt contient `kaolin/non_commercial/` sous NVIDIA Source Code License for
Kaolin (`LICENSE.NSCL`). Il ne contient aujourd'hui que l'ancienne copie de FlexiCubes, dépréciée
(en-tête lu : « Any use … without an express license agreement from NVIDIA … is strictly prohibited » ;
son docstring dit qu'elle a été déplacée « due to license changes »). Règle proposée : **ne jamais
importer `kaolin.non_commercial`** ; le gardien doit décider si la présence de ce code dans la roue
installée pose un problème de distribution.

## 3. Représentation à optimiser

### 3.1 FlexiCubes via Kaolin (piste prioritaire de l'utilisateur)

Shen et al., « Flexible Isosurface Extraction for Gradient-Based Mesh Optimization », ACM TOG 42(4),
SIGGRAPH 2023, DOI 10.1145/3592430.

Licences :
- `nv-tlabs/FlexiCubes` : Apache-2.0 (vérifié par l'agent principal, puis `LICENSE.txt` relu ici).
- `kaolin.ops.conversions.FlexiCubes` : **en-tête du fichier source lu** (`kaolin/ops/conversions/flexicubes/flexicubes.py`,
  branche `master`) : « Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES … Licensed under the Apache
  License, Version 2.0 ». C'est la bonne classe. L'ancienne `kaolin.non_commercial.FlexiCubes` est
  propriétaire et ne doit pas être utilisée.
- **Non vérifié** : que la roue **0.18.0** installée contient bien la version Apache sous
  `kaolin/ops/conversions/` (j'ai lu `master`, pas l'archive de la roue). À contrôler par le gardien en
  ouvrant le fichier dans la roue.
- Les exemples officiels installent nvdiffrast pour le rendu : à remplacer.

Faisabilité Windows / torch 2.6 / CUDA 12.4 : **oui sur le papier**. La roue Kaolin 0.18.0 cp312 Windows
existe pour torch 2.6.0 cu124, FlexiCubes est en PyTorch pur (tables + opérations tensorielles, pas
d'extension CUDA propre), et le rasteriseur `cuda` de Kaolin est dans la même roue. Une seule dépendance
binaire à risque : la roue Kaolin elle-même. Pas besoin de `torch_scatter` côté Kaolin (les exemples
nv-tlabs l'utilisent, à vérifier si on reprend leur code).

Rasteriseur de remplacement recommandé : **le rasteriseur `cuda` de Kaolin** (même paquet, même
format de tenseurs ; topologie variable à chaque itération sans coût de reconstruction de scène) :
`rasterize` + `interpolate` pour la profondeur et les normales, `dibr_soft_mask` pour le gradient de
silhouette. Le gradient sur les bords nets étant moins bon que l'antialiasing de nvdiffrast, compenser par
le masque doux DIB-R et par la perte de profondeur. Mitsuba 3 en second choix pour un contrôle de
qualité ou une passe finale (topologie figée).

Coût mémoire (**estimation**, non mesurée ; grille dense de Kaolin) : paramètres par cube = SDF aux sommets
+ déformation (3) + poids `beta` (12), `alpha` (8), `gamma_f` (1), en float32, plus 2 états Adam.

| Résolution | Cubes | Paramètres + Adam (approx.) | Verdict 16 Go |
|---|---|---|---|
| 128³ | 2,1 M | ~0,8 Go | Confortable |
| 192³ | 7,1 M | ~2,7 Go | Confortable |
| 256³ | 16,8 M | ~6,5 Go (~2 Go si on n'optimise que SDF + déformation) | Faisable, à mesurer (pics lors de l'extraction en mode `training`) |
| 384³ et plus | ≥ 57 M | > 20 Go en grille dense | GPU cloud, ou grille éparse / bande étroite autour de la surface |

Ajouter le rendu (quelques centaines de Mo en 512² à 1024²) et l'estimateur de normales s'il n'est pas
déchargé avant (voir section 4).

**Limite importante pour le visage** : la résolution est globale. Une tête qui fait 5 % de la hauteur du
personnage ne couvre qu'environ 13 voxels en 256³ : pas assez pour des traits de visage. FlexiCubes
convient donc à la **silhouette, aux accessoires et à la topologie** ; le visage demande soit une
**seconde grille locale** sur la boîte englobante de la tête (même code, résolution 128³ sur la tête,
puis recollage), soit une passe de déformation de sommets à haute résolution (3.2).

Initialisation de la grille à partir du mesh de l'étape 2 :
1. Normaliser le mesh dans la boîte de la grille (avec une marge) ; garder la transformation.
2. Calculer une distance signée aux sommets de la grille : distance point-mesh
   (`kaolin.metrics.trianglemesh.point_to_mesh_distance`) et signe par nombre d'enroulement généralisé
   (robuste aux meshes non étanches ; `kaolin.ops.mesh.check_sign` suppose un mesh étanche). Options à
   faire auditer : gpytoolbox (MIT, LICENSE lu) ou Open3D (MIT déclaré).
3. Tronquer la SDF à une bande de quelques voxels (stabilité et mémoire), poids `alpha/beta/gamma` à 1,
   déformation à 0. Une première extraction sans optimisation doit redonner le mesh d'entrée : test de
   non-régression.
4. Construire un **masque de voxels visibles** depuis la caméra de l'image (rendu de l'index de face, ou
   lancer de rayons) ; geler la SDF hors de ce masque et hors d'une bande autour de la silhouette, pour
   ne pas toucher les faces cachées.
5. Pertes : silhouette contre le masque affiné (`refine_mask`), profondeur relative invariante en
   échelle et décalage, normales estimées dans la zone visible, plus `l_dev` (régularisation fournie par
   FlexiCubes), une régularisation de la SDF (eikonale ou lissage), et une perte de fidélité à la SDF
   initiale hors zone supervisée.
6. Texture : FlexiCubes interpole des `voxelgrid_features` (couleur échantillonnée sur le mesh d'origine) ;
   sinon re-projeter la texture de l'étape 3 après coup.

Lien avec TRELLIS : TRELLIS décode ses meshes par FlexiCubes (copie dans son propre dépôt, licence de
cette copie **non vérifiée**). On pourrait initialiser directement depuis sa SDF éparse plutôt que
depuis le mesh, mais cela couple les étapes 1 et 4 ; l'initialisation depuis le mesh de l'étape 2 est
indépendante du générateur. L'usage de FlexiCubes par **TRELLIS.2** n'a pas été vérifié par cet agent
(TRELLIS.2 annonce une autre représentation, « O-Voxel ») : à confirmer par la veille de l'étape 1.

Maturité : article SIGGRAPH 2023, intégré à Kaolin (maintenu par NVIDIA, 0.18.0 compatible torch 2.3 à 2.8),
utilisé en production dans TRELLIS. Pas d'outil prêt à l'emploi pour notre cas « une image » : la boucle
est à écrire (quelques centaines de lignes).

### 3.2 Déformation de sommets avec Large Steps et remaillage continu

| Brique | Licence | Source | Rôle |
|---|---|---|---|
| `largesteps` (Nicolet et al., SIGGRAPH Asia 2021) | BSD-3 | LICENSE lu | Paramétrisation `u = (I + λL)v` : gradients lissés par le Laplacien, évite le mesh emmêlé |
| `cholespy` | BSD-3 | LICENSE lu | Solveur de Cholesky requis par largesteps (et par Mitsuba) |
| Continuous Remeshing (Palfinger, Profactor) | MIT | LICENSE lu | Remaillage adaptatif pendant l'optimisation (densifie là où il y a du détail : visage). Cœur en PyTorch + `torch_scatter` ; nvdiffrast seulement dans l'exemple |
| gpytoolbox (remaillage Botsch-Kobbelt) | MIT (composants Apache-2.0, MIT, BSD-2) | LICENSE lu | Remaillage entre deux phases, utilisé dans le tutoriel Mitsuba |

Avantages : résolution locale arbitraire (idéal pour le visage), topologie conservée (pas de trou créé),
mémoire faible (meshes de 100 k à 500 k sommets : bien sous 16 Go, **estimation**). Inconvénient : ne peut
ni retirer ni déplacer un accessoire détaché ; ne corrige que des écarts modérés.

Note : `torch_scatter` (MIT déclaré) n'a pas toujours de roue Windows pour torch 2.6 cu124 ; à vérifier,
ou réécrire avec `torch.Tensor.index_add_` / `scatter_reduce` natifs.

## 4. Estimateurs monoculaires (cibles de profondeur et de normales)

Usage : une passe unique sur l'image (et sur le recadrage de la tête) avant la boucle, puis
déchargement du modèle (règle GPU : un seul modèle lourd chargé à la fois).

| Modèle | Sortie | Licence code | Licence poids | Source | Taille | Retenu |
|---|---|---|---|---|---|---|
| **MoGe-2 `vitl-normal`** (Microsoft) | Géométrie + **normales** | MIT (+ code DINOv2 Apache-2.0) | MIT | Code : LICENSE lu ; poids : déclaré HF (`Ruicheng/moge-2-vitl-normal`) | ~0,3 B (DINOv2-L), **estimation** 2 à 3 Go | **Oui, premier choix normales** (léger, licence la plus simple). Variantes `vitb-normal`, `vits-normal` |
| **Depth Anything V2 Small** | Profondeur relative | Apache-2.0 | Apache-2.0 | README + HF déclarés | ~25 M | Oui (profondeur, très léger) |
| **Depth Anything 3 : DA3-Small, DA3-Base, DA3Mono-Large, DA3Metric-Large** | Profondeur | Apache-2.0 | Apache-2.0 | déclaré (tableau README, HF) | DA3Mono-Large ~0,35 B | Oui, **seulement ces variantes** |
| Lotus normal v1-1 (`jingheya/lotus-normal-g-v1-1`, `-d-`) | Normales | À vérifier | Apache-2.0 (déclaré HF) | déclaré | Dérivé de SD2 (~0,9 B) | Candidat, conditions de SD2 à clarifier |
| StableNormal (`Stable-X/stable-normal-v0-1`) | Normales | Apache-2.0 | Apache-2.0 (déclaré HF) | déclaré | Dérivé de Stable Diffusion | Candidat ; vérifier ce que son pipeline télécharge en plus (détoureur, DINO) |
| Marigold normals / depth v1-1 (`prs-eth`) | Normales, profondeur | Apache-2.0 | CreativeML Open RAIL++-M | déclaré (README, HF) | 0,9 B, **estimation** 4 à 6 Go en fp16 | Candidat de secours ; restrictions d'usage RAIL à transmettre en aval |
| Depth Anything V2 Base / Large / Giant | — | Apache-2.0 | **CC-BY-NC-4.0** | déclaré | — | **Éliminé** |
| DA3-Large, DA3-Giant, DA3Nested (y compris 1.1) | — | Apache-2.0 | **CC BY-NC 4.0** | déclaré | — | **Éliminé** |
| DSINE (Imperial College) | Normales | **Non commerciale** (« solely for non-commercial, internal or academic research purposes ») | idem | LICENSE lu | — | **Éliminé** |
| Sapiens (Meta) normales / profondeur | — | **CC-BY-NC-4.0** | idem | LICENSE lu | — | **Éliminé** (dommage : spécialisé humains) |
| Lotus-2 | — | — | Basé sur FLUX.1-dev (non commercial) | déclaré | — | **Éliminé** |

Points d'attention transverses :
- **Stable Diffusion 2 / 2.1 a disparu de HuggingFace** (`stabilityai/stable-diffusion-2-1` renvoie
  401/404 ; rapports d'utilisateurs, une citation non vérifiée évoque l'AI Act européen). Les dérivés
  (Marigold, Lotus, StableNormal) restent téléchargeables, sous OpenRAIL++-M hérité. Signal de risque
  réglementaire à faire évaluer.
- **Données d'entraînement** : Marigold et Lotus sont entraînés sur Hypersim (CC BY-SA 3.0) et Virtual
  KITTI 2 (CC BY-NC-SA 3.0 d'après l'article Marigold, non relu ici). DA3 : « public academic datasets
  only ». MoGe-2 : non documenté sur la fiche. La question « des poids entraînés sur des données non
  commerciales » est juridique ; je la transmets au gardien.
- Les concept arts peints sont hors distribution pour tous ces modèles : leurs normales sur un visage
  stylisé de 40 px seront peu fiables. Les pertes de normales doivent être pondérées et limitées aux zones
  de confiance.

## 5. Pertes perceptuelles et par features

| Brique | Licence | Source | Usage | Remarque |
|---|---|---|---|---|
| **DINOv2** (Meta) | Apache-2.0 (code et poids) | LICENSE lu | Perte de features entre rendu et image (contours, parties), robuste au style | Premier choix |
| CLIP OpenAI | MIT | LICENSE lu | Perte sémantique grossière | Déjà proche de l'usage recherche du projet (le modèle utilisé, `sentence-transformers/clip-ViT-B-32`, est à auditer à part) |
| LPIPS | BSD-2 | LICENSE lu | Perte perceptuelle sur la couleur | Les réseaux de base (AlexNet/VGG de torchvision) sont pré-entraînés sur ImageNet, dont les conditions visent la recherche non commerciale : à trancher par le gardien |
| DINOv3 (Meta) | DINOv3 License (personnalisée) | LICENSE lu | Idem DINOv2, meilleur | Pas de clause non commerciale, licence mondiale, entité contractante Meta Platforms Ireland pour l'EEE ; mais : interdiction ITAR et usages militaires, pas de rétro-ingénierie, redistribution uniquement sous la même licence, résiliation possible, droit californien, accès sur demande. Acceptable a priori, à faire valider |

Ces pertes sont utiles pour la couleur et les détails ; pour la géométrie, la silhouette et les normales
restent les signaux principaux. Elles coûtent peu (un encodeur ViT gelé, **estimation** < 2 Go).

## 6. Raffinement par diffusion (score distillation)

| Option | Licence | Verdict |
|---|---|---|
| Zero123++ v1.1 / v1.2 (multi-vues conditionné par l'image) | Code Apache-2.0, **poids CC-BY-NC 4.0** (README) | **Éliminé** |
| Stable Zero123 | Licence non commerciale Stability (connu, non relu ici) | **Éliminé** |
| FLUX.1-dev (et Elevate3D, Lotus-2, Kiss3DGen qui en dépendent) | Non commerciale | **Éliminé** |
| SDXL base 1.0 (texte → image, SDS guidé par texte) | CreativeML Open RAIL++-M (déclaré), mais la fiche dit aussi « intended for research purposes only » | Douteux, à auditer ; non conditionné par l'image : faible valeur pour coller à un concept précis |
| Stable Diffusion 2.1 | OpenRAIL++-M, **retiré de HuggingFace** | À écarter tant que le statut n'est pas clair |
| Wonder3D, Wonder3D++ | **AGPL-3.0** (README de la branche Wonder3D_Plus), base SD | Pas interdit par nos règles mais copyleft réseau : tout produit ou service qui inclut le code ou le modèle doit être publié. Déconseillé |

Conclusion : **pas de SDS conditionné par l'image sous licence permissive identifié**. Le SDS guidé par
texte invente du détail plutôt que de coller à l'image et coûte cher (des milliers d'itérations). Piste
de basse priorité.

## 7. Méthodes complètes publiées

| Méthode | Licence | Source | Dépendances bloquantes | Verdict |
|---|---|---|---|---|
| **DetailGen3D** (VAST, 3DV 2026) : mesh grossier + image → mesh détaillé, par flux génératif dans l'espace latent de TripoSG | Code MIT | Code : README/étiquette ; poids MIT déclarés HF (`VAST-AI/DetailGen3D`, 0,4 B) | Base TripoSG (MIT déclaré, candidat de l'étape 1) ; remerciements à HunyuanDiT et FlashVDM : **vérifier qu'aucun code Hunyuan n'est repris** | **Candidat sérieux**, complémentaire : ajoute du détail plausible, pas un alignement pixel. Recouvre peut-être l'étape 2 |
| Unique3D (ISOMER) : reconstruction de mesh depuis des normales multi-vues | Code MIT (étiquette) | déclaré | `requirements.txt` lu : **nvdiffrast** et PyTorch3D, plus rembg et un modèle image → multi-vues dérivé de SD1.5 | Le code ISOMER est une bonne référence d'algorithme (remaillage + normales), mais à porter sur Kaolin ; ses modèles multi-vues ne sont pas retenus |
| CraftsMan3D (raffineur géométrique par normales) | MIT (README) | déclaré | Le code du raffineur ne semble pas publié | Inspiration uniquement |
| Hi3DGen (Stable-X, « normal bridging ») | MIT | déclaré | Basé sur TRELLIS ; le README dit avoir retiré kaolin, nvdiffrast et flexicube | Générateur (étape 1), pas un raffineur |
| Elevate3D (SIGGRAPH 2025) | MIT (étiquette) | déclaré | **FLUX** (variante non précisée, probablement dev), 48 Go recommandés | **Éliminé** en l'état |
| Wonder3D++ | AGPL-3.0 | déclaré | — | Déconseillé (voir 6) |
| threestudio (Magic123, DreamCraft3D…) | Apache-2.0 (connu, non relu) | — | nvdiffrast, Zero123 et DeepFloyd IF non commerciaux | **Éliminé** pour le produit |

Aucune méthode publiée ne fait exactement « coller un mesh existant à une seule image de concept » sous
licence propre et sans nvdiffrast. **La solution réaliste est une boucle maison** assemblée à partir des
briques des sections 2 à 5.

## 8. Tableau comparatif des chaînes proposées

| Rang | Chaîne | Corrige | Faisabilité 16 Go Windows | Risque licence | Risque technique |
|---|---|---|---|---|---|
| 1 | **FlexiCubes (Kaolin) + rasteriseur Kaolin `cuda`** + silhouette + profondeur DA-V2-S/DA3Mono-L + normales MoGe-2, visibilité gelée | Silhouette, accessoires mal placés (topologie), signature | Oui jusqu'à 256³ (estimation), roue Windows officielle | Faible (Apache-2.0, à condition d'éviter `kaolin.non_commercial`) | Moyen : boucle à écrire, gradients de silhouette DIB-R moins précis que nvdiffrast, résolution globale insuffisante pour le visage |
| 2 | **Large Steps + Continuous Remeshing** sur le même rasteriseur Kaolin, recadrage de la tête | Visage, contours fins, petits écarts | Oui, faible mémoire | Faible (BSD-3, MIT) | Faible à moyen : `torch_scatter` sous Windows, pas de changement de topologie |
| 3 | Pertes DINOv2 (et LPIPS) en complément des rangs 1 et 2 | Détail couleur et cohérence des parties | Oui | Faible (DINOv2) ; ImageNet pour LPIPS à trancher | Faible |
| 4 | DetailGen3D en amont ou en parallèle | Détail géométrique plausible | Probable (0,4 B + VAE TripoSG, à mesurer) | Moyen (ascendance TripoSG / Hunyuan à auditer) | Moyen : ne garantit pas l'alignement à l'image |
| 5 | Mitsuba 3 (projectif) + Large Steps | Comme 2, avec une visibilité exacte | Oui (roues Windows) | Faible (BSD-3) | Moyen : interop PyTorch, plus lent ; utile comme référence de qualité |
| — | SDS (SDXL texte) | Détail inventé | Oui en fp16 | Moyen (RAIL++ + « research purposes ») | Élevé (hallucinations, long) | Non recommandé |

Motivation du classement :
1. **FlexiCubes en tête**, comme demandé, parce que les conditions tiennent : licence Apache-2.0 lue dans
   le fichier source de Kaolin, roue Windows cp312 pour torch 2.6 cu124 publiée, rasteriseur Apache dans
   le même paquet, mémoire compatible 16 Go jusqu'à 256³. C'est la seule brique qui traite les accessoires
   mal placés, parce qu'elle peut changer la topologie. Réserve : il faut confirmer le contenu exact de la
   roue 0.18.0 et mesurer la mémoire réelle.
2. La déformation de sommets vient en second car c'est la seule à donner du détail local au visage ; les
   deux passes sont complémentaires (FlexiCubes grossier puis sommets fins).
3. Les estimateurs ne sont classés qu'à licence permissive : MoGe-2 (normales, MIT) et DA-V2 Small / DA3
   Apache. Les modèles dérivés de SD2 passent après, à cause du retrait de SD2 et de l'OpenRAIL.

## 9. Proposition d'expérience (pour `/banc-essai`, après audit)

Sur 3 images connues (`3dfab…` pour l'accessoire, une petite tête, `1464…` pour la signature) et un
mesh de l'étape 2 :
1. Optimisation de la caméra seule (silhouette).
2. FlexiCubes 192³ puis 256³ : IoU de silhouette avant / après, VRAM de pointe, temps.
3. Passe sommets sur le recadrage de la tête.
4. Rendus avant / après depuis la vue de l'image **et depuis le dos** (contrôle que les faces cachées ne
   bougent pas). Sorties sous `data/bench/`.

## Sources consultées le 2026-11-06

- nvdiffrast `LICENSE.txt` et README : https://github.com/NVlabs/nvdiffrast
- Kaolin `LICENSE`, `tools/requirements.txt`, `kaolin/ops/conversions/flexicubes/flexicubes.py`,
  `kaolin/non_commercial/flexicubes/flexicubes.py` : https://github.com/NVIDIAGameWorks/kaolin ;
  installation : https://kaolin.readthedocs.io/en/latest/notes/installation.html ; roues :
  https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html ; API rendu :
  https://kaolin.readthedocs.io/en/latest/modules/kaolin.render.mesh.html ; API FlexiCubes :
  https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.conversions.html
- FlexiCubes `LICENSE.txt` et README : https://github.com/nv-tlabs/FlexiCubes
- warp-lang `LICENSE.md` : https://github.com/NVIDIA/warp
- Mitsuba 3 `LICENSE` : https://github.com/mitsuba-renderer/mitsuba3 ; PyPI : https://pypi.org/project/mitsuba/ ;
  tutoriel : https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html
- PyTorch3D `LICENSE`, `INSTALL.md` : https://github.com/facebookresearch/pytorch3d
- DRTK `LICENSE`, README, installation : https://github.com/facebookresearch/DRTK
- Large Steps `LICENSE` : https://github.com/rgl-epfl/large-steps-pytorch ; cholespy `LICENSE` :
  https://github.com/rgl-epfl/cholespy
- Continuous Remeshing `LICENSE` : https://github.com/Profactor/continuous-remeshing
- gpytoolbox `LICENSE.MIT` : https://github.com/sgsellan/gpytoolbox
- Depth Anything V2 : https://github.com/DepthAnything/Depth-Anything-V2 ,
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small
- Depth Anything 3 : https://github.com/ByteDance-Seed/Depth-Anything-3 ,
  https://huggingface.co/depth-anything/DA3MONO-LARGE
- MoGe `LICENSE` : https://github.com/microsoft/MoGe ; https://huggingface.co/Ruicheng/moge-2-vitl-normal
- Marigold : https://github.com/prs-eth/Marigold , https://huggingface.co/prs-eth/marigold-normals-v1-1 ,
  https://huggingface.co/prs-eth/marigold-depth-v1-1
- StableNormal : https://github.com/Stable-X/StableNormal , https://huggingface.co/Stable-X/stable-normal-v0-1
- Lotus : https://huggingface.co/jingheya/lotus-normal-g-v1-1
- DSINE `LICENSE` : https://github.com/baegwangbin/DSINE
- Sapiens `LICENSE` : https://github.com/facebookresearch/sapiens
- DINOv2 `LICENSE` : https://github.com/facebookresearch/dinov2 ; DINOv3 `LICENSE.md` :
  https://github.com/facebookresearch/dinov3
- CLIP `LICENSE` : https://github.com/openai/CLIP ; LPIPS `LICENSE` :
  https://github.com/richzhang/PerceptualSimilarity
- Zero123++ : https://github.com/SUDO-AI-3D/zero123plus ; SDXL :
  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 ; retrait de SD2 :
  https://discuss.huggingface.co/t/error-404-https-huggingface-co-stabilityai-stable-diffusion-2/170425
- DetailGen3D : https://github.com/VAST-AI-Research/DetailGen3D , https://huggingface.co/VAST-AI/DetailGen3D
- Unique3D (`requirements.txt`, `mesh_reconstruction/render.py`) : https://github.com/AiuniAI/Unique3D
- Wonder3D / Wonder3D++ : https://github.com/xxlong0/Wonder3D
- CraftsMan3D : https://github.com/wyysf-98/CraftsMan ; Hi3DGen : https://github.com/Stable-X/Hi3DGen ;
  Elevate3D : https://github.com/ryunuri/elevate3d
