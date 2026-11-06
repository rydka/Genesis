# Veille : optimiseur de mesh (étape 2 du pipeline 3D)

Date : 2026-11-06. Agent : Veille modèles. Rien n'a été installé ni testé : ce document trie des
candidats. Rien ne passe à l'intégration avant l'audit de `/gardien-licences` et une ligne validée par un
humain dans `LICENSES.md`.

## Périmètre

Pipeline envisagé : générateur de mesh → **optimiseur de mesh** → optimiseur de texture → optimiseur local
par rapprochement à l'image. Ce document ne traite que l'étape 2.

Entrée : mesh brut d'un générateur image → 3D (0,5 à 1,7 M de faces, marching cubes), avec morceaux
flottants, petits trous, excroissances (signature d'artiste, voir `docs/experiments.md`), parfois une
plaque plate. Sortie : mesh exploitable, selon l'usage :

- **Jeu** (« jouable ») : low poly (quelques milliers à quelques dizaines de milliers de faces), quads ou
  triangles propres, UV pour l'étape texture.
- **Impression** de figurines : étanche, 2-manifold, sans auto-intersection, épaisseurs suffisantes ;
  la topologie et les UV comptent peu, la densité peut rester élevée.

Sous-tâches : nettoyage, réparation (étanchéité), décimation, retopologie, UV.

## Niveaux de vérification des licences

Le registre exige de distinguer ce qui a été lu de ce qui est déclaré :

- **L** : texte du fichier LICENSE lu (au moins l'en-tête et les clauses de restriction), soit dans
  `.venv\Lib\site-packages\*.dist-info`, soit via l'API GitHub le 2026-11-06.
- **S** : licence détectée par GitHub (SPDX) à partir du fichier du dépôt, texte non relu.
- **D** : seulement déclarée (classifieur PyPI, étiquette HuggingFace, badge de README, page web).
- **Aucune** : pas de fichier de licence trouvé dans le dépôt, donc par défaut tous droits réservés.

## Rappel GPL / LGPL / AGPL (analyse technique, pas un avis juridique)

- Les **sorties** (meshes produits) d'un outil GPL ne sont pas couvertes par la GPL.
- **Outil externe** : appeler Blender, MeshLab, QuadWild ou MeshFix en sous-processus, avec échange par
  fichiers, ne lie pas notre code. C'est déjà la règle pour Blender dans `LICENSES.md`.
- **Intégration** : `import pymeshlab` (GPL-3) ou `import pymeshfix` (GPL-3 / AGPL-3, voir plus bas)
  dans `fantasy_rag` fait de l'ensemble une œuvre combinée. Si Genesis est un jour distribué aux clients
  (logiciel installé), tout le code devrait être publié sous GPL. En SaaS, la GPL n'est pas déclenchée,
  mais l'**AGPL** l'est. À éviter dans le produit ; tolérable dans des scripts de recherche sous
  `tools/` qui ne sont pas livrés, à confirmer avec le gardien.
- **MPL-2.0** (libigl cœur, fTetWild, Eigen) : copyleft par fichier, compatible avec un produit fermé si
  les fichiers MPL modifiés restent publiés.

## Constat sur l'environnement actuel

`.venv` contient déjà, sans ligne dans `LICENSES.md` :

| Paquet | Version | Licence (niveau) | Remarque |
|---|---|---|---|
| `pymeshlab` | 2025.7.post1 | GPL-3.0 (**L**, `dist-info/licenses/LICENSE`) | Requis par aucun paquet (`pip show`) ; probablement un reste des dépendances de Hunyuan3D. Embarque des greffons dont `filter_mesh_alpha_wrap` (CGAL, GPL). Non importé par `fantasy_rag` (recherche faite hors `third_party/`). |
| `xatlas` (python) | 0.0.11 | MIT (**L**) | Requis par aucun paquet. |
| `open3d` | 0.19.0 | MIT (**L**) | Absent du registre. |
| `trimesh` | 4.8.1 | MIT (**L**) | Ligne « À VÉRIFIER » dans le registre : le fichier dit MIT. |

À signaler au gardien : ces lignes doivent être créées ou complétées, et `pymeshlab` ne doit pas être
importé par le code du produit.

## Candidats éliminés d'emblée

| Candidat | Rôle | Motif (niveau) |
|---|---|---|
| MeshAnything / MeshAnything V2 (`buaacyw`) | Mesh « artiste » depuis nuage de points | Code **S-Lab License 1.0, non commercial** (**L**). Piège : les poids `Yiwen-ntu/MeshAnythingV2` sont étiquetés MIT sur HF (**D**) ; le code d'inférence reste non commercial. |
| BPT (`Tencent-Hunyuan/bpt`) | Idem | **Tencent Hunyuan 3D 2.0 Community License, exclut l'UE** (**L**). Même piège : poids `whaohan/bpt` étiquetés MIT sur HF (**D**). |
| Hunyuan3D-PolyGen et autres travaux Tencent Hunyuan | Idem | Même famille de licences excluant l'UE ; non retenu. |
| FastMesh (3DV 2026) | Idem | **S-Lab License 1.0, non commercial** (**L**). Reprend aussi MeshAnything, BPT, Michelangelo. |
| EdgeRunner (NVIDIA) | Idem | NVIDIA Source Code License : « non-commercially … and with NVIDIA Processors » (**L**). |
| LLaMA-Mesh (NVIDIA) | Mesh en texte via LLM | NVIDIA License, §3.3 non commercial (**L**). |
| PartField (NVIDIA) | Segmentation en parties | NVIDIA License, §3.3 non commercial (**L**). |
| MeshLib (MeshInspector) | Réparation, décimation, remaillage | « NON-COMMERCIAL & education LICENSE AGREEMENT » (**L**) ; licence commerciale payante à part. |
| Semantic-Visibility-UV-Param (ICLR 2026) | UV appris | Aucun fichier de licence : tous droits réservés. |
| QuadGPT, SATO, Mesh-Pro, Meshtron, DreamUV, SeamGen, MeshQuery | Quads / UV appris | Pas de code ni de poids publiés trouvés (SATO : article sous CC BY-NC-SA). |

## Fiches des candidats retenus ou à arbitrer

### A. Bibliothèques Python intégrables (licence permissive)

**trimesh** : MIT (**L**, installé). Nettoyage de base : composantes connexes (`split`), faces
dégénérées et dupliquées, fusion de sommets, orientation, `fill_holes` (petits trous simples seulement),
tests `is_watertight` / `is_winding_consistent`. CPU, déjà utilisé par `mesh_utils.repair_mesh`. Ne
répare pas les trous complexes ni les auto-intersections. Jeu et impression (diagnostic).

**open3d** : MIT (**L**, installé). `cluster_connected_triangles` (suppression des îlots par aire),
`remove_non_manifold_edges`, `simplify_quadric_decimation`, lissage Taubin. CPU, gère 1,7 M de faces.
Jeu (décimation) et impression (nettoyage).

**fast-simplification** (pyvista) : MIT (**S** GitHub, **D** PyPI). Décimation quadrique très rapide
(dérivée de Fast-Quadric-Mesh-Simplification). Dépôt actif (novembre 2026). Pas de garantie de manifold en
sortie. Jeu.

**pyfqmr** : MIT (**S**, **D**). Même algorithme, alternative. Actif (juillet 2026).

**meshoptimizer** (zeux) : MIT (**S**). Référence industrielle du jeu vidéo : simplification avec
préservation des attributs (UV, normales), LOD, optimisation du cache de sommets. Bindings Python
`meshoptimizer` (PyPI, **D** MIT, version alpha 0.2.30a0). Très actif. Jeu (LOD une fois les UV posées).

**xatlas** (jpcy + bindings `xatlas-python`) : MIT (**L** pour les bindings installés, **S** pour la
bibliothèque C++). Dépliage UV automatique et empaquetage d'atlas ; c'est ce qu'utilisent la plupart des
pipelines image → 3D. Lent sur des meshes de plus de quelques centaines de milliers de faces : à appliquer
après décimation. Bibliothèque C++ peu active (2024) mais stable. Jeu (UV).

**manifold3d** (elalish/manifold) : Apache-2.0 (**S**, **D**). Booléens robustes et garantis manifold ;
exige une entrée déjà manifold. Utile pour l'impression (ajout d'un socle, union de parties, vérification).
Très actif. Ne répare pas un mesh cassé.

**mesh2sdf** (wang-ps) : MIT (**L**, fichier nommé `LISCENCE`). Convertit un mesh non étanche en SDF
puis, avec un marching cubes (scikit-image, BSD), donne un mesh **étanche**. Perte des détails fins selon
la résolution de la grille. Impression.

**point-cloud-utils** : MIT (**S**, **D**). `make_mesh_watertight` (méthode « Manifold » de Huang et al.,
dont le dépôt d'origine `hjwdzh/Manifold` est BSD-3, **L**). Impression.

**CuMesh** (JeffreyXiang, auteur de TRELLIS) : MIT (**L**), sous-module `cubvh` (MIT, **S**). Bibliothèque
**GPU** (PyTorch + CUDA) qui couvre presque toute l'étape : suppression des petites composantes,
faces dégénérées et dupliquées, réparation des arêtes non manifold, orientation, `fill_holes`,
décimation, remaillage étanche par UDF + Dual Contouring, dépliage UV (xatlas accéléré). Utilisée par le
post-traitement de TRELLIS.2, lui-même candidat générateur. Risques : compilation depuis les sources
(CUDA Toolkit ≥ 12.4, PyTorch ≥ 2.4), pas de roue Windows annoncée ; jeune (222 étoiles, dernière
activité mai 2026). VRAM non annoncée, a priori faible devant celle du générateur. Jeu et impression.

**gpytoolbox** : double licence (**L** README) : module principal MIT, module `gpytoolbox.copyleft`
GPL-3. N'importer que le module principal. Remaillage isotrope, réparation légère. Optionnel.

**libigl** (C++ et bindings Python) : cœur MPL-2.0, parties copyleft GPL-3 (**S** : fichiers
`LICENSE.MPL2` et `LICENSE.GPL` présents ; PyPI déclare MPL-2.0). Paramétrisation (LSCM, ARAP),
nombreux outils. À n'utiliser qu'en vérifiant que la fonction appelée n'est pas dans la partie GPL.

**pmp-library** : MIT (**S**). Remaillage isotrope, décimation, lissage, de qualité académique. Pas de
paquet Python officiel trouvé (le paquet PyPI `pmp` semble sans rapport) : intégration en C++ ou via
binaire.

**geogram** (Inria) : BSD-3 (**L**). Contient `vorpalite` : remaillage CVT de qualité, réparation,
décimation, en ligne de commande. Très actif. Remaillage triangle propre, jeu et impression.

**fTetWild** : MPL-2.0 (**S**). Tétraédrisation robuste par enveloppe : sa surface extérieure donne un
mesh étanche et sans auto-intersection même depuis une soupe de triangles. Lourd (CPU, minutes sur
1,7 M de faces, à mesurer). Impression, en dernier recours.

**OpenVDB** : Apache-2.0 (**S**). Remaillage voxel (étanche par construction). Pas de roue Python
officielle sur PyPI ; accessible via Blender (modificateur Remesh, mode Voxel).

### B. Outils externes en sous-processus

**Blender 4.5 LTS** (installé : 4.5.15, build 2026-10-22) : GPL-3.0, déjà au registre comme outil externe.
Ses sorties sont libres. Utilisable sans interface (`blender -b -P script.py`). Fournit d'un bloc :
- nettoyage : fusion par distance, suppression des éléments isolés, remplissage des trous,
  « Make Manifold » de l'extension 3D Print Toolbox (GPL, extension officielle à installer) ;
- étanchéité : Remesh **Voxel** (OpenVDB) ;
- décimation : modificateur Decimate (collapse, planar, un-subdivide) ;
- retopologie : Remesh **QuadriFlow** intégré (code QuadriFlow BSD-3, **L**) ; exige une entrée
  manifold, donc souvent précédé d'un Remesh Voxel ;
- UV : Smart UV Project, Unwrap (angle based, conformal, minimum stretch depuis 4.3), empaquetage.
Avantage : un seul outil déjà présent et déjà au registre. Inconvénient : qualité de retopologie
moyenne sur des personnages (boucles de bord non alignées sur l'anatomie), lenteur sur 1,7 M de faces
à mesurer.

**Instant Meshes** (wjakob) : BSD-3 (**L**). Retopologie quad-dominante ou triangles isotropes par champs
de directions, robuste sur des meshes denses et bruités, binaire Windows précompilé, ligne de commande
(`-o sortie.obj -f faces -D` pour quads purs). Dépôt figé depuis 2022 mais stable et très utilisé. Le
binding `pynanoinstantmeshes` (utilisé par Stable Fast 3D) a une roue Windows cp312 et un classifieur
BSD (**D**), mais **aucun fichier LICENSE** dans son dépôt (`vork/PyNanoInstantMeshes`) : à clarifier
avant tout import. Jeu (retopologie).

**QuadriFlow** (autonome) : BSD-3 (**L**). Même algorithme que dans Blender, avec garanties sur les
singularités. Le binding `pyquadriflow` (satabol) n'a **pas de fichier de licence** : passer par
Blender ou par le binaire compilé.

**QuadWild / QuadWild-BiMDF** : GPL-3.0 (**S**). Meilleure qualité académique de quads guidés par les
lignes caractéristiques (orienté CAO, mais BiMDF améliore les formes organiques). Outil externe
uniquement. Compilation Windows non triviale (Dockerfile Linux fourni pour BiMDF). Jeu.

**AutoRemesher** (huxingyi) : MIT (**S**), dépendances embarquées : Eigen (MPL-2), TBB (Apache-2),
meshoptimizer (MIT), Qt pour l'interface (LGPL, à vérifier). Remailleur quad automatique, très actif
(octobre 2026), 3,6 k étoiles. Application graphique : mode batch à vérifier. Jeu.

**MeshFix** (Attene) / **PyMeshFix** (pyvista) : MeshFix GPL-3 (fichier `gpl-3.0.txt` présent) ;
PyMeshFix : fichier GPL-3 (**L**) mais classifieur PyPI **AGPL-3** (**D**), incohérence à lever. Très
efficace pour fermer les trous et supprimer les auto-intersections en gardant la surface. Uniquement en
sous-processus, jamais importé dans le produit. Impression.

**MeshLab / PyMeshLab** : GPL-3 (**L**). Le plus complet (décimation quadrique avec préservation des
UV, Alpha Wrap CGAL pour l'étanchéité, nettoyage, isotropic remeshing). En sous-processus uniquement
(`meshlabserver` ou un script Python séparé lancé comme processus, non importé par `fantasy_rag`). À
confirmer par le gardien : un script `tools/` qui importe pymeshlab est-il acceptable s'il n'est jamais
livré ?

**Quad Remesher** (Exoside) : propriétaire, payant. Licence Pro perpétuelle ~110 USD HT par poste
(commerciale), Indie non commerciale ; la documentation dit que la licence standard **n'autorise pas**
l'usage comme SDK ni l'intégration dans un processus automatisé sans accord d'Exoside (**D**, pages
web ; EULA non lu). Meilleure qualité de quads « artiste » disponible hors IA. Seulement si un contrat
d'automatisation est négocié. Jeu.

**NeurCross** (SIGGRAPH 2025) : AGPL-3.0 (**S**). Optimisation neuronale par mesh (pas de poids
pré-entraînés) d'un champ de croix pour la génération de quads ; environnement ancien (CUDA 11.7,
PyTorch 1.13). Outil externe de recherche au mieux ; AGPL exclut tout usage intégré en SaaS.

**CurioMesh**, **UVgami** (add-ons Blender) : GPL-3 (**S**), utilisables comme add-ons d'un Blender
externe. UVgami peut appeler PartUV/PartField (non commercial) : ne pas activer ce moteur. Faible
maturité (CurioMesh « prototype »).

### C. Approches apprises (mesh « artiste » ou retopologie par IA)

Constat : **aucune approche apprise de retopologie n'est aujourd'hui propre pour un usage commercial
dans l'UE.** Les deux seules à licence de code permissive ont des dépendances ou des poids non clarifiés.

**TreeMeshGPT** (Sea AI Lab, CVPR 2025) : code MIT (**L**). Transformer autorégressif conditionné par
un nuage de points échantillonné sur le mesh dense (le script décime d'abord l'entrée vers ~6 000 faces).
Sortie : mesh triangles « artiste », **5 500 faces max** (7 bits) ou **11 000** (9 bits) : échelle jeu,
pas impression. Points à risque :
- poids sur Google Drive **sans licence indiquée** ;
- code « construit sur PivotMesh » (`whaohan/pivotmesh`) dont le dépôt **n'a pas de fichier de
  licence** ; l'auteur de PivotMesh est aussi celui de BPT (licence Tencent excluant l'UE) ;
- données d'entraînement non précisées dans le README (Objaverse probable, licences mixtes dont NC) ;
- VRAM et temps non annoncés ; dépôt inactif depuis mai 2025.
Classement : intéressant à tester **en recherche uniquement** après audit ; non utilisable en produit en
l'état.

**DeepMesh** (Tsinghua / ShengShu, ICCV 2025) : fichier LICENSE **Apache-2.0** (**L**), badge README
**MIT** (**D**), poids `zzzrw/DeepMesh` étiquetés MIT sur HF (**D**). Modèle 0,5 B conditionné par nuage
de points avec normales, affiné par DPO. Mais le README indique que le code « est basé sur » **BPT**
(licence Tencent excluant l'UE, **L**), **Michelangelo** (GPL-3, **S**, présent dans le dossier `miche/`),
LLaMA-Mesh (NVIDIA non commercial) et MeshAnything V2 (S-Lab non commercial). Testé sur Linux avec
A100/A800/A6000, flash-attention requis. Classement : **risque élevé** ; à considérer comme exclu tant
qu'un audit fichier par fichier n'a pas montré qu'aucun code BPT ou Michelangelo n'est repris, et que les
poids ne dérivent pas d'un modèle Tencent.

**MeshXL** (OpenMeshLab) : code MIT (**S**), poids `CH3COOK/mesh-xl-*` sans étiquette de licence (**D**).
Génération de mesh (non conditionnée ou par texte), pas un remailleur de mesh dense : hors besoin.

**meshgpt-pytorch** (lucidrains) : MIT (**S**), sans poids officiels. Base possible pour un futur
fine-tuning maison sur des données propres (voir le rôle Données et fine-tuning), pas un outil prêt.

**PartUV** (SIGGRAPH Asia 2025) : projet Apache-2.0 (**L**) mais le même fichier LICENSE inclut la
licence NVIDIA non commerciale pour **PartField**, utilisé pour la segmentation, et le README renvoie à
**UVPackMaster** (payant) pour l'empaquetage. Non utilisable tel quel ; la partie géométrique Apache
pourrait l'être si l'on remplace PartField par une segmentation propre. À garder en veille pour les UV.

## Tableau comparatif

| Candidat | Sous-tâches | Licence (niveau) | Mode d'usage | Jeu / Impr. | Matériel local | Maturité |
|---|---|---|---|---|---|---|
| trimesh | nettoyage, diagnostic | MIT (L) | import | les deux | CPU, installé | élevée |
| open3d | nettoyage, décimation | MIT (L) | import | les deux | CPU, installé | élevée |
| fast-simplification | décimation | MIT (S) | import | jeu | CPU | bonne, actif |
| meshoptimizer | décimation, LOD | MIT (S) | import (binding alpha) | jeu | CPU | très élevée |
| xatlas | UV | MIT (L) | import | jeu | CPU, installé | stable |
| CuMesh | nettoyage, trous, étanchéité, décimation, UV | MIT (L) | import | les deux | GPU CUDA, build Windows incertain | jeune |
| mesh2sdf + marching cubes | étanchéité | MIT (L) | import | impression | CPU | moyenne |
| manifold3d | booléens, validation | Apache-2.0 (S) | import | impression | CPU | élevée |
| geogram / vorpalite | remaillage, réparation | BSD-3 (L) | binaire | les deux | CPU | élevée |
| Blender 4.5 | tout (voxel, Decimate, QuadriFlow, UV) | GPL-3 (registre) | sous-processus | les deux | installé | très élevée |
| Instant Meshes | retopologie quads | BSD-3 (L) | binaire | jeu | CPU, binaire Windows | stable, figé |
| QuadriFlow | retopologie quads | BSD-3 (L) | via Blender | jeu | CPU | stable, figé |
| QuadWild-BiMDF | retopologie quads | GPL-3 (S) | binaire externe | jeu | CPU, build difficile | recherche |
| AutoRemesher | retopologie quads | MIT (S) + Qt | application | jeu | CPU | actif |
| PyMeshFix / MeshFix | trous, auto-intersections | GPL-3 / AGPL-3 (L / D) | sous-processus seulement | impression | CPU | élevée |
| PyMeshLab | tout | GPL-3 (L) | sous-processus seulement | les deux | CPU, installé | élevée |
| fTetWild | étanchéité robuste | MPL-2.0 (S) | binaire | impression | CPU, lent | élevée |
| Quad Remesher | retopologie quads | propriétaire (D) | add-on payant, automatisation sous accord | jeu | CPU | élevée |
| TreeMeshGPT | mesh « artiste » | code MIT (L), poids sans licence | recherche | jeu | GPU, VRAM non annoncée | inactif |
| DeepMesh | mesh « artiste » | Apache-2.0 (L) vs MIT (D), dérive de BPT et Michelangelo | exclu en l'état | jeu | GPU, testé sur A100 | moyenne |
| PartUV | UV | Apache-2.0 + PartField NC (L) | exclu en l'état | jeu | GPU | recherche |

## Recommandation par sous-tâche

Ordre de préférence : permissif importable > outil externe déjà au registre (Blender) > binaire
permissif > outil GPL en sous-processus > payant. Les temps sur 1,7 M de faces ne sont pas annoncés par
les projets : à mesurer par `/banc-essai`, sans juger la qualité (rôle `/evaluateur`).

1. **Nettoyage** : **trimesh + open3d** (déjà installés, MIT lus). Suppression des composantes sous
   un seuil d'aire relative, faces dégénérées et dupliquées, arêtes non manifold, petits trous. Les
   excroissances (signature) et les plaques plates ne sont détectées par aucun outil : ce sont des
   heuristiques à écrire (aire relative d'un composant, planarité d'un amas de faces) ou à traiter en
   amont au détourage. Alternative GPU tout-en-un : **CuMesh**, si sa compilation sous Windows passe.

2. **Réparation / étanchéité (impression)** : en premier **Blender Remesh Voxel** (OpenVDB, sous-processus,
   déjà au registre) ou **mesh2sdf + marching cubes** en Python (MIT), puis validation **manifold3d**.
   Le remaillage voxel ferme tout mais lisse les détails fins (doigts, lame) : la taille de voxel est le
   réglage clé. En dernier recours pour des cas pathologiques : **fTetWild** (MPL-2). MeshFix et
   PyMeshLab Alpha Wrap seulement en sous-processus, et seulement si le gardien l'accepte.

3. **Décimation** : **fast-simplification** (MIT) ou `open3d.simplify_quadric_decimation` (déjà là)
   pour passer de 1,7 M à quelques centaines de milliers ou dizaines de milliers de faces ;
   **meshoptimizer** pour les LOD de jeu une fois les UV posées (préserve les attributs). Blender
   Decimate en solution de repli dans le même sous-processus que le reste.

4. **Retopologie (jeu)** : **Instant Meshes** (BSD-3 lu, binaire Windows, ligne de commande) en premier,
   **QuadriFlow via Blender** en second (BSD-3 lu, après Remesh Voxel). À comparer ensuite avec
   **QuadWild-BiMDF** (GPL-3, externe) et **AutoRemesher** (MIT) si la qualité des boucles ne suffit
   pas. Quad Remesher seulement avec un accord d'automatisation d'Exoside. Aucune IA de retopologie n'est
   recommandée en production ; **TreeMeshGPT** peut être testé en recherche après audit des poids et de
   PivotMesh, pour des versions très low poly (≤ 11 000 faces).
   Pour l'impression, la retopologie est inutile : décimation modérée d'un mesh étanche.

5. **UV (pour l'étape texture)** : **xatlas** (MIT, installé) après décimation ; **Blender Smart UV
   Project / Minimum Stretch** en alternative dans le même sous-processus Blender. PartUV à suivre si sa
   segmentation PartField est remplaçable.

Chaîne minimale proposée pour un premier banc, tout en composants permissifs ou déjà au registre :
trimesh/open3d (nettoyage) → Blender en un seul sous-processus (Remesh Voxel → Decimate ou QuadriFlow →
Smart UV) ; variante « sans Blender » : open3d → mesh2sdf → fast-simplification → Instant Meshes →
xatlas.

## Dépendances à risque repérées

- `pymeshlab` (GPL-3) installé dans `.venv` sans ligne au registre et sans dépendant : à déclarer ou
  désinstaller ; ne pas l'importer dans `fantasy_rag`.
- `pymeshfix` : GPL-3 dans le dépôt, AGPL-3 sur PyPI. Si AGPL, même un usage SaaS intégré est exclu.
- `pynanoinstantmeshes` et `pyquadriflow` : bindings **sans fichier LICENSE** alors que les
  bibliothèques d'origine sont BSD-3.
- libigl et gpytoolbox : parties GPL dans des paquets majoritairement permissifs ; vérifier chaque
  fonction appelée.
- AutoRemesher : interface Qt (LGPL ou GPL selon les modules) dans un binaire MIT.
- CuMesh : compilation CUDA, pas de roue Windows ; sous-module `cubvh` (fourche `trellis.2`).
- Modèles appris : poids étiquetés MIT sur HF alors que le code est non commercial ou excluant l'UE
  (MeshAnything V2, BPT) ; DeepMesh dérivé de BPT et Michelangelo ; TreeMeshGPT sur PivotMesh sans
  licence ; données d'entraînement de type Objaverse (licences mixtes).
- Quad Remesher : EULA à lire ; automatisation non couverte par la licence standard selon Exoside.

## Candidats à faire auditer par `/gardien-licences`

Priorité 1 (recommandés) : open3d, trimesh (compléter la ligne), xatlas (bindings + bibliothèque
C++), fast-simplification, meshoptimizer (et son binding Python), manifold3d, mesh2sdf (+ scikit-image),
Instant Meshes (binaire Windows officiel), QuadriFlow tel qu'embarqué dans Blender, CuMesh (+ cubvh),
extension Blender 3D Print Toolbox.

Priorité 2 (alternatives) : geogram/vorpalite, fTetWild, point-cloud-utils, pyfqmr, pmp-library,
AutoRemesher (avec Qt), QuadWild-BiMDF (externe GPL), gpytoolbox (module MIT seul), libigl (partie MPL).

Priorité 3 (décision de principe sur la GPL / AGPL en sous-processus) : PyMeshLab (déjà installé),
MeshFix / PyMeshFix.

Recherche uniquement (à auditer avant tout test) : TreeMeshGPT (poids Google Drive, PivotMesh, données),
DeepMesh (fichier par fichier : `miche/`, code BPT, origine des poids).

Hors open source, décision humaine : Quad Remesher (EULA et accord d'automatisation).

## Sources consultées le 2026-11-06

- Fichiers LICENSE locaux : `.venv\Lib\site-packages\{pymeshlab,trimesh,xatlas,open3d}*.dist-info`.
- API GitHub (`repos/<dépôt>` et `repos/<dépôt>/license`) pour tous les dépôts cités.
- API PyPI (`pypi.org/pypi/<paquet>/json`) et API HuggingFace (`huggingface.co/api/models/<id>`).
- README : DeepMesh, TreeMeshGPT, FastMesh, CuMesh, gpytoolbox, PartUV.
- Pages Exoside : https://exoside.com/quadremesher/quadremesher-buy/ et https://exoside.com/?p=5390.
- Articles : FastMesh (arXiv 2508.19188), QuadGPT, SATO (arXiv 2604.09132), Mesh-Pro (arXiv 2603.00526),
  NeurCross (arXiv 2405.13745), PartUV (arXiv 2511.16659).
