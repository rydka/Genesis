# Registre des licences

Genesis est un projet de recherche à visée commerciale. La structure et la clientèle sont en **Europe**.
Un composant (bibliothèque, modèle, poids, jeu de données) n'entre dans le projet que s'il a une ligne
dans ce registre. Cette ligne est tenue par l'agent Gardien des licences (`/gardien-licences`).

Ce registre est une analyse technique, pas un avis juridique. Avant toute commercialisation, un juriste
doit valider les lignes marquées « OUI ».

## Légende

- **Commercial UE** : `OUI` (usage commercial possible dans l'UE), `NON`, `À VÉRIFIER`.
- **Vérifié** : date et source réellement lue (fichier LICENSE, fiche du modèle, conditions du dataset).
  Une licence « déclarée » sur une page HuggingFace ne suffit pas : il faut lire le fichier et regarder ce que
  les scripts officiels téléchargent en plus.
- **Validé par** : nom de la personne qui a validé la ligne. Un agent ne remplit jamais cette colonne.

## Modèles et poids

| Composant | Rôle | Statut | Licence code | Licence poids | Territoire | Commercial UE | Vérifié | Validé par |
|---|---|---|---|---|---|---|---|---|
| Hunyuan3D-2.1 (Tencent) | Image → forme 3D | Référence de recherche, **à remplacer** | Tencent Hunyuan 3D 2.1 Community | Idem | Exclut UE, Royaume-Uni, Corée du Sud | **NON** (aucun droit accordé dans l'UE, y compris pour le fine-tuning et la distillation) | 2026-11-04, `third_party/Hunyuan3D-2.1/LICENSE` | |
| MatAnyone2 | Matting d'alpha | **Retiré** le 2026-11-04 | S-Lab License 1.0 (non commercial) | Idem | Monde | **NON** | 2026-11-04, `LICENSE.txt` du dépôt | |
| BiRefNet (`birefnet-general` via rembg) | Détourage (par défaut) | Utilisé | MIT (déclaré) | MIT (déclaré, `ZhengPeng7/BiRefNet`) | Monde | À VÉRIFIER | 2026-11-04, licence déclarée HF uniquement ; vérifier l'origine du fichier ONNX téléchargé par rembg | |
| isnet-anime, u2net (via rembg) | Détourage (comparaison) | Outils de comparaison | À VÉRIFIER | À VÉRIFIER | | À VÉRIFIER | | |
| CLIP ViT-B-32 (`sentence-transformers/clip-ViT-B-32`) | Recherche texte → 3D | Utilisé | À VÉRIFIER | À VÉRIFIER | | À VÉRIFIER | | |
| StdGEN | Génération de personnages | Expérimental | Apache-2.0 | Apache-2.0 (déclaré, `hyz317/StdGEN`) | Monde | À VÉRIFIER (dépendances téléchargées non auditées) | 2026-11-04, `third_party/StdGEN/LICENSE` + HF | |
| TRELLIS.2-4B, génération (`microsoft/TRELLIS.2-4B` : `ss_flow_*`, `slat_flow_img2shape_*`, `shape_dec_next_*`, `tex_dec_next_*`, `slat_flow_imgshape2tex_*` ; plus `ss_dec_conv3d_16l8_fp16` de `microsoft/TRELLIS-image-large`) | Image → mesh PBR | Candidat n° 1 (non intégré) | MIT (lu, `LICENSE` de `microsoft/TRELLIS.2`) | MIT (fiche lue : `license: mit` et « This model is released under the MIT License » ; aucun fichier LICENSE dans les dépôts de poids ; idem `TRELLIS-image-large`) | Monde | À VÉRIFIER : utilisable seulement après correctifs. Tel que publié, `from_pretrained` instancie `briaai/RMBG-2.0` (**NON**) même si l'image a un alpha ; `o_voxel/__init__.py` importe `postprocess`, qui importe `nvdiffrast` (**NON**) et `cumesh` (**NON** tel qu'épinglé, voir CuMesh), et `trellis2/models/sc_vaes/fdg_vae.py` importe `o_voxel` ; `trellis2/representations/mesh/base.py` importe `cumesh`. Reste DINOv3 (juriste) et plyfile (GPL-3). Voir « Conditions d'intégration de TRELLIS.2 » plus bas | 2026-11-06, `LICENSE` GitHub, fiches HF, `pipeline.json`, code lu (commit de `main` du 2026-07-10) | |
| TRELLIS.2-4B, texturation (`texturing_pipeline.json` : `shape_enc_next_*`, `tex_dec_next_*`, `slat_flow_imgshape2tex_dit_1_3B_512/1024`) | Texture d'un mesh donné depuis une image | Candidat n° 1 de l'étape texture (non intégré) | MIT (lu) | MIT (fiche lue, même dépôt) | Monde | À VÉRIFIER : mêmes bloquants que la génération ; en plus, `trellis2/pipelines/trellis2_texturing.py` importe `nvdiffrast.torch` en tête de fichier (cuisson UV dans `postprocess_mesh`) et `cumesh` pour le dépliage UV : les deux doivent être remplacés | 2026-11-06, `texturing_pipeline.json`, `trellis2_texturing.py` lus | |
| DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`) | Encodeur d'image de TRELLIS.2 et Pixal3D | Dépendance des candidats | DINOv3 License (Meta, propre) | Idem ; dépôt HF à accès validé manuellement (`gated: manual`) | Monde (« non-exclusive, worldwide ») ; contractant Meta Platforms Ireland pour l'EEE ; droit et tribunaux de Californie | À VÉRIFIER (juriste). Pas de clause non commerciale. Clause armes et export, §1.b.v : « You agree not to use, or permit others to use, DINO Materials for any activities subject to the International Traffic in Arms Regulations (ITAR) or end uses prohibited by Trade Controls, including those related to military or warfare purposes, nuclear industries or applications, espionage, or the development or use of guns or illegal weapons. » ; « Trade Controls » inclut les sanctions de l'UE. Aussi : redistribution seulement sous cette licence avec copie (§1.b.i), pas de rétro-ingénierie (§1.b.iv), résiliation par Meta en cas de manquement (§6). La mention « Built with DINOv3 » citée par la veille n'apparaît pas dans le `LICENSE.md` lu | 2026-11-06, `LICENSE.md` de `facebookresearch/dinov3` + API HF | |
| DINOv2 (`facebookresearch/dinov2`, poids via `torch.hub`) | Encodeur (Direct3D-S2, MoGe-2, pertes par features) | Dépendance des candidats | Apache-2.0 (lu) | Apache-2.0 (README lu : « DINOv2 code and model weights are released under the Apache License 2.0 ») | Monde | OUI | 2026-11-06, `LICENSE` et README GitHub | |
| MoGe-2 (`Ruicheng/moge-2-vitl-normal`, `-vitb-normal`, `-vits-normal`, `moge-2-vitl`) | Normales et géométrie monoculaires (optimisation par l'image) ; Pixal3D charge `moge-2-vitl` | Candidat (non intégré) | MIT, sauf `moge/model/modules/dinov2` en Apache-2.0 (lu) | MIT (déclaré : `license: mit` seul dans la fiche ; `base_model: facebook/dinov2-large`) ; fichier `model.pt` (pickle) | Monde | À VÉRIFIER : la configuration d'entraînement publiée (`configs/train/v2.json`) liste des jeux de données dont plusieurs ont des conditions non commerciales connues (ARKitScenes, Waymo, ScanNet++, Argoverse2, Synthia…), conditions non relues ici ; question « poids MIT entraînés sur des données non commerciales » pour le juriste. Charger avec `weights_only` ou convertir en safetensors | 2026-11-06, `LICENSE` et README GitHub, API HF, `configs/train/v2.json` | |
| Pixal3D (`TencentARC/Pixal3D`, code `TencentARC/Pixal3D`) | Image → mesh PBR (simple ou multi-vues) | Repli, **décision humaine requise** | MIT depuis le commit `5098ba1f` du 2026-05-20 (lu) ; avant : licence propre « academic purposes only » et « Pixal3D IS NOT INTENDED FOR USE WITHIN THE EUROPEAN UNION » (commit `d42dcaad`) | MIT (fichier `LICENSE` HF lu, révision `b648b3c5` du 2026-05-20) ; la révision HF `60f4ffb2` du 2026-05-12 portait `license: other`, `pixal3d-license` et le même texte « academic purposes only » et hors UE ; poids simple vue déposés le 2026-04-30, poids `_mv` le 2026-08-31 | Monde selon la MIT ; mais la fiche HF garde `extra_gated_eu_disallowed: true` (sans effet tant que le dépôt n'est pas « gated ») | À VÉRIFIER (juriste et humain). Le texte en vigueur accorde les droits MIT sans exclusion, mais le drapeau UE maintenu après le passage en MIT contredit l'intention affichée, et un ayant droit Tencent peut verrouiller l'accès UE à tout moment. Hérite aussi de tous les bloquants de TRELLIS.2 (`import o_voxel`, RMBG-2.0 dans `pipeline.json`) et charge DINOv3 depuis le miroir non officiel `camenduru/dinov3-vitl16-pretrain-lvd1689m` (non « gated », sous DINOv3 License déclarée) : utiliser le dépôt `facebook/...`. Si retenu : épingler une révision ≥ `b648b3c5`, archiver la `LICENSE` MIT et demander une confirmation écrite à Tencent ARC sur l'usage dans l'UE | 2026-11-06, historique Git et HF lus, API HF, `inference.py` | |
| Direct3D-S2 (`wushuang98/Direct3D-S2`, v1.0 et v1.1) | Image → mesh (géométrie seule) | Repli (GPU cloud en 1024) | MIT (lu, `LICENSE.txt`) | MIT (déclaré : fiche `license: mit` et « See LICENSE », aucun fichier LICENSE dans le dépôt de poids) ; fichiers `.ckpt` (pickle) | Monde | À VÉRIFIER : `direct3d_s2/utils/__init__.py` importe `fill_hole.py` sans condition, qui importe **pymeshfix** (GPL-3 / AGPL-3) et **python-igraph** (GPL, déclaré PyPI) dans le même processus ; à retirer avant tout usage produit. Détourage `ZhengPeng7/BiRefNet` (ligne BiRefNet ci-dessus), DINOv2 (OUI), `spatial_sparse_attention` en Apache-2.0 (en-tête lu) | 2026-11-06, `LICENSE.txt`, `requirements.txt`, `pipeline.py`, `utils/__init__.py` lus ; API HF | |
| RMBG-2.0 (`briaai/RMBG-2.0`) | Détoureur par défaut de TRELLIS.2 et Pixal3D | **Interdit** : ne pas télécharger | Idem poids | « The model is released under a CC BY-NC 4.0 license for non-commercial use. » ; « Commercial use is subject to a commercial agreement with BRIA. » (fiche HF) | Monde | **NON** | 2026-11-06, fiche HF (`license_name: bria-rmbg-2.0`, accès « gated ») | |
| TripoSG (`VAST-AI/TripoSG`) | Image → mesh | **Retiré des candidats** le 2026-11-06 | MIT déclaré, mais `triposg/models/transformers/triposg_transformer.py` : « This code is based on Tencent HunyuanDiT […] Portions of this code are copied or adapted from HunyuanDiT » | MIT (déclaré) | L'en-tête embarque la Tencent Hunyuan Community License : « THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION » ; « “Territory” shall mean the worldwide territory, excluding the territory of the European Union » | **NON** (code dérivé de Hunyuan, aucun droit dans l'UE ; en plus, d'après la veille : diso en CC BY-NC 4.0, RMBG-1.4 non commercial) | 2026-11-06, en-tête du fichier lu sur GitHub | |
| Step1X-3D (`stepfun-ai/Step1X-3D`) | Image → mesh texturé | **Retiré des candidats** le 2026-11-06 | Apache-2.0 déclaré, mais `step1x3d_geometry/models/autoencoders/volume_decoders.py` : « Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT » | Apache-2.0 (déclaré) | Licences Hunyuan 3D (excluent l'UE) | **NON** (code Hunyuan ; la veille a lu le même en-tête dans `step1x3d_texture/differentiable_renderer/mesh_render.py`, plus `custom_rasterizer` Hunyuan et nvdiffrast) | 2026-11-06, en-tête de `volume_decoders.py` lu sur GitHub | |

### Conditions d'intégration de TRELLIS.2 (et de Pixal3D)

Constat de l'audit du code (2026-11-06) : sans correctifs, aucun chemin de génération ni de texturation ne
s'importe sans composant interdit. Avant toute intégration :

1. Ne pas installer nvdiffrast ni nvdiffrec (`setup.sh` sans `--nvdiffrast` ni `--nvdiffrec`).
2. Rendre l'import de `o_voxel.postprocess` paresseux (il importe `nvdiffrast` et `cumesh` dès
   `import o_voxel`) et réécrire la cuisson UV de `trellis2_texturing.py` sans nvdiffrast.
3. Retirer `rembg_model` de `pipeline.json` et `texturing_pipeline.json` (copie locale) et faire accepter
   `None` par `from_pretrained` : sinon RMBG-2.0 est téléchargé à chaque chargement. Fournir nos RGBA birefnet.
4. CuMesh : ne pas utiliser la version épinglée (voir sa ligne) ; la remplacer ou la reconstruire sur
   une version de cubvh validée.
5. Télécharger DINOv3 depuis `facebook/dinov3-vitl16-pretrain-lvd1689m` après acceptation de la licence
   par un humain, jamais depuis un miroir.
6. Ne pas appeler les rendus de visualisation (`trellis2/renderers/*`, `utils3d.torch.rasterization`),
   qui importent nvdiffrast ou nvdiffrec.

Composants importés par les chemins de génération et de texturation une fois ces points traités : code
TRELLIS.2 et o-voxel (MIT), FlexGEMM (MIT) et Triton (MIT), flash-attn (BSD-3), transformers (Apache-2.0),
OpenCV (Apache-2.0), trimesh (MIT), utils3d (MIT), easydict (LGPL-3.0) et plyfile (GPL-3.0). Aucun autre
composant non commercial n'a été trouvé (recherche de `import`, `from_pretrained`, `hf_hub_download`,
`torch.hub` et des URL dans `trellis2/` et `o-voxel/`, hors `trainers/` et `data_toolkit/`, non audités).

## Candidats au remplacement de Hunyuan3D (non intégrés)

TRELLIS.2, Pixal3D et Direct3D-S2 ont désormais leur ligne dans le tableau ci-dessus ; TripoSG et
Step1X-3D y sont passés à **NON**. Les candidats ci-dessous n'ont qu'une licence **déclarée** et restent
à auditer avant toute intégration.

| Candidat | Licence déclarée | Points d'attention |
|---|---|---|
| TRELLIS (`microsoft/TRELLIS-image-large`) | MIT | Export GLB : nvdiffrast et rasteriseur gaussien Inria (non commerciaux), rembg `u2net`, pymeshfix (d'après la veille). À VÉRIFIER. |
| Hi3DGen (`Stable-X/trellis-normal-v0-1`, `Stable-X/yoso-normal-v1-8-1`) | MIT, Apache-2.0 | Hors périmètre de l'audit du 2026-11-06. À VÉRIFIER : base Stable Diffusion des poids de normales. |
| MV-Adapter (`huanngzh/mv-adapter`) | Apache-2.0 | Hors périmètre. À VÉRIFIER : nvdiffrast partout, SDXL (OpenRAIL++-M), big-lama, RealESRGAN, pymeshlab. |
| SUPRA-Tex (`supra3d/supra-tex`) | Code MIT ; poids sans licence | Hors périmètre. À VÉRIFIER : Qwen-Image-2.1 (licence de recherche), nvdiffrast. |

## Bibliothèques

| Bibliothèque | Licence | Remarque |
|---|---|---|
| PyTorch, NumPy, SciPy | BSD | |
| OpenCV (≥ 4.5) | Apache-2.0 | |
| onnxruntime-gpu | MIT | |
| rembg | À VÉRIFIER | |
| pymatting | À VÉRIFIER | |
| trimesh | MIT (lu) | Voir la ligne détaillée dans la section suivante. |
| sentence-transformers, Streamlit, ChromaDB | À VÉRIFIER | |
| Blender | GPL-3.0 | Utilisé comme outil externe (sous-processus) pour le rendu. Ne pas l'embarquer ni lier son code au produit. |

## Bibliothèques et outils du pipeline de génération 3D

Audit du 2026-11-06 à partir des veilles `docs/veille/2026-11-06_*.md`. « Lu » : fichier de licence ou
en-tête lu ; « déclaré » : métadonnées PyPI, GitHub ou HuggingFace seulement. Rien de cette section n'est
installé, sauf mention contraire.

| Composant | Rôle | Statut | Licence | Commercial UE | Vérifié | Remarque | Validé par |
|---|---|---|---|---|---|---|---|
| nvdiffrast (NVlabs) | Rasterisation différentiable | **Exclu, y compris pour les tests internes** (décision de l'utilisateur, 2026-11-06) | Nvidia Source Code License (1-Way Commercial) | **NON** | 2026-11-06, `LICENSE.txt` lu | §3.3 : « The Work and any derivative works thereof only may be used or intended for use non-commercially […] "non-commercially" means for research or evaluation purposes only ». Importé par TRELLIS.2, o-voxel, Pixal3D, utils3d (`torch.rasterization`), les exemples Kaolin, FlexiCubes et Large Steps. Remplacé par le rasteriseur `cuda` de Kaolin. | |
| nvdiffrec (NVlabs, et le fork `JeffreyXiang/nvdiffrec` installé par TRELLIS.2) | Rendu PBR de visualisation | Interdit | NVIDIA Source Code License for nvdiffrec | **NON** | 2026-11-06, `LICENSE.txt` amont lu ; le fork n'a pas de licence propre (GitHub : NOASSERTION) | §3.3 : « The Work and any derivative works thereof only may be used or intended for use non-commercially. » Importé seulement par `trellis2/renderers/pbr_mesh_renderer.py`. | |
| CuMesh (`JeffreyXiang/CuMesh`, sous-module `cubvh` épinglé sur `JeffreyXiang/cubvh@ce92267a`) | Nettoyage, remaillage, BVH, dépliage UV (GPU) ; importé par TRELLIS.2 | Interdit tel qu'épinglé | MIT (lu) pour CuMesh ; mais le sous-module `cubvh` contient `LICENSE_NVIDIA` (NVIDIA Source Code License for instant neural graphics primitives) | **NON** (tel qu'épinglé) | 2026-11-06, `LICENSE`, `.gitmodules`, `setup.py`, `cumesh/__init__.py` de CuMesh ; `LICENSE_NVIDIA` et historique de `cubvh` | `setup.py` compile `third_party/cubvh/src/bvh.cu` dans l'extension `cumesh._cubvh`, et `cumesh/__init__.py` l'importe toujours (`from .bvh import cuBVH`). Le mainteneur amont de cubvh écrit (commit `191b0206`, 2026-08-26) : « src/bvh.cu, include/gpu/bvh.cuh, include/gpu/triangle.cuh, include/gpu/bounding_box.cuh and part of include/gpu/common.h borrowed code from NVlabs/instant-ngp » ; licence de ce code, §3.3 : « may be used or intended for use non-commercially ». Le commit épinglé (2025-12-30) précède la réécriture amont (`c7379c00`, 2026-08-26, annoncée comme indépendante et préparée avec un agent d'IA) : version réécrite **À VÉRIFIER** par un humain avant de l'adopter. Autres points : la simplification a pour « reference implementation » pamo (AGPL-3.0, lu), à vérifier ; xatlas embarqué en MIT (lu). | |
| FlexGEMM (`JeffreyXiang/FlexGEMM`) | Convolutions creuses (backend par défaut de TRELLIS.2) | Candidat | MIT | OUI | 2026-11-06, `LICENSE` lu, arborescence vérifiée | Dépend de Triton (MIT, lu), sans roue Windows officielle. | |
| o-voxel (dans `microsoft/TRELLIS.2`) | Conversion mesh ↔ O-Voxel, post-traitement | Candidat, inutilisable tel quel | MIT (licence du dépôt ; pas de fichier propre) ; sous-module Eigen (MPL-2.0) | OUI pour son code ; voir remarque | 2026-11-06, code lu | `o_voxel/__init__.py` importe `postprocess`, qui importe `nvdiffrast` (NON) et `cumesh` (NON tel qu'épinglé). Dépend de plyfile (GPL-3.0). | |
| utils3d (`EasternJournalist/utils3d@9a4eb15e`) | Utilitaires 3D (TRELLIS.2, Direct3D-S2) | Candidat | MIT | OUI | 2026-11-06, `LICENSE` et `pyproject.toml` lus au commit épinglé | Ne pas appeler `utils3d.torch.rasterization` (importe nvdiffrast, chargé à la demande). Dépend de moderngl (MIT) et plyfile (GPL-3.0). | |
| flash-attn | Attention (TRELLIS.2) | Candidat | BSD-3-Clause | OUI | 2026-11-06, `LICENSE` lu | Alternatives prévues par le code : xformers (BSD, déclaré) ou `sdpa` de PyTorch. | |
| plyfile | Lecture et écriture PLY (o-voxel, utils3d, `trellis2/utils/mesh_utils.py`) | Dépendance des candidats | GPL-3.0 ou ultérieure | À VÉRIFIER | 2026-11-06, `COPYING` lu | Copyleft fort importé dans le même processus. Décision de principe (comme pour pymeshlab) ; remplaçable par l'écriture PLY de trimesh. | |
| easydict | Dictionnaires (échantillonneur de TRELLIS.2) | Dépendance des candidats | LGPL-3.0 | À VÉRIFIER | 2026-11-06, `LICENSE` lu | Copyleft faible ; trivial à remplacer par une classe maison. | |
| Kaolin 0.18.0, roue `kaolin-0.18.0-cp312-cp312-win_amd64.whl` (index `torch-2.6.0_cu124`) | FlexiCubes, rasteriseur `cuda`, cuisson des textures | Candidat (étapes optimisation par l'image et texture) | Apache-2.0 (`dist-info/LICENSE`), **et** `dist-info/LICENSE.NSCL` (NVIDIA Source Code License for Kaolin, §3.3 non commercial) | À VÉRIFIER | 2026-11-06, roue téléchargée et ouverte (sans exécution) : `METADATA`, `LICENSE`, `LICENSE.NSCL`, sources Python ; en-têtes C++/CUDA du tag `v0.18.0` | `kaolin.ops.conversions.FlexiCubes` est bien la version Apache dans la roue (« Copyright (c) 2025 NVIDIA […] Licensed under the Apache License, Version 2.0 »). Mais `kaolin/__init__.py` exécute `from . import non_commercial` à chaque `import kaolin`, et `kaolin/non_commercial/flexicubes/*.py` porte « Any use, reproduction, disclosure or distribution of this software […] without an express license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited » : la roue livre et charge ce code. Deux en-têtes compilés dans `_C.pyd` ne sont pas Apache : `csrc/ops/spc/point_utils.h` (même mention « strictly prohibited ») et `csrc/ops/conversions/unbatched_mcube/*.h` (EULA NVIDIA). Rasteriseur : `rasterize(..., backend='cuda')` par défaut ; `easy_render` choisit nvdiffrast s'il est installé (il ne le sera pas). Remèdes à faire valider : supprimer `kaolin/non_commercial` et la ligne d'import, ou obtenir l'avis d'un juriste. | |
| `kaolin.non_commercial` (ancienne FlexiCubes) | — | Interdit | NVIDIA, propriétaire | **NON** | 2026-11-06, en-tête lu dans la roue | Ne jamais importer `kaolin.non_commercial` ni l'appeler. | |
| warp-lang | Dépendance de Kaolin | Dépendance | Apache-2.0 depuis la version 1.6.2 (2025-03-07) ; avant : « NVIDIA Software License » | OUI (version ≥ 1.6.2) | 2026-11-06, `LICENSE.md` et `CHANGELOG.md` lus | Épingler `warp-lang>=1.6.2`. | |
| usd-core | Dépendance de Kaolin | Dépendance | Tomorrow Open Source Technology License 1.0 (Apache-2.0 avec une clause de marques différente) | OUI | 2026-11-06, `LICENSE.txt` d'OpenUSD lu | | |
| pygltflib | Dépendance de Kaolin | Dépendance | MIT | OUI | 2026-11-06, `LICENSE` lu | La veille citait aussi plyfile comme dépendance de Kaolin : ce n'est pas le cas (absent de `METADATA`). | |
| largesteps | Optimisation de sommets (Large Steps) | Candidat | BSD-3-Clause (variante avec clause « Enhancements ») | OUI | 2026-11-06, `LICENSE` lu | | |
| cholespy | Solveur de Cholesky (largesteps) | Candidat | BSD-3-Clause | OUI | 2026-11-06, `LICENSE` lu | | |
| continuous-remeshing (Profactor) | Remaillage pendant l'optimisation | Candidat | MIT | OUI | 2026-11-06, `LICENSE` lu | nvdiffrast seulement dans l'exemple (veille) : ne pas le reprendre. | |
| torch_scatter | Dépendance de continuous-remeshing | Candidat | MIT | OUI | 2026-11-06, `LICENSE` lu | Roue `2.1.2+pt26cu124` cp312 Windows publiée sur `data.pyg.org`. | |
| Mitsuba 3, Dr.Jit | Rendu différentiable de référence | Candidat (repli) | BSD-3-Clause (variante « Enhancements ») | OUI | 2026-11-06, `LICENSE` des deux dépôts lus | Bibliothèques embarquées (Embree, OpenEXR, libpng, libjpeg-turbo, zlib, pugixml, nanobind…) connues comme permissives, licences non relues une à une. | |
| trimesh | Nettoyage et diagnostic | Utilisé (4.8.1 dans `.venv`) | MIT | OUI | 2026-11-06, `dist-info/licenses/LICENSE.md` lu | Déjà dans `config/requirements/base.txt`. | |
| open3d | Nettoyage, décimation | Installé (0.19.0), non utilisé | MIT | OUI | 2026-11-06, `dist-info/LICENSE.txt` lu ; `3rdparty/README.md` du tag `v0.19.0` | Bibliothèques liées permissives d'après ce README (libzmq en LGPL-3.0 avec exception de liaison statique). | |
| xatlas (bibliothèque C++ `jpcy/xatlas` et bindings `xatlas` 0.0.11 de `mworchel/xatlas-python`) | Dépliage UV | Installé, non utilisé | MIT (les deux) | OUI | 2026-11-06, `LICENSE` du dépôt C++ et `dist-info/licenses/LICENSE` locale lus | | |
| fast-simplification (pyvista) | Décimation | Candidat | MIT | OUI | 2026-11-06, `LICENSE` lu | | |
| meshoptimizer (bibliothèque C++ `zeux/meshoptimizer`) | Simplification, LOD | Candidat | MIT | OUI | 2026-11-06, `LICENSE.md` lu | | |
| meshoptimizer (binding Python PyPI 0.2.30a0) | Idem en Python | Candidat | MIT (classifieur PyPI seulement) | À VÉRIFIER | 2026-11-06, sdist ouverte : ni fichier LICENSE ni auteur | Le README renvoie à un dossier `python/` absent du dépôt `zeux/meshoptimizer` : origine du binding inconnue. Préférer un binding dont la provenance est établie. | |
| manifold3d | Booléens, validation manifold | Candidat | Apache-2.0 | OUI | 2026-11-06, `LICENSE` lu | | |
| mesh2sdf | Étanchéité par SDF | Candidat | MIT | OUI | 2026-11-06, fichier `LISCENCE` lu | Pas de roue Windows cp312 sur PyPI (compilation). | |
| Instant Meshes (binaire Windows officiel, `instant-meshes-windows.zip`, 2019-11-03) | Retopologie quads (sous-processus) | Candidat | BSD-3-Clause (variante « Enhancements ») | OUI | 2026-11-06, `LICENSE.txt` lu ; sous-modules : nanogui (BSD), pcg32 (Apache-2.0), dset (zlib), tbb (Apache-2.0), pss (BSD-3) lus | Binaire non reproductible : noter son empreinte au téléchargement. | |
| QuadriFlow tel qu'embarqué dans Blender 4.5 (`extern/quadriflow`) | Retopologie quads via Blender (sous-processus) | Candidat | BSD-3 (`LICENSE.txt`) ; `README.blender` indique « SPDX:MIT » | OUI | 2026-11-06, `LICENSE.txt` et `README.blender` de la branche `blender-v4.5-release` lus | Utilisé dans un Blender externe (GPL-3.0), déjà au registre. Embarque lemon 1.3.1 (Boost), pcg32, pss. | |
| pymeshlab | Traitement de mesh | **Installé dans `.venv` sans ligne au registre** (2025.7.post1, installé explicitement : marqueur `REQUESTED`, aucun paquet ne le requiert) ; importé nulle part dans `fantasy_rag`, `tools`, `tests` ni `config` | GPL-3.0 | À VÉRIFIER | 2026-11-06, `dist-info/licenses/LICENSE` lu, `pip show` | Copyleft fort : ne jamais l'importer dans `fantasy_rag`. Décision humaine : le désinstaller, ou l'admettre comme outil externe en sous-processus comme Blender. | |
| pymeshfix | Bouchage de trous (Direct3D-S2, TRELLIS v1) | Candidat écarté du produit | Fichier `LICENSE` en GPL-3.0, classifieur PyPI 0.18.1 en AGPL-3.0 | À VÉRIFIER | 2026-11-06, `LICENSE` et `pyproject.toml` lus | Incohérence non levée : la traiter comme AGPL-3.0 (copyleft réseau, donc aussi en SaaS). Ne jamais l'importer dans le produit ; sous-processus seulement après décision humaine. | |

## Données

| Source | Usage autorisé | Remarque |
|---|---|---|
| Images de test dans `data/` (concept arts trouvés en ligne) | Évaluation interne uniquement | Œuvres d'artistes tiers, certaines signées. Interdites pour l'entraînement, les démos publiques ou le marketing sans droits écrits. |
| Futurs jeux de données de fine-tuning | À définir | Chaque échantillon doit avoir sa provenance et sa licence (voir le skill `donnees-provenance`). Objaverse et similaires mélangent des licences par objet, dont des licences non commerciales. |
