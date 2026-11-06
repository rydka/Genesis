# Veille : étape 3, optimiseur de texture (2026-11-06)

Agent : Veille modèles. Périmètre : étape 3 du pipeline envisagé (générateur de mesh, optimiseur de mesh,
**optimiseur de texture**, optimiseur local par rapprochement à l'image). Entrée : un mesh nettoyé
(avec ou sans UV) et l'image de concept d'origine, une seule vue, détourée. Sortie attendue : texture
fidèle à l'image sur les faces visibles, faces cachées complétées de façon cohérente, PBR si possible,
coutures propres.

Rien n'a été installé ni exécuté. Tout ce qui suit est à faire auditer par `/gardien-licences` avant
toute intégration.

## Niveau de preuve

- **Lu (fichier)** : fichier LICENSE ou code source lu sur le dépôt (via GitHub raw ou `gh api`) le
  2026-11-06.
- **Déclaré** : licence affichée par l'API GitHub (champ `license`), par l'étiquette HuggingFace ou par un
  README, sans lecture du fichier.
- **Secondaire** : article de blog ou résumé tiers. À confirmer.

## Dépendances transverses à risque

Ce sont elles qui éliminent ou bloquent la plupart des pipelines de texture. À vérifier dans chaque
candidat, y compris ce que les scripts téléchargent.

| Dépendance | Licence | Niveau de preuve | Conséquence |
|---|---|---|---|
| **nvdiffrast** (NVlabs) | NVIDIA Source Code License (1-Way Commercial), section 3.3 : usage « non-commercially » seulement, défini comme « research or evaluation purposes only » | **Lu** (`LICENSE.txt`) | **Interdit pour le produit.** Présent dans TRELLIS.2, MV-Adapter, SUPRA-Tex, UniTEX. Doit être remplacé (PyTorch3D, Kaolin, rasteriseur maison, bake Blender). L'usage en banc d'évaluation interne relève du gardien. |
| **nvdiffrec** (NVlabs, et le fork `JeffreyXiang/nvdiffrec` installé par TRELLIS.2) | NVIDIA Source Code License for nvdiffrec, non commercial | **Lu** (`LICENSE.txt` amont) ; fork : non lu (GitHub : NOASSERTION) | Interdit. Le pipeline de texturation TRELLIS.2 ne l'importe pas (lu dans le code), seul le moteur de rendu PBR de démo l'importe. |
| `custom_rasterizer` et `differentiable_renderer` de Hunyuan3D-2 | Tencent Hunyuan 3D 2.0 Community License : « Territory » exclut l'UE, le Royaume-Uni et la Corée du Sud | **Lu** (`LICENSE` de Hunyuan3D-2) ; l'appartenance de ces modules à cette licence est une déduction (ils viennent de ce dépôt) | Élimine Step1X-3D Texture, MVPainter, MaterialMVP en l'état. |
| **RMBG-2.0** (`briaai/RMBG-2.0`) | `bria-rmbg-2.0`, usage commercial soumis à accord avec Bria | Déclaré (étiquette HF) | Détoureur par défaut de TRELLIS.2 (texturation), SUPRA-Tex, UniTEX. À contourner : fournir notre RGBA (birefnet). |
| **FLUX.1-dev** | FLUX.1 [dev] Non-Commercial License | Connu, non relu aujourd'hui | Élimine UniTEX (lu dans `pipeline.py`). FLUX.1-schnell est apache-2.0 (déclaré HF). |
| **Qwen-Image-2.1** | `qwen-research` (Qwen Research License, non commercial selon des sources secondaires) | Déclaré (étiquette HF) + secondaire | Élimine le frontal matériaux de SUPRA-Tex et l'option Qwen de StableGen. Les Qwen-Image antérieurs seraient apache-2.0 (secondaire). |
| **DINOv3** (`facebook/dinov3-vitl16-pretrain-lvd1689m`) | DINOv3 License (Meta) : licence mondiale, sans restriction d'usage commercial ; mention « Built with DINOv3 » obligatoire ; interdits liés aux contrôles à l'export (ITAR, usages militaires) ; accès HF validé manuellement | **Lu** (page de licence Meta) | Acceptable a priori, à valider. Requis par TRELLIS.2 (texturation) et SUPRA-Tex. |
| Stable Diffusion XL base 1.0 | CreativeML OpenRAIL++-M | Déclaré (étiquette HF) | Commercial possible, avec restrictions d'usage (annexe A) à répercuter aux utilisateurs. Base de MV-Adapter, StableGen. |
| Stable Diffusion 1.5 | CreativeML OpenRAIL-M | Déclaré (étiquette HF) | Idem. Base de SyncMVD, Paint3D. |
| Stable Diffusion 2.1 base | OpenRAIL++ (connu) | **Le dépôt HF `stabilityai/stable-diffusion-2-1-base` répond 401** le 2026-11-06 | Variante basse VRAM de MV-Adapter : source des poids à retrouver et à auditer. |
| big-lama (LaMa, inpainting) | Code Apache-2.0 (déclaré GitHub) | Poids non audités | Les poids sont entraînés sur Places2, dont les conditions sont restrictives : à vérifier. Utilisé par MV-Adapter. |
| RealESRGAN_x2plus (super-résolution) | Code BSD-3-Clause (déclaré GitHub) | Poids non audités | Données d'entraînement (DIV2K, Flickr2K, OST) à conditions académiques : à vérifier. Utilisé par MV-Adapter. |
| pymeshlab | GPL-3.0 (connu) | Non relu | Importé par MV-Adapter. GPL : à isoler (sous-processus) ou remplacer. |
| ComfyUI | GPL-3.0 | Déclaré (GitHub) | Backend de StableGen. Utilisable comme outil externe, comme Blender. |

Alternatives propres pour la rasterisation et le bake : PyTorch3D (BSD, **lu** en-tête de `LICENSE`),
Kaolin (Apache-2.0, déclaré), xatlas et xatlas-python (MIT, déclaré), CuMesh et FlexGEMM (MIT, déclaré),
OpenCV `inpaint` (Apache-2.0, déjà au registre), Blender 4.5 (GPL-3.0, déjà au registre comme outil
externe). Mitsuba 3 : licence non déterminée par l'API GitHub, à vérifier si on y pense.

## Éliminés d'emblée

| Candidat | Motif | Preuve |
|---|---|---|
| Hunyuan3D-Paint 2.0 / 2.1, RomanTex, MaterialMVP | Licences Tencent Hunyuan excluant l'UE, ou code `custom_rasterizer` / `DifferentiableRenderer` de Hunyuan3D-2 dans le dépôt (MaterialMVP) | Licence Hunyuan3D-2 lue ; arborescence MaterialMVP lue |
| NaTex (Tencent Hunyuan, CVPR 2026) | Aucun code ni licence trouvés ; origine Tencent Hunyuan | Recherche web |
| Step1X-3D Texture | Code Apache-2.0, mais le bake installe `custom_rasterizer` et `differentiable_renderer` repris de Hunyuan3D 2.0 (README). Environ 27 Go pour forme + texture. Récupérable seulement en remplaçant ces deux modules. | README lu ; licence Hunyuan3D-2 lue |
| MVPainter (amap-cvlab) | Apache-2.0 (**lu**), poids apache-2.0 (déclaré HF), mais vendorise le `custom_rasterizer` Hunyuan ; 40 Go de VRAM minimum ; Linux | `LICENSE`, arborescence et `INSTALL.md` lus |
| UniTEX (CVPR 2026) | Code Apache-2.0 (déclaré), mais `pipeline.py` charge `black-forest-labs/FLUX.1-dev` et `briaai/RMBG-2.0` ; nvdiffrast | `pipeline.py` lu |
| TEXGen (CVMI-Lab) | Aucun fichier de licence (tous droits réservés par défaut) ; 24 Go en inférence ; exige des UV | GitHub API et README |
| MeshGen (heheyas) | Aucun fichier de licence | GitHub API |
| Bria FIBO 2.0, Qwen-Image-2.1 (options de StableGen) | CC BY-NC 4.0 et licence de recherche | README StableGen, étiquette HF |

## Fiches des candidats retenus

### 1. TRELLIS.2, pipeline de texturation (`Trellis2TexturingPipeline`)

- **Licence** : code MIT (**lu**, `LICENSE`, Copyright Microsoft) ; poids `microsoft/TRELLIS.2-4B` MIT
  (déclaré HF). Déjà listé comme candidat générateur au registre, mais **la texturation est une ligne
  distincte** à auditer (autres checkpoints : `slat_flow_imgshape2tex_dit_1_3B_512/1024`,
  `shape_enc_next`, `tex_dec_next`).
- **Dépendances repérées (lues dans `trellis2_texturing.py`, `texturing_pipeline.json`, `setup.sh`)** :
  - `nvdiffrast` : importé et utilisé **uniquement** dans `postprocess_mesh` pour rasteriser l'espace UV
    lors du bake. Interdit : à remplacer par un rasteriseur UV maison ou PyTorch3D. Attention, `o_voxel`
    (`o-voxel/o_voxel/postprocess.py`) importe aussi nvdiffrast : l'import doit être rendu optionnel.
  - `briaai/RMBG-2.0` comme détoureur par défaut : le code ne l'appelle que si l'image n'a pas d'alpha
    utile ; avec notre RGBA birefnet, il n'est pas appelé, mais il faut le retirer de la configuration
    pour qu'il ne soit pas téléchargé.
  - DINOv3 ViT-L/16 comme encodeur d'image (licence Meta, voir plus haut).
  - CuMesh (MIT, déclaré) pour le dépliage UV si le mesh n'en a pas ; FlexGEMM (MIT, déclaré) ;
    o-voxel (dans le dépôt, MIT) ; OpenCV `inpaint` (Telea) pour combler les trous de l'atlas.
  - Pas de modèle de diffusion d'image tiers (pas de SD, ni de Flux).
- **Entrées et sorties** : un mesh (trimesh) et une image ; sortie GLB avec base color, metallic,
  roughness et alpha (PBR natif). Les faces cachées sont générées en 3D natif (latent de texture
  conditionné par la forme), donc cohérentes sans multi-vues.
- **VRAM et temps** : officiel « au moins 24 Go », testé sur A100/H100. Rapports communautaires
  (secondaires) : 16 Go suffisent en 512, voire en 1024 avec des textures de 2K. **À mesurer** sur la
  RTX 2000 Ada. Windows non pris en charge officiellement (extensions CUDA à compiler ; WSL rapporté
  comme fonctionnel).
- **Réponse aux défauts connus** : bonne cohérence globale et dos plausibles. La texture passe par une
  grille voxel (512 ou 1024), donc le détail fin (petit visage d'environ 40 px, motifs) risque d'être
  lissé : prévoir une passe de projection directe de l'image sur les faces visibles (étape 4 ou
  candidat 3). Le canal alpha peut aider pour les effets semi-transparents, sans garantie.
- **Maturité** : article arXiv 2512.14692 (décembre 2025), dépôt Microsoft très actif, exemple
  `example_texturing.py` et démo `app_texturing.py`.

### 2. MV-Adapter, texture depuis une image (`scripts/texture_i2tex.py`)

- **Licence** : code Apache-2.0 (déclaré GitHub, fichier non relu) ; poids `huanngzh/mv-adapter`
  apache-2.0 (déclaré HF).
- **Dépendances repérées (lues dans `texture_i2tex.py` et `requirements.txt`)** :
  - Base `stabilityai/stable-diffusion-xl-base-1.0` (OpenRAIL++-M, déclaré) et VAE
    `madebyollin/sdxl-vae-fp16-fix` (MIT, déclaré). Variante `sd21` : base SD 2.1 inaccessible sur HF.
  - `nvdiffrast` partout dans `mvadapter/utils/mesh_utils` (rendu des cartes de normales et de position,
    projection, UV) sans alternative : remplacement plus lourd que pour TRELLIS.2, mais circonscrit à
    une classe `NVDiffRastContextWrapper`.
  - `RealESRGAN_x2plus.pth` et `big-lama.pt` (poids à auditer, voir plus haut) ; `ZhengPeng7/BiRefNet`
    seulement avec `--remove_bg` (MIT, déclaré) ; `pymeshlab` (GPL) ; CV-CUDA (licence non déterminée).
- **Entrées et sorties** : image et mesh, génération de 6 vues conditionnées par la géométrie, puis
  projection, inpainting et bake vers un GLB. **Albedo seulement, pas de PBR** : à compléter
  (Material Anything ou Marigold-IID, voir plus bas).
- **VRAM** : plus de 16 Go recommandés pour la version SDXL conditionnée par la géométrie ; moins de
  10 Go en SD 2.1. Sur 16 Go, SDXL est limite : offload ou GPU cloud à prévoir.
- **Réponse aux défauts** : la vue de face reste proche de l'image (projection), mais les vues générées
  par SDXL peuvent dériver de style sur les dos et les côtés ; coutures traitées par fusion de Poisson
  et inpainting.
- **Maturité** : ICCV 2025, pipeline complet publié en mai 2025, dernière activité juin 2025 ; dépôt
  populaire, bien documenté, nœuds ComfyUI.

### 3. Projection classique de l'image et bake (référence sans risque)

- **Composants** : Blender 4.5 en sous-processus (projection caméra de l'image de concept sur les faces
  visibles, bake vers l'atlas), xatlas (MIT, déclaré) pour les UV si besoin, OpenCV ou PyTorch3D (BSD)
  pour les masques de visibilité et la dilatation des coutures. Aucun poids.
- **Rôle** : donne la fidélité maximale sur la face vue, y compris le visage, et sert de témoin pour
  juger les candidats 1 et 2. Ne résout pas les faces cachées : à combiner (par exemple, faces visibles
  par projection, reste par TRELLIS.2).
- **Faisabilité** : Windows natif, CPU ou GPU léger, s'appuie sur `tools/debug/render_glb_views.py`.
- **Licence** : Blender déjà au registre (GPL-3.0, outil externe) ; xatlas à ajouter.

### À surveiller : SUPRA-Tex (Huawei, octobre 2026)

- Code MIT (**lu**, `LICENSE` et `THIRD_PARTY_NOTICES.md`) ; reprend du code TRELLIS.2 (MIT) et un
  code Tencent sous MIT. Sorti le 2026-10-19, très actif.
- Fait exactement le besoin (texturer un mesh existant depuis une image, PBR, faces cachées complétées,
  option de vue arrière générée) et vise la fidélité aux détails de l'image.
- **Bloquants** : le frontal matériaux est un adaptateur sur **Qwen-Image-2.1** (licence de recherche) ;
  la fiche HF `supra3d/supra-tex` n'a **aucune licence déclarée** pour les poids ; nvdiffrast ;
  RMBG-2.0 optionnel ; Linux uniquement ; testé sur 24 Go avec offload. Le modèle de complétion
  (2,1 B) accepte des cartes albedo/MRA fournies par nous, ce qui contournerait Qwen, mais seulement
  si la licence des poids de complétion est clarifiée et qu'ils ne dérivent pas d'un modèle interdit.

### Composants PBR et alternatives propres moins prioritaires

| Candidat | Licence | Preuve | Intérêt et limites |
|---|---|---|---|
| Material Anything (3DTopia) | Code MIT ; poids `material_estimator` et `material_refiner` apache-2.0 | Déclaré (GitHub, HF) | Ajoute albedo, rugosité, métal et bump à un mesh déjà texturé (albedo seul), en complément de MV-Adapter ou de la projection. PyTorch3D et Blender, pas de nvdiffrast dans `requirements.txt`. Modèle de base des poids non identifié. Linux, Blender 3.2. |
| Marigold-IID appearance et lighting v1-1 (ETH) | Code Apache-2.0 ; poids OpenRAIL++ | Déclaré | Décomposition intrinsèque (albedo, rugosité, métal) des vues, pour retirer l'éclairage peint dans le concept art. |
| StableDelight (`Stable-X/yoso-delight-v0-4-base`) | Poids apache-2.0 ; dépôt sans licence déclarée | Déclaré (HF) | Retrait des reflets avant projection. Licence du code à vérifier. |
| SyncMVD | Code MIT ; SD 1.5 + ControlNet v1.1 (OpenRAIL) ; PyTorch3D ; xatlas | Déclaré | Chaîne entièrement propre, mais conditionné par le texte : il faudrait ajouter IP-Adapter (`h94/IP-Adapter`, apache-2.0 déclaré) pour suivre l'image. 2023, peu actif. |
| Paint3D | Code Apache-2.0 ; SD 1.5, ControlNet ; Kaolin (Apache-2.0) | Déclaré | Image ou texte vers texture UV, avec inpainting en espace UV. Inactif depuis novembre 2024. |
| StableGen (extension Blender) | GPL-3.0 ; backend ComfyUI GPL-3.0 ; SDXL + ControlNet + IP-Adapter ; PBR via Marigold et StableDelight | Déclaré (GitHub) | Windows natif, environ 8 Go en SDXL, projection et bake dans Blender. Bon outil de prototypage. Exclure ses options Qwen-Image-2.1 et FIBO ; vérifier les checkpoints téléchargés. |

## Tableau comparatif

| Critère | TRELLIS.2 texturation | MV-Adapter i2tex | Projection Blender | SUPRA-Tex |
|---|---|---|---|---|
| Licence code / poids | MIT lu / MIT déclaré | Apache-2.0 déclaré / apache-2.0 déclaré | GPL (outil) + MIT / aucun poids | MIT lu / **non déclarée** |
| Bloquants à lever | nvdiffrast (bake UV), retirer RMBG-2.0 | nvdiffrast (partout), poids LaMa et ESRGAN, pymeshlab | Aucun | Qwen-Image-2.1, licence des poids, nvdiffrast |
| Modèle de base | Aucun (DINOv3 en encodeur) | SDXL (OpenRAIL++-M) | Aucun | Qwen-Image-2.1, DINOv3, TRELLIS.2 |
| Faces cachées | Oui, natif 3D | Oui, 6 vues générées | Non | Oui, natif 3D |
| PBR | Oui (base color, métal, rugosité, alpha) | Non (albedo) | Non | Oui |
| Fidélité à l'image (face vue) | Moyenne (grille voxel) | Bonne | Maximale | Très bonne (annoncée) |
| VRAM / plateforme | 24 Go officiel, 16 Go probable en 512, à mesurer ; Linux | Plus de 16 Go en SDXL ; Linux | Faible ; Windows | 24 Go avec offload ; Linux |
| Maturité | Décembre 2025, très actif | Mai 2025, stable | Outils mûrs | Octobre 2026, très récent |

## Classement motivé

1. **TRELLIS.2 texturation.** Seul candidat qui coche en même temps image et mesh en entrée, PBR natif,
   faces cachées cohérentes, licence MIT lue, et aucun modèle de diffusion tiers. Le seul bloquant
   technique (nvdiffrast) se limite à la rasterisation UV du bake, remplaçable sans toucher au modèle.
   Point faible attendu : le détail fin, d'où l'intérêt de le coupler à la projection directe.
2. **MV-Adapter i2tex.** Meilleure fidélité sur la face vue grâce à la projection multi-vues, chaîne
   de licences permissive en apparence (Apache, OpenRAIL++-M), communauté large. Mais nvdiffrast est
   omniprésent, il n'y a pas de PBR, plusieurs poids auxiliaires sont à auditer et la VRAM est limite.
3. **Projection classique dans Blender.** Sans risque de licence, Windows natif, fidélité maximale sur
   les faces visibles. Sert de témoin et de composant de fusion avec le candidat 1 ; ne traite pas les
   faces cachées.

SUPRA-Tex serait premier sur le papier, mais reste bloqué tant que la licence de ses poids n'est pas
déclarée et que son frontal dépend de Qwen-Image-2.1.

## Pour le gardien des licences

À auditer avant toute intégration, par ordre de priorité :

1. TRELLIS.2 texturation : checkpoints de texturation de `microsoft/TRELLIS.2-4B` (fiche et fichier de
   licence HF), DINOv3 License (attribution, contrôle à l'export, acceptation HF), CuMesh, FlexGEMM,
   o-voxel, utils3d ; confirmer qu'aucun autre composant non commercial n'est importé une fois
   nvdiffrast retiré.
2. MV-Adapter : fichier `LICENSE` du dépôt, poids `huanngzh/mv-adapter`, SDXL base 1.0 (texte
   OpenRAIL++-M et annexe A), `madebyollin/sdxl-vae-fp16-fix`, `big-lama.pt` (Places2),
   `RealESRGAN_x2plus.pth`, CV-CUDA, pymeshlab (GPL).
3. Outils de bake et de rasterisation : xatlas / xatlas-python, PyTorch3D, Kaolin.
4. PBR complémentaire : poids Material Anything, Marigold-IID (OpenRAIL++), StableDelight.
5. SUPRA-Tex : licence des poids de complétion et du matériel adapter, provenance (dérivés de
   Qwen-Image-2.1 ?).
6. À trancher : peut-on installer nvdiffrast pour une évaluation interne seulement (sa licence autorise
   « research or evaluation purposes »), en attendant le remplacement ?

## Sources

- TRELLIS.2 : https://github.com/microsoft/TRELLIS.2 (`LICENSE`, `setup.sh`, `example_texturing.py`,
  `trellis2/pipelines/trellis2_texturing.py`), https://huggingface.co/microsoft/TRELLIS.2-4B
  (`texturing_pipeline.json`)
- nvdiffrast : https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt ; nvdiffrec :
  https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt
- DINOv3 License : https://ai.meta.com/resources/models-and-libraries/dinov3-license/
- Hunyuan3D-2 : https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE
- Step1X-3D : https://github.com/stepfun-ai/Step1X-3D
- MV-Adapter : https://github.com/huanngzh/MV-Adapter (`scripts/texture_i2tex.py`, `requirements.txt`,
  `mvadapter/utils/mesh_utils/render.py`)
- MVPainter : https://github.com/amap-cvlab/MV-Painter
- UniTEX : https://github.com/YixunLiang/UniTEX (`pipeline.py`)
- TEXGen : https://github.com/CVMI-Lab/TEXGen ; NaTex : https://arxiv.org/abs/2511.16317
- SUPRA-Tex : https://github.com/Supra-3D/supra-tex (`LICENSE`, `THIRD_PARTY_NOTICES.md`,
  `deployment/docs/DEPENDENCIES.md`, `docs/texturing.md`), https://supra-3d.github.io/supratex/
- Qwen-Image-2.1 (secondaire) :
  https://runtimewire.com/article/alibaba-qwen-image-2-1-transparent-editing-research-license
- StableGen : https://github.com/sakalond/StableGen ; Material Anything :
  https://github.com/3DTopia/MaterialAnything ; SyncMVD : https://github.com/LIU-Yuxin/SyncMVD ;
  Paint3D : https://github.com/OpenTexture/Paint3D
- Hi3D V1.0 (géométrie seule, hors périmètre) : https://huggingface.co/hi3d-official/Hi3Dv1.0
- VRAM TRELLIS.2 (secondaire) : https://reddit.sentinel-team.org/posts/1pqxkag/snapshots/2025-12-20T08%3A31%3A16.315468Z
