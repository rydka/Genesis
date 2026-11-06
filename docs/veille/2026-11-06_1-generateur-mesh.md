# Veille : générateur image → mesh 3D (étape 1 du pipeline)

Date : 2026-11-06. Agent : Veille modèles. Rien n'a été installé ni téléchargé (hors lecture de fichiers
texte sur GitHub et HuggingFace).

Pipeline visé : **générateur de mesh** → optimiseur de mesh → optimiseur de texture → optimiseur local
par rapprochement à l'image. Ce document ne couvre que l'étape 1 (remplacer Hunyuan3D-2.1, interdit dans
l'UE) et note pour chaque candidat ce qu'il produit, puisque les étapes 2 à 4 en dépendent.

Besoins (voir `docs/experiments.md`) : une image de concept art fantasy détourée par notre birefnet ;
défauts connus à surveiller : accessoires fins perdus, petites têtes peu détaillées, sujets coupés par le
cadre (plaques), effets semi-transparents. Machine : RTX 2000 Ada 16 Go sous Windows.

**Ce document est une analyse technique, pas un avis juridique.** Rien n'entre dans le projet avant
l'audit de `/gardien-licences` et la validation d'un humain dans `LICENSES.md`.

## Niveau de preuve

- **Vérifié (fichier lu)** : j'ai lu le fichier LICENSE du dépôt ou de la fiche HF (via l'API GitHub ou le
  fichier brut HF) le 2026-11-06, et/ou le code qui télécharge les dépendances.
- **Déclaré** : seulement l'étiquette HF, le badge du README ou un article tiers.
- Les chiffres de VRAM et de temps sont ceux **annoncés** par les auteurs, sauf mention contraire. Aucun
  n'a été mesuré sur notre machine.

## Tableau comparatif (candidats retenus)

| Candidat | Date / activité | Licence code | Licence poids | Dépendances à risque téléchargées par les scripts officiels | VRAM annoncée / 16 Go ? | Sortie |
|---|---|---|---|---|---|---|
| **TRELLIS.2-4B** (Microsoft) | déc. 2025, dernier push juil. 2026, ~12,7 k étoiles | MIT (vérifié, `LICENSE`) | MIT (déclaré HF) | **RMBG-2.0** (BRIA, non commercial) par défaut ; **DINOv3** (licence Meta propre) ; **nvdiffrast** (NVIDIA, non commercial) dans l'export GLB ; nvdiffrec (NVIDIA, non commercial) pour le rendu | 24 Go minimum (README) ; 16 Go en 1024 plausible avec des forks communautaires, non vérifié | Mesh triangulaire haute densité + attributs PBR (base color, roughness, metallic, **opacité**), surfaces ouvertes ; GLB avec UV et textures après `to_glb` |
| **Pixal3D** (Tencent ARC, Tsinghua) | SIGGRAPH 2026, mai 2026, multi-vues sept. 2026, ~2,6 k étoiles | MIT (vérifié, `LICENSE` + `NOTICE`) **depuis le 2026-05-20 seulement** | MIT (vérifié, `LICENSE` HF) mais fiche HF avec `extra_gated_eu_disallowed: true` | Les mêmes que TRELLIS.2 (RMBG-2.0 dans `pipeline.json`, nvdiffrast via TRELLIS.2) + DINOv3 via un **miroir non officiel** (`camenduru/...`) + MoGe-2 (MIT) + NAF (Apache-2.0) | Non chiffrée ; mode `--low_vram` (1024 par défaut), le code indique « ~18 GB » sans ce mode | Comme TRELLIS.2 (même représentation), 1536 par défaut ; entrée image unique **ou multi-vues avec caméras** |
| **Direct3D-S2** (DreamTech) | mai 2025, dernier push sept. 2025, v1.2 « personnages » annoncée, non publiée | MIT (vérifié, `LICENSE.txt`) | MIT (déclaré HF) | BiRefNet `ZhengPeng7/BiRefNet` (MIT déclaré, celui qu'on utilise déjà) ; DINOv2 (Apache-2.0) ; torchsparse (MIT) ; **pymeshfix (GPL-3.0)** pour boucher les trous | 10 Go en 512, ~24 Go en 1024 (README) : **1024 = GPU cloud** | Géométrie seule (SDF → marching cubes), pas d'UV ni de texture |
| **Hi3DGen / Stable3DGen** (Stable-X) | avr. 2025, dernier push juil. 2025 | MIT (déclaré README, qui dit avoir retiré kaolin, nvdiffrast et flexicubes pour l'usage commercial) | MIT (`trellis-normal-v0-1`, déclaré HF) | StableNormal / `yoso-normal-v1-8-1` (Apache-2.0 déclaré ; base Stable Diffusion à vérifier) ; BiRefNet | Non chiffrée (base TRELLIS v1, ~16 Go) | Géométrie seule, détail via cartes de normales |
| **TRELLIS** (v1, `TRELLIS-image-large`) | déc. 2024, toujours maintenu | MIT (vérifié via l'API GitHub) | MIT (déclaré HF) | rembg `u2net` par défaut ; **nvdiffrast** et le rasteriseur gaussien **Inria (non commercial)** dans l'export GLB ; kaolin (Apache-2.0) ; pymeshfix (GPL-3.0) | 16 Go | Gaussiennes, champ de radiance ou mesh ; GLB texturé par cuisson |

## Fiches

### 1. TRELLIS.2-4B (Microsoft)

- **Licence** : `LICENSE` du dépôt `microsoft/TRELLIS.2` lu : MIT, Microsoft. Poids `microsoft/TRELLIS.2-4B` :
  MIT déclaré sur HF, non gated. Le README signale lui-même que nvdiffrast et nvdiffrec ont leur propre
  licence. Le `sparse_structure_decoder` est chargé depuis `microsoft/TRELLIS-image-large` (MIT déclaré).
- **Dépendances téléchargées (vérifié dans le code)** :
  - `pipeline.json` (HF) : `rembg_model` = `briaai/RMBG-2.0` (licence `bria-rmbg-2.0`, non commerciale sans
    accord BRIA) ; `image_cond_model` = `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated, DINOv3 License).
  - `o-voxel/o_voxel/postprocess.py` importe `nvdiffrast.torch` (rasterisation pour la cuisson des
    textures dans `to_glb`). `LICENSE.txt` de nvdiffrast lu : « Nvidia Source Code License (1-Way
    Commercial) », section 3.3 : usage non commercial uniquement. **Interdit pour le produit.**
  - `setup.sh` installe aussi un fork de nvdiffrec (`JeffreyXiang/nvdiffrec`, branche `renderutils`),
    licence NVIDIA non commerciale (lue sur `NVlabs/nvdiffrec`), utilisée pour le rendu PBR de
    visualisation.
  - CuMesh, FlexGEMM, o-voxel (Jeffrey Xiang) : MIT (API GitHub) ; utils3d : MIT.
- **DINOv3 License** (lue) : licence propre à Meta, mondiale, commerciale autorisée ; signataire Meta
  Platforms Ireland pour l'EEE ; interdit la rétro-ingénierie, impose le respect des contrôles à
  l'export et exclut les usages militaires et le « développement ou l'utilisation d'armes à feu ou
  d'armes illégales ». Pas d'exclusion de l'UE. Droit californien. À faire lire par un juriste (armes
  fantasy ≠ armes à feu, mais la formulation mérite un avis).
- **Faisabilité** : 24 Go minimum annoncés, testé A100/H100 ; Linux. Temps annoncés sur H100 : ~3 s en
  512³, ~17 s en 1024³, ~60 s en 1536³. Des forks et installeurs Windows communautaires annoncent 8 à 16 Go
  avec déchargement CPU : **non vérifié**, et ces forks doivent eux-mêmes être audités.
- **Entrées / sorties** : image unique (RGBA acceptée : le détourage est sauté si l'alpha est fourni, à
  confirmer à l'intégration). Sortie O-Voxel → mesh à topologie arbitraire, **surfaces ouvertes**
  (vêtements, capes), attributs PBR dont **l'opacité**. `to_glb` remaille, décime, déplie les UV et cuit
  les textures (c'est là qu'intervient nvdiffrast). Il existe aussi un pipeline de **texturage d'une forme
  donnée** (`example_texturing.py`), utile à l'étape 3.
- **Défauts connus** : surfaces ouvertes et opacité sont la meilleure réponse disponible aux capes et aux
  effets semi-transparents ; 1536³ devrait aider les petites têtes et les lames fines. Aucune preuve sur nos
  images : à tester au banc.
- **Maturité** : forte (dépôt le plus suivi, code d'entraînement publié, ~2,7 M téléchargements HF).
- **Pour l'utiliser commercialement** : passer nos RGBA birefnet (pas RMBG-2.0), extraire le mesh sans
  `to_glb` (ou réécrire la cuisson sans nvdiffrast, sujet de l'étape 3), ne pas installer nvdiffrec.

### 2. Pixal3D (Tencent ARC, Tsinghua, Victoria University of Wellington)

- **Licence** : `LICENSE` GitHub et `LICENSE` HF lus : MIT, Copyright 2026 Tencent. `NOTICE` lu : liste
  dinov2 (Apache-2.0), TRELLIS.2, Direct3D-S2 et MoGe (MIT).
- **Alerte historique (vérifié dans l'historique git)** : le premier `LICENSE` (commit `d42dcaad`,
  2026-05-12) était une licence propre : « academic purposes only », « refrain from using it for any
  commercial or production purposes » et « **Pixal3D IS NOT INTENDED FOR USE WITHIN THE EUROPEAN UNION** ».
  Il a été remplacé par la MIT au commit `5098ba1f` du 2026-05-20. Les poids HF ont été déposés à partir du
  2026-04-30 ; la fiche HF garde aujourd'hui `extra_gated_eu_disallowed: true` (sans effet tant que le dépôt
  n'est pas gated, mais c'est un signal d'intention contraire à la MIT). **Ambiguïté à trancher par le
  gardien et un juriste** ; si c'est retenu, épingler une révision postérieure au 2026-05-20.
- **Dépendances téléchargées (vérifié dans le code)** : `pipeline.json` HF : `rembg_model` =
  `briaai/RMBG-2.0` (non commercial) ; `inference.py` charge DINOv3 depuis
  **`camenduru/dinov3-vitl16-pretrain-lvd1689m`**, un miroir non gated d'un modèle que Meta distribue sous
  accord (DINOv3 License) : la chaîne de distribution est à vérifier, mieux vaut pointer vers le dépôt
  officiel `facebook/...` ; MoGe-2 `Ruicheng/moge-2-vitl` (MIT déclaré, code MoGe MIT lu) ; NAF
  `valeoai/NAF` via `torch.hub` (Apache-2.0 lu) ; natten. L'installation suit celle de TRELLIS.2, donc
  **nvdiffrast** et les autres extensions sont aussi là ; le `NOTICE` de Pixal3D ne mentionne ni DINOv3,
  ni RMBG-2.0, ni nvdiffrast. `utils3d` est installé depuis une wheel d'une release GitHub personnelle.
- **Faisabilité** : aucune VRAM chiffrée ; le code dit qu'en mode normal tous les modèles (« ~18 GB »)
  sont sur le GPU, et qu'en `--low_vram` le pic est « un modèle de flux + un DINOv3 » (flux de 1,3 B
  paramètres en bf16). **16 Go plausible en `--low_vram` à 1024, non vérifié.** Windows : intégration
  ComfyUI communautaire seulement.
- **Entrées / sorties** : image unique, ou **multi-vues** (`inference_mv.py`, poids `*_mv` séparés,
  `transforms.json` à la convention Blender/NeRF, la première vue doit être la face avant). Les vues avec
  alpha sont utilisées telles quelles. Sortie GLB avec PBR, même représentation que TRELLIS.2, 1536 par
  défaut.
- **Défauts connus** : la rétroprojection des pixels vers la 3D vise justement la fidélité à l'image
  (détails fins, accessoires, visage) ; le multi-vues est une piste pour les petites têtes et le dos des
  personnages. Pas de preuve sur concept art : à tester.
- **Maturité** : jeune (six mois), 27 issues ouvertes, code d'entraînement publié, documentation correcte.

### 3. Direct3D-S2 (DreamTech)

- **Licence** : `LICENSE.txt` lu : MIT, DreamTechAI 2025. Poids `wushuang98/Direct3D-S2` : MIT déclaré HF.
- **Dépendances (vérifié dans le code)** : détourage `ZhengPeng7/BiRefNet` (MIT déclaré, déjà dans notre
  registre) ; DINOv2 `dinov2_vitl14_reg` (Apache-2.0) ; torchsparse (MIT) ; extension `udf_ext` du dépôt
  (MIT du dépôt) ; `direct3d_s2/utils/fill_hole.py` importe **pymeshfix (GPL-3.0)**, pyvista et igraph :
  remplaçable, mais à isoler si on garde le bouchage de trous.
- **Faisabilité** : 512 → 10 Go, 1024 → ~24 Go (README). Les auteurs déconseillent le 512 (étape
  intermédiaire, qualité nettement plus basse). Donc **1024 sur GPU cloud**. Windows : issues #11 et #12 du
  dépôt.
- **Entrées / sorties** : image unique ; géométrie seule (SDF clairsemé 1024³ → marching cubes), mesh
  fermé, sans UV ni texture : les étapes 2 et 3 devraient tout faire.
- **Défauts connus** : très bonne netteté géométrique annoncée ; le SDF fermé ne représente pas les
  surfaces ouvertes ni les effets translucides (risque de nappes épaisses). v1.2 « génération de
  personnages » annoncée en juin 2025, jamais publiée.
- **Maturité** : moyenne, dépôt peu actif depuis septembre 2025. C'est la base du `paper` branch de
  Pixal3D.

### 4. Hi3DGen / Stable3DGen (Stable-X), alternative géométrique

- Code adapté de TRELLIS, MIT déclaré ; le README dit avoir **retiré kaolin, nvdiffrast et flexicubes**
  pour permettre l'usage commercial (non vérifié fichier par fichier). Poids `Stable-X/trellis-normal-v0-1`
  (MIT déclaré), estimateur de normales `Stable-X/yoso-normal-v1-8-1` (Apache-2.0 déclaré) chargé via
  `torch.hub` depuis `hugoycj/StableNormal` (Apache-2.0) : vérifier si ces poids dérivent de Stable
  Diffusion (licence OpenRAIL++, restrictions d'usage). Détourage BiRefNet.
- Sortie : géométrie seule, détaillée via normales, résolution TRELLIS v1 (64³ clairsemé), donc moins fine
  que TRELLIS.2. Tient a priori sur 16 Go. Intérêt : solution de repli « propre » si l'export de TRELLIS.2
  pose problème.

### 5. TRELLIS v1 (`microsoft/TRELLIS-image-large`), repli

- MIT (code et poids déclarés). 16 Go annoncés. Mais l'export GLB (`postprocessing_utils.py`) importe
  nvdiffrast et passe par le rasteriseur gaussien de mip-splatting, sous **licence Gaussian-Splatting
  d'Inria (non commerciale)** ; pymeshfix (GPL-3.0) ; détourage rembg `u2net` (le détoureur qui produit
  des halos chez nous). Résolution plus faible que TRELLIS.2. Utile seulement comme référence ou repli.

## Écartés d'emblée

| Modèle | Raison (vérifié sauf mention) |
|---|---|
| Hunyuan3D-2.0, 2.1, 2mini, 2mv, Omni, Part, 3.x | Licences Tencent qui excluent l'UE ; 3.x non publié |
| **TripoSG** (VAST) | Poids MIT, mais `triposg/models/transformers/triposg_transformer.py` est « based on Tencent HunyuanDiT » et embarque la Tencent Hunyuan Community License (« DOES NOT APPLY IN THE EUROPEAN UNION ») ; `NOTICE` cite aussi FlashVDM (Tencent) ; extraction de surface par **diso (CC BY-NC 4.0)** ; script officiel avec RMBG-1.4 (non commercial). À retirer de la liste des candidats de `LICENSES.md`. |
| **Step1X-3D** (StepFun) | Apache-2.0 déclaré, mais `step1x3d_geometry/models/autoencoders/volume_decoders.py` et `step1x3d_texture/differentiable_renderer/mesh_render.py` portent l'en-tête de licence Hunyuan 3D (« TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT ») ; `custom_rasterizer` repris de Hunyuan3D ; nvdiffrast ; 27 à 29 Go annoncés. À retirer de la liste des candidats. |
| UltraShape 1.0 | Fiche HF : `base_model: tencent/Hunyuan3D-2.1` (dérivé d'un modèle interdit) |
| PartCrafter | MIT déclaré, mais construit sur TripoSG d'après l'article (non vérifié dans le code) : même problème |
| PartPacker (NVIDIA) | NVIDIA Source Code License, non commerciale (déclaré) |
| UniForge3D, ShapeR (Meta) | CC BY-NC 4.0 (étiquette HF) |
| SPAR3D, Stable Fast 3D | Stability AI Community License (gratuite sous 1 M$ de revenu annuel, puis licence payante) ; génération plus ancienne et moins détaillée. Non prioritaire |
| Sparc3D, Rodin, Tripo, Meshy, Hunyuan 3.x | Fermés ou API seulement |

## Hors périmètre mais à signaler aux autres étapes

- **Hi3D v1.0** (OCEANO HK, 2026-10-23) : raffineur « deuxième étage » (image + mesh grossier → géométrie
  jusqu'à 1536³), basé sur Sparc3D. Licence propre « Hi3D Open License 1.0 » (lue) : mondiale, commerciale
  autorisée, mais mention obligatoire « Powered by Hi3D 1.0 » et licence séparée au-delà de 200 000 MAU ou
  1 M$ de revenu mensuel. Validé seulement sur une H20 96 Go, Linux. Le générateur grossier de la démo est
  **Hunyuan3D-2 mini** (interdit) : il faudrait lui fournir notre propre mesh grossier, et vérifier
  qu'il n'a pas été entraîné sur des sorties Hunyuan. Pertinent pour l'étape 2.
- **HairDeck** (Tsinghua, oct. 2026, Apache-2.0 déclaré) : cartes de cheveux stylisées avec UV depuis une
  image ; piste pour le point faible cheveux. Non audité.
- **AniGen** (VAST, mars 2026, MIT déclaré, base TRELLIS) : mesh riggé (squelette + skinning) ; 18 Go
  minimum ; le README signale que `extensions/CUBVH/` dérive d'instant-ngp (**non commercial**) ; nvdiffrast.
- **Idol** (Speridlabs, oct. 2026, Apache-2.0 déclaré) : gaussiennes seulement (pas de mesh), 7,2 Go ;
  dépend de DINOv3 et du VAE téléchargé depuis `black-forest-labs/FLUX.2-dev`, dépôt sous **FLUX
  Non-Commercial License** (étiquette HF). Non retenu pour l'étape 1.
- **SAM 3D Objects** (Meta) : SAM License (lue, mêmes termes que DINOv3 : commerciale, mondiale) ;
  reconstruction d'objets dans des photos, gaussiennes et mesh. Pas adapté en priorité au concept art.

## Classement motivé

1. **TRELLIS.2-4B**. Le plus mûr, licence MIT vérifiée, la seule représentation qui gère surfaces
   ouvertes et opacité (capes, effets), PBR et UV en sortie, résolution jusqu'à 1536³. Ses dépendances à
   risque sont identifiées et remplaçables : notre birefnet à la place de RMBG-2.0, une extraction de mesh
   sans nvdiffrast, pas de nvdiffrec. Points ouverts : DINOv3 License (juriste) et tenue sur 16 Go (24 Go
   officiels, à mesurer au banc avant d'envisager le cloud).
2. **Pixal3D**. Sur le papier le meilleur pour nos défauts (alignement pixel → 3D pour les accessoires fins
   et les visages, multi-vues avec caméras, 1536 par défaut, mode basse VRAM), et il hérite de la sortie de
   TRELLIS.2. Classé deuxième à cause du **risque juridique** : licence académique hors UE jusqu'au
   2026-05-20, drapeau « EU disallowed » encore sur la fiche HF, DINOv3 tiré d'un miroir non officiel. S'il
   passe l'audit, il pourrait passer premier après un banc comparatif.
3. **Direct3D-S2**. La chaîne de licences la plus propre (MIT, BiRefNet MIT, DINOv2 Apache-2.0 ; seul
   pymeshfix GPL, facile à retirer), géométrie très nette. Mais géométrie seule, SDF fermé (mauvais pour
   capes et effets), 1024 = GPU cloud, dépôt peu actif.

Repli : Hi3DGen (géométrie seule, licence nettoyée, tient sur 16 Go), puis TRELLIS v1.

Pour l'étape 1 comme pour les suivantes, la quasi-totalité de la chaîne TRELLIS (TRELLIS, TRELLIS.2,
Pixal3D, Hi3DGen, AniGen) réutilise nvdiffrast pour la cuisson des textures : l'étape 3 devra prévoir une
cuisson sans nvdiffrast (rasteriseur sous licence permissive ou Blender en sous-processus).

## Dépendances à risque repérées (toutes chaînes)

| Dépendance | Licence (lue sauf mention) | Qui la tire | Remède |
|---|---|---|---|
| nvdiffrast | NVIDIA Source Code License, non commerciale (§3.3) | TRELLIS, TRELLIS.2, Pixal3D, Step1X-3D, AniGen | Ne pas installer ; extraire le mesh sans `to_glb` ; cuisson à refaire à l'étape 3 |
| nvdiffrec (et fork `JeffreyXiang/nvdiffrec`) | NVIDIA, non commerciale | TRELLIS.2, Pixal3D | Ne pas installer (rendu de visualisation seulement) |
| briaai/RMBG-2.0 | bria-rmbg-2.0, non commerciale (étiquette HF, gated) | TRELLIS.2, Pixal3D (`pipeline.json`) | Fournir nos RGBA birefnet, retirer le `rembg_model` |
| briaai/RMBG-1.4 | bria-rmbg-1.4, non commerciale (fiche HF lue) | TripoSG | Candidat écarté |
| diso | CC BY-NC 4.0 (`LICENSE` lu) | TripoSG | Candidat écarté |
| Code HunyuanDiT / FlashVDM / Hunyuan3D | Tencent, exclut l'UE (en-têtes lus) | TripoSG, Step1X-3D | Candidats écartés |
| Rasteriseur gaussien (mip-splatting / Inria) | Gaussian-Splatting License, non commerciale | TRELLIS v1 | Ne pas utiliser l'export GLB officiel |
| CUBVH (dérivé d'instant-ngp) | non commerciale (déclaré README AniGen) | AniGen | — |
| DINOv3 | DINOv3 License (Meta, propre ; commerciale, clauses armes et export) | TRELLIS.2, Pixal3D, Idol | Avis juridique ; télécharger depuis `facebook/...`, pas un miroir |
| pymeshfix | GPL-3.0 (API GitHub) | Direct3D-S2, TRELLIS v1 | Retirer ou isoler le bouchage de trous |
| FLUX.2-dev (VAE) | FLUX Non-Commercial License (étiquette HF) | Idol | Candidat non retenu |

Remarque commune : tous ces modèles sont entraînés sur Objaverse(-XL) ou des jeux similaires aux licences
mêlées par objet. C'est la licence des poids qui nous lie, mais le point est à noter pour le juriste et pour
le futur fine-tuning (voir le skill `donnees-provenance`).

## Ce que chaque candidat retenu fournit aux étapes 2 à 4

| Candidat | Topologie | UV | Texture | Autres sorties | Résolution |
|---|---|---|---|---|---|
| TRELLIS.2 | Triangles denses (millions de faces avant décimation), surfaces ouvertes possibles, pas forcément étanche | Oui après `to_glb` (dépliage CuMesh) | PBR : base color, roughness, metallic, opacité (cuisson via nvdiffrast) | Voxels O-Voxel avec attributs | 512³, 1024³, 1536³ |
| Pixal3D | Idem TRELLIS.2 | Idem | Idem | Mode multi-vues | 1024 (basse VRAM) à 1536 |
| Direct3D-S2 | Triangles marching cubes, fermé | Non | Non | — | 512³ ou 1024³ |
| Hi3DGen | Triangles, fermé | Non | Non | Carte de normales intermédiaire | ~TRELLIS v1 |
| TRELLIS v1 | Triangles (FlexiCubes) | Oui via export (xatlas) | Couleur cuite depuis les gaussiennes (rasteriseur non commercial) | Gaussiennes, champ de radiance | 64³ clairsemé |

## À transmettre à `/gardien-licences`

Audit demandé, dans cet ordre : TRELLIS.2-4B, Pixal3D, Direct3D-S2, puis Hi3DGen en repli. Points à
trancher : l'historique de licence et le drapeau UE de Pixal3D ; la DINOv3 License ; la licence réelle des
poids StableNormal/yoso ; et retirer TripoSG et Step1X-3D des candidats de `LICENSES.md`. Rien ne passe à
l'intégration avant cet audit et la validation d'un humain.

## Sources

- Dépôts lus via l'API GitHub : `microsoft/TRELLIS.2`, `microsoft/TRELLIS`, `TencentARC/Pixal3D`,
  `DreamTechAI/Direct3D-S2`, `VAST-AI-Research/TripoSG`, `stepfun-ai/Step1X-3D`, `Stable-X/Hi3DGen`,
  `NVlabs/nvdiffrast`, `NVlabs/nvdiffrec`, `facebookresearch/dinov3`, `facebookresearch/sam-3d-objects`,
  `microsoft/MoGe`, `valeoai/NAF`, `SarahWeiii/diso`, `autonomousvision/mip-splatting`,
  `VAST-AI-Research/AniGen`, `speridlabs/idol-sdk`, `hi3d-official/Hi3Dv1.0`.
- Fiches HF : https://huggingface.co/microsoft/TRELLIS.2-4B, https://huggingface.co/TencentARC/Pixal3D,
  https://huggingface.co/wushuang98/Direct3D-S2, https://huggingface.co/VAST-AI/TripoSG,
  https://huggingface.co/stepfun-ai/Step1X-3D, https://huggingface.co/hi3d-official/Hi3Dv1.0,
  https://huggingface.co/speridlabs/idol, https://huggingface.co/infinith/UltraShape,
  https://huggingface.co/VAST-AI/AniGen, https://huggingface.co/hyz317/HairDeck.
- Chronologie secondaire (non utilisée comme preuve de licence) :
  https://app.cinevva.com/guides/ai-3d-model-generation-timeline-2026,
  https://startupfortune.com/pixal3d-goes-mit-open-source-for-3d-asset-creation/.
