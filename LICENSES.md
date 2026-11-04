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

## Candidats au remplacement de Hunyuan3D (non intégrés)

Licences seulement **déclarées** sur HuggingFace le 2026-11-04. À auditer avant toute intégration.

| Candidat | Licence déclarée | Points d'attention |
|---|---|---|
| TRELLIS.2 (`microsoft/TRELLIS.2-4B`) | MIT | Les scripts officiels peuvent télécharger un détoureur non commercial : utiliser notre propre détourage. |
| TRELLIS (`microsoft/TRELLIS-image-large`) | MIT | Même remarque. |
| TripoSG (`VAST-AI/TripoSG`) | MIT | Le script officiel utilise un détoureur non commercial : utiliser notre propre détourage. |
| Direct3D-S2 (`wushuang98/Direct3D-S2`) | MIT | Environ 24 Go de VRAM en 1024 : GPU cloud nécessaire. |
| Step1X-3D (`stepfun-ai/Step1X-3D`) | Apache-2.0 | Le code semble reprendre des composants Hunyuan : à vérifier fichier par fichier. |

## Bibliothèques

| Bibliothèque | Licence | Remarque |
|---|---|---|
| PyTorch, NumPy, SciPy | BSD | |
| OpenCV (≥ 4.5) | Apache-2.0 | |
| onnxruntime-gpu | MIT | |
| rembg | À VÉRIFIER | |
| pymatting | À VÉRIFIER | |
| trimesh | À VÉRIFIER | |
| sentence-transformers, Streamlit, ChromaDB | À VÉRIFIER | |
| Blender | GPL-3.0 | Utilisé comme outil externe (sous-processus) pour le rendu. Ne pas l'embarquer ni lier son code au produit. |

## Données

| Source | Usage autorisé | Remarque |
|---|---|---|
| Images de test dans `data/` (concept arts trouvés en ligne) | Évaluation interne uniquement | Œuvres d'artistes tiers, certaines signées. Interdites pour l'entraînement, les démos publiques ou le marketing sans droits écrits. |
| Futurs jeux de données de fine-tuning | À définir | Chaque échantillon doit avoir sa provenance et sa licence (voir le skill `donnees-provenance`). Objaverse et similaires mélangent des licences par objet, dont des licences non commerciales. |
