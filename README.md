# Fantasy RAG

Recherche et génération d'assets 3D fantastiques.

- **Recherche** : indexe des fichiers STL via un rendu Blender multi-vues et des embeddings CLIP. On cherche en langage naturel (français ou anglais) ; CLIP place texte et images dans le même espace vectoriel.
- **Génération** : transforme une image de concept en mesh 3D avec Hunyuan3D-2.1, après détourage automatique (birefnet) et affinage du masque.

> **Licences.** Le projet vise un usage commercial dans l'UE. La licence de Hunyuan3D-2.1 ne s'applique pas
> dans l'UE : il ne sert plus que de référence de recherche et doit être remplacé. Avant d'ajouter un
> modèle, une dépendance ou un jeu de données, consulter [LICENSES.md](LICENSES.md).
>
> Le développement est organisé en agents pour Google Antigravity : voir [.agents/README.md](.agents/README.md).
> Les résultats des expériences sont dans [docs/experiments.md](docs/experiments.md).

## Pipelines

```
Recherche
  Fichiers STL
    -> Rendu Blender (6 vues PNG 1024x1024)
    -> Embedding CLIP (moyenne des 6 vues, 512 dim)
    -> ChromaDB (stockage vectoriel persistant)
    -> Recherche par similarité cosinus (texte -> vecteur CLIP)

Génération
  Image de concept
    -> Détourage rembg (birefnet-general) + affinage du masque
    -> Hunyuan3D-2.1 (image -> mesh, octree 384)
    -> Suppression des morceaux flottants
    -> Export GLB / STL / OBJ / PLY
```

## Prérequis

- Python 3.10+
- [Blender](https://www.blender.org/) 4.4+ dans le PATH (rendu des STL)
- Génération 3D : GPU NVIDIA (testé sur 16 Go de VRAM), CUDA 12, et
  [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) cloné dans `third_party/`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/Mac

pip install -r config/requirements/base.txt         # recherche
pip install -r config/requirements/development.txt  # pytest, black, flake8, mypy
pip install -r config/requirements/generation.txt   # génération 3D (GPU)
```

Le modèle CLIP (`clip-ViT-B-32`, ~600 Mo) et les poids Hunyuan3D sont téléchargés automatiquement au premier usage.

## Utilisation

```bash
# Interface (onglets Recherche et Génération 3D)
streamlit run apps/streamlit/app.py

# Tests
python -m pytest tests/ -v
```

Indexer des assets depuis Python :

```python
from pathlib import Path
from fantasy_rag.database.indexing_pipeline import IndexingPipeline

pipeline = IndexingPipeline()
pipeline.initialize()
pipeline.process_directory(Path("ressources/data/assets"))
```

## Structure du projet

```
fantasy_rag/              # Package principal
├── asset_management/     # Chargement et indexation STL
├── rendering/            # Rendu multi-vues Blender
├── embeddings/           # Embedder CLIP + recherche ChromaDB
├── database/             # Gestion ChromaDB + pipeline d'indexation
├── generation/           # Détourage, Hunyuan3D (formes), StdGEN (personnages)
├── diffrender/           # Rasteriseur différentiable maison (pur PyTorch), voir son README
├── config/               # Configuration (settings.py)
└── utils/                # Utilitaires (fichiers, mesh, images)
apps/streamlit/app.py     # Interface utilisateur
tests/                    # Tests pytest
tools/debug/              # Outils d'expérimentation (comparaison de détourage,
                          #   balayage d'hyperparamètres, rendu de GLB)
tools/bench/              # Mesures (temps et mémoire du rasteriseur différentiable), recalage
                          #   par la silhouette (fit_silhouette, fit_flexicubes), planches de
                          #   zooms visage et mains (zoom_sheet)
config/requirements/      # Dépendances (base, development, generation)
```

Non versionnés : `.venv/`, `data/` (images et sorties de test), `ressources/` (base ChromaDB, rendus), `third_party/` (Hunyuan3D, StdGEN).

### Recalage d'un mesh sur la silhouette de son image (banc d'essai)

`tools/bench/fit_flexicubes.py` recale la forme d'un mesh généré sur la silhouette de son image
(FlexiCubes, puis finition par remaillage continu ; détails dans `fantasy_rag/diffrender/README.md`).
**Règle : recaler sur le masque qui a servi à générer le mesh.** Pour les meshes Hunyuan3D, c'est le
masque affiné `birefnet-general_refined.png` (défaut `--target refined` ; `--mask` reçoit l'alpha brut
`birefnet-general_raw.png`, le masque affiné est trouvé à côté). `--target raw` recale sur l'alpha
brut, ce qui retire aussi ce que l'affinage avait comblé et sur quoi le générateur a modelé (la main
tendue de 3dfab en v2).

La finition (`--finish remesh` par défaut, `vertex`, ou `none`) ne touche que le **contour** : seuls
les sommets qui se projettent à moins de `--finish-contour-px` (8 px par défaut, atténuation lisse sur
les 4 derniers, `--finish-contour-transition-px`) du bord de la silhouette FlexiCubes ou de la cible
bougent ou sont remaillés ; l'intérieur de la silhouette (visage, mains vues de face, plis) reste
exactement celui de FlexiCubes. `--finish-contour-px 0` rend l'ancien comportement (campagne v3).

```powershell
.venv\Scripts\python.exe tools\bench\fit_flexicubes.py --mesh M.glb --mask data\bg_comparison\<stem>\birefnet-general_raw.png --camera data\bench\<date>_fit-silhouette\<stem>\camera.json --out data\bench\<date>_fit-v3\<stem> --device cuda --blender
```

## Configuration

Les chemins et paramètres sont définis dans `fantasy_rag/config/settings.py` :

| Paramètre | Valeur par défaut |
|-----------|-------------------|
| Modèle CLIP | `clip-ViT-B-32` (512 dim) |
| Résolution Blender | 1024x1024 |
| Base ChromaDB | `ressources/data/database` |
| Métrique | cosinus |

Les paramètres de génération (`ShapeConfig` dans `fantasy_rag/generation/shape_generator.py`) sont réglables depuis l'interface : modèle de détourage, affinage du masque, suppression des morceaux flottants, nombre d'étapes, graine.

## Commandes Make

Disponibles sur Linux/Mac ou en CI (`make` n'est pas installé par défaut sous Windows) :

```bash
make help         # Liste des commandes
make install      # Dépendances de base
make install-gen  # Dépendances de génération 3D
make test         # Tests
make lint         # flake8
make format       # black
make run          # Interface Streamlit
make clean        # Caches Python et pytest
```
