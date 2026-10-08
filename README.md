# Fantasy RAG

Recherche et génération d'assets 3D fantastiques.

- **Recherche** : indexe des fichiers STL via un rendu Blender multi-vues et des embeddings CLIP. On cherche en langage naturel (français ou anglais) ; CLIP place texte et images dans le même espace vectoriel.
- **Génération** : transforme une image de concept en mesh 3D avec Hunyuan3D-2.1, après détourage automatique (birefnet) et affinage du masque.

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
├── config/               # Configuration (settings.py)
└── utils/                # Utilitaires (fichiers, mesh, images)
apps/streamlit/app.py     # Interface utilisateur
tests/                    # Tests pytest
tools/debug/              # Outils d'expérimentation (comparaison de détourage,
                          #   balayage d'hyperparamètres, rendu de GLB)
config/requirements/      # Dépendances (base, development, generation)
```

Non versionnés : `.venv/`, `data/` (images et sorties de test), `ressources/` (base ChromaDB, rendus), `third_party/` (Hunyuan3D, StdGEN).

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
