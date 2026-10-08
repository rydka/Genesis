# Fantasy RAG

Recherche visuelle d'assets 3D fantastiques par similarite semantique. Indexe des fichiers STL via rendu Blender multi-vues et embeddings CLIP. Les utilisateurs cherchent en langage naturel (francais ou anglais), CLIP mappe texte et images dans le meme espace vectoriel.

## Pipeline

```
Fichiers STL
  -> Blender Renderer (6 vues PNG 1024x1024)
  -> CLIP Embedding (moyenne des 6 vues, 512 dim)
  -> ChromaDB (stockage vectoriel persistant)
  -> Recherche par similarite cosine (texte -> vecteur CLIP)
```

## Prerequis

- Python 3.10+
- [Blender](https://www.blender.org/) 4.5+ (accessible dans le PATH, pour le rendu)
- GPU recommande pour le rendu Blender (CPU fonctionne aussi)

## Installation

```bash
cd Genesis

# Activer l'environnement virtuel
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Installer les dependances
pip install -r config/requirements/base.txt

# Developpement (pytest, black, flake8, mypy)
pip install -r config/requirements/development.txt
```

Le modele CLIP (`clip-ViT-B-32`, ~600 MB) est telecharge automatiquement au premier usage.

## Utilisation

```bash
# Lancer l'interface de recherche
streamlit run apps/streamlit/app.py

# Lancer les tests
python -m pytest tests/ -v

# Indexer des assets (depuis Python)
from fantasy_rag.database.indexing_pipeline import IndexingPipeline
pipeline = IndexingPipeline()
pipeline.initialize()
pipeline.process_directory(Path("ressources/data/assets"))
```

## Structure du projet

```
Genesis/
├── fantasy_rag/            # Package principal
│   ├── asset_management/   # Chargement et indexation STL
│   ├── rendering/          # Rendu multi-vues Blender (6 vues, 1024x1024)
│   ├── embeddings/         # CLIP embedder + recherche ChromaDB
│   ├── database/           # ChromaDB manager + pipeline d'indexation
│   ├── config/             # Configuration
│   └── utils/              # Utilitaires (fichiers, mesh, images)
├── apps/
│   └── streamlit/app.py    # Interface de recherche
├── tests/                  # Tests pytest
│   ├── test_clip_embedder.py
│   └── test_search.py
└── config/
    └── requirements/       # Fichiers de dependances
```

## Configuration

Fichier : `config/development.yaml`

| Parametre | Valeur par defaut |
|-----------|-------------------|
| CLIP model | `clip-ViT-B-32` (512 dim) |
| Blender resolution | 1024x1024 |
| ChromaDB persist | `ressources/data/database` |
| Distance metric | cosine |

## Commandes Make

```bash
make help              # Liste des commandes
make install           # Dependances de base
make test              # Lancer les tests
make lint              # Verifier le code (flake8)
make format            # Formater (black)
make clean             # Nettoyer caches
```
