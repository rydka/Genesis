# Fantasy RAG - Système de Génération 3D

## 🏰 Architecture Professionnelle

Fantasy RAG est maintenant organisé avec une architecture professionnelle et modulaire.

## 🚀 Utilisation Rapide

```bash
# Interface Streamlit (recommandée)
python launch_fantasy_rag.py streamlit

# Interface Web Flask
python launch_fantasy_rag.py web

# Interface CLI
python launch_fantasy_rag.py cli

# Via Makefile
make run-streamlit
make help
```

## 📂 Structure du Projet

```
Genesis/
├── 📦 fantasy_rag/           # Package principal
├── 🧪 tests/                 # Tests organisés
├── 🛠️ tools/                 # Outils de développement
├── 🎮 demos/                 # Démonstrations
├── 🖥️ apps/                  # Applications
│   ├── streamlit/            # Interface Streamlit modulaire
│   ├── web/                  # Application web Flask
│   └── cli/                  # Interface ligne de commande
├── 📋 scripts/               # Scripts d'automatisation
├── 📚 docs/                  # Documentation
├── ⚙️ config/                # Configuration
└── 📦 data/                  # Données et cache
```

## 🔧 Développement

```bash
make install       # Installer les dépendances
make test          # Lancer les tests
make lint          # Vérifier la qualité
make format        # Formater le code
make clean         # Nettoyer
```

## 📚 Documentation

- [Plan de Refactoring](docs/ARCHITECTURE_REFACTORING_PLAN.md)
- [Architecture Complète](ARCHITECTURE_REFACTORING_COMPLETE.md)
- [Guide Développeur](docs/developer_guide/)
- [Guide Utilisateur](docs/user_guide/)

## 🎯 Fonctionnalités

- **Recherche Sémantique** : Recherche d'assets 3D par description
- **Génération MeshFormer** : Génération de mesh 3D avec IA
- **Rendu Multi-Vues** : Rendu Blender 6 vues automatique
- **Interface Moderne** : Streamlit avec navigation modulaire
- **CLI Professionnel** : Interface ligne de commande complète
- **Architecture Modulaire** : Code organisé et maintenable

Transformé d'une collection de scripts vers une architecture professionnelle ! 🎉
