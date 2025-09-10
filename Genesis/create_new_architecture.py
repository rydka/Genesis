#!/usr/bin/env python3
"""
Script de Création Nouvelle Architecture Fantasy RAG
===================================================

Crée la nouvelle structure de dossiers professionnelle et migre les fichiers existants.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def create_directory_structure():
    """Crée la nouvelle structure de dossiers"""
    
    print("🏗️ Création de la nouvelle structure de dossiers...")
    
    # Structure des dossiers à créer
    directories = [
        # Tests organisés
        "tests",
        "tests/unit",
        "tests/integration", 
        "tests/e2e",
        "tests/fixtures",
        "tests/fixtures/sample_stl",
        "tests/fixtures/sample_descriptions",
        
        # Outils et utilitaires
        "tools",
        "tools/setup",
        "tools/data",
        "tools/debug",
        "tools/maintenance",
        
        # Démonstrations
        "demos",
        "demos/basic",
        "demos/advanced",
        "demos/interactive",
        "demos/interactive/jupyter_notebooks",
        "demos/interactive/streamlit_demos",
        
        # Applications
        "apps",
        "apps/web",
        "apps/web/static",
        "apps/web/templates",
        "apps/streamlit",
        "apps/streamlit/pages",
        "apps/streamlit/components",
        "apps/cli",
        "apps/cli/commands",
        
        # Scripts d'automatisation
        "scripts",
        "scripts/deployment",
        "scripts/ci_cd",
        "scripts/monitoring",
        
        # Documentation
        "docs",
        "docs/api",
        "docs/user_guide",
        "docs/developer_guide",
        "docs/architecture",
        
        # Configuration
        "config",
        "config/requirements",
        
        # Données (remplace ressources)
        "data",
        "data/assets",
        "data/models", 
        "data/cache",
        "data/exports"
    ]
    
    # Créer les dossiers
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Créer __init__.py pour les packages Python
        if any(part in directory for part in ["tests", "tools", "demos", "apps"]):
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Package module."""\n')
        
        print(f"   ✅ {directory}/")
    
    print(f"✅ Structure créée : {len(directories)} dossiers")

def get_file_migrations() -> Dict[str, List[Tuple[str, str]]]:
    """Définit les migrations de fichiers"""
    
    return {
        "Tests": [
            ("test_*.py", "tests/unit/"),
            ("test_*_integration.py", "tests/integration/"),
            ("test_streamlit*.py", "tests/e2e/"),
            ("test_web*.py", "tests/e2e/"),
        ],
        
        "Setup/Installation": [
            ("setup_*.py", "tools/setup/"),
            ("requirements*.txt", "config/requirements/"),
            ("*.bat", "tools/setup/"),
            ("*.ps1", "tools/setup/"),
        ],
        
        "Outils de données": [
            ("process_*.py", "tools/data/"),
            ("regenerate_*.py", "tools/data/"),
            ("clean_*.py", "tools/data/"),
            ("improve_*.py", "tools/data/"),
            ("generate_*.py", "tools/data/"),
        ],
        
        "Debug/Fix": [
            ("debug_*.py", "tools/debug/"),
            ("fix_*.py", "tools/debug/"),
            ("check_*.py", "tools/debug/"),
        ],
        
        "Démos": [
            ("demo_*.py", "demos/advanced/"),
        ],
        
        "Applications": [
            ("streamlit_*.py", "apps/streamlit/"),
            ("launch_*.py", "apps/"),
        ],
        
        "Documentation": [
            ("*.md", "docs/"),
            ("*.ipynb", "demos/interactive/jupyter_notebooks/"),
        ]
    }

def migrate_files():
    """Migre les fichiers selon la nouvelle structure"""
    
    print("\n📦 Migration des fichiers...")
    
    migrations = get_file_migrations()
    migrated_count = 0
    
    for category, patterns in migrations.items():
        print(f"\n📂 {category}:")
        
        for pattern, destination in patterns:
            # Trouver les fichiers correspondants
            if "*" in pattern:
                import glob
                files = glob.glob(pattern)
            else:
                files = [pattern] if Path(pattern).exists() else []
            
            for file_path in files:
                source = Path(file_path)
                if source.exists() and source.is_file():
                    dest_dir = Path(destination)
                    dest_file = dest_dir / source.name
                    
                    # Éviter les conflits
                    if dest_file.exists():
                        print(f"   ⚠️ Conflit : {source.name} existe déjà dans {destination}")
                        continue
                    
                    try:
                        shutil.move(str(source), str(dest_file))
                        print(f"   ✅ {source.name} → {destination}")
                        migrated_count += 1
                    except Exception as e:
                        print(f"   ❌ Erreur migration {source.name}: {e}")
    
    print(f"\n✅ Migration terminée : {migrated_count} fichiers migrés")

def create_entry_points():
    """Crée les points d'entrée principaux"""
    
    print("\n🚀 Création des points d'entrée...")
    
    # Point d'entrée Streamlit principal
    streamlit_main = Path("apps/streamlit/main.py")
    with open(streamlit_main, 'w', encoding='utf-8') as f:
        f.write('''#!/usr/bin/env python3
"""
Fantasy RAG - Application Streamlit Principale
==============================================

Point d'entrée principal pour l'interface Streamlit modulaire.
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter le package fantasy_rag au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    """Point d'entrée principal"""

    st.set_page_config(
        page_title="Fantasy RAG",
        page_icon="🏰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Navigation
    st.sidebar.title("Fantasy RAG")
    
    # Import des pages
    try:
        from .pages import search, generation, analytics, admin

        pages = {
            "Recherche": search.show,
            "Generation": generation.show,
            "Analytics": analytics.show,
            "Admin": admin.show
        }

        # Sélection de page
        page = st.sidebar.selectbox("Navigation", list(pages.keys()))

        # Affichage de la page
        pages[page]()

    except ImportError as e:
        st.error(f"Erreur import pages: {e}")
        st.info("Utilisation du mode de compatibilité...")

        # Fallback vers l'ancienne interface
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from streamlit_app import main as old_main
            old_main()
        except ImportError:
            st.error("Interface non disponible")

if __name__ == "__main__":
    main()
''')
    
    # Point d'entrée CLI
    cli_main = Path("apps/cli/fantasy_cli.py")
    with open(cli_main, 'w', encoding='utf-8') as f:
        f.write('''#!/usr/bin/env python3
"""
Fantasy RAG - Interface Ligne de Commande
=========================================

Interface CLI pour Fantasy RAG avec commandes modulaires.
"""

import click
import sys
from pathlib import Path

# Ajouter le package fantasy_rag au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Fantasy RAG - Système RAG pour assets 3D fantastiques"""
    pass

@cli.command()
@click.argument('query')
@click.option('--language', default='français', help='Langue de recherche')
@click.option('--max-results', default=5, help='Nombre maximum de résultats')
def search(query, language, max_results):
    """Recherche d'assets par texte"""
    click.echo(f"Recherche: {query}")

    try:
        from fantasy_rag.embeddings import MultilingualSearchEngine
        # Implémentation recherche
        click.echo("Recherche terminée")
    except Exception as e:
        click.echo(f"Erreur: {e}")

@cli.command()
@click.argument('prompt')
@click.option('--style', default='fantastique', help='Style de génération')
@click.option('--complexity', default=5, help='Complexité (1-10)')
def generate(prompt, style, complexity):
    """Génération de mesh 3D"""
    click.echo(f"Génération: {prompt}")

    try:
        from fantasy_rag.generation import MeshGenerator
        # Implémentation génération
        click.echo("Génération terminée")
    except Exception as e:
        click.echo(f"Erreur: {e}")

@cli.command()
@click.argument('stl_path')
def process(stl_path):
    """Traitement d'un fichier STL"""
    click.echo(f"Traitement: {stl_path}")

    try:
        from fantasy_rag.database import IndexingPipeline
        # Implémentation traitement
        click.echo("Traitement terminé")
    except Exception as e:
        click.echo(f"Erreur: {e}")

if __name__ == "__main__":
    cli()
''')
    
    # Makefile pour automatisation
    makefile = Path("Makefile")
    with open(makefile, 'w', encoding='utf-8') as f:
        f.write('''# Fantasy RAG - Makefile
# Commandes d'automatisation

.PHONY: help install test lint format run-web run-streamlit clean

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\\033[36m%-20s\\033[0m %s\\n", $$1, $$2}'

install:  ## Installe les dépendances
	pip install -r config/requirements/base.txt

install-dev:  ## Installe les dépendances de développement
	pip install -r config/requirements/development.txt

test:  ## Lance tous les tests
	python -m pytest tests/ -v

test-unit:  ## Lance les tests unitaires
	python -m pytest tests/unit/ -v

test-integration:  ## Lance les tests d'intégration
	python -m pytest tests/integration/ -v

lint:  ## Vérifie le code avec flake8
	flake8 fantasy_rag/ apps/ tools/

format:  ## Formate le code avec black
	black fantasy_rag/ apps/ tools/

run-web:  ## Lance l'application web
	python apps/web/app.py

run-streamlit:  ## Lance l'application Streamlit
	streamlit run apps/streamlit/main.py

run-cli:  ## Lance l'interface CLI
	python apps/cli/fantasy_cli.py --help

setup:  ## Configuration initiale
	python tools/setup/install_dependencies.py

clean:  ## Nettoie les fichiers temporaires
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf data/cache/*

demo:  ## Lance une démonstration
	python demos/basic/demo_search.py

health:  ## Vérifie la santé du système
	python tools/debug/validate_system.py
''')
    
    print("   ✅ apps/streamlit/main.py")
    print("   ✅ apps/cli/fantasy_cli.py") 
    print("   ✅ Makefile")

def create_config_files():
    """Crée les fichiers de configuration"""
    
    print("\n⚙️ Création des fichiers de configuration...")
    
    # Requirements séparés
    base_req = Path("config/requirements/base.txt")
    with open(base_req, 'w', encoding='utf-8') as f:
        f.write('''# Fantasy RAG - Dépendances de base
streamlit>=1.28.0
flask>=2.3.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
trimesh>=3.21.0
numpy>=1.24.0
pillow>=9.5.0
requests>=2.31.0
pydantic>=2.0.0
''')
    
    dev_req = Path("config/requirements/development.txt")
    with open(dev_req, 'w', encoding='utf-8') as f:
        f.write('''-r base.txt

# Développement
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.7.0
flake8>=6.0.0
mypy>=1.5.0
jupyter>=1.0.0
ipython>=8.14.0

# Debug et profiling
memory-profiler>=0.61.0
line-profiler>=4.1.0
''')
    
    prod_req = Path("config/requirements/production.txt")
    with open(prod_req, 'w', encoding='utf-8') as f:
        f.write('''-r base.txt

# Production
gunicorn>=21.2.0
redis>=4.6.0
celery>=5.3.0
prometheus-client>=0.17.0
''')
    
    # Configuration YAML
    dev_config = Path("config/development.yaml")
    with open(dev_config, 'w', encoding='utf-8') as f:
        f.write('''# Configuration développement
database:
  chromadb:
    persist_directory: "data/chromadb_dev"
    host: "localhost"
    port: 8000

models:
  ollama:
    base_url: "http://localhost:11434"
    model: "llama3.2-vision:latest"
  
  embeddings:
    model: "paraphrase-multilingual-MiniLM-L12-v2"
    device: "cpu"

rendering:
  blender:
    executable: "blender"
    resolution: [1024, 1024]
    samples: 64

logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

cache:
  enabled: true
  directory: "data/cache"
  max_size_gb: 5
''')
    
    print("   ✅ config/requirements/base.txt")
    print("   ✅ config/requirements/development.txt")
    print("   ✅ config/requirements/production.txt")
    print("   ✅ config/development.yaml")

def main():
    """Fonction principale de refactorisation"""
    
    print("🏗️ REFACTORISATION ARCHITECTURE FANTASY RAG")
    print("=" * 60)
    print("Transformation vers une architecture professionnelle")
    print("=" * 60)
    
    try:
        # Étape 1: Créer la structure
        create_directory_structure()
        
        # Étape 2: Migrer les fichiers
        migrate_files()
        
        # Étape 3: Créer les points d'entrée
        create_entry_points()
        
        # Étape 4: Créer la configuration
        create_config_files()
        
        print("\n" + "=" * 60)
        print("🎉 REFACTORISATION TERMINÉE AVEC SUCCÈS !")
        print("=" * 60)
        
        print("\n✅ NOUVELLE STRUCTURE CRÉÉE:")
        print("   📁 tests/ - Tests organisés par type")
        print("   📁 tools/ - Outils de développement")
        print("   📁 demos/ - Démonstrations")
        print("   📁 apps/ - Applications (web, streamlit, cli)")
        print("   📁 scripts/ - Scripts d'automatisation")
        print("   📁 docs/ - Documentation")
        print("   📁 config/ - Configuration par environnement")
        print("   📁 data/ - Données et cache")
        
        print("\n🚀 PROCHAINES ÉTAPES:")
        print("   1. Tester la nouvelle structure")
        print("   2. Mettre à jour les imports")
        print("   3. Lancer: make install")
        print("   4. Lancer: make test")
        print("   5. Lancer: make run-streamlit")
        
        print("\n💡 COMMANDES UTILES:")
        print("   make help          # Voir toutes les commandes")
        print("   make run-streamlit # Lancer Streamlit")
        print("   make test          # Lancer les tests")
        print("   make health        # Vérifier le système")
        
    except Exception as e:
        print(f"\n❌ ERREUR REFACTORISATION: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
