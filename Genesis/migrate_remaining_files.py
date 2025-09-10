#!/usr/bin/env python3
"""
Migration des Fichiers Restants
===============================

Migre les fichiers restants vers la nouvelle architecture.
"""

import shutil
import sys
from pathlib import Path

def migrate_remaining_files():
    """Migre les fichiers restants"""
    
    print("📦 Migration des fichiers restants...")
    
    # Fichiers à migrer
    migrations = [
        # Fichiers de données restants
        ("create_clean_200_word_description.py", "tools/data/"),
        ("quick_fix_descriptions.py", "tools/data/"),
        ("render_stl_working.py", "tools/debug/"),
        ("start_ollama.py", "tools/setup/"),
        
        # Fichiers de configuration
        ("Genesis.py", "apps/"),
        ("Genesis.pyproj", "config/"),
        
        # Déplacer ressources vers data
        ("ressources/", "data/legacy_resources/"),
        ("models/", "data/models/"),
    ]
    
    migrated = 0
    
    for source, destination in migrations:
        source_path = Path(source)
        dest_dir = Path(destination)
        
        if source_path.exists():
            # Créer le dossier de destination
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                if source_path.is_dir():
                    # Copier le dossier
                    dest_path = dest_dir / source_path.name
                    if not dest_path.exists():
                        shutil.copytree(source_path, dest_path)
                        print(f"   ✅ {source} → {destination}")
                        migrated += 1
                else:
                    # Déplacer le fichier
                    dest_file = dest_dir / source_path.name
                    if not dest_file.exists():
                        shutil.move(str(source_path), str(dest_file))
                        print(f"   ✅ {source} → {destination}")
                        migrated += 1
                    
            except Exception as e:
                print(f"   ❌ Erreur migration {source}: {e}")
    
    print(f"✅ Migration terminée : {migrated} éléments migrés")

def create_compatibility_files():
    """Crée des fichiers de compatibilité"""
    
    print("\n🔗 Création des fichiers de compatibilité...")
    
    # Lanceur Streamlit compatible
    launcher = Path("launch_streamlit_new.py")
    with open(launcher, 'w', encoding='utf-8') as f:
        f.write('''#!/usr/bin/env python3
"""
Lanceur Streamlit - Nouvelle Architecture
=========================================

Lance l'application Streamlit avec la nouvelle architecture modulaire.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Lance l'application Streamlit"""
    
    print("🚀 Lancement Fantasy RAG - Nouvelle Architecture")
    print("=" * 60)
    
    # Chemin vers la nouvelle application
    app_path = Path("apps/streamlit/main.py")
    
    if not app_path.exists():
        print("❌ Application Streamlit non trouvée")
        print("   Utilisation du mode de compatibilité...")
        
        # Fallback vers l'ancienne interface
        old_app = Path("apps/streamlit_app.py")
        if old_app.exists():
            app_path = old_app
        else:
            print("❌ Aucune interface Streamlit disponible")
            return False
    
    try:
        # Lancer Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
        print(f"🌐 Commande: {' '.join(cmd)}")
        print("   URL: http://localhost:8501")
        print("   Appuyez sur Ctrl+C pour arrêter")
        
        subprocess.run(cmd)
        return True
        
    except KeyboardInterrupt:
        print("\\n👋 Arrêt de l'application")
        return True
    except Exception as e:
        print(f"❌ Erreur lancement: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
''')
    
    # README pour la nouvelle architecture
    readme = Path("README_NEW_ARCHITECTURE.md")
    with open(readme, 'w', encoding='utf-8') as f:
        f.write('''# Fantasy RAG - Nouvelle Architecture

## 🏗️ Structure Refactorisée

Le projet Fantasy RAG a été refactorisé vers une architecture professionnelle et modulaire.

### 📂 Nouvelle Structure

```
Genesis/
├── 📦 fantasy_rag/           # Package principal (inchangé)
├── 🧪 tests/                 # Tests organisés
│   ├── unit/                 # Tests unitaires
│   ├── integration/          # Tests d'intégration
│   └── e2e/                  # Tests end-to-end
├── 🛠️ tools/                 # Outils de développement
│   ├── setup/                # Scripts d'installation
│   ├── data/                 # Outils de données
│   ├── debug/                # Outils de debug
│   └── maintenance/          # Maintenance
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

### 🚀 Utilisation

#### Commandes Make
```bash
make help          # Voir toutes les commandes
make install       # Installer les dépendances
make test          # Lancer les tests
make run-streamlit # Lancer Streamlit
make run-web       # Lancer l'application web
make health        # Vérifier le système
```

#### Applications
```bash
# Streamlit (nouvelle architecture)
python launch_streamlit_new.py

# Streamlit (compatibilité)
python apps/launch_streamlit.py

# Web Flask
python apps/launch_web_interface.py

# CLI
python apps/cli/fantasy_cli.py --help
```

### 🔄 Migration

Les anciens scripts ont été organisés :
- `test_*.py` → `tests/unit/`
- `setup_*.py` → `tools/setup/`
- `debug_*.py` → `tools/debug/`
- `demo_*.py` → `demos/advanced/`
- `process_*.py` → `tools/data/`

### 💡 Avantages

- **Organisation claire** : Chaque fichier a sa place
- **Modularité** : Composants indépendants
- **Maintenance facile** : Structure intuitive
- **Production ready** : Architecture professionnelle
- **Tests organisés** : Par module et type
- **Configuration flexible** : Par environnement

### 🔧 Développement

```bash
# Installation développement
make install-dev

# Tests
make test-unit
make test-integration

# Qualité code
make lint
make format

# Nettoyage
make clean
```

Cette nouvelle architecture rend le projet plus maintenable et professionnel !
''')
    
    print("   ✅ launch_streamlit_new.py")
    print("   ✅ README_NEW_ARCHITECTURE.md")

def main():
    """Fonction principale"""
    
    print("🔄 MIGRATION FICHIERS RESTANTS")
    print("=" * 50)
    
    try:
        # Migrer les fichiers restants
        migrate_remaining_files()
        
        # Créer les fichiers de compatibilité
        create_compatibility_files()
        
        print("\n" + "=" * 50)
        print("🎉 MIGRATION TERMINÉE !")
        print("=" * 50)
        
        print("\n✅ NOUVELLE ARCHITECTURE PRÊTE :")
        print("   📁 Structure professionnelle créée")
        print("   📦 Fichiers migrés et organisés")
        print("   🔗 Compatibilité maintenue")
        print("   📚 Documentation mise à jour")
        
        print("\n🚀 TESTER LA NOUVELLE ARCHITECTURE :")
        print("   python launch_streamlit_new.py")
        print("   make run-streamlit")
        print("   make test")
        print("   make help")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR MIGRATION: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
