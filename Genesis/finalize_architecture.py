#!/usr/bin/env python3
"""
Finalisation de l'Architecture
==============================

Finalise la migration vers la nouvelle architecture.
"""

import shutil
import sys
from pathlib import Path

def move_remaining_files():
    """Déplace les fichiers restants"""
    
    print("📦 Finalisation de la migration...")
    
    # Fichiers à déplacer
    moves = [
        ("create_clean_200_word_description.py", "tools/data/"),
        ("quick_fix_descriptions.py", "tools/data/"),
        ("render_stl_working.py", "tools/debug/"),
        ("start_ollama.py", "tools/setup/"),
        ("Genesis.py", "apps/"),
        ("Genesis.pyproj", "config/"),
    ]
    
    moved = 0
    
    for source, destination in moves:
        source_path = Path(source)
        dest_dir = Path(destination)
        
        if source_path.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / source_path.name
            
            try:
                if not dest_file.exists():
                    shutil.move(str(source_path), str(dest_file))
                    print(f"   ✅ {source} → {destination}")
                    moved += 1
                else:
                    print(f"   ⚠️  {source} → {destination} (déjà existant)")
            except Exception as e:
                print(f"   ❌ Erreur {source}: {e}")
    
    print(f"✅ {moved} fichiers déplacés")

def create_final_launcher():
    """Crée le lanceur final"""
    
    launcher = Path("launch_fantasy_rag.py")
    with open(launcher, 'w', encoding='utf-8') as f:
        f.write('''#!/usr/bin/env python3
"""
Lanceur Fantasy RAG - Architecture Professionnelle
==================================================

Point d'entrée principal pour Fantasy RAG avec nouvelle architecture.
"""

import subprocess
import sys
import argparse
from pathlib import Path

def launch_streamlit():
    """Lance l'interface Streamlit"""
    print("🚀 Lancement Fantasy RAG - Interface Streamlit")
    print("   URL: http://localhost:8502")
    
    cmd = [sys.executable, "-m", "streamlit", "run", "apps/streamlit/main.py"]
    subprocess.run(cmd)

def launch_web():
    """Lance l'interface web Flask"""
    print("🌐 Lancement Fantasy RAG - Interface Web")
    print("   URL: http://localhost:5000")
    
    cmd = [sys.executable, "apps/launch_web_interface.py"]
    subprocess.run(cmd)

def launch_cli():
    """Lance l'interface CLI"""
    print("💻 Fantasy RAG - Interface CLI")
    
    cmd = [sys.executable, "apps/cli/fantasy_cli.py", "--help"]
    subprocess.run(cmd)

def main():
    """Fonction principale"""
    
    parser = argparse.ArgumentParser(
        description="Fantasy RAG - Système de Génération 3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python launch_fantasy_rag.py streamlit    # Interface web moderne
  python launch_fantasy_rag.py web          # Interface web Flask
  python launch_fantasy_rag.py cli          # Interface ligne de commande
  
  make run-streamlit                         # Via Makefile
  make help                                  # Voir toutes les commandes
        """
    )
    
    parser.add_argument(
        'interface',
        choices=['streamlit', 'web', 'cli'],
        default='streamlit',
        nargs='?',
        help='Interface à lancer (défaut: streamlit)'
    )
    
    args = parser.parse_args()
    
    print("🏰 FANTASY RAG - ARCHITECTURE PROFESSIONNELLE")
    print("=" * 60)
    
    if args.interface == 'streamlit':
        launch_streamlit()
    elif args.interface == 'web':
        launch_web()
    elif args.interface == 'cli':
        launch_cli()

if __name__ == "__main__":
    main()
''')
    
    print(f"✅ Lanceur principal créé: {launcher}")

def create_readme_update():
    """Met à jour le README principal"""
    
    readme = Path("README.md")
    content = '''# Fantasy RAG - Système de Génération 3D

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
'''
    
    with open(readme, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ README principal mis à jour")

def main():
    """Fonction principale"""
    
    print("🎯 FINALISATION ARCHITECTURE FANTASY RAG")
    print("=" * 50)
    
    try:
        # Déplacer les fichiers restants
        move_remaining_files()
        
        # Créer le lanceur final
        create_final_launcher()
        
        # Mettre à jour le README
        create_readme_update()
        
        print("\n" + "=" * 50)
        print("🎉 ARCHITECTURE FINALISÉE !")
        print("=" * 50)
        
        print("\n✅ NOUVELLE ARCHITECTURE COMPLÈTE :")
        print("   📁 Tous les fichiers organisés")
        print("   🚀 Lanceur principal créé")
        print("   📚 Documentation mise à jour")
        print("   🔧 Outils de développement prêts")
        
        print("\n🎯 UTILISATION :")
        print("   python launch_fantasy_rag.py streamlit")
        print("   make run-streamlit")
        print("   make help")
        
        print("\n🌐 INTERFACE STREAMLIT :")
        print("   http://localhost:8502")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR FINALISATION: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
