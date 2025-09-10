#!/usr/bin/env python3
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
