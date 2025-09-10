#!/usr/bin/env python3
"""
Lanceur pour l'interface Streamlit Fantasy RAG
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_flask_api():
    """Vérifie si l'API Flask est en cours d'exécution"""
    try:
        response = requests.get("http://localhost:5000/api/stats", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Lance l'interface Streamlit"""
    
    print("🚀 Lancement de Fantasy RAG - Interface Streamlit")
    print("=" * 60)
    
    # Vérification de l'API Flask
    print("🔍 Vérification de l'API Flask...")
    
    if not check_flask_api():
        print("❌ L'API Flask n'est pas accessible sur http://localhost:5000")
        print("\n💡 Pour démarrer l'API Flask :")
        print("   1. Ouvrez un nouveau terminal")
        print("   2. Naviguez vers le dossier Genesis")
        print("   3. Exécutez : python -m fantasy_rag.web.app")
        print("   4. Attendez que le serveur démarre")
        print("   5. Relancez ce script")
        
        choice = input("\n❓ Voulez-vous continuer quand même ? (y/N): ")
        if choice.lower() != 'y':
            return 1
    else:
        print("✅ API Flask accessible")
    
    # Lancement de Streamlit
    print("\n🌐 Lancement de l'interface Streamlit...")
    print("   URL: http://localhost:8501")
    print("   Appuyez sur Ctrl+C pour arrêter")
    
    try:
        # Commande Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=false"
        ]
        
        # Lancement
        subprocess.run(cmd, cwd=Path.cwd())
        
    except KeyboardInterrupt:
        print("\n👋 Interface Streamlit arrêtée")
        return 0
    except Exception as e:
        print(f"\n❌ Erreur lors du lancement : {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
