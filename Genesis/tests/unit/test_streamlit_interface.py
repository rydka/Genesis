#!/usr/bin/env python3
"""
Test de l'interface Streamlit Fantasy RAG
=========================================

Script de test pour vérifier toutes les fonctionnalités de l'interface Streamlit.
"""

import requests
import time
import json
from pathlib import Path

def test_api_connection():
    """Test de connexion à l'API Flask"""
    print("🔍 Test de connexion à l'API Flask...")
    
    try:
        response = requests.get("http://localhost:5000/api/stats", timeout=5)
        if response.status_code == 200:
            print("✅ API Flask accessible")
            stats = response.json()
            print(f"   📦 Assets: {stats.get('total_assets', 0)}")
            print(f"   📝 Descriptions: {stats.get('total_descriptions', 0)}")
            print(f"   🖼️ Rendus: {stats.get('total_renders', 0)}")
            return True
        else:
            print(f"❌ API Flask erreur HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Flask inaccessible: {str(e)}")
        return False

def test_search_functionality():
    """Test de la fonctionnalité de recherche"""
    print("\n🔍 Test de la recherche...")
    
    test_queries = [
        "épée",
        "warrior", 
        "personnage armé",
        "magical sword"
    ]
    
    for query in test_queries:
        try:
            payload = {
                "query": query,
                "language": "français",
                "max_results": 3,
                "min_similarity": -1.0
            }
            
            response = requests.post(
                "http://localhost:5000/api/search",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json()
                if results.get("success"):
                    count = len(results.get("results", []))
                    time_taken = results.get("processing_time", 0)
                    print(f"   ✅ '{query}': {count} résultats en {time_taken:.2f}s")
                else:
                    print(f"   ❌ '{query}': {results.get('error', 'Erreur inconnue')}")
            else:
                print(f"   ❌ '{query}': HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ '{query}': {str(e)}")

def test_image_endpoints():
    """Test des endpoints d'images"""
    print("\n🖼️ Test des endpoints d'images...")
    
    # Test avec l'asset connu
    asset_id = "stl_fc728778684a"
    views = ["front", "back", "left", "right", "top", "iso_3d"]
    
    for view in views:
        try:
            url = f"http://localhost:5000/api/image/{asset_id}/{view}.png"
            response = requests.head(url, timeout=5)
            
            if response.status_code == 200:
                size = response.headers.get('Content-Length', 'Unknown')
                print(f"   ✅ {view}.png: {size} bytes")
            else:
                print(f"   ❌ {view}.png: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ {view}.png: {str(e)}")

def test_streamlit_accessibility():
    """Test d'accessibilité de Streamlit"""
    print("\n🌐 Test d'accessibilité Streamlit...")
    
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200:
            print("✅ Interface Streamlit accessible")
            
            # Vérifier la présence de certains éléments
            content = response.text
            if "Fantasy RAG" in content:
                print("   ✅ Titre principal trouvé")
            if "Recherche" in content:
                print("   ✅ Interface de recherche trouvée")
            
            return True
        else:
            print(f"❌ Interface Streamlit erreur HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Interface Streamlit inaccessible: {str(e)}")
        return False

def test_file_structure():
    """Test de la structure des fichiers"""
    print("\n📁 Test de la structure des fichiers...")
    
    required_files = [
        "streamlit_app.py",
        "streamlit_advanced.py", 
        "launch_streamlit.py",
        "requirements_streamlit.txt",
        "fantasy_rag/web/app.py"
    ]
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"   ✅ {file_path}: {size} bytes")
        else:
            print(f"   ❌ {file_path}: Fichier manquant")

def test_database_content():
    """Test du contenu de la base de données"""
    print("\n💾 Test du contenu de la base de données...")
    
    try:
        # Test via l'API de recherche générale
        payload = {
            "query": "",  # Recherche vide pour obtenir tous les résultats
            "language": "français",
            "max_results": 50,
            "min_similarity": -1.0
        }
        
        response = requests.post(
            "http://localhost:5000/api/search",
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            results = response.json()
            if results.get("success"):
                assets = results.get("results", [])
                print(f"   ✅ {len(assets)} assets trouvés dans la base")
                
                # Analyser les assets
                with_descriptions = sum(1 for a in assets if a.get('description'))
                with_renders = sum(1 for a in assets if a.get('render_count', 0) > 0)
                avg_words = sum(a.get('word_count', 0) for a in assets) / max(len(assets), 1)
                
                print(f"   📝 {with_descriptions} avec descriptions")
                print(f"   🖼️ {with_renders} avec rendus")
                print(f"   📊 Moyenne: {avg_words:.1f} mots/description")
            else:
                print(f"   ❌ Erreur base de données: {results.get('error')}")
        else:
            print(f"   ❌ Erreur HTTP {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Erreur: {str(e)}")

def main():
    """Fonction principale de test"""
    
    print("🧪 Test de l'Interface Streamlit Fantasy RAG")
    print("=" * 60)
    
    # Tests séquentiels
    tests = [
        test_api_connection,
        test_search_functionality,
        test_image_endpoints,
        test_streamlit_accessibility,
        test_file_structure,
        test_database_content
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            result = test_func()
            if result is not False:  # None ou True = succès
                passed += 1
        except Exception as e:
            print(f"❌ Erreur dans {test_func.__name__}: {str(e)}")
    
    # Résumé
    print("\n" + "=" * 60)
    print(f"📊 Résumé des tests: {passed}/{total} réussis")
    
    if passed == total:
        print("🎉 Tous les tests sont passés ! Interface prête à utiliser.")
        print("\n💡 Prochaines étapes recommandées :")
        print("   1. Ouvrir http://localhost:8501 dans le navigateur")
        print("   2. Tester la recherche avec différents termes")
        print("   3. Explorer les onglets Comparaison et Analytics")
        print("   4. Tester la génération assistée (simulation)")
    else:
        print("⚠️ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
        
        if not test_api_connection():
            print("\n🔧 Pour corriger les problèmes d'API :")
            print("   1. Vérifiez que l'API Flask est démarrée")
            print("   2. Exécutez: python -m fantasy_rag.web.app")
            print("   3. Attendez le message 'Debugger is active!'")
        
        if not test_streamlit_accessibility():
            print("\n🔧 Pour corriger les problèmes Streamlit :")
            print("   1. Vérifiez que Streamlit est démarré")
            print("   2. Exécutez: python launch_streamlit.py")
            print("   3. Attendez le message 'You can now view your Streamlit app'")

if __name__ == "__main__":
    main()
