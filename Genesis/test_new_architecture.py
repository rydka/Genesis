#!/usr/bin/env python3
"""
Test de la Nouvelle Architecture
===============================

Valide que la nouvelle architecture fonctionne correctement.
"""

import sys
import os
from pathlib import Path

def test_directory_structure():
    """Teste que la structure de dossiers existe"""
    
    print("🏗️ Test Structure de Dossiers")
    print("-" * 40)
    
    required_dirs = [
        "tests/unit",
        "tests/integration", 
        "tests/e2e",
        "tools/setup",
        "tools/data",
        "tools/debug",
        "apps/streamlit",
        "apps/web",
        "apps/cli",
        "config/requirements",
        "docs",
        "data"
    ]
    
    success = 0
    total = len(required_dirs)
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"   ✅ {dir_path}")
            success += 1
        else:
            print(f"   ❌ {dir_path}")
    
    print(f"\n📊 Résultat: {success}/{total} dossiers créés")
    return success == total

def test_entry_points():
    """Teste que les points d'entrée existent"""
    
    print("\n🚀 Test Points d'Entrée")
    print("-" * 40)
    
    entry_points = [
        "apps/streamlit/main.py",
        "apps/cli/fantasy_cli.py",
        "Makefile",
        "config/development.yaml",
        "config/requirements/base.txt"
    ]
    
    success = 0
    total = len(entry_points)
    
    for entry in entry_points:
        if Path(entry).exists():
            print(f"   ✅ {entry}")
            success += 1
        else:
            print(f"   ❌ {entry}")
    
    print(f"\n📊 Résultat: {success}/{total} points d'entrée créés")
    return success == total

def test_imports():
    """Teste que les imports fonctionnent"""
    
    print("\n📦 Test Imports")
    print("-" * 40)
    
    imports_to_test = [
        ("fantasy_rag.embeddings", "MultilingualSearchEngine"),
        ("fantasy_rag.generation", "MeshFormerIntegration"),
        ("fantasy_rag.rendering", "BlenderRenderer"),
        ("fantasy_rag.descriptions", "DescriptionGenerator")
    ]
    
    success = 0
    total = len(imports_to_test)
    
    for module, class_name in imports_to_test:
        try:
            exec(f"from {module} import {class_name}")
            print(f"   ✅ {module}.{class_name}")
            success += 1
        except ImportError as e:
            print(f"   ❌ {module}.{class_name} - {e}")
    
    print(f"\n📊 Résultat: {success}/{total} imports réussis")
    return success == total

def test_streamlit_pages():
    """Teste que les pages Streamlit existent"""
    
    print("\n📄 Test Pages Streamlit")
    print("-" * 40)
    
    pages = [
        "apps/streamlit/pages/search.py",
        "apps/streamlit/pages/generation.py", 
        "apps/streamlit/pages/analytics.py",
        "apps/streamlit/pages/admin.py"
    ]
    
    success = 0
    total = len(pages)
    
    for page in pages:
        if Path(page).exists():
            print(f"   ✅ {page}")
            success += 1
        else:
            print(f"   ❌ {page}")
    
    print(f"\n📊 Résultat: {success}/{total} pages créées")
    return success == total

def test_makefile_commands():
    """Teste que le Makefile contient les bonnes commandes"""
    
    print("\n🔧 Test Commandes Makefile")
    print("-" * 40)
    
    makefile = Path("Makefile")
    if not makefile.exists():
        print("   ❌ Makefile non trouvé")
        return False
    
    content = makefile.read_text(encoding='utf-8')
    
    required_commands = [
        "install:",
        "test:",
        "run-streamlit:",
        "run-web:",
        "help:"
    ]
    
    success = 0
    total = len(required_commands)
    
    for cmd in required_commands:
        if cmd in content:
            print(f"   ✅ {cmd}")
            success += 1
        else:
            print(f"   ❌ {cmd}")
    
    print(f"\n📊 Résultat: {success}/{total} commandes trouvées")
    return success == total

def main():
    """Fonction principale"""
    
    print("🧪 TEST NOUVELLE ARCHITECTURE FANTASY RAG")
    print("=" * 60)
    
    tests = [
        ("Structure Dossiers", test_directory_structure),
        ("Points d'Entrée", test_entry_points),
        ("Imports Modules", test_imports),
        ("Pages Streamlit", test_streamlit_pages),
        ("Commandes Makefile", test_makefile_commands)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Erreur dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé final
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSÉ" if result else "❌ ÉCHOUÉ"
        print(f"   {status:<10} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 SCORE GLOBAL: {passed}/{total} tests passés")
    
    if passed == total:
        print("\n🎉 NOUVELLE ARCHITECTURE VALIDÉE !")
        print("   Tous les tests sont passés avec succès.")
        print("   L'architecture est prête à être utilisée.")
    else:
        print(f"\n⚠️  ARCHITECTURE PARTIELLEMENT VALIDÉE")
        print(f"   {total - passed} test(s) ont échoué.")
        print("   Vérifiez les erreurs ci-dessus.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
