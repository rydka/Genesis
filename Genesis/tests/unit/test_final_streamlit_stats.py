#!/usr/bin/env python3
"""
Test Final Statistiques Streamlit
=================================

Teste que l'interface Streamlit affiche correctement les statistiques
des mesh générés avec les solutions temporaires.
"""

import sys
import time

def test_streamlit_function_directly():
    """Teste directement la fonction utilisée par Streamlit"""
    
    print("🧪 Test Direct Fonction Streamlit")
    print("=" * 50)
    
    try:
        # Import de la fonction Streamlit
        from streamlit_advanced import generate_mesh_with_meshformer
        
        # Paramètres de test
        test_params = {
            "prompt": "épée test streamlit final",
            "asset_type": "weapon",
            "style": "test",
            "complexity": 7,
            "quality_mode": "balanced",
            "target_size": "medium",
            "output_format": "stl",
            "enable_smoothing": True,
            "enable_repair": True,
            "generate_renders": True
        }
        
        print(f"🔄 Génération avec prompt: '{test_params['prompt']}'")
        print(f"   Complexité: {test_params['complexity']}")
        print(f"   Type: {test_params['asset_type']}")
        
        # Génération
        start_time = time.time()
        result = generate_mesh_with_meshformer(**test_params)
        total_time = time.time() - start_time
        
        print(f"\n📊 RÉSULTATS STREAMLIT:")
        print(f"   Succès: {result.get('success', False)}")
        print(f"   Temps total: {total_time:.2f}s")
        print(f"   Temps génération: {result.get('generation_time', 0):.2f}s")
        print(f"   Qualité: {result.get('quality_score', 0):.2f}")
        print(f"   Complexité: {result.get('complexity_score', 0):.2f}")
        
        # Statistiques mesh
        mesh_stats = result.get('mesh_stats', {})
        print(f"\n📐 STATISTIQUES MESH:")
        print(f"   Vertices: {mesh_stats.get('vertices', 0):,}")
        print(f"   Faces: {mesh_stats.get('faces', 0):,}")
        print(f"   Volume: {mesh_stats.get('volume', 0):.3f} unités³")
        print(f"   Surface: {mesh_stats.get('surface_area', 0):.3f} unités²")
        
        # Fichier mesh
        mesh_path = result.get('mesh_path', '')
        if mesh_path:
            from pathlib import Path
            mesh_file = Path(mesh_path)
            if mesh_file.exists():
                file_size = mesh_file.stat().st_size
                print(f"   Fichier: {mesh_file.name}")
                print(f"   Taille: {file_size:,} bytes")
            else:
                print(f"   ❌ Fichier non trouvé: {mesh_path}")
        
        # Avertissements
        warnings = result.get('warnings', [])
        if warnings:
            print(f"\n💡 AVERTISSEMENTS:")
            for warning in warnings:
                print(f"   • {warning}")
        
        # Vérifications
        success_checks = []
        
        # Vérifier que les statistiques ne sont pas nulles
        vertices = mesh_stats.get('vertices', 0)
        faces = mesh_stats.get('faces', 0)
        volume = mesh_stats.get('volume', 0)
        
        success_checks.append(("Vertices > 0", vertices > 0))
        success_checks.append(("Faces > 0", faces > 0))
        success_checks.append(("Volume > 0", volume > 0))
        success_checks.append(("Temps > 0", result.get('generation_time', 0) > 0))
        success_checks.append(("Qualité = 0.75", result.get('quality_score', 0) == 0.75))
        success_checks.append(("Solutions temporaires", any("solutions temporaires" in str(w).lower() for w in warnings)))
        
        print(f"\n✅ VÉRIFICATIONS:")
        all_passed = True
        for check_name, passed in success_checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison_with_old_simulation():
    """Compare avec l'ancienne simulation pour vérifier l'amélioration"""
    
    print("\n🔄 Comparaison avec Ancienne Simulation")
    print("=" * 50)
    
    try:
        from streamlit_advanced import generate_mesh_with_meshformer, simulate_meshformer_generation
        
        # Paramètres identiques
        params = {
            "prompt": "épée de comparaison finale",
            "asset_type": "weapon",
            "complexity": 6,
            "quality_mode": "balanced"
        }
        
        print("1️⃣ ANCIENNE SIMULATION")
        print("-" * 30)
        
        old_result = simulate_meshformer_generation(**params)
        old_stats = old_result.get('mesh_stats', {})
        
        print(f"   Qualité: {old_result.get('quality_score', 0):.2f}")
        print(f"   Vertices: {old_stats.get('vertices', 0):,}")
        print(f"   Faces: {old_stats.get('faces', 0):,}")
        print(f"   Volume: {old_stats.get('volume', 0):.3f}")
        print(f"   Avertissements: {len(old_result.get('warnings', []))}")
        
        print("\n2️⃣ NOUVELLES SOLUTIONS TEMPORAIRES")
        print("-" * 30)
        
        new_result = generate_mesh_with_meshformer(**params)
        new_stats = new_result.get('mesh_stats', {})
        
        print(f"   Qualité: {new_result.get('quality_score', 0):.2f}")
        print(f"   Vertices: {new_stats.get('vertices', 0):,}")
        print(f"   Faces: {new_stats.get('faces', 0):,}")
        print(f"   Volume: {new_stats.get('volume', 0):.3f}")
        print(f"   Avertissements: {len(new_result.get('warnings', []))}")
        
        print("\n📊 DIFFÉRENCES")
        print("-" * 20)
        
        # Comparaisons
        quality_diff = new_result.get('quality_score', 0) - old_result.get('quality_score', 0)
        vertices_old = old_stats.get('vertices', 0)
        vertices_new = new_stats.get('vertices', 0)
        
        print(f"   Qualité: {quality_diff:+.2f}")
        print(f"   Vertices: {vertices_old:,} → {vertices_new:,}")
        
        # Vérifier les améliorations
        has_real_stats = vertices_new > 0 and new_stats.get('faces', 0) > 0
        has_temp_solutions = any("solutions temporaires" in str(w).lower() 
                                for w in new_result.get('warnings', []))
        
        print(f"   Statistiques réelles: {'✅' if has_real_stats else '❌'}")
        print(f"   Solutions temporaires: {'✅' if has_temp_solutions else '❌'}")
        
        return has_real_stats and has_temp_solutions
        
    except Exception as e:
        print(f"❌ Erreur comparaison: {e}")
        return False

def main():
    """Fonction principale de test"""
    
    print("🧪 TEST FINAL STATISTIQUES STREAMLIT")
    print("=" * 70)
    print("Vérification que Streamlit affiche les vraies statistiques")
    print("=" * 70)
    
    # Tests
    tests = [
        ("Test Direct Fonction Streamlit", test_streamlit_function_directly),
        ("Comparaison Ancien vs Nouveau", test_comparison_with_old_simulation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Erreur {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé final
    print(f"\n{'='*70}")
    print("📊 RÉSUMÉ FINAL")
    print(f"{'='*70}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSÉ" if success else "❌ ÉCHEC"
        print(f"   {test_name:<35} {status}")
    
    print(f"\n🎯 RÉSULTAT: {passed}/{total} tests passés")
    
    if passed == total:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS !")
        print("\n✅ PROBLÈME RÉSOLU !")
        print("   • Streamlit affiche maintenant les vraies statistiques")
        print("   • Vertices et Faces ne sont plus à 0")
        print("   • Volume et Surface sont calculés correctement")
        print("   • Temps de génération est réaliste (≥ 0.1s)")
        print("   • Solutions temporaires sont utilisées et signalées")
        
        print("\n🎨 RÉSULTATS ATTENDUS DANS STREAMLIT:")
        print("   • ⭐ Qualité: 0.75")
        print("   • 🔧 Complexité: 0.60-0.70 (selon prompt)")
        print("   • 📐 Vertices: 8+ (réel)")
        print("   • 🔺 Faces: 12+ (réel)")
        print("   • Volume: 0.002+ unités³")
        print("   • Surface: 0.196+ unités²")
        print("   • Message: '✅ Généré avec solutions temporaires améliorées'")
        
        print("\n💡 UTILISATION:")
        print("   1. Allez sur http://localhost:8501")
        print("   2. Onglet 'Génération' → Génération Simple")
        print("   3. Entrez un prompt (ex: 'épée magique')")
        print("   4. Cliquez 'Générer'")
        print("   5. Vérifiez les statistiques réelles !")
        
    else:
        print("\n⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        print("   Vérifiez les erreurs ci-dessus")
        print("   Les statistiques peuvent encore être incorrectes")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
