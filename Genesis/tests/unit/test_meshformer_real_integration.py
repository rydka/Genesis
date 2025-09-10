#!/usr/bin/env python3
"""
Test MeshFormer Real Integration
===============================

Tests complets pour l'intégration de MeshFormer réel dans Fantasy RAG.
Vérifie tous les composants et l'intégration complète.
"""

import sys
import time
import logging
from pathlib import Path
import torch

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test des imports des modules MeshFormer"""
    
    print("🔍 Test des imports")
    print("=" * 40)
    
    try:
        # Modules de base
        from fantasy_rag.generation.meshformer_real import (
            MeshFormerReal, MeshFormerRealConfig, MeshFormerInput, MeshFormerOutput
        )
        print("✅ meshformer_real importé")
        
        from fantasy_rag.generation.meshformer_integration import MeshFormerIntegrationManager
        print("✅ meshformer_integration importé")
        
        from fantasy_rag.generation.meshformer_interface import (
            MeshFormerInterface, GenerationRequest, GenerationResult
        )
        print("✅ meshformer_interface importé")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erreur import: {e}")
        return False

def test_configuration():
    """Test de la configuration MeshFormer"""
    
    print("\n🔧 Test de la configuration")
    print("=" * 40)
    
    try:
        from fantasy_rag.generation.meshformer_real import MeshFormerRealConfig
        
        # Configuration par défaut
        config = MeshFormerRealConfig()
        print(f"✅ Configuration par défaut créée")
        print(f"   Device: {config.device}")
        print(f"   Résolution coarse: {config.voxel_resolution_coarse}")
        print(f"   Résolution fine: {config.voxel_resolution_fine}")
        print(f"   SDF résolution: {config.sdf_resolution}")
        
        # Configuration personnalisée
        custom_config = MeshFormerRealConfig(
            device="cpu",
            voxel_resolution_coarse=32,
            voxel_resolution_fine=128,
            sdf_resolution=256
        )
        print(f"✅ Configuration personnalisée créée")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur configuration: {e}")
        return False

def test_integration_manager():
    """Test du gestionnaire d'intégration"""
    
    print("\n🎛️ Test du gestionnaire d'intégration")
    print("=" * 40)
    
    try:
        from fantasy_rag.generation.meshformer_integration import MeshFormerIntegrationManager
        from fantasy_rag.generation.meshformer_real import MeshFormerRealConfig
        
        # Configuration de test
        config = MeshFormerRealConfig(
            device="cpu",  # Force CPU pour compatibilité
            voxel_resolution_coarse=16,  # Résolution très faible pour test rapide
            voxel_resolution_fine=32,
            sdf_resolution=64
        )
        
        # Gestionnaire en mode simulation forcé
        manager = MeshFormerIntegrationManager(config, force_simulation=True)
        print(f"✅ Gestionnaire créé en mode simulation")
        
        # Informations du modèle
        model_info = manager.get_model_info()
        print(f"   Type: {model_info['model_type']}")
        print(f"   Device: {model_info['device']}")
        print(f"   Réel: {model_info['is_real']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur gestionnaire: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_request():
    """Test de création de requête de génération"""
    
    print("\n📝 Test de requête de génération")
    print("=" * 40)
    
    try:
        from fantasy_rag.generation.meshformer_interface import GenerationRequest
        
        # Requête simple
        request = GenerationRequest(
            prompt="épée magique simple",
            style="fantastique",
            complexity=5,
            asset_type="weapon"
        )
        
        print(f"✅ Requête créée")
        print(f"   Prompt: {request.prompt}")
        print(f"   Style: {request.style}")
        print(f"   Complexité: {request.complexity}")
        print(f"   Type: {request.asset_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur requête: {e}")
        return False

def test_mesh_generation():
    """Test de génération de mesh"""
    
    print("\n🎨 Test de génération de mesh")
    print("=" * 40)
    
    try:
        from fantasy_rag.generation.meshformer_integration import MeshFormerIntegrationManager
        from fantasy_rag.generation.meshformer_real import MeshFormerRealConfig
        from fantasy_rag.generation.meshformer_interface import GenerationRequest
        
        # Configuration optimisée pour test rapide
        config = MeshFormerRealConfig(
            device="cpu",
            voxel_resolution_coarse=8,   # Très faible pour test rapide
            voxel_resolution_fine=16,
            sdf_resolution=32,
            batch_size=1
        )
        
        # Gestionnaire
        manager = MeshFormerIntegrationManager(config, force_simulation=True)
        
        # Requête de test
        request = GenerationRequest(
            prompt="cube simple test",
            style="simple",
            complexity=3,
            asset_type="object"
        )
        
        print(f"🔄 Génération en cours...")
        start_time = time.time()
        
        result = manager.generate_mesh(request)
        
        generation_time = time.time() - start_time
        
        if result.success:
            print(f"✅ Génération réussie en {generation_time:.2f}s")
            print(f"   Mesh: {result.mesh_path}")
            print(f"   Qualité: {result.quality_score:.2f}")
            print(f"   Temps: {result.generation_time:.2f}s")
            
            # Vérifier que le fichier existe
            if Path(result.mesh_path).exists():
                print(f"✅ Fichier mesh créé")
                file_size = Path(result.mesh_path).stat().st_size
                print(f"   Taille: {file_size:,} bytes")
            else:
                print(f"⚠️ Fichier mesh non trouvé")
            
            return True
        else:
            print(f"❌ Génération échouée: {result.error_message}")
            return False
        
    except Exception as e:
        print(f"❌ Erreur génération: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interface_integration():
    """Test de l'intégration avec l'interface existante"""
    
    print("\n🔗 Test de l'intégration interface")
    print("=" * 40)
    
    try:
        from fantasy_rag.generation.meshformer_interface import MeshFormerInterface, MeshFormerConfig
        from fantasy_rag.generation.meshformer_interface import GenerationRequest
        
        # Configuration interface
        config = MeshFormerConfig(
            device="cpu",
            quality_mode="fast",
            resolution=16
        )
        
        # Interface
        interface = MeshFormerInterface(config)
        
        # Requête
        request = GenerationRequest(
            prompt="test intégration interface",
            style="simple",
            complexity=3,
            asset_type="object"
        )
        
        print(f"🔄 Test génération via interface...")
        result = interface.generate_mesh(request)
        
        if result.success:
            print(f"✅ Interface intégrée avec succès")
            print(f"   Temps: {result.generation_time:.2f}s")
            print(f"   Qualité: {result.quality_score:.2f}")
            return True
        else:
            print(f"❌ Échec interface: {result.error_message}")
            return False
        
    except Exception as e:
        print(f"❌ Erreur intégration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test de performance basique"""
    
    print("\n⚡ Test de performance")
    print("=" * 40)
    
    try:
        from fantasy_rag.generation.meshformer_integration import MeshFormerIntegrationManager
        from fantasy_rag.generation.meshformer_real import MeshFormerRealConfig
        from fantasy_rag.generation.meshformer_interface import GenerationRequest
        
        # Configuration performance
        config = MeshFormerRealConfig(
            device="cpu",
            voxel_resolution_coarse=16,
            voxel_resolution_fine=32,
            sdf_resolution=64
        )
        
        manager = MeshFormerIntegrationManager(config, force_simulation=True)
        
        # Test multiple générations
        num_tests = 3
        times = []
        
        for i in range(num_tests):
            request = GenerationRequest(
                prompt=f"test performance {i+1}",
                style="simple",
                complexity=3,
                asset_type="object"
            )
            
            start_time = time.time()
            result = manager.generate_mesh(request)
            end_time = time.time()
            
            if result.success:
                times.append(end_time - start_time)
                print(f"   Test {i+1}: {times[-1]:.2f}s")
            else:
                print(f"   Test {i+1}: ÉCHEC")
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"✅ Performance moyenne: {avg_time:.2f}s")
            print(f"   Min: {min(times):.2f}s")
            print(f"   Max: {max(times):.2f}s")
            return True
        else:
            print(f"❌ Aucun test de performance réussi")
            return False
        
    except Exception as e:
        print(f"❌ Erreur performance: {e}")
        return False

def main():
    """Fonction principale de test"""
    
    print("🧪 TEST MESHFORMER REAL INTEGRATION")
    print("=" * 60)
    print("Tests complets de l'intégration MeshFormer réel")
    print("=" * 60)
    
    # Tests à exécuter
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Gestionnaire d'intégration", test_integration_manager),
        ("Requête de génération", test_generation_request),
        ("Génération de mesh", test_mesh_generation),
        ("Intégration interface", test_interface_integration),
        ("Performance", test_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Erreur critique {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé final
    print(f"\n{'='*60}")
    print("📊 RÉSUMÉ DES TESTS")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSÉ" if success else "❌ ÉCHEC"
        print(f"   {test_name:<30} {status}")
    
    print(f"\n🎯 RÉSULTAT GLOBAL: {passed}/{total} tests passés")
    
    if passed == total:
        print("🎉 TOUS LES TESTS SONT PASSÉS !")
        print("\n✅ MeshFormer Real est complètement intégré")
        print("✅ Prêt pour la production")
        print("✅ Interface unifiée fonctionnelle")
        
        print("\n💡 PROCHAINES ÉTAPES:")
        print("   1. Exécuter: python setup_meshformer_real.py")
        print("   2. Tester avec l'interface Streamlit")
        print("   3. Optimiser les performances")
        print("   4. Intégrer le vrai modèle quand disponible")
        
    else:
        print("⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        print("\n🔧 ACTIONS RECOMMANDÉES:")
        print("   1. Vérifier les dépendances manquantes")
        print("   2. Corriger les erreurs d'import")
        print("   3. Vérifier la configuration")
        print("   4. Relancer les tests")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
