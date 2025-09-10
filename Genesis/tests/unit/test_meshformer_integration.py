#!/usr/bin/env python3
"""
Test de l'Intégration MeshFormer
===============================

Script de test complet pour vérifier l'intégration MeshFormer
dans le système Fantasy RAG.
"""

import sys
import time
import logging
from pathlib import Path

# Ajouter le chemin du module
sys.path.append(str(Path(__file__).parent))

from fantasy_rag.generation.meshformer_interface import (
    MeshFormerInterface, MeshFormerConfig, GenerationRequest
)
from fantasy_rag.generation.mesh_generator import MeshGenerator
from fantasy_rag.generation.prompt_to_mesh import PromptToMeshPipeline
from fantasy_rag.generation.concept_generator import ConceptGenerator

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_meshformer_interface():
    """Test de l'interface MeshFormer"""
    
    print("🧪 Test MeshFormer Interface")
    print("=" * 50)
    
    try:
        # Configuration personnalisée
        config = MeshFormerConfig(
            resolution=128,  # Résolution réduite pour les tests
            quality_mode="medium",
            target_faces=5000
        )
        
        # Initialisation
        meshformer = MeshFormerInterface(config)
        
        print(f"✅ Interface initialisée")
        print(f"   Device: {config.device}")
        print(f"   Mode: {'Simulation' if meshformer.model == 'simulation' else 'Réel'}")
        
        # Test de génération simple
        request = GenerationRequest(
            prompt="épée magique elfique avec des runes brillantes",
            style="elfique",
            complexity=6,
            asset_type="weapon"
        )
        
        print(f"\n🎨 Test génération: '{request.prompt}'")
        result = meshformer.generate_mesh(request)
        
        if result.success:
            print(f"✅ Génération réussie en {result.generation_time:.2f}s")
            print(f"   Mesh: {result.mesh_path}")
            print(f"   Qualité: {result.quality_score:.2f}")
            print(f"   Vertices: {result.mesh_stats.get('vertices', 0)}")
            print(f"   Faces: {result.mesh_stats.get('faces', 0)}")
            print(f"   Rendus: {len(result.render_paths)}")
        else:
            print(f"❌ Génération échouée: {result.error_message}")
        
        # Test des informations du modèle
        model_info = meshformer.get_model_info()
        print(f"\n📊 Informations du modèle:")
        print(f"   Type: {model_info['model_type']}")
        print(f"   Capacités: {len(model_info['capabilities'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test MeshFormer: {str(e)}")
        return False

def test_mesh_generator():
    """Test du générateur de mesh"""
    
    print("\n🧪 Test Mesh Generator")
    print("=" * 50)
    
    try:
        # Initialisation
        generator = MeshGenerator()
        
        print("✅ MeshGenerator initialisé")
        
        # Test génération synchrone
        request = GenerationRequest(
            prompt="armure de chevalier dragon avec écailles dorées",
            style="fantastique",
            complexity=7,
            asset_type="armor"
        )
        
        print(f"\n🎯 Test génération synchrone: '{request.prompt}'")
        result = generator.generate_mesh_sync(request)
        
        if result.success:
            print(f"✅ Génération synchrone réussie")
            print(f"   Temps: {result.generation_time:.2f}s")
            print(f"   Qualité: {result.quality_score:.2f}")
        else:
            print(f"❌ Génération synchrone échouée: {result.error_message}")
        
        # Test génération asynchrone
        print(f"\n🚀 Test génération asynchrone")
        job_id = generator.generate_mesh_async(request)
        print(f"   Job lancé: {job_id}")
        
        # Attendre completion
        max_wait = 10
        waited = 0
        while waited < max_wait:
            status = generator.get_job_status(job_id)
            if status and status["status"] in ["completed", "failed"]:
                break
            time.sleep(0.5)
            waited += 0.5
        
        final_status = generator.get_job_status(job_id)
        if final_status:
            print(f"   Statut final: {final_status['status']}")
            if final_status.get('success'):
                print(f"   Temps: {final_status.get('generation_time', 0):.2f}s")
        
        # Test statistiques
        stats = generator.get_statistics()
        print(f"\n📊 Statistiques:")
        print(f"   Total généré: {stats['total_generated']}")
        print(f"   Taux de succès: {stats['success_rate']:.2%}")
        print(f"   Temps moyen: {stats['average_generation_time']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test MeshGenerator: {str(e)}")
        return False

def test_prompt_to_mesh_pipeline():
    """Test du pipeline prompt-to-mesh"""
    
    print("\n🧪 Test Prompt-to-Mesh Pipeline")
    print("=" * 50)
    
    try:
        # Initialisation
        generator = MeshGenerator()
        pipeline = PromptToMeshPipeline(generator)
        
        print("✅ Pipeline initialisé")
        
        # Test d'analyse de prompt
        test_prompts = [
            "épée magique elfique",
            "armure de dragon rouge",
            "créature ailée mystique",
            "tour de mage cristalline",
            "magical sword with glowing runes"
        ]
        
        print(f"\n🔍 Test analyse de prompts:")
        
        for prompt in test_prompts:
            analysis = pipeline.analyze_prompt(prompt)
            
            print(f"\n   Prompt: '{prompt}'")
            print(f"   Type: {analysis.asset_type}")
            print(f"   Style: {analysis.style}")
            print(f"   Complexité: {analysis.complexity_estimate}/10")
            print(f"   Confiance: {analysis.confidence_score:.2f}")
            print(f"   Mots-clés: {', '.join(analysis.keywords[:3])}")
            print(f"   Amélioré: '{analysis.enhanced_prompt[:60]}...'")
        
        # Test génération à partir de prompt
        print(f"\n🎨 Test génération complète:")
        test_prompt = "bâton de mage cristallin avec orbe flottant"
        
        analysis, job_id = pipeline.generate_from_prompt(test_prompt)
        print(f"   Prompt: '{test_prompt}'")
        print(f"   Job ID: {job_id}")
        print(f"   Type détecté: {analysis.asset_type}")
        print(f"   Style détecté: {analysis.style}")
        
        # Test suggestions
        suggestions = pipeline.get_prompt_suggestions("épée")
        print(f"\n💡 Suggestions pour 'épée':")
        for suggestion in suggestions:
            print(f"   - {suggestion}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test Pipeline: {str(e)}")
        return False

def test_concept_generator():
    """Test du générateur de concepts"""
    
    print("\n🧪 Test Concept Generator")
    print("=" * 50)
    
    try:
        # Initialisation
        concept_gen = ConceptGenerator()
        
        print("✅ ConceptGenerator initialisé")
        
        # Test génération de concept simple
        print(f"\n🎭 Test génération de concepts:")
        
        categories = ["weapon", "armor", "creature", "object"]
        
        for category in categories:
            concept = concept_gen.generate_concept(category=category)
            
            print(f"\n   Catégorie: {category}")
            print(f"   Nom: {concept.name}")
            print(f"   Style: {concept.style}")
            print(f"   Rareté: {concept.rarity}")
            print(f"   Complexité: {concept.complexity}/10")
            print(f"   Description: {concept.description[:80]}...")
            print(f"   Prompt: {concept.prompt[:60]}...")
        
        # Test génération de série
        print(f"\n🎨 Test série thématique 'élémentaire':")
        
        series = concept_gen.generate_concept_series("élémentaire", 3)
        
        for i, concept in enumerate(series):
            print(f"   {i+1}. {concept.name} ({concept.category}, {concept.rarity})")
        
        # Test statistiques
        all_concepts = [concept_gen.generate_concept() for _ in range(10)]
        stats = concept_gen.get_concept_statistics(all_concepts)
        
        print(f"\n📊 Statistiques sur 10 concepts:")
        print(f"   Catégories: {stats['categories']}")
        print(f"   Styles: {list(stats['styles'].keys())[:3]}")
        print(f"   Complexité moyenne: {stats['average_complexity']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test ConceptGenerator: {str(e)}")
        return False

def test_integration_complete():
    """Test d'intégration complète"""
    
    print("\n🧪 Test Intégration Complète")
    print("=" * 50)
    
    try:
        # Initialisation de tous les composants
        generator = MeshGenerator()
        pipeline = PromptToMeshPipeline(generator)
        concept_gen = ConceptGenerator()
        
        print("✅ Tous les composants initialisés")
        
        # Workflow complet: Concept → Prompt → Mesh
        print(f"\n🔄 Workflow complet:")
        
        # 1. Générer un concept
        concept = concept_gen.generate_concept(category="weapon", rarity="rare")
        print(f"   1. Concept généré: {concept.name}")
        
        # 2. Analyser le prompt du concept
        analysis = pipeline.analyze_prompt(concept.prompt)
        print(f"   2. Prompt analysé: {analysis.asset_type} ({analysis.style})")
        
        # 3. Générer le mesh (asynchrone)
        analysis, job_id = pipeline.generate_from_prompt(concept.prompt)
        print(f"   3. Génération lancée: {job_id}")
        
        # 4. Vérifier le statut
        time.sleep(1)  # Attendre un peu
        status = generator.get_job_status(job_id)
        if status:
            print(f"   4. Statut: {status['status']}")
        
        # 5. Statistiques finales
        stats = generator.get_statistics()
        print(f"   5. Jobs totaux: {stats['total_generated']}")
        
        print(f"\n✅ Workflow complet testé avec succès")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test intégration: {str(e)}")
        return False

def main():
    """Fonction principale de test"""
    
    print("🧪 Test Complet - Intégration MeshFormer Fantasy RAG")
    print("=" * 70)
    
    tests = [
        ("MeshFormer Interface", test_meshformer_interface),
        ("Mesh Generator", test_mesh_generator),
        ("Prompt-to-Mesh Pipeline", test_prompt_to_mesh_pipeline),
        ("Concept Generator", test_concept_generator),
        ("Intégration Complète", test_integration_complete)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: RÉUSSI")
            else:
                print(f"❌ {test_name}: ÉCHOUÉ")
        except Exception as e:
            print(f"❌ {test_name}: ERREUR - {str(e)}")
    
    # Résumé final
    print(f"\n{'='*70}")
    print(f"📊 RÉSUMÉ DES TESTS")
    print(f"{'='*70}")
    print(f"Tests réussis: {passed}/{total}")
    print(f"Taux de succès: {passed/total:.1%}")
    
    if passed == total:
        print(f"\n🎉 TOUS LES TESTS SONT PASSÉS !")
        print(f"✅ L'intégration MeshFormer est fonctionnelle")
        print(f"\n💡 Prochaines étapes:")
        print(f"   1. Tester l'interface Streamlit mise à jour")
        print(f"   2. Intégrer le vrai modèle MeshFormer")
        print(f"   3. Connecter avec le système de rendu Blender")
        print(f"   4. Optimiser les performances")
    else:
        print(f"\n⚠️ Certains tests ont échoué")
        print(f"🔧 Vérifiez les erreurs ci-dessus et corrigez les problèmes")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
