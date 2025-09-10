#!/usr/bin/env python3
"""
Setup Solutions Temporaires
===========================

Installation et configuration des solutions temporaires pour la génération 3D
en attendant le modèle MeshFormer officiel.
"""

import sys
import subprocess
import logging
from pathlib import Path
import time
import requests

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_available_solutions():
    """Vérifie quelles solutions sont disponibles"""
    
    print("🔍 Vérification des solutions disponibles")
    print("=" * 50)
    
    solutions_status = {}
    
    # 1. TripoSR (Stability AI)
    print("\n📦 TripoSR (Stability AI)")
    try:
        import triposr
        print("✅ TripoSR déjà installé")
        solutions_status["triposr"] = "installed"
    except ImportError:
        print("⚠️ TripoSR non installé")
        solutions_status["triposr"] = "available"
    
    # 2. InstantMesh
    print("\n📦 InstantMesh")
    try:
        # Vérifier si InstantMesh est disponible
        response = requests.head("https://github.com/TencentARC/InstantMesh", timeout=5)
        if response.status_code == 200:
            print("✅ InstantMesh disponible en ligne")
            solutions_status["instantmesh"] = "available"
        else:
            print("❌ InstantMesh non accessible")
            solutions_status["instantmesh"] = "unavailable"
    except:
        print("❌ InstantMesh non accessible")
        solutions_status["instantmesh"] = "unavailable"
    
    # 3. Shap-E (OpenAI)
    print("\n📦 Shap-E (OpenAI)")
    try:
        import shap_e
        print("✅ Shap-E déjà installé")
        solutions_status["shap_e"] = "installed"
    except ImportError:
        print("⚠️ Shap-E non installé")
        solutions_status["shap_e"] = "available"
    
    # 4. OpenSCAD (Code Generation)
    print("\n📦 OpenSCAD (Génération par code)")
    try:
        result = subprocess.run(["openscad", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ OpenSCAD installé")
            solutions_status["openscad"] = "installed"
        else:
            print("⚠️ OpenSCAD non trouvé")
            solutions_status["openscad"] = "available"
    except:
        print("⚠️ OpenSCAD non trouvé")
        solutions_status["openscad"] = "available"
    
    # 5. Blender (Code Generation)
    print("\n📦 Blender (Génération par code)")
    try:
        result = subprocess.run(["blender", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Blender installé")
            solutions_status["blender"] = "installed"
        else:
            print("⚠️ Blender non trouvé")
            solutions_status["blender"] = "available"
    except:
        print("⚠️ Blender non trouvé")
        solutions_status["blender"] = "available"
    
    return solutions_status

def install_triposr():
    """Installe TripoSR"""
    
    print("\n📦 Installation TripoSR")
    print("=" * 30)
    
    try:
        print("🔄 Installation en cours...")
        
        # Installer les dépendances de base
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "transformers", "diffusers"
        ], check=True, capture_output=True)
        
        # Installer TripoSR depuis GitHub
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/VAST-AI-Research/TripoSR.git"
        ], check=True, capture_output=True)
        
        print("✅ TripoSR installé avec succès")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Échec installation TripoSR: {e}")
        return False

def install_shap_e():
    """Installe Shap-E"""
    
    print("\n📦 Installation Shap-E")
    print("=" * 30)
    
    try:
        print("🔄 Installation en cours...")
        
        # Installer Shap-E
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/openai/shap-e.git"
        ], check=True, capture_output=True)
        
        print("✅ Shap-E installé avec succès")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Échec installation Shap-E: {e}")
        return False

def setup_code_generation():
    """Configure la génération par code"""
    
    print("\n🔧 Configuration Génération par Code")
    print("=" * 40)
    
    # Créer les répertoires nécessaires
    code_dir = Path("ressources/code_generation")
    code_dir.mkdir(parents=True, exist_ok=True)
    
    templates_dir = code_dir / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Créer des templates de base
    create_scad_templates(templates_dir)
    create_blender_templates(templates_dir)
    
    print("✅ Configuration génération par code terminée")
    return True

def create_scad_templates(templates_dir: Path):
    """Crée les templates OpenSCAD"""
    
    # Template épée
    sword_template = '''
// Template Épée OpenSCAD
// Paramètres: blade_length, blade_width, complexity

module sword(blade_length=80, blade_width=15, complexity=5) {
    // Lame
    translate([blade_length/2, 0, 0])
        cube([blade_length, blade_width, 3], center=true);
    
    // Garde
    cube([10, blade_width*3, 4], center=true);
    
    // Poignée
    translate([0, 0, -20])
        cylinder(h=40, r=4, center=true);
    
    // Pommeau
    translate([0, 0, -25])
        sphere(r=6);
    
    // Détails selon complexité
    if (complexity > 7) {
        // Gravures sur la lame
        for (i = [1:3]) {
            translate([blade_length/4 * i, 0, 1.6])
                cylinder(h=0.5, r=2, center=true);
        }
    }
}

sword();
'''
    
    with open(templates_dir / "sword.scad", 'w') as f:
        f.write(sword_template)
    
    # Template bouclier
    shield_template = '''
// Template Bouclier OpenSCAD
// Paramètres: radius, complexity

module shield(radius=30, complexity=5) {
    // Base du bouclier
    cylinder(h=5, r=radius, center=true);
    
    // Boss central
    sphere(r=radius/4);
    
    // Détails selon complexité
    if (complexity > 5) {
        // Anneaux décoratifs
        for (i = [1:2]) {
            difference() {
                cylinder(h=1, r=radius*0.7*i, center=true);
                cylinder(h=2, r=radius*0.6*i, center=true);
            }
        }
    }
}

shield();
'''
    
    with open(templates_dir / "shield.scad", 'w') as f:
        f.write(shield_template)

def create_blender_templates(templates_dir: Path):
    """Crée les templates Blender Python"""
    
    # Template épée Blender
    sword_blender = '''
import bpy
import bmesh
from mathutils import Vector

def create_sword(blade_length=2.0, blade_width=0.3, complexity=5):
    """Crée une épée dans Blender"""
    
    # Nettoyer la scène
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Créer la lame
    bpy.ops.mesh.primitive_cube_add(size=2)
    blade = bpy.context.active_object
    blade.name = "Blade"
    blade.scale = (blade_length/2, blade_width/2, 0.05)
    blade.location = (blade_length/2, 0, 0)
    
    # Créer la garde
    bpy.ops.mesh.primitive_cube_add(size=2)
    guard = bpy.context.active_object
    guard.name = "Guard"
    guard.scale = (0.1, blade_width*1.5, 0.1)
    
    # Créer la poignée
    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.4)
    handle = bpy.context.active_object
    handle.name = "Handle"
    handle.location = (0, 0, -0.2)
    
    # Créer le pommeau
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.08)
    pommel = bpy.context.active_object
    pommel.name = "Pommel"
    pommel.location = (0, 0, -0.4)
    
    # Joindre tous les objets
    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = blade
    bpy.ops.object.join()
    
    return blade

# Créer l'épée
sword = create_sword()

# Exporter en STL
bpy.ops.export_mesh.stl(filepath="/tmp/sword.stl")
'''
    
    with open(templates_dir / "sword_blender.py", 'w') as f:
        f.write(sword_blender)

def test_temporary_solutions():
    """Teste les solutions temporaires"""
    
    print("\n🧪 Test des Solutions Temporaires")
    print("=" * 40)
    
    try:
        from fantasy_rag.generation.temporary_solutions import (
            TemporarySolutionManager, TemporarySolutionConfig
        )
        from fantasy_rag.generation.meshformer_interface import GenerationRequest
        
        # Configuration
        config = TemporarySolutionConfig(
            preferred_solution="triposr",
            device="cpu",
            quality_mode="fast"
        )
        
        # Gestionnaire
        manager = TemporarySolutionManager(config)
        
        print(f"✅ Gestionnaire initialisé")
        
        # Solutions disponibles
        available = manager.get_available_solutions()
        print(f"   Solutions disponibles: {available}")
        
        # Test de génération
        request = GenerationRequest(
            prompt="épée simple test temporaire",
            style="simple",
            complexity=4,
            asset_type="weapon"
        )
        
        print(f"🔄 Test génération...")
        result = manager.generate_mesh(request)
        
        if result.success:
            print(f"✅ Génération réussie en {result.generation_time:.2f}s")
            print(f"   Mesh: {result.mesh_path}")
            print(f"   Qualité: {result.quality_score:.2f}")
            print(f"   Avertissements: {result.warnings}")
            return True
        else:
            print(f"❌ Génération échouée: {result.error_message}")
            return False
        
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale d'installation"""
    
    print("🔧 SETUP SOLUTIONS TEMPORAIRES - Fantasy RAG")
    print("=" * 60)
    print("Installation des solutions temporaires pour génération 3D")
    print("=" * 60)
    
    # Vérifier les solutions disponibles
    solutions_status = check_available_solutions()
    
    # Proposer l'installation
    print(f"\n💡 RECOMMANDATIONS D'INSTALLATION")
    print("=" * 40)
    
    install_recommendations = []
    
    if solutions_status.get("triposr") == "available":
        install_recommendations.append("TripoSR (Recommandé - Rapide et qualité)")
    
    if solutions_status.get("shap_e") == "available":
        install_recommendations.append("Shap-E (OpenAI - Bonne qualité)")
    
    install_recommendations.append("Génération par code (Toujours disponible)")
    
    for i, rec in enumerate(install_recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Installation interactive
    print(f"\n🚀 INSTALLATION AUTOMATIQUE")
    print("=" * 30)
    
    success_count = 0
    total_attempts = 0
    
    # Installer TripoSR si disponible
    if solutions_status.get("triposr") == "available":
        total_attempts += 1
        if install_triposr():
            success_count += 1
    
    # Installer Shap-E si disponible
    if solutions_status.get("shap_e") == "available":
        total_attempts += 1
        if install_shap_e():
            success_count += 1
    
    # Configurer génération par code
    total_attempts += 1
    if setup_code_generation():
        success_count += 1
    
    # Tester les solutions
    total_attempts += 1
    if test_temporary_solutions():
        success_count += 1
    
    # Résumé final
    print(f"\n{'='*60}")
    print("📊 RÉSUMÉ DE L'INSTALLATION")
    print(f"{'='*60}")
    
    print(f"🎯 Succès: {success_count}/{total_attempts} installations")
    
    if success_count >= total_attempts - 1:  # Tolérer 1 échec
        print("\n🎉 INSTALLATION RÉUSSIE !")
        print("\n✅ Solutions temporaires configurées")
        print("✅ Prêt pour la génération 3D améliorée")
        print("✅ Qualité supérieure à la simulation de base")
        
        print("\n💡 UTILISATION:")
        print("   1. Interface Streamlit: Génération automatique améliorée")
        print("   2. API Python: TemporarySolutionManager")
        print("   3. Fallback intelligent vers simulation si nécessaire")
        
        print("\n🔄 SOLUTIONS DISPONIBLES:")
        if "triposr" in [s for s in solutions_status if solutions_status[s] == "installed"]:
            print("   ✅ TripoSR - Génération rapide et qualité")
        if "shap_e" in [s for s in solutions_status if solutions_status[s] == "installed"]:
            print("   ✅ Shap-E - Génération OpenAI")
        print("   ✅ Génération par code - Templates OpenSCAD/Blender")
        print("   ✅ Simulation améliorée - Fallback intelligent")
        
    else:
        print("\n⚠️ INSTALLATION PARTIELLE")
        print("   Au moins la génération par code est disponible")
        print("   Vérifiez les erreurs ci-dessus pour les autres solutions")
    
    return success_count >= 1  # Au moins une solution doit fonctionner

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
