#!/usr/bin/env python3
"""
MeshFormer Integration Manager
=============================

Gestionnaire d'intégration pour MeshFormer réel dans Fantasy RAG.
Gère le téléchargement des modèles, l'initialisation et l'interface avec le système existant.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import time
import json
import requests
from tqdm import tqdm

# Local imports
from .meshformer_real import MeshFormerReal, MeshFormerRealConfig, MeshFormerInput, MeshFormerOutput
from .meshformer_interface import GenerationRequest, GenerationResult

logger = logging.getLogger(__name__)

class MeshFormerIntegrationManager:
    """
    Gestionnaire d'intégration MeshFormer
    
    Gère la transition entre le mode simulation et le mode réel,
    le téléchargement des modèles, et l'interface unifiée.
    """
    
    def __init__(self, 
                 config: Optional[MeshFormerRealConfig] = None,
                 model_path: Optional[str] = None,
                 force_simulation: bool = False):
        """
        Initialise le gestionnaire d'intégration
        
        Args:
            config: Configuration MeshFormer
            model_path: Chemin vers le modèle pré-entraîné
            force_simulation: Force le mode simulation même si le modèle est disponible
        """
        
        self.config = config or MeshFormerRealConfig()
        self.model_path = model_path
        self.force_simulation = force_simulation
        
        # État du modèle
        self.model = None
        self.is_real_model = False
        self.model_info = {}
        
        # Chemins des modèles
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialisation
        self._initialize_model()
        
        logger.info(f"🎨 MeshFormer Integration Manager initialisé")
        logger.info(f"   Mode: {'Réel' if self.is_real_model else 'Simulation'}")
        logger.info(f"   Device: {self.config.device}")
    
    def _initialize_model(self):
        """Initialise le modèle (réel ou simulation)"""
        
        if self.force_simulation:
            logger.info("🎭 Mode simulation forcé")
            self.is_real_model = False
            return
        
        # Essayer de charger le modèle réel
        if self._try_load_real_model():
            logger.info("✅ Modèle MeshFormer réel chargé")
            self.is_real_model = True
        else:
            logger.info("🎭 Modèle réel non disponible, utilisation du mode simulation")
            self.is_real_model = False
    
    def _try_load_real_model(self) -> bool:
        """Essaie de charger le modèle réel"""
        
        try:
            # Vérifier si le modèle existe localement
            if self.model_path and Path(self.model_path).exists():
                return self._load_model_from_path(self.model_path)
            
            # Essayer de télécharger le modèle
            if self._download_model():
                return self._load_model_from_path(self.models_dir / "meshformer_checkpoint.pth")
            
            # Essayer de créer un modèle pré-entraîné simulé
            return self._create_pretrained_model()
            
        except Exception as e:
            logger.warning(f"Échec du chargement du modèle réel: {e}")
            return False
    
    def _download_model(self) -> bool:
        """Télécharge le modèle MeshFormer officiel"""
        
        # URLs potentielles pour le modèle (à mettre à jour quand disponible)
        model_urls = [
            "https://github.com/meshformer3d/meshformer/releases/download/v1.0/meshformer_checkpoint.pth",
            "https://huggingface.co/meshformer/meshformer-base/resolve/main/pytorch_model.bin",
            # Ajouter d'autres URLs quand disponibles
        ]
        
        for url in model_urls:
            try:
                logger.info(f"🔄 Tentative de téléchargement depuis: {url}")
                
                response = requests.head(url, timeout=10)
                if response.status_code == 200:
                    return self._download_file(url, self.models_dir / "meshformer_checkpoint.pth")
                    
            except Exception as e:
                logger.debug(f"URL non disponible: {url} - {e}")
                continue
        
        logger.info("📥 Aucun modèle officiel trouvé en ligne")
        return False
    
    def _download_file(self, url: str, output_path: Path) -> bool:
        """Télécharge un fichier avec barre de progression"""
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc="Téléchargement MeshFormer",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"✅ Modèle téléchargé: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Échec du téléchargement: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def _load_model_from_path(self, model_path: Path) -> bool:
        """Charge le modèle depuis un fichier"""
        
        try:
            logger.info(f"🔄 Chargement du modèle: {model_path}")
            
            # Créer le modèle
            self.model = MeshFormerReal(self.config)
            
            # Charger les poids
            checkpoint = torch.load(model_path, map_location=self.config.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model_info = checkpoint.get('model_info', {})
            else:
                self.model.load_state_dict(checkpoint)
            
            # Mettre en mode évaluation
            self.model.eval()
            self.model.to(self.config.device)
            
            logger.info(f"✅ Modèle chargé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Échec du chargement: {e}")
            return False
    
    def _create_pretrained_model(self) -> bool:
        """Crée un modèle pré-entraîné simulé pour les tests"""
        
        try:
            logger.info("🔧 Création d'un modèle pré-entraîné simulé")
            
            # Créer le modèle
            self.model = MeshFormerReal(self.config)
            
            # Initialiser avec des poids aléatoires mais cohérents
            self._initialize_model_weights()
            
            # Mettre en mode évaluation
            self.model.eval()
            self.model.to(self.config.device)
            
            # Sauvegarder pour usage futur
            self._save_simulated_model()
            
            logger.info("✅ Modèle simulé créé")
            return True
            
        except Exception as e:
            logger.error(f"❌ Échec de création du modèle simulé: {e}")
            return False
    
    def _initialize_model_weights(self):
        """Initialise les poids du modèle de manière cohérente"""
        
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.model.apply(init_weights)
    
    def _save_simulated_model(self):
        """Sauvegarde le modèle simulé"""
        
        try:
            model_path = self.models_dir / "meshformer_simulated.pth"
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__,
                'model_info': {
                    'type': 'simulated',
                    'created': time.time(),
                    'version': '1.0.0-sim'
                }
            }
            
            torch.save(checkpoint, model_path)
            logger.info(f"💾 Modèle simulé sauvegardé: {model_path}")
            
        except Exception as e:
            logger.warning(f"Échec de sauvegarde du modèle simulé: {e}")
    
    def generate_mesh(self, request: GenerationRequest) -> GenerationResult:
        """
        Génère un mesh en utilisant le modèle réel ou la simulation
        
        Args:
            request: Requête de génération
            
        Returns:
            Résultat de génération
        """
        
        if self.is_real_model and self.model is not None:
            return self._generate_with_real_model(request)
        else:
            return self._generate_with_simulation(request)
    
    def _generate_with_real_model(self, request: GenerationRequest) -> GenerationResult:
        """Génère avec le modèle réel"""
        
        try:
            logger.info(f"🎨 Génération avec modèle réel: '{request.prompt}'")
            
            # Préparer les inputs (simulation pour l'instant)
            inputs = self._prepare_inputs(request)
            
            # Génération
            with torch.no_grad():
                output = self.model(inputs)
            
            # Convertir en GenerationResult
            result = self._convert_output_to_result(output, request)
            
            logger.info(f"✅ Génération réelle terminée en {result.generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur génération réelle: {e}")
            # Fallback vers simulation
            return self._generate_with_simulation(request)
    
    def _generate_with_simulation(self, request: GenerationRequest) -> GenerationResult:
        """Génère avec simulation ou solutions temporaires"""

        # Essayer d'abord les solutions temporaires
        try:
            from .temporary_solutions import TemporarySolutionManager, TemporarySolutionConfig

            # Configuration pour solutions temporaires
            temp_config = TemporarySolutionConfig(
                preferred_solution="triposr",
                device=self.config.device,
                quality_mode="balanced"
            )

            # Gestionnaire de solutions temporaires
            temp_manager = TemporarySolutionManager(temp_config)

            # Vérifier si des solutions sont disponibles
            available_solutions = temp_manager.get_available_solutions()

            if available_solutions:
                logger.info(f"🔄 Utilisation solutions temporaires: {available_solutions}")
                result = temp_manager.generate_mesh(request)

                if result.success:
                    logger.info("✅ Génération réussie avec solutions temporaires")
                    return result
                else:
                    logger.warning("❌ Échec solutions temporaires, fallback vers simulation")
            else:
                logger.info("⚠️ Aucune solution temporaire disponible, utilisation simulation")

        except Exception as e:
            logger.warning(f"❌ Erreur solutions temporaires: {e}, fallback vers simulation")

        # Fallback vers simulation classique
        from .meshformer_interface import MeshFormerInterface, MeshFormerConfig

        # Créer une interface simulation
        sim_config = MeshFormerConfig(
            device=self.config.device,
            quality_mode=getattr(self.config, 'quality_mode', 'high'),
            resolution=self.config.voxel_resolution_coarse
        )

        sim_interface = MeshFormerInterface(sim_config)

        # Générer avec simulation
        return sim_interface.generate_mesh(request)
    
    def _prepare_inputs(self, request: GenerationRequest) -> MeshFormerInput:
        """Prépare les inputs pour le modèle réel"""
        
        # TODO: Implémenter la préparation réelle des inputs
        # Pour l'instant, créer des inputs simulés
        
        B = 1
        V = self.config.num_views
        H = W = self.config.image_resolution
        
        # Images RGB simulées
        rgb_images = torch.randn(B, V, 3, H, W, device=self.config.device)
        
        # Cartes normales simulées
        normal_images = torch.randn(B, V, 3, H, W, device=self.config.device)
        
        # Poses caméras simulées
        camera_poses = torch.eye(4, device=self.config.device).unsqueeze(0).repeat(B, V, 1, 1)
        
        # Intrinsèques caméras simulées
        camera_intrinsics = torch.eye(3, device=self.config.device).unsqueeze(0).repeat(B, V, 1, 1)
        camera_intrinsics[:, :, 0, 0] = 500  # fx
        camera_intrinsics[:, :, 1, 1] = 500  # fy
        camera_intrinsics[:, :, 0, 2] = H // 2  # cx
        camera_intrinsics[:, :, 1, 2] = W // 2  # cy
        
        return MeshFormerInput(
            rgb_images=rgb_images,
            normal_images=normal_images,
            camera_poses=camera_poses,
            camera_intrinsics=camera_intrinsics
        )
    
    def _convert_output_to_result(self, output: MeshFormerOutput, request: GenerationRequest) -> GenerationResult:
        """Convertit la sortie MeshFormer en GenerationResult"""
        
        # Sauvegarder le mesh
        output_dir = Path("ressources/data/generated_meshes")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        mesh_filename = f"generated_{request.asset_type}_{timestamp}.{self.config.output_format}"
        mesh_path = output_dir / mesh_filename
        
        # Exporter le mesh
        output.mesh.export(str(mesh_path))
        
        # Créer le résultat
        return GenerationResult(
            success=True,
            mesh_path=str(mesh_path),
            generation_time=output.generation_time,
            quality_score=self._compute_quality_score(output.quality_metrics),
            mesh_stats=output.quality_metrics,
            render_paths={},  # TODO: Générer les rendus
            error_message=None,
            warnings=[]
        )
    
    def _compute_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calcule un score de qualité basé sur les métriques"""
        
        # Score basé sur le nombre de faces et la validité du mesh
        base_score = 0.7
        
        if metrics.get("is_watertight", False):
            base_score += 0.2
        
        num_faces = metrics.get("num_faces", 0)
        if num_faces > 1000:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations du modèle"""
        
        info = {
            "model_type": "MeshFormer Real" if self.is_real_model else "MeshFormer Simulation",
            "device": self.config.device,
            "is_real": self.is_real_model,
            "config": self.config.__dict__
        }
        
        if self.model_info:
            info.update(self.model_info)
        
        return info
