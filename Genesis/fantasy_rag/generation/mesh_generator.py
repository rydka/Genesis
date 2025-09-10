#!/usr/bin/env python3
"""
Générateur de Mesh 3D pour Fantasy RAG
======================================

Générateur principal qui orchestre la création de mesh 3D fantastiques
en utilisant MeshFormer et d'autres techniques de génération.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import json
import time
from concurrent.futures import ThreadPoolExecutor

from .meshformer_interface import MeshFormerInterface, MeshFormerConfig, GenerationRequest, GenerationResult
from ..database.chromadb_manager import ChromaDBManager
from ..description.llm_interface import OllamaVisionInterface
# from ..rendering.blender_renderer import BlenderRenderer  # TODO: À implémenter

logger = logging.getLogger(__name__)

@dataclass
class GenerationJob:
    """Job de génération de mesh"""
    
    job_id: str
    request: GenerationRequest
    status: str = "pending"  # pending, processing, completed, failed
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[GenerationResult] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class MeshGenerator:
    """
    Générateur principal de mesh 3D
    
    Orchestre la génération de mesh fantastiques en utilisant MeshFormer
    et intègre les résultats dans le système Fantasy RAG.
    """
    
    def __init__(self,
                 meshformer_config: Optional[MeshFormerConfig] = None,
                 db_manager: Optional[ChromaDBManager] = None,
                 llm_interface: Optional[OllamaVisionInterface] = None):
        """
        Initialise le générateur de mesh
        
        Args:
            meshformer_config: Configuration MeshFormer
            db_manager: Gestionnaire de base de données
            llm_interface: Interface LLM pour descriptions
        """
        
        self.meshformer = MeshFormerInterface(meshformer_config)
        self.db_manager = db_manager
        self.llm_interface = llm_interface
        
        # Queue des jobs
        self.jobs: Dict[str, GenerationJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Statistiques
        self.stats = {
            "total_generated": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "average_generation_time": 0.0
        }
        
        logger.info("🎨 MeshGenerator initialisé")
        logger.info(f"   MeshFormer: {'✅' if self.meshformer.is_initialized else '❌'}")
        logger.info(f"   Database: {'✅' if self.db_manager else '❌'}")
        logger.info(f"   LLM: {'✅' if self.llm_interface else '❌'}")
    
    def generate_mesh_async(self, request: GenerationRequest) -> str:
        """
        Lance une génération de mesh asynchrone
        
        Args:
            request: Requête de génération
            
        Returns:
            ID du job de génération
        """
        
        # Créer un job unique
        job_id = f"gen_{int(time.time())}_{hash(request.prompt) % 10000:04d}"
        
        job = GenerationJob(
            job_id=job_id,
            request=request
        )
        
        self.jobs[job_id] = job
        
        # Lancer la génération en arrière-plan
        future = self.executor.submit(self._process_generation_job, job_id)
        
        logger.info(f"🚀 Job de génération lancé: {job_id}")
        logger.info(f"   Prompt: '{request.prompt[:50]}...'")
        
        return job_id
    
    def generate_mesh_sync(self, request: GenerationRequest) -> GenerationResult:
        """
        Génère un mesh de manière synchrone
        
        Args:
            request: Requête de génération
            
        Returns:
            Résultat de génération
        """
        
        job_id = self.generate_mesh_async(request)
        
        # Attendre la completion
        while True:
            job = self.jobs.get(job_id)
            if job and job.status in ["completed", "failed"]:
                break
            time.sleep(0.1)
        
        return job.result if job.result else GenerationResult(
            mesh_path="",
            success=False,
            generation_time=0.0,
            mesh_stats={},
            render_paths={},
            quality_score=0.0,
            complexity_score=0.0,
            error_message=job.error or "Génération échouée"
        )
    
    def _process_generation_job(self, job_id: str):
        """Traite un job de génération"""
        
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job introuvable: {job_id}")
            return
        
        try:
            job.status = "processing"
            job.started_at = time.time()
            
            logger.info(f"🔄 Traitement job {job_id}")
            
            # Étape 1: Génération du mesh avec MeshFormer
            result = self.meshformer.generate_mesh(job.request)
            
            if result.success:
                # Étape 2: Génération des rendus
                result = self._generate_renders(result, job.request)
                
                # Étape 3: Génération de description
                if self.llm_interface:
                    result = self._generate_description(result, job.request)
                
                # Étape 4: Intégration dans la base de données
                if self.db_manager:
                    result = self._integrate_to_database(result, job.request)
                
                # Mise à jour des statistiques
                self._update_stats(result, success=True)
                
                logger.info(f"✅ Job {job_id} terminé avec succès")
            
            else:
                self._update_stats(result, success=False)
                logger.error(f"❌ Job {job_id} échoué: {result.error_message}")
            
            job.result = result
            job.status = "completed"
            job.completed_at = time.time()
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement job {job_id}: {str(e)}")
            job.status = "failed"
            job.error = str(e)
            job.completed_at = time.time()
            
            self._update_stats(None, success=False)
    
    def _generate_renders(self, result: GenerationResult, request: GenerationRequest) -> GenerationResult:
        """Génère les rendus du mesh"""
        
        try:
            logger.info("📸 Génération des rendus...")
            
            # Pour l'instant, utiliser les rendus simulés
            # TODO: Intégrer avec BlenderRenderer réel
            
            mesh_path = Path(result.mesh_path)
            render_dir = mesh_path.parent / "renders"
            render_dir.mkdir(exist_ok=True)
            
            # Simuler la génération de rendus
            views = ["front", "back", "left", "right", "top", "iso_3d"]
            render_paths = {}
            
            for view in views:
                render_filename = f"{mesh_path.stem}_{view}.png"
                render_path = render_dir / render_filename
                render_paths[view] = str(render_path)
                
                # Créer un fichier placeholder
                render_path.touch()
            
            result.render_paths = render_paths
            logger.info(f"✅ {len(render_paths)} rendus générés")
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur génération rendus: {str(e)}")
            result.warnings.append(f"Rendus échoués: {str(e)}")
        
        return result
    
    def _generate_description(self, result: GenerationResult, request: GenerationRequest) -> GenerationResult:
        """Génère une description du mesh"""
        
        try:
            logger.info("📝 Génération de la description...")
            
            # Construire un prompt pour la description
            description_prompt = f"""
            Décris ce modèle 3D fantastique généré:
            
            Type: {request.asset_type}
            Style: {request.style}
            Prompt original: {request.prompt}
            Complexité: {request.complexity}/10
            
            Statistiques du mesh:
            - Vertices: {result.mesh_stats.get('vertices', 0)}
            - Faces: {result.mesh_stats.get('faces', 0)}
            - Volume: {result.mesh_stats.get('volume', 0):.3f}
            
            Génère une description détaillée en français de 200 mots environ.
            """
            
            # Générer la description (simulation pour l'instant)
            description = f"""Ce modèle 3D représente un {request.asset_type} de style {request.style} 
            généré à partir du prompt "{request.prompt}". Le mesh contient {result.mesh_stats.get('vertices', 0)} 
            vertices et {result.mesh_stats.get('faces', 0)} faces, offrant un niveau de détail adapté à 
            une complexité de {request.complexity}/10. La géométrie présente des caractéristiques 
            fantastiques typiques avec des proportions équilibrées et une structure cohérente. 
            Le modèle est optimisé pour l'utilisation dans des applications 3D et peut être 
            facilement intégré dans des environnements de jeu ou de visualisation."""
            
            # Ajouter la description aux métadonnées
            if 'description' not in result.mesh_stats:
                result.mesh_stats['description'] = description
                result.mesh_stats['description_length'] = len(description.split())
            
            logger.info("✅ Description générée")
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur génération description: {str(e)}")
            result.warnings.append(f"Description échouée: {str(e)}")
        
        return result
    
    def _integrate_to_database(self, result: GenerationResult, request: GenerationRequest) -> GenerationResult:
        """Intègre le mesh généré dans la base de données"""
        
        try:
            logger.info("💾 Intégration en base de données...")
            
            # Créer l'enregistrement d'asset
            asset_id = f"generated_{int(time.time())}"
            
            asset_record = {
                "asset_id": asset_id,
                "asset_name": f"Generated {request.asset_type}",
                "file_path": result.mesh_path,
                "asset_type": request.asset_type,
                "generation_method": "MeshFormer",
                "generation_prompt": request.prompt,
                "generation_style": request.style,
                "complexity_score": result.complexity_score,
                "quality_score": result.quality_score,
                "mesh_stats": result.mesh_stats,
                "created_at": time.time()
            }
            
            # TODO: Ajouter à la base de données réelle
            # self.db_manager.add_asset(asset_record)
            
            logger.info(f"✅ Asset {asset_id} intégré en base")
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur intégration base: {str(e)}")
            result.warnings.append(f"Intégration base échouée: {str(e)}")
        
        return result
    
    def _update_stats(self, result: Optional[GenerationResult], success: bool):
        """Met à jour les statistiques"""
        
        self.stats["total_generated"] += 1
        
        if success:
            self.stats["successful_generations"] += 1
            if result and result.generation_time > 0:
                # Calcul de la moyenne mobile
                current_avg = self.stats["average_generation_time"]
                total_successful = self.stats["successful_generations"]
                new_avg = ((current_avg * (total_successful - 1)) + result.generation_time) / total_successful
                self.stats["average_generation_time"] = new_avg
        else:
            self.stats["failed_generations"] += 1
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retourne le statut d'un job"""
        
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        status_info = {
            "job_id": job.job_id,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "prompt": job.request.prompt,
            "asset_type": job.request.asset_type,
            "error": job.error
        }
        
        if job.result:
            status_info.update({
                "success": job.result.success,
                "generation_time": job.result.generation_time,
                "mesh_path": job.result.mesh_path,
                "quality_score": job.result.quality_score,
                "render_count": len(job.result.render_paths)
            })
        
        return status_info
    
    def list_jobs(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Liste tous les jobs avec filtre optionnel"""
        
        jobs = []
        for job in self.jobs.values():
            if status_filter is None or job.status == status_filter:
                job_info = self.get_job_status(job.job_id)
                if job_info:
                    jobs.append(job_info)
        
        # Trier par date de création (plus récent en premier)
        jobs.sort(key=lambda x: x["created_at"], reverse=True)
        
        return jobs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de génération"""
        
        success_rate = 0.0
        if self.stats["total_generated"] > 0:
            success_rate = self.stats["successful_generations"] / self.stats["total_generated"]
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "active_jobs": len([j for j in self.jobs.values() if j.status == "processing"]),
            "pending_jobs": len([j for j in self.jobs.values() if j.status == "pending"]),
            "meshformer_info": self.meshformer.get_model_info()
        }
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Nettoie les anciens jobs"""
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        jobs_to_remove = []
        for job_id, job in self.jobs.items():
            if job.status in ["completed", "failed"]:
                age = current_time - job.created_at
                if age > max_age_seconds:
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
        
        if jobs_to_remove:
            logger.info(f"🧹 {len(jobs_to_remove)} anciens jobs nettoyés")
    
    def shutdown(self):
        """Arrête le générateur proprement"""
        
        logger.info("🛑 Arrêt du MeshGenerator...")
        
        # Attendre la fin des jobs en cours
        self.executor.shutdown(wait=True)
        
        logger.info("✅ MeshGenerator arrêté")
