#!/usr/bin/env python3
"""
Pipeline Prompt-to-Mesh pour Fantasy RAG
========================================

Pipeline complet pour convertir des prompts textuels en mesh 3D fantastiques
en utilisant MeshFormer et des techniques d'amélioration de prompts.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

from .meshformer_interface import GenerationRequest, GenerationResult
from .mesh_generator import MeshGenerator
from ..description.llm_interface import OllamaVisionInterface

logger = logging.getLogger(__name__)

@dataclass
class PromptAnalysis:
    """Analyse d'un prompt de génération"""
    
    # Éléments détectés
    asset_type: str
    style: str
    complexity_estimate: int
    
    # Mots-clés extraits
    keywords: List[str]
    descriptors: List[str]
    
    # Améliorations suggérées
    enhanced_prompt: str
    confidence_score: float
    
    # Métadonnées
    language: str
    original_prompt: str

class PromptToMeshPipeline:
    """
    Pipeline complet Prompt-to-Mesh
    
    Analyse les prompts, les améliore, et génère des mesh 3D fantastiques
    optimisés pour le système Fantasy RAG.
    """
    
    def __init__(self,
                 mesh_generator: MeshGenerator,
                 llm_interface: Optional[OllamaVisionInterface] = None):
        """
        Initialise le pipeline
        
        Args:
            mesh_generator: Générateur de mesh
            llm_interface: Interface LLM pour amélioration de prompts
        """
        
        self.mesh_generator = mesh_generator
        self.llm_interface = llm_interface
        
        # Dictionnaires de classification
        self.asset_types = {
            # Armes
            "weapon": ["épée", "sword", "lance", "spear", "arc", "bow", "dague", "dagger", 
                      "hache", "axe", "masse", "mace", "bâton", "staff", "wand", "baguette"],
            
            # Armures
            "armor": ["armure", "armor", "casque", "helmet", "bouclier", "shield", 
                     "plastron", "breastplate", "gantelet", "gauntlet", "heaume"],
            
            # Personnages
            "character": ["personnage", "character", "guerrier", "warrior", "mage", "wizard",
                         "chevalier", "knight", "archer", "rogue", "voleur", "paladin"],
            
            # Créatures
            "creature": ["dragon", "créature", "creature", "monstre", "monster", "bête", "beast",
                        "démon", "demon", "ange", "angel", "esprit", "spirit", "fantôme", "ghost"],
            
            # Objets
            "object": ["objet", "object", "artefact", "artifact", "relique", "relic", "gemme", "gem",
                      "cristal", "crystal", "potion", "livre", "book", "parchemin", "scroll"],
            
            # Architecture
            "architecture": ["tour", "tower", "château", "castle", "temple", "church", "maison", "house",
                           "pont", "bridge", "porte", "gate", "mur", "wall", "colonne", "pillar"]
        }
        
        self.style_keywords = {
            "fantastique": ["magique", "magical", "enchanté", "enchanted", "mystique", "mystic"],
            "médiéval": ["médiéval", "medieval", "ancien", "ancient", "traditionnel", "traditional"],
            "moderne": ["moderne", "modern", "contemporain", "contemporary", "actuel", "current"],
            "futuriste": ["futuriste", "futuristic", "sci-fi", "technologique", "technological"],
            "steampunk": ["steampunk", "vapeur", "steam", "mécanique", "mechanical", "engrenage"],
            "cyberpunk": ["cyberpunk", "cyber", "néon", "neon", "digital", "numérique"]
        }
        
        logger.info("🔄 Pipeline Prompt-to-Mesh initialisé")
    
    def analyze_prompt(self, prompt: str, language: str = "français") -> PromptAnalysis:
        """
        Analyse un prompt pour extraire les informations clés
        
        Args:
            prompt: Prompt à analyser
            language: Langue du prompt
            
        Returns:
            Analyse détaillée du prompt
        """
        
        logger.info(f"🔍 Analyse du prompt: '{prompt[:50]}...'")
        
        prompt_lower = prompt.lower()
        
        # Détection du type d'asset
        asset_type = self._detect_asset_type(prompt_lower)
        
        # Détection du style
        style = self._detect_style(prompt_lower)
        
        # Estimation de la complexité
        complexity = self._estimate_complexity(prompt_lower)
        
        # Extraction des mots-clés
        keywords = self._extract_keywords(prompt_lower)
        descriptors = self._extract_descriptors(prompt_lower)
        
        # Amélioration du prompt
        enhanced_prompt = self._enhance_prompt(prompt, asset_type, style, keywords)
        
        # Score de confiance
        confidence = self._calculate_confidence(asset_type, style, keywords)
        
        analysis = PromptAnalysis(
            asset_type=asset_type,
            style=style,
            complexity_estimate=complexity,
            keywords=keywords,
            descriptors=descriptors,
            enhanced_prompt=enhanced_prompt,
            confidence_score=confidence,
            language=language,
            original_prompt=prompt
        )
        
        logger.info(f"✅ Analyse terminée: {asset_type} ({style}, complexité {complexity})")
        
        return analysis
    
    def _detect_asset_type(self, prompt_lower: str) -> str:
        """Détecte le type d'asset dans le prompt"""
        
        scores = {}
        
        for asset_type, keywords in self.asset_types.items():
            score = 0
            for keyword in keywords:
                if keyword in prompt_lower:
                    score += 1
            scores[asset_type] = score
        
        # Retourner le type avec le meilleur score
        best_type = max(scores, key=scores.get)
        
        # Si aucun mot-clé trouvé, utiliser "generic"
        if scores[best_type] == 0:
            return "generic"
        
        return best_type
    
    def _detect_style(self, prompt_lower: str) -> str:
        """Détecte le style dans le prompt"""
        
        scores = {}
        
        for style, keywords in self.style_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in prompt_lower:
                    score += 1
            scores[style] = score
        
        # Retourner le style avec le meilleur score
        best_style = max(scores, key=scores.get)
        
        # Si aucun mot-clé trouvé, utiliser "fantastique" par défaut
        if scores[best_style] == 0:
            return "fantastique"
        
        return best_style
    
    def _estimate_complexity(self, prompt_lower: str) -> int:
        """Estime la complexité basée sur le prompt"""
        
        complexity = 5  # Base
        
        # Mots indiquant une complexité élevée
        high_complexity_words = [
            "détaillé", "detailed", "complexe", "complex", "orné", "ornate",
            "sculpté", "carved", "gravé", "engraved", "décoré", "decorated",
            "élaboré", "elaborate", "sophistiqué", "sophisticated"
        ]
        
        # Mots indiquant une complexité faible
        low_complexity_words = [
            "simple", "basique", "basic", "minimal", "épuré", "clean",
            "lisse", "smooth", "uni", "plain"
        ]
        
        for word in high_complexity_words:
            if word in prompt_lower:
                complexity += 1
        
        for word in low_complexity_words:
            if word in prompt_lower:
                complexity -= 1
        
        # Longueur du prompt influence la complexité
        if len(prompt_lower.split()) > 10:
            complexity += 1
        elif len(prompt_lower.split()) < 5:
            complexity -= 1
        
        return max(1, min(10, complexity))
    
    def _extract_keywords(self, prompt_lower: str) -> List[str]:
        """Extrait les mots-clés importants"""
        
        # Mots vides à ignorer
        stop_words = {
            "le", "la", "les", "un", "une", "des", "du", "de", "et", "avec", "pour",
            "dans", "sur", "par", "ce", "cette", "ces", "qui", "que", "dont",
            "the", "a", "an", "and", "with", "for", "in", "on", "by", "this", "that"
        }
        
        words = re.findall(r'\b\w+\b', prompt_lower)
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        return keywords[:10]  # Limiter à 10 mots-clés
    
    def _extract_descriptors(self, prompt_lower: str) -> List[str]:
        """Extrait les descripteurs (adjectifs)"""
        
        # Descripteurs communs pour les assets fantastiques
        fantasy_descriptors = [
            "magique", "magical", "enchanté", "enchanted", "mystique", "mystic",
            "ancien", "ancient", "légendaire", "legendary", "puissant", "powerful",
            "brillant", "bright", "sombre", "dark", "lumineux", "glowing",
            "doré", "golden", "argenté", "silver", "cristallin", "crystalline",
            "éthéré", "ethereal", "spectral", "ghostly", "divin", "divine"
        ]
        
        found_descriptors = []
        for descriptor in fantasy_descriptors:
            if descriptor in prompt_lower:
                found_descriptors.append(descriptor)
        
        return found_descriptors
    
    def _enhance_prompt(self, original_prompt: str, asset_type: str, style: str, keywords: List[str]) -> str:
        """Améliore le prompt pour une meilleure génération"""
        
        # Ajouter des détails spécifiques au type d'asset
        enhancements = {
            "weapon": "avec des détails métalliques fins et une ergonomie réaliste",
            "armor": "avec des plaques articulées et des détails de protection",
            "character": "avec des proportions anatomiques correctes et des détails vestimentaires",
            "creature": "avec une anatomie fantastique cohérente et des textures organiques",
            "object": "avec des détails fonctionnels et une finition soignée",
            "architecture": "avec des proportions architecturales et des détails structurels"
        }
        
        # Ajouter des détails de style
        style_enhancements = {
            "fantastique": "dans un style fantastique avec des éléments magiques",
            "médiéval": "dans un style médiéval authentique avec des matériaux d'époque",
            "moderne": "dans un style moderne avec des lignes épurées",
            "futuriste": "dans un style futuriste avec des éléments technologiques",
            "steampunk": "dans un style steampunk avec des mécanismes à vapeur",
            "cyberpunk": "dans un style cyberpunk avec des éléments néon et digitaux"
        }
        
        enhanced = original_prompt
        
        # Ajouter l'amélioration du type d'asset
        if asset_type in enhancements:
            enhanced += f" {enhancements[asset_type]}"
        
        # Ajouter l'amélioration du style
        if style in style_enhancements:
            enhanced += f", {style_enhancements[style]}"
        
        # Ajouter des détails de qualité
        enhanced += ", haute qualité, détails fins, rendu professionnel"
        
        return enhanced
    
    def _calculate_confidence(self, asset_type: str, style: str, keywords: List[str]) -> float:
        """Calcule un score de confiance pour l'analyse"""
        
        confidence = 0.5  # Base
        
        # Bonus si type d'asset détecté
        if asset_type != "generic":
            confidence += 0.2
        
        # Bonus si style détecté
        if style != "fantastique":  # fantastique est le défaut
            confidence += 0.1
        
        # Bonus basé sur le nombre de mots-clés
        confidence += min(0.2, len(keywords) * 0.02)
        
        return min(1.0, confidence)
    
    def generate_from_prompt(self, 
                           prompt: str, 
                           language: str = "français",
                           user_id: Optional[str] = None,
                           **kwargs) -> Tuple[PromptAnalysis, str]:
        """
        Génère un mesh à partir d'un prompt
        
        Args:
            prompt: Prompt de génération
            language: Langue du prompt
            user_id: ID utilisateur (optionnel)
            **kwargs: Paramètres additionnels
            
        Returns:
            Tuple (analyse du prompt, ID du job de génération)
        """
        
        logger.info(f"🎨 Génération à partir du prompt: '{prompt}'")
        
        # Analyser le prompt
        analysis = self.analyze_prompt(prompt, language)
        
        # Créer la requête de génération
        request = GenerationRequest(
            prompt=analysis.enhanced_prompt,
            style=analysis.style,
            complexity=analysis.complexity_estimate,
            asset_type=analysis.asset_type,
            language=language,
            user_id=user_id,
            **kwargs
        )
        
        # Lancer la génération
        job_id = self.mesh_generator.generate_mesh_async(request)
        
        logger.info(f"🚀 Job de génération lancé: {job_id}")
        logger.info(f"   Type: {analysis.asset_type}")
        logger.info(f"   Style: {analysis.style}")
        logger.info(f"   Complexité: {analysis.complexity_estimate}/10")
        logger.info(f"   Confiance: {analysis.confidence_score:.2f}")
        
        return analysis, job_id
    
    def generate_variations(self, 
                          base_prompt: str, 
                          num_variations: int = 3,
                          language: str = "français") -> List[Tuple[PromptAnalysis, str]]:
        """
        Génère plusieurs variations d'un prompt
        
        Args:
            base_prompt: Prompt de base
            num_variations: Nombre de variations
            language: Langue
            
        Returns:
            Liste des analyses et IDs de jobs
        """
        
        logger.info(f"🎭 Génération de {num_variations} variations")
        
        variations = []
        base_analysis = self.analyze_prompt(base_prompt, language)
        
        # Créer des variations en modifiant la complexité et le style
        complexity_variations = [
            max(1, base_analysis.complexity_estimate - 2),
            base_analysis.complexity_estimate,
            min(10, base_analysis.complexity_estimate + 2)
        ]
        
        for i in range(num_variations):
            # Modifier légèrement le prompt
            variation_prompt = base_prompt
            if i > 0:
                variation_prompt += f" (variation {i+1})"
            
            # Utiliser une complexité différente
            complexity = complexity_variations[i % len(complexity_variations)]
            
            request = GenerationRequest(
                prompt=variation_prompt,
                style=base_analysis.style,
                complexity=complexity,
                asset_type=base_analysis.asset_type,
                language=language
            )
            
            job_id = self.mesh_generator.generate_mesh_async(request)
            
            # Créer une analyse modifiée
            variation_analysis = PromptAnalysis(
                asset_type=base_analysis.asset_type,
                style=base_analysis.style,
                complexity_estimate=complexity,
                keywords=base_analysis.keywords,
                descriptors=base_analysis.descriptors,
                enhanced_prompt=variation_prompt,
                confidence_score=base_analysis.confidence_score,
                language=language,
                original_prompt=variation_prompt
            )
            
            variations.append((variation_analysis, job_id))
        
        logger.info(f"✅ {len(variations)} variations lancées")
        
        return variations
    
    def get_prompt_suggestions(self, partial_prompt: str, language: str = "français") -> List[str]:
        """
        Suggère des améliorations pour un prompt partiel
        
        Args:
            partial_prompt: Prompt partiel
            language: Langue
            
        Returns:
            Liste de suggestions
        """
        
        analysis = self.analyze_prompt(partial_prompt, language)
        
        suggestions = []
        
        # Suggestions basées sur le type d'asset
        if analysis.asset_type == "weapon":
            suggestions.extend([
                "épée magique avec des runes gravées",
                "lance de guerre ornée de gemmes",
                "arc elfique en bois ancien"
            ])
        elif analysis.asset_type == "armor":
            suggestions.extend([
                "armure de chevalier avec blason",
                "casque de guerre décoré",
                "bouclier magique lumineux"
            ])
        
        # Suggestions basées sur le style
        if analysis.style == "fantastique":
            suggestions.extend([
                "avec des effets magiques",
                "orné de cristaux mystiques",
                "dans un style elfique"
            ])
        
        return suggestions[:5]  # Limiter à 5 suggestions
