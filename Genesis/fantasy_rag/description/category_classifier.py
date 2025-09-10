#!/usr/bin/env python3
"""
Category Classifier for Fantasy Assets

This module handles automatic classification of fantasy assets into categories.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from .prompt_templates import AssetCategory

@dataclass
class ClassificationResult:
    """Result of asset classification"""
    category: AssetCategory
    confidence: float
    keywords_found: List[str]

class CategoryClassifier:
    """
    Classifier for fantasy asset categories
    """
    
    # Keywords for each category
    CATEGORY_KEYWORDS = {
        AssetCategory.WEAPON: [
            "sword", "épée", "blade", "lame", "axe", "hache", "bow", "arc",
            "staff", "bâton", "dagger", "poignard", "mace", "masse", "spear", "lance"
        ],
        AssetCategory.ARMOR: [
            "armor", "armure", "helmet", "casque", "shield", "bouclier", 
            "breastplate", "plastron", "gauntlet", "gantelet"
        ],
        AssetCategory.CHARACTER: [
            "character", "personnage", "warrior", "guerrier", "mage", "wizard",
            "knight", "chevalier", "rogue", "voleur", "human", "humain", "elf", "elfe"
        ],
        AssetCategory.CREATURE: [
            "dragon", "beast", "bête", "monster", "monstre", "demon", "démon",
            "creature", "créature", "animal"
        ],
        AssetCategory.MAGICAL_ITEM: [
            "crystal", "cristal", "orb", "orbe", "amulet", "amulette", 
            "ring", "anneau", "potion", "scroll", "parchemin"
        ]
    }
    
    def __init__(self):
        """Initialize classifier"""
        pass
    
    def classify_by_name(self, asset_name: str) -> ClassificationResult:
        """
        Classify asset by name
        
        Args:
            asset_name: Name of the asset
            
        Returns:
            ClassificationResult
        """
        name_lower = asset_name.lower()
        
        best_category = AssetCategory.UNKNOWN
        best_score = 0.0
        found_keywords = []
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = 0
            category_keywords = []
            
            for keyword in keywords:
                if keyword in name_lower:
                    score += 1
                    category_keywords.append(keyword)
            
            if score > best_score:
                best_score = score
                best_category = category
                found_keywords = category_keywords
        
        confidence = min(1.0, best_score / 3.0)  # Normalize to 0-1
        
        return ClassificationResult(
            category=best_category,
            confidence=confidence,
            keywords_found=found_keywords
        )
    
    def classify_by_description(self, description: str) -> ClassificationResult:
        """
        Classify asset by description text
        
        Args:
            description: Generated description text
            
        Returns:
            ClassificationResult
        """
        desc_lower = description.lower()
        
        category_scores = {}
        all_found_keywords = {}
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = 0
            found_keywords = []
            
            for keyword in keywords:
                count = desc_lower.count(keyword)
                if count > 0:
                    score += count
                    found_keywords.extend([keyword] * count)
            
            category_scores[category] = score
            all_found_keywords[category] = found_keywords
        
        # Find best category
        best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
        best_score = category_scores[best_category]
        
        if best_score == 0:
            best_category = AssetCategory.UNKNOWN
            confidence = 0.0
            found_keywords = []
        else:
            total_keywords = sum(category_scores.values())
            confidence = best_score / max(1, total_keywords)
            found_keywords = all_found_keywords[best_category]
        
        return ClassificationResult(
            category=best_category,
            confidence=confidence,
            keywords_found=found_keywords
        )
