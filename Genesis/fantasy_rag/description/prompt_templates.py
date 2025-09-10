#!/usr/bin/env python3
"""
Prompt Templates for Fantasy Asset Description Generation

This module contains specialized prompt templates for generating detailed descriptions
of fantasy 3D assets using vision-language models.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class DescriptionLanguage(Enum):
    FRENCH = "français"
    ENGLISH = "english"

class AssetCategory(Enum):
    WEAPON = "weapon"
    ARMOR = "armor"
    CHARACTER = "character"
    CREATURE = "creature"
    ACCESSORY = "accessory"
    ARCHITECTURE = "architecture"
    MAGICAL_ITEM = "magical_item"
    UNKNOWN = "unknown"

@dataclass
class PromptContext:
    """Context information for prompt generation"""
    asset_name: str
    view_count: int = 6
    target_length: int = 200  # words
    language: DescriptionLanguage = DescriptionLanguage.FRENCH
    category_hint: Optional[AssetCategory] = None
    special_instructions: Optional[str] = None

class FantasyPromptTemplates:
    """
    Collection of prompt templates for fantasy asset description generation
    """
    
    # Base system prompts
    SYSTEM_PROMPTS = {
        DescriptionLanguage.FRENCH: """Tu es un expert en objets fantastiques et médiévaux. 
Tu analyses des modèles 3D d'assets fantastiques à partir de vues multiples haute résolution.
Ton rôle est de créer des descriptions détaillées, immersives et précises qui capturent 
l'essence magique et l'artisanat de chaque objet.

Concentre-toi sur :
- Les détails visuels observables
- Les matériaux et textures apparents  
- L'artisanat et les techniques de fabrication
- Les éléments décoratifs et symboliques
- L'usage probable et la fonction
- L'atmosphère et l'esthétique fantastique""",

        DescriptionLanguage.ENGLISH: """You are an expert in fantasy and medieval objects.
You analyze 3D fantasy asset models from multiple high-resolution views.
Your role is to create detailed, immersive and accurate descriptions that capture
the magical essence and craftsmanship of each object.

Focus on:
- Observable visual details
- Apparent materials and textures
- Craftsmanship and manufacturing techniques  
- Decorative and symbolic elements
- Probable usage and function
- Fantasy atmosphere and aesthetics"""
    }
    
    # Multi-view analysis prompts
    MULTIVIEW_PROMPTS = {
        DescriptionLanguage.FRENCH: """Analyse cet objet fantastique à partir de ces {view_count} vues haute résolution :

🔍 **ANALYSE VISUELLE DÉTAILLÉE**
Examine attentivement chaque vue pour identifier :
- **Forme générale** : silhouette, proportions, symétrie
- **Matériaux** : métal, bois, cuir, pierre, cristal, etc.
- **Textures** : lisse, rugueuse, gravée, polie, patinée
- **Détails décoratifs** : ornements, gravures, incrustations, gemmes
- **Éléments fonctionnels** : poignées, lames, mécanismes, attaches
- **Signes d'usure** : éraflures, patine, traces d'utilisation

🎨 **DESCRIPTION CRÉATIVE** ({target_length} mots)
Rédigez une description immersive qui :
1. **Présente l'objet** (nom, type, fonction principale)
2. **Décrit l'apparence** (forme, taille, couleurs dominantes)
3. **Détaille les matériaux** (composition, qualité, finition)
4. **Explore l'artisanat** (techniques, style, époque)
5. **Évoque l'usage** (fonction, maniement, efficacité)
6. **Capture l'atmosphère** (mystique, guerrière, noble, etc.)

Style : descriptif, immersif, précis, évocateur du fantastique médiéval.""",

        DescriptionLanguage.ENGLISH: """Analyze this fantasy object from these {view_count} high-resolution views:

🔍 **DETAILED VISUAL ANALYSIS**
Carefully examine each view to identify:
- **General shape**: silhouette, proportions, symmetry
- **Materials**: metal, wood, leather, stone, crystal, etc.
- **Textures**: smooth, rough, engraved, polished, weathered
- **Decorative details**: ornaments, engravings, inlays, gems
- **Functional elements**: handles, blades, mechanisms, attachments
- **Signs of wear**: scratches, patina, usage marks

🎨 **CREATIVE DESCRIPTION** ({target_length} words)
Write an immersive description that:
1. **Introduces the object** (name, type, main function)
2. **Describes appearance** (shape, size, dominant colors)
3. **Details materials** (composition, quality, finish)
4. **Explores craftsmanship** (techniques, style, era)
5. **Evokes usage** (function, handling, effectiveness)
6. **Captures atmosphere** (mystical, warrior, noble, etc.)

Style: descriptive, immersive, precise, evocative of medieval fantasy."""
    }
    
    # Category-specific prompts
    CATEGORY_PROMPTS = {
        AssetCategory.WEAPON: {
            DescriptionLanguage.FRENCH: """Cet objet est une **ARME FANTASTIQUE**. 
Concentrez-vous particulièrement sur :
- Le type d'arme (épée, hache, arc, bâton, etc.)
- La lame/partie offensive (forme, tranchant, pointe)
- La poignée/manche (ergonomie, matériaux, protection)
- L'équilibre et la maniabilité apparente
- Les enchantements ou runes visibles
- L'efficacité au combat suggérée""",
            
            DescriptionLanguage.ENGLISH: """This object is a **FANTASY WEAPON**.
Focus particularly on:
- Weapon type (sword, axe, bow, staff, etc.)
- Blade/offensive part (shape, edge, point)
- Handle/grip (ergonomics, materials, protection)
- Apparent balance and maneuverability
- Visible enchantments or runes
- Suggested combat effectiveness"""
        },
        
        AssetCategory.CHARACTER: {
            DescriptionLanguage.FRENCH: """Cet objet représente un **PERSONNAGE FANTASTIQUE**.
Concentrez-vous particulièrement sur :
- La race/espèce (humain, elfe, nain, orc, etc.)
- La posture et l'expression
- Les vêtements et armures portés
- Les accessoires et armes équipés
- Les traits distinctifs (cicatrices, tatouages, etc.)
- La classe/profession suggérée (guerrier, mage, etc.)""",
            
            DescriptionLanguage.ENGLISH: """This object represents a **FANTASY CHARACTER**.
Focus particularly on:
- Race/species (human, elf, dwarf, orc, etc.)
- Posture and expression
- Clothing and armor worn
- Accessories and weapons equipped
- Distinctive features (scars, tattoos, etc.)
- Suggested class/profession (warrior, mage, etc.)"""
        }
    }
    
    # Quality and length instructions
    QUALITY_INSTRUCTIONS = {
        DescriptionLanguage.FRENCH: """
📏 **CONTRAINTES DE LONGUEUR STRICTES** :
- IMPÉRATIF : Exactement {target_length} mots (±10 mots maximum)
- Comptez vos mots pendant que vous écrivez
- Arrêtez-vous dès que vous atteignez {target_length} mots
- NE DÉPASSEZ JAMAIS {target_length} + 10 mots

📝 **CONTRAINTES DE QUALITÉ** :
- Vocabulaire riche et varié
- Phrases fluides et bien construites
- Éviter les répétitions
- Ton immersif et captivant
- Précision technique sans jargon excessif

🚫 **À ÉVITER ABSOLUMENT** :
- Descriptions génériques ou vagues
- Répétitions de mots ou d'idées
- Spéculations sur l'histoire ou l'origine
- Comparaisons avec des œuvres connues
- Informations non observables dans les images
- Dépassement de la limite de {target_length} mots""",

        DescriptionLanguage.ENGLISH: """
📏 **STRICT LENGTH CONSTRAINTS**:
- MANDATORY: Exactly {target_length} words (±10 words maximum)
- Count your words as you write
- Stop as soon as you reach {target_length} words
- NEVER EXCEED {target_length} + 10 words

📝 **QUALITY CONSTRAINTS**:
- Rich and varied vocabulary
- Fluid and well-constructed sentences
- Avoid repetitions
- Immersive and captivating tone
- Technical precision without excessive jargon

🚫 **ABSOLUTELY AVOID**:
- Generic or vague descriptions
- Word or idea repetitions
- Speculation about history or origin
- Comparisons with known works
- Information not observable in images"""
    }
    
    def __init__(self):
        """Initialize the prompt templates"""
        pass
    
    def generate_system_prompt(self, language: DescriptionLanguage) -> str:
        """Generate system prompt for the given language"""
        return self.SYSTEM_PROMPTS[language]
    
    def generate_analysis_prompt(self, context: PromptContext) -> str:
        """Generate the main analysis prompt"""

        # Start with strict length reminder
        length_reminder = {
            DescriptionLanguage.FRENCH: f"🚨 RAPPEL CRITIQUE : Votre réponse doit faire EXACTEMENT {context.target_length} mots (±10 maximum). Comptez vos mots !",
            DescriptionLanguage.ENGLISH: f"🚨 CRITICAL REMINDER: Your response must be EXACTLY {context.target_length} words (±10 maximum). Count your words!"
        }

        base_prompt = self.MULTIVIEW_PROMPTS[context.language].format(
            view_count=context.view_count,
            target_length=context.target_length
        )

        # Add category-specific instructions if available
        if context.category_hint and context.category_hint in self.CATEGORY_PROMPTS:
            category_prompt = self.CATEGORY_PROMPTS[context.category_hint][context.language]
            base_prompt = f"{category_prompt}\n\n{base_prompt}"

        # Add quality instructions
        quality_prompt = self.QUALITY_INSTRUCTIONS[context.language].format(
            target_length=context.target_length
        )

        # Add special instructions if provided
        if context.special_instructions:
            base_prompt = f"{base_prompt}\n\n🎯 **INSTRUCTIONS SPÉCIALES** :\n{context.special_instructions}"

        return f"{length_reminder[context.language]}\n\n{base_prompt}\n\n{quality_prompt}"
    
    def generate_complete_prompt(self, context: PromptContext) -> Dict[str, str]:
        """Generate complete prompt with system and user parts"""
        return {
            "system": self.generate_system_prompt(context.language),
            "user": self.generate_analysis_prompt(context)
        }
    
    @staticmethod
    def get_supported_languages() -> List[DescriptionLanguage]:
        """Get list of supported languages"""
        return list(DescriptionLanguage)
    
    @staticmethod
    def get_supported_categories() -> List[AssetCategory]:
        """Get list of supported asset categories"""
        return list(AssetCategory)
