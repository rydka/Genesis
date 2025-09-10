#!/usr/bin/env python3
"""
Générateur de Concepts Fantastiques
===================================

Génère automatiquement des concepts d'assets fantastiques innovants
en combinant des éléments existants et en créant de nouvelles idées.
"""

import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ConceptTemplate:
    """Template pour la génération de concepts"""
    
    name: str
    category: str
    base_elements: List[str]
    modifiers: List[str]
    styles: List[str]
    rarity: str  # common, uncommon, rare, epic, legendary
    
@dataclass
class GeneratedConcept:
    """Concept généré"""
    
    name: str
    description: str
    category: str
    style: str
    complexity: int
    rarity: str
    elements: List[str]
    prompt: str
    metadata: Dict[str, Any]

class ConceptGenerator:
    """
    Générateur de concepts fantastiques
    
    Crée automatiquement des idées d'assets 3D fantastiques en combinant
    des éléments, styles et modificateurs selon des règles créatives.
    """
    
    def __init__(self):
        """Initialise le générateur de concepts"""
        
        self.templates = self._load_concept_templates()
        self.element_library = self._load_element_library()
        self.style_library = self._load_style_library()
        
        logger.info("🎭 ConceptGenerator initialisé")
        logger.info(f"   Templates: {len(self.templates)}")
        logger.info(f"   Éléments: {len(self.element_library)}")
        logger.info(f"   Styles: {len(self.style_library)}")
    
    def _load_concept_templates(self) -> Dict[str, List[ConceptTemplate]]:
        """Charge les templates de concepts"""
        
        templates = {
            "weapon": [
                ConceptTemplate(
                    name="Épée Élémentaire",
                    category="weapon",
                    base_elements=["lame", "garde", "pommeau"],
                    modifiers=["feu", "glace", "foudre", "terre", "air"],
                    styles=["elfique", "nain", "humain", "démoniaque", "angélique"],
                    rarity="uncommon"
                ),
                ConceptTemplate(
                    name="Arme Hybride",
                    category="weapon",
                    base_elements=["épée", "lance", "hache", "masse"],
                    modifiers=["transformable", "magique", "maudit", "béni"],
                    styles=["steampunk", "cristallin", "organique", "mécanique"],
                    rarity="rare"
                ),
                ConceptTemplate(
                    name="Artefact Légendaire",
                    category="weapon",
                    base_elements=["relique", "fragment", "essence"],
                    modifiers=["ancien", "perdu", "interdit", "divin"],
                    styles=["mystique", "éthéré", "cosmique", "temporel"],
                    rarity="legendary"
                )
            ],
            
            "armor": [
                ConceptTemplate(
                    name="Armure Adaptative",
                    category="armor",
                    base_elements=["plastron", "épaulières", "casque", "gantelet"],
                    modifiers=["vivante", "cristalline", "énergétique", "spectrale"],
                    styles=["bio-organique", "technologique", "magique", "élémentaire"],
                    rarity="rare"
                ),
                ConceptTemplate(
                    name="Protection Rituelle",
                    category="armor",
                    base_elements=["amulette", "tatouage", "aura", "bouclier"],
                    modifiers=["protecteur", "régénérant", "invisible", "réfléchissant"],
                    styles=["tribal", "runique", "céleste", "infernal"],
                    rarity="epic"
                )
            ],
            
            "creature": [
                ConceptTemplate(
                    name="Hybride Élémentaire",
                    category="creature",
                    base_elements=["corps", "ailes", "queue", "cornes"],
                    modifiers=["feu", "eau", "terre", "air", "lumière", "ombre"],
                    styles=["draconique", "félin", "aviaire", "reptilien"],
                    rarity="uncommon"
                ),
                ConceptTemplate(
                    name="Entité Mystique",
                    category="creature",
                    base_elements=["essence", "forme", "manifestation"],
                    modifiers=["changeante", "éthérée", "temporelle", "dimensionnelle"],
                    styles=["spectral", "cristallin", "énergétique", "cosmique"],
                    rarity="legendary"
                )
            ],
            
            "object": [
                ConceptTemplate(
                    name="Artefact Utilitaire",
                    category="object",
                    base_elements=["cristal", "orbe", "bâton", "livre"],
                    modifiers=["stockage", "communication", "divination", "transformation"],
                    styles=["ancien", "moderne", "alien", "magique"],
                    rarity="common"
                ),
                ConceptTemplate(
                    name="Relique de Pouvoir",
                    category="object",
                    base_elements=["gemme", "couronne", "sceptre", "amulette"],
                    modifiers=["contrôle", "amplification", "canalisation", "protection"],
                    styles=["royal", "démoniaque", "angélique", "primordial"],
                    rarity="epic"
                )
            ]
        }
        
        return templates
    
    def _load_element_library(self) -> Dict[str, List[str]]:
        """Charge la bibliothèque d'éléments"""
        
        return {
            "materials": [
                "mithril", "adamantium", "cristal", "obsidienne", "ivoire",
                "os de dragon", "bois elfique", "métal stellaire", "pierre de lune",
                "écaille de serpent", "plume de phénix", "corne de licorne"
            ],
            
            "colors": [
                "doré", "argenté", "rouge sang", "bleu saphir", "vert émeraude",
                "violet améthyste", "noir obsidien", "blanc nacré", "cuivré",
                "iridescent", "luminescent", "opalescent"
            ],
            
            "effects": [
                "luisant", "fumant", "étincelant", "vibrant", "pulsant",
                "flottant", "tournoyant", "ondulant", "scintillant",
                "rayonnant", "crépitant", "murmurant"
            ],
            
            "patterns": [
                "runes gravées", "motifs celtiques", "spirales", "fractales",
                "entrelacs", "géométrie sacrée", "constellations", "veines",
                "écailles", "plumes", "flammes", "vagues"
            ],
            
            "powers": [
                "téléportation", "invisibilité", "régénération", "vol",
                "contrôle élémentaire", "vision nocturne", "télépathie",
                "manipulation temporelle", "bouclier énergétique", "métamorphose"
            ]
        }
    
    def _load_style_library(self) -> Dict[str, Dict[str, Any]]:
        """Charge la bibliothèque de styles"""
        
        return {
            "elfique": {
                "characteristics": ["élégant", "organique", "fluide", "naturel"],
                "materials": ["bois", "cristal", "mithril", "pierre de lune"],
                "colors": ["vert", "doré", "argenté", "blanc"],
                "complexity": 7
            },
            
            "nain": {
                "characteristics": ["robuste", "géométrique", "fonctionnel", "orné"],
                "materials": ["fer", "acier", "pierre", "gemmes"],
                "colors": ["brun", "gris", "rouge", "doré"],
                "complexity": 6
            },
            
            "démoniaque": {
                "characteristics": ["menaçant", "asymétrique", "corrompu", "chaotique"],
                "materials": ["obsidienne", "os", "métal noir", "soufre"],
                "colors": ["rouge", "noir", "violet", "vert acide"],
                "complexity": 8
            },
            
            "angélique": {
                "characteristics": ["pur", "symétrique", "lumineux", "harmonieux"],
                "materials": ["or", "cristal", "plumes", "lumière"],
                "colors": ["blanc", "doré", "bleu ciel", "argenté"],
                "complexity": 8
            },
            
            "steampunk": {
                "characteristics": ["mécanique", "industriel", "cuivre", "engrenages"],
                "materials": ["cuivre", "laiton", "acier", "verre"],
                "colors": ["cuivre", "laiton", "brun", "vert-de-gris"],
                "complexity": 9
            },
            
            "cristallin": {
                "characteristics": ["transparent", "facetté", "réfractant", "géométrique"],
                "materials": ["cristal", "gemmes", "verre", "quartz"],
                "colors": ["transparent", "iridescent", "bleu", "violet"],
                "complexity": 5
            }
        }
    
    def generate_concept(self, 
                        category: Optional[str] = None,
                        rarity: Optional[str] = None,
                        style: Optional[str] = None) -> GeneratedConcept:
        """
        Génère un nouveau concept
        
        Args:
            category: Catégorie spécifique (optionnel)
            rarity: Rareté spécifique (optionnel)
            style: Style spécifique (optionnel)
            
        Returns:
            Concept généré
        """
        
        # Sélectionner une catégorie
        if category is None:
            category = random.choice(list(self.templates.keys()))
        
        # Sélectionner un template
        available_templates = self.templates.get(category, [])
        if not available_templates:
            raise ValueError(f"Aucun template disponible pour la catégorie: {category}")
        
        # Filtrer par rareté si spécifiée
        if rarity:
            filtered_templates = [t for t in available_templates if t.rarity == rarity]
            if filtered_templates:
                available_templates = filtered_templates
        
        template = random.choice(available_templates)
        
        # Sélectionner un style
        if style is None:
            style = random.choice(template.styles)
        
        # Générer le concept
        concept = self._generate_from_template(template, style)
        
        logger.info(f"🎨 Concept généré: {concept.name} ({concept.category}, {concept.rarity})")
        
        return concept
    
    def _generate_from_template(self, template: ConceptTemplate, style: str) -> GeneratedConcept:
        """Génère un concept à partir d'un template"""
        
        # Sélectionner des éléments de base
        base_elements = random.sample(template.base_elements, 
                                    min(len(template.base_elements), 
                                        random.randint(1, 3)))
        
        # Sélectionner des modificateurs
        modifiers = random.sample(template.modifiers,
                                min(len(template.modifiers),
                                    random.randint(1, 2)))
        
        # Sélectionner des éléments additionnels
        materials = random.sample(self.element_library["materials"], 
                                random.randint(1, 2))
        colors = random.sample(self.element_library["colors"],
                             random.randint(1, 2))
        effects = random.sample(self.element_library["effects"],
                              random.randint(0, 2))
        patterns = random.sample(self.element_library["patterns"],
                               random.randint(0, 1))
        
        # Générer le nom
        name = self._generate_name(template, modifiers, materials, style)
        
        # Générer la description
        description = self._generate_description(
            template, base_elements, modifiers, materials, colors, effects, patterns, style
        )
        
        # Générer le prompt
        prompt = self._generate_prompt(name, description, style, template.category)
        
        # Calculer la complexité
        style_info = self.style_library.get(style, {"complexity": 5})
        base_complexity = style_info.get("complexity", 5)
        rarity_bonus = {"common": 0, "uncommon": 1, "rare": 2, "epic": 3, "legendary": 4}
        complexity = min(10, base_complexity + rarity_bonus.get(template.rarity, 0))
        
        # Créer les métadonnées
        metadata = {
            "template": template.name,
            "generation_method": "procedural",
            "elements_used": {
                "base": base_elements,
                "modifiers": modifiers,
                "materials": materials,
                "colors": colors,
                "effects": effects,
                "patterns": patterns
            },
            "style_info": style_info
        }
        
        return GeneratedConcept(
            name=name,
            description=description,
            category=template.category,
            style=style,
            complexity=complexity,
            rarity=template.rarity,
            elements=base_elements + modifiers + materials,
            prompt=prompt,
            metadata=metadata
        )
    
    def _generate_name(self, template: ConceptTemplate, modifiers: List[str], 
                      materials: List[str], style: str) -> str:
        """Génère un nom pour le concept"""
        
        # Préfixes basés sur le style
        style_prefixes = {
            "elfique": ["Lame", "Éclat", "Chant", "Souffle"],
            "nain": ["Marteau", "Forge", "Roc", "Fer"],
            "démoniaque": ["Fléau", "Corruption", "Tourment", "Chaos"],
            "angélique": ["Lumière", "Grâce", "Bénédiction", "Pureté"],
            "steampunk": ["Mécanisme", "Vapeur", "Engrenage", "Piston"],
            "cristallin": ["Prisme", "Facette", "Résonance", "Éclat"]
        }
        
        # Suffixes basés sur les modificateurs
        modifier_suffixes = {
            "feu": "Ardent", "glace": "Glacial", "foudre": "Foudroyant",
            "magique": "Enchanté", "maudit": "Maudit", "béni": "Béni",
            "ancien": "Antique", "perdu": "Oublié", "divin": "Divin"
        }
        
        prefix = random.choice(style_prefixes.get(style, ["Artefact"]))
        
        if modifiers:
            modifier = random.choice(modifiers)
            suffix = modifier_suffixes.get(modifier, modifier.title())
        else:
            suffix = random.choice(materials).title()
        
        return f"{prefix} {suffix}"
    
    def _generate_description(self, template: ConceptTemplate, base_elements: List[str],
                            modifiers: List[str], materials: List[str], colors: List[str],
                            effects: List[str], patterns: List[str], style: str) -> str:
        """Génère une description détaillée"""
        
        description_parts = []
        
        # Introduction
        category_intros = {
            "weapon": "Cette arme fantastique",
            "armor": "Cette protection magique",
            "creature": "Cette créature mystique",
            "object": "Cet artefact précieux"
        }
        
        intro = category_intros.get(template.category, "Cet objet")
        description_parts.append(intro)
        
        # Matériaux et couleurs
        if materials and colors:
            material = random.choice(materials)
            color = random.choice(colors)
            description_parts.append(f"est forgée en {material} {color}")
        
        # Éléments de base
        if base_elements:
            elements_desc = ", ".join(base_elements)
            description_parts.append(f"et présente {elements_desc}")
        
        # Modificateurs
        if modifiers:
            modifier_desc = " et ".join(modifiers)
            description_parts.append(f"avec des propriétés {modifier_desc}")
        
        # Motifs
        if patterns:
            pattern = random.choice(patterns)
            description_parts.append(f"ornée de {pattern}")
        
        # Effets
        if effects:
            effect = random.choice(effects)
            description_parts.append(f"Elle est {effect}")
        
        # Style
        style_descriptions = {
            "elfique": "dans le plus pur style elfique, alliant grâce et efficacité",
            "nain": "selon la tradition naine, privilégiant robustesse et fonctionnalité",
            "démoniaque": "d'inspiration démoniaque, dégageant une aura menaçante",
            "angélique": "d'essence angélique, rayonnant de pureté et de lumière",
            "steampunk": "de facture steampunk, mêlant mécanique et esthétique",
            "cristallin": "de nature cristalline, jouant avec la lumière et la transparence"
        }
        
        if style in style_descriptions:
            description_parts.append(style_descriptions[style])
        
        # Rareté
        rarity_endings = {
            "common": "Un objet relativement courant mais néanmoins remarquable.",
            "uncommon": "Un artefact peu commun aux propriétés intéressantes.",
            "rare": "Une pièce rare recherchée par les collectionneurs.",
            "epic": "Un objet épique aux pouvoirs considérables.",
            "legendary": "Un artefact légendaire aux capacités extraordinaires."
        }
        
        description_parts.append(rarity_endings.get(template.rarity, ""))
        
        return " ".join(description_parts)
    
    def _generate_prompt(self, name: str, description: str, style: str, category: str) -> str:
        """Génère un prompt optimisé pour MeshFormer"""
        
        # Extraire les éléments clés de la description
        key_elements = []
        
        # Ajouter le nom et la catégorie
        key_elements.append(f"{category} fantastique nommé {name}")
        
        # Ajouter le style
        key_elements.append(f"style {style}")
        
        # Extraire les matériaux et couleurs de la description
        materials_found = []
        for material in self.element_library["materials"]:
            if material in description.lower():
                materials_found.append(material)
        
        if materials_found:
            key_elements.append(f"matériaux: {', '.join(materials_found[:2])}")
        
        # Ajouter des détails de qualité
        key_elements.extend([
            "haute qualité",
            "détails fins",
            "rendu professionnel",
            "éclairage dramatique"
        ])
        
        return ", ".join(key_elements)
    
    def generate_concept_series(self, theme: str, count: int = 5) -> List[GeneratedConcept]:
        """
        Génère une série de concepts sur un thème
        
        Args:
            theme: Thème de la série
            count: Nombre de concepts
            
        Returns:
            Liste de concepts cohérents
        """
        
        logger.info(f"🎭 Génération série '{theme}' ({count} concepts)")
        
        # Définir les paramètres selon le thème
        theme_configs = {
            "élémentaire": {
                "modifiers": ["feu", "eau", "terre", "air"],
                "styles": ["elfique", "cristallin"],
                "categories": ["weapon", "armor", "object"]
            },
            "royal": {
                "modifiers": ["royal", "noble", "précieux"],
                "styles": ["angélique", "elfique"],
                "categories": ["weapon", "armor", "object"]
            },
            "sombre": {
                "modifiers": ["maudit", "sombre", "corrompu"],
                "styles": ["démoniaque"],
                "categories": ["weapon", "creature", "object"]
            }
        }
        
        config = theme_configs.get(theme, {
            "modifiers": [],
            "styles": list(self.style_library.keys()),
            "categories": list(self.templates.keys())
        })
        
        concepts = []
        
        for i in range(count):
            category = random.choice(config["categories"])
            style = random.choice(config["styles"])
            
            concept = self.generate_concept(category=category, style=style)
            concepts.append(concept)
        
        logger.info(f"✅ Série '{theme}' générée: {len(concepts)} concepts")
        
        return concepts
    
    def save_concept_library(self, concepts: List[GeneratedConcept], 
                           output_path: str = "ressources/data/concept_library.json"):
        """Sauvegarde une bibliothèque de concepts"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir en dictionnaire
        concepts_data = []
        for concept in concepts:
            concept_dict = {
                "name": concept.name,
                "description": concept.description,
                "category": concept.category,
                "style": concept.style,
                "complexity": concept.complexity,
                "rarity": concept.rarity,
                "elements": concept.elements,
                "prompt": concept.prompt,
                "metadata": concept.metadata
            }
            concepts_data.append(concept_dict)
        
        # Sauvegarder
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(concepts_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Bibliothèque sauvegardée: {output_file} ({len(concepts)} concepts)")
    
    def get_concept_statistics(self, concepts: List[GeneratedConcept]) -> Dict[str, Any]:
        """Calcule des statistiques sur les concepts"""
        
        if not concepts:
            return {}
        
        # Compter par catégorie
        categories = {}
        styles = {}
        rarities = {}
        complexities = []
        
        for concept in concepts:
            categories[concept.category] = categories.get(concept.category, 0) + 1
            styles[concept.style] = styles.get(concept.style, 0) + 1
            rarities[concept.rarity] = rarities.get(concept.rarity, 0) + 1
            complexities.append(concept.complexity)
        
        return {
            "total_concepts": len(concepts),
            "categories": categories,
            "styles": styles,
            "rarities": rarities,
            "average_complexity": sum(complexities) / len(complexities),
            "complexity_range": [min(complexities), max(complexities)]
        }
