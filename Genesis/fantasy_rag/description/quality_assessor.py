#!/usr/bin/env python3
"""
Quality Assessor for Fantasy Asset Descriptions

This module evaluates the quality of generated descriptions.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .description_parser import DescriptionParser, ParsedDescription
from .multilingual_handler import MultilingualHandler
from .prompt_templates import DescriptionLanguage

@dataclass
class QualityMetrics:
    """Quality metrics for a description"""
    overall_score: float
    length_score: float
    language_score: float
    content_score: float
    structure_score: float
    issues: List[str]
    recommendations: List[str]

class QualityAssessor:
    """
    Assessor for description quality
    """
    
    def __init__(self, target_length: int = 200):
        """
        Initialize quality assessor
        
        Args:
            target_length: Target word count
        """
        self.target_length = target_length
        self.parser = DescriptionParser(target_length)
        self.multilingual_handler = MultilingualHandler()
    
    def assess_quality(self, 
                      text: str, 
                      expected_language: DescriptionLanguage) -> QualityMetrics:
        """
        Assess overall quality of description
        
        Args:
            text: Description text
            expected_language: Expected language
            
        Returns:
            QualityMetrics object
        """
        # Parse description
        parsed = self.parser.parse_description(text)
        
        # Assess different aspects
        length_score = self._assess_length(parsed.word_count)
        language_score = self._assess_language(text, expected_language)
        content_score = self._assess_content(text)
        structure_score = self._assess_structure(parsed)
        
        # Calculate overall score
        weights = {
            'length': 0.25,
            'language': 0.20,
            'content': 0.35,
            'structure': 0.20
        }
        
        overall_score = (
            length_score * weights['length'] +
            language_score * weights['language'] +
            content_score * weights['content'] +
            structure_score * weights['structure']
        )
        
        # Collect issues and recommendations
        issues = parsed.issues.copy()
        recommendations = self._generate_recommendations(
            length_score, language_score, content_score, structure_score
        )
        
        return QualityMetrics(
            overall_score=overall_score,
            length_score=length_score,
            language_score=language_score,
            content_score=content_score,
            structure_score=structure_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _assess_length(self, word_count: int) -> float:
        """Assess length quality"""
        target = self.target_length
        diff = abs(word_count - target)
        
        if diff <= target * 0.05:  # Within 5%
            return 100.0
        elif diff <= target * 0.10:  # Within 10%
            return 90.0
        elif diff <= target * 0.20:  # Within 20%
            return 70.0
        else:
            return max(0, 50 - (diff / target * 100))
    
    def _assess_language(self, text: str, expected_language: DescriptionLanguage) -> float:
        """Assess language quality"""
        detection = self.multilingual_handler.detect_language(text)
        
        if detection.detected_language == expected_language:
            return min(100.0, detection.confidence * 100 + 50)
        else:
            return max(0, 30 - detection.confidence * 50)
    
    def _assess_content(self, text: str) -> float:
        """Assess content quality"""
        score = 100.0
        text_lower = text.lower()
        
        # Check for fantasy/descriptive keywords
        fantasy_keywords = [
            "metal", "métal", "wood", "bois", "leather", "cuir", "crystal", "cristal",
            "ornate", "orné", "carved", "gravé", "polished", "poli", "ancient", "ancien",
            "magical", "magique", "enchanted", "enchanté", "detailed", "détaillé"
        ]
        
        keyword_count = sum(1 for keyword in fantasy_keywords if keyword in text_lower)
        if keyword_count < 3:
            score -= 20
        
        # Check for generic phrases (negative)
        generic_phrases = [
            "this object", "cet objet", "this item", "cet article",
            "appears to be", "semble être", "looks like", "ressemble à"
        ]
        
        generic_count = sum(1 for phrase in generic_phrases if phrase in text_lower)
        score -= generic_count * 10
        
        return max(0, score)
    
    def _assess_structure(self, parsed: ParsedDescription) -> float:
        """Assess structure quality"""
        score = 100.0
        
        # Check sentence count
        sentence_count = len(parsed.sentences)
        if sentence_count < 3:
            score -= 30
        elif sentence_count > 15:
            score -= 20
        
        # Check sentence length variation
        if sentence_count > 1:
            lengths = [len(s.split()) for s in parsed.sentences]
            avg_length = sum(lengths) / len(lengths)
            
            # Check for variation
            variation = sum(abs(l - avg_length) for l in lengths) / len(lengths)
            if variation < 2:  # Too uniform
                score -= 15
        
        return max(0, score)
    
    def _generate_recommendations(self, 
                                length_score: float,
                                language_score: float, 
                                content_score: float,
                                structure_score: float) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if length_score < 70:
            recommendations.append("Adjust description length to target word count")
        
        if language_score < 70:
            recommendations.append("Improve language consistency and fluency")
        
        if content_score < 70:
            recommendations.append("Add more descriptive and fantasy-specific details")
        
        if structure_score < 70:
            recommendations.append("Improve sentence structure and variation")
        
        return recommendations
