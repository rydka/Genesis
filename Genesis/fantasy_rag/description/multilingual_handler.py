#!/usr/bin/env python3
"""
Multilingual Handler for Fantasy Asset Descriptions

This module handles multilingual aspects of description generation.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from .prompt_templates import DescriptionLanguage

@dataclass
class LanguageDetection:
    """Result of language detection"""
    detected_language: DescriptionLanguage
    confidence: float

class MultilingualHandler:
    """
    Handler for multilingual description processing
    """
    
    # Language indicators
    LANGUAGE_INDICATORS = {
        DescriptionLanguage.FRENCH: [
            "le", "la", "les", "un", "une", "des", "du", "de", "et", "est", "avec",
            "cette", "cet", "ces", "dans", "sur", "pour", "par", "très", "plus"
        ],
        DescriptionLanguage.ENGLISH: [
            "the", "a", "an", "and", "is", "with", "this", "that", "these", "those",
            "in", "on", "for", "by", "very", "more", "from", "to", "of", "at"
        ]
    }
    
    def __init__(self):
        """Initialize multilingual handler"""
        pass
    
    def detect_language(self, text: str) -> LanguageDetection:
        """
        Detect language of text
        
        Args:
            text: Text to analyze
            
        Returns:
            LanguageDetection result
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        language_scores = {}
        
        for language, indicators in self.LANGUAGE_INDICATORS.items():
            score = 0
            for indicator in indicators:
                score += text_lower.count(indicator)
            language_scores[language] = score
        
        if not language_scores or max(language_scores.values()) == 0:
            return LanguageDetection(
                detected_language=DescriptionLanguage.ENGLISH,  # Default
                confidence=0.0
            )
        
        best_language = max(language_scores.keys(), key=lambda k: language_scores[k])
        total_score = sum(language_scores.values())
        confidence = language_scores[best_language] / max(1, total_score)
        
        return LanguageDetection(
            detected_language=best_language,
            confidence=confidence
        )
    
    def validate_language(self, text: str, expected_language: DescriptionLanguage) -> bool:
        """
        Validate that text is in expected language
        
        Args:
            text: Text to validate
            expected_language: Expected language
            
        Returns:
            True if language matches expectation
        """
        detection = self.detect_language(text)
        return (detection.detected_language == expected_language and 
                detection.confidence > 0.3)
    
    def get_language_name(self, language: DescriptionLanguage) -> str:
        """Get human-readable language name"""
        return language.value.title()
