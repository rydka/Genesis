#!/usr/bin/env python3
"""
Description Parser for Fantasy Asset Descriptions

This module handles parsing and standardization of generated descriptions.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ParsedDescription:
    """Parsed and standardized description"""
    text: str
    word_count: int
    sentences: List[str]
    quality_score: float
    issues: List[str]

class DescriptionParser:
    """
    Parser for fantasy asset descriptions
    """
    
    def __init__(self, target_length: int = 200):
        """
        Initialize parser
        
        Args:
            target_length: Target word count for descriptions
        """
        self.target_length = target_length
    
    def parse_description(self, text: str) -> ParsedDescription:
        """
        Parse and analyze a description
        
        Args:
            text: Raw description text
            
        Returns:
            ParsedDescription object
        """
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Count words and sentences
        words = cleaned_text.split()
        word_count = len(words)
        sentences = self._split_sentences(cleaned_text)
        
        # Analyze quality
        quality_score, issues = self._analyze_quality(cleaned_text, word_count)
        
        return ParsedDescription(
            text=cleaned_text,
            word_count=word_count,
            sentences=sentences,
            quality_score=quality_score,
            issues=issues
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)
        
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _analyze_quality(self, text: str, word_count: int) -> Tuple[float, List[str]]:
        """Analyze description quality"""
        issues = []
        score = 100.0
        
        # Check word count
        length_diff = abs(word_count - self.target_length)
        if length_diff > self.target_length * 0.2:  # More than 20% off
            issues.append(f"Length deviation: {length_diff} words")
            score -= min(30, length_diff / self.target_length * 100)
        
        # Check for repetitions
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only check longer words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        repeated_words = [w for w, count in word_freq.items() if count > 3]
        if repeated_words:
            issues.append(f"Repeated words: {', '.join(repeated_words[:3])}")
            score -= len(repeated_words) * 5
        
        # Check sentence structure
        sentences = self._split_sentences(text)
        if len(sentences) < 3:
            issues.append("Too few sentences")
            score -= 15
        
        avg_sentence_length = word_count / max(1, len(sentences))
        if avg_sentence_length > 30:
            issues.append("Sentences too long")
            score -= 10
        elif avg_sentence_length < 8:
            issues.append("Sentences too short")
            score -= 10
        
        return max(0, score), issues
