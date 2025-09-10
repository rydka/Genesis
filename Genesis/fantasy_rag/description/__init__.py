#!/usr/bin/env python3
"""
Fantasy Assets Description Generator Module

This module handles the generation of detailed descriptions for 3D fantasy assets
using vision-language models (VLM) like Llama3.2-Vision via Ollama.
"""

from .llm_interface import OllamaVisionInterface
from .prompt_templates import FantasyPromptTemplates
from .description_parser import DescriptionParser
from .category_classifier import CategoryClassifier
from .multilingual_handler import MultilingualHandler
from .quality_assessor import QualityAssessor

__all__ = [
    "OllamaVisionInterface",
    "FantasyPromptTemplates", 
    "DescriptionParser",
    "CategoryClassifier",
    "MultilingualHandler",
    "QualityAssessor"
]
