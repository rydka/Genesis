#!/usr/bin/env python3
"""
Ollama Vision Interface for Fantasy Asset Description Generation

This module provides an interface to Ollama's vision-language models
for generating detailed descriptions of 3D fantasy assets.
"""

import logging
import base64
import time
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass
import requests
import json

from .prompt_templates import FantasyPromptTemplates, PromptContext, DescriptionLanguage

logger = logging.getLogger(__name__)

@dataclass
class VisionResponse:
    """Response from vision-language model"""
    description: str
    language: DescriptionLanguage
    processing_time: float
    word_count: int
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class OllamaVisionInterface:
    """
    Interface to Ollama vision-language models for fantasy asset description
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model_name: str = "llama3.2-vision",
                 timeout: int = 120,
                 temperature: float = 0.7,
                 max_tokens: int = 2000):
        """
        Initialize Ollama Vision Interface
        
        Args:
            base_url: Ollama server URL
            model_name: Vision model name
            timeout: Request timeout in seconds
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.prompt_templates = FantasyPromptTemplates()
        
        logger.info(f"Ollama Vision Interface initialized:")
        logger.info(f"  Server: {self.base_url}")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Timeout: {self.timeout}s")
    
    def verify_connection(self) -> bool:
        """
        Verify connection to Ollama server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model_name in model_names:
                    logger.info(f"✅ Ollama connection verified, model {self.model_name} available")
                    return True
                else:
                    logger.warning(f"⚠️ Model {self.model_name} not found. Available: {model_names}")
                    return False
            else:
                logger.error(f"❌ Ollama server error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {str(e)}")
            return False
    
    def encode_image(self, image_path: Path) -> str:
        """
        Encode image to base64 for Ollama
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, 'rb') as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                logger.debug(f"Encoded image: {image_path.name} ({len(encoded)} chars)")
                return encoded
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {str(e)}")
            raise
    
    def prepare_multiview_images(self, view_paths: Dict[str, Path]) -> List[str]:
        """
        Prepare multiple view images for analysis
        
        Args:
            view_paths: Dictionary mapping view names to image paths
            
        Returns:
            List of base64 encoded images
        """
        encoded_images = []
        
        # Process views in priority order
        priority_order = ["front", "iso_3d", "left", "right", "top", "back"]
        
        for view_name in priority_order:
            if view_name in view_paths:
                try:
                    encoded = self.encode_image(view_paths[view_name])
                    encoded_images.append(encoded)
                    logger.debug(f"Added view: {view_name}")
                except Exception as e:
                    logger.warning(f"Failed to encode view {view_name}: {str(e)}")
        
        # Add any remaining views
        for view_name, path in view_paths.items():
            if view_name not in priority_order:
                try:
                    encoded = self.encode_image(path)
                    encoded_images.append(encoded)
                    logger.debug(f"Added additional view: {view_name}")
                except Exception as e:
                    logger.warning(f"Failed to encode view {view_name}: {str(e)}")
        
        logger.info(f"Prepared {len(encoded_images)} images for analysis")
        return encoded_images
    
    def generate_description(self, 
                           view_paths: Dict[str, Path],
                           context: PromptContext) -> VisionResponse:
        """
        Generate description from multiple view images
        
        Args:
            view_paths: Dictionary mapping view names to image paths
            context: Prompt context with generation parameters
            
        Returns:
            VisionResponse with generated description
        """
        start_time = time.time()
        
        logger.info(f"Generating description for: {context.asset_name}")
        logger.info(f"  Language: {context.language.value}")
        logger.info(f"  Target length: {context.target_length} words")
        logger.info(f"  Views: {list(view_paths.keys())}")
        
        try:
            # Check if we have multiple views - use sequential approach for single-image models
            if len(view_paths) > 1:
                return self._generate_description_sequential(view_paths, context, start_time)

            # Single image approach
            encoded_images = self.prepare_multiview_images(view_paths)

            if not encoded_images:
                return VisionResponse(
                    description="",
                    language=context.language,
                    processing_time=time.time() - start_time,
                    word_count=0,
                    success=False,
                    error_message="No valid images provided"
                )

            # Generate prompts
            prompts = self.prompt_templates.generate_complete_prompt(context)

            # Prepare Ollama request
            request_data = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": prompts["system"]
                    },
                    {
                        "role": "user",
                        "content": prompts["user"],
                        "images": encoded_images
                    }
                ],
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "stop": []  # Don't stop early
                },
                "stream": False
            }
            
            # Make request to Ollama
            logger.debug("Sending request to Ollama...")
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                timeout=self.timeout
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                description = result.get('message', {}).get('content', '').strip()
                
                if description:
                    word_count = len(description.split())
                    
                    logger.info(f"✅ Description generated successfully")
                    logger.info(f"   Processing time: {processing_time:.1f}s")
                    logger.info(f"   Word count: {word_count}")
                    
                    return VisionResponse(
                        description=description,
                        language=context.language,
                        processing_time=processing_time,
                        word_count=word_count,
                        success=True,
                        metadata={
                            "model": self.model_name,
                            "temperature": self.temperature,
                            "views_processed": len(encoded_images),
                            "prompt_tokens": len(prompts["system"]) + len(prompts["user"])
                        }
                    )
                else:
                    return VisionResponse(
                        description="",
                        language=context.language,
                        processing_time=processing_time,
                        word_count=0,
                        success=False,
                        error_message="Empty response from model"
                    )
            else:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                
                return VisionResponse(
                    description="",
                    language=context.language,
                    processing_time=processing_time,
                    word_count=0,
                    success=False,
                    error_message=error_msg
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error generating description: {str(e)}"
            logger.error(error_msg)
            
            return VisionResponse(
                description="",
                language=context.language,
                processing_time=processing_time,
                word_count=0,
                success=False,
                error_message=error_msg
            )

    def _generate_description_sequential(self,
                                       view_paths: Dict[str, Path],
                                       context: PromptContext,
                                       start_time: float) -> VisionResponse:
        """
        Generate description by analyzing views sequentially (for single-image models)
        """
        logger.info("Using sequential analysis (single image per request)")

        # Select the most important view for analysis
        priority_views = ["front", "iso_3d", "left", "right", "top", "back"]
        selected_view = None
        selected_path = None

        for view_name in priority_views:
            if view_name in view_paths:
                selected_view = view_name
                selected_path = view_paths[view_name]
                break

        if not selected_path:
            # Fallback to first available view
            selected_view = list(view_paths.keys())[0]
            selected_path = list(view_paths.values())[0]

        logger.info(f"Selected primary view: {selected_view}")

        # Generate description using single view
        single_view_paths = {selected_view: selected_path}

        # Use single image approach
        encoded_images = self.prepare_multiview_images(single_view_paths)

        if not encoded_images:
            return VisionResponse(
                description="",
                language=context.language,
                processing_time=time.time() - start_time,
                word_count=0,
                success=False,
                error_message="Failed to encode selected view"
            )

        # Generate prompts with note about multiple views
        prompts = self.prompt_templates.generate_complete_prompt(context)

        # Add note about view selection
        view_note = f"\n\nNote: Analyzing the {selected_view} view (primary view selected from {len(view_paths)} available views)."
        prompts["user"] += view_note

        # Prepare Ollama request
        request_data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": prompts["system"]
                },
                {
                    "role": "user",
                    "content": prompts["user"],
                    "images": encoded_images
                }
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            },
            "stream": False
        }

        # Make request to Ollama
        logger.debug("Sending sequential request to Ollama...")
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=request_data,
            timeout=self.timeout
        )

        processing_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            description = result.get('message', {}).get('content', '').strip()

            if description:
                word_count = len(description.split())

                logger.info(f"✅ Sequential description generated successfully")
                logger.info(f"   Processing time: {processing_time:.1f}s")
                logger.info(f"   Word count: {word_count}")
                logger.info(f"   Primary view: {selected_view}")

                return VisionResponse(
                    description=description,
                    language=context.language,
                    processing_time=processing_time,
                    word_count=word_count,
                    success=True,
                    metadata={
                        "model": self.model_name,
                        "temperature": self.temperature,
                        "views_available": len(view_paths),
                        "primary_view": selected_view,
                        "analysis_mode": "sequential"
                    }
                )
            else:
                return VisionResponse(
                    description="",
                    language=context.language,
                    processing_time=processing_time,
                    word_count=0,
                    success=False,
                    error_message="Empty response from model"
                )
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return VisionResponse(
                description="",
                language=context.language,
                processing_time=processing_time,
                word_count=0,
                success=False,
                error_message=f"Ollama API error: {response.status_code} - {response.text}"
            )

    def generate_multilingual_descriptions(self,
                                         view_paths: Dict[str, Path],
                                         asset_name: str,
                                         target_length: int = 200) -> Dict[str, VisionResponse]:
        """
        Generate descriptions in multiple languages
        
        Args:
            view_paths: Dictionary mapping view names to image paths
            asset_name: Name of the asset
            target_length: Target word count
            
        Returns:
            Dictionary mapping language codes to VisionResponse objects
        """
        results = {}
        
        for language in [DescriptionLanguage.FRENCH, DescriptionLanguage.ENGLISH]:
            context = PromptContext(
                asset_name=asset_name,
                view_count=len(view_paths),
                target_length=target_length,
                language=language
            )
            
            response = self.generate_description(view_paths, context)
            results[language.value] = response
            
            # Small delay between requests
            time.sleep(1)
        
        return results


def main():
    """Test the Ollama Vision Interface"""
    logging.basicConfig(level=logging.INFO)
    
    interface = OllamaVisionInterface()
    
    print("🧠 Ollama Vision Interface Test")
    print("=" * 50)
    
    # Test connection
    if interface.verify_connection():
        print("✅ Ollama connection successful")
    else:
        print("❌ Ollama connection failed")
        return 1
    
    print(f"📋 Configuration:")
    print(f"   Model: {interface.model_name}")
    print(f"   Server: {interface.base_url}")
    print(f"   Temperature: {interface.temperature}")
    
    return 0

if __name__ == "__main__":
    exit(main())
