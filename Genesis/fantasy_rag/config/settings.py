"""
Configuration settings for Fantasy Assets RAG system
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESOURCES_DIR = PROJECT_ROOT / "ressources"
DATA_DIR = RESOURCES_DIR / "data"
ASSETS_DIR = DATA_DIR / "assets"
LIBRARY_DIR = RESOURCES_DIR / "library"

# Fantasy RAG specific paths
FANTASY_RAG_DIR = PROJECT_ROOT / "fantasy_rag"
RENDERS_DIR = DATA_DIR / "renders"
CACHE_DIR = DATA_DIR / "cache"
DATABASE_DIR = DATA_DIR / "database"
LOGS_DIR = DATA_DIR / "logs"

@dataclass
class RenderingConfig:
    """Configuration for 6-view rendering system"""
    
    # Resolution and quality
    resolution: Tuple[int, int] = (1024, 1024)
    samples: int = 128
    format: str = "PNG"
    
    # 6 views configuration
    views: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.views is None:
            self.views = {
                "front": {
                    "rotation": (90, 0, 0),
                    "description": "Vue de face - détails frontaux, ornements",
                    "priority": "HIGH",
                    "usage": ["description", "meshformer_ref"]
                },
                "back": {
                    "rotation": (90, 0, 180),
                    "description": "Vue arrière - détails dorsaux, attaches",
                    "priority": "HIGH", 
                    "usage": ["description", "completeness"]
                },
                "left": {
                    "rotation": (90, 0, 90),
                    "description": "Profil gauche - silhouette, proportions",
                    "priority": "MEDIUM",
                    "usage": ["description", "proportions"]
                },
                "right": {
                    "rotation": (90, 0, -90),
                    "description": "Profil droit - symétrie, variations",
                    "priority": "MEDIUM",
                    "usage": ["description", "symmetry_check"]
                },
                "top": {
                    "rotation": (0, 0, 0),
                    "description": "Vue du dessus - forme générale, structure",
                    "priority": "HIGH",
                    "usage": ["description", "meshformer_ref", "structure"]
                },
                "iso_3d": {
                    "rotation": (60, 0, 45),
                    "description": "Vue isométrique 3D - perspective globale",
                    "priority": "CRITICAL",
                    "usage": ["description", "meshformer_ref", "thumbnail"]
                }
            }

@dataclass
class LLMConfig:
    """Configuration for local LLM (Ollama)"""
    
    model_name: str = "llama3.2-vision"
    base_url: str = "http://localhost:11434"
    description_length: int = 200  # words
    languages: List[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["français", "english"]

@dataclass
class EmbeddingConfig:
    """Configuration for embeddings"""
    
    text_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vision_model: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512
    batch_size: int = 32

@dataclass
class DatabaseConfig:
    """Configuration for ChromaDB"""
    
    persist_directory: str = str(DATABASE_DIR)
    collection_name: str = "fantasy_assets"
    distance_metric: str = "cosine"
    
@dataclass
class BlenderConfig:
    """Configuration for Blender rendering"""
    
    blender_executable: str = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"  # Full path to Blender
    headless: bool = True
    engine: str = "CYCLES"
    device: str = "GPU"  # or "CPU"
    
# Global configuration instance
@dataclass
class FantasyRAGConfig:
    """Main configuration class"""
    
    rendering: RenderingConfig = None
    llm: LLMConfig = None
    embedding: EmbeddingConfig = None
    database: DatabaseConfig = None
    blender: BlenderConfig = None
    
    # Performance settings
    max_concurrent_renders: int = 4
    cache_enabled: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.rendering is None:
            self.rendering = RenderingConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.database is None:
            self.database = DatabaseConfig()
        if self.blender is None:
            self.blender = BlenderConfig()

# Create global config instance
settings = FantasyRAGConfig()

# Ensure directories exist
def create_directories():
    """Create necessary directories"""
    directories = [
        RENDERS_DIR,
        CACHE_DIR, 
        DATABASE_DIR,
        LOGS_DIR,
        ASSETS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
if __name__ == "__main__":
    create_directories()
    print("✅ Directories created successfully")
    print(f"📁 Assets directory: {ASSETS_DIR}")
    print(f"📁 Renders directory: {RENDERS_DIR}")
    print(f"📁 Database directory: {DATABASE_DIR}")
