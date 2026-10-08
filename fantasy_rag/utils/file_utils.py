"""
File utilities for Fantasy Assets RAG system
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib
import json
from datetime import datetime

def get_stl_files(directory: Path) -> List[Path]:
    """
    Get all STL files from a directory recursively
    
    Args:
        directory: Directory to search
        
    Returns:
        List of STL file paths
    """
    stl_files = []
    if directory.exists():
        # Get all files and filter by extension (case-insensitive)
        all_files = list(directory.rglob("*"))
        stl_files = [f for f in all_files if f.suffix.lower() == '.stl']
    return sorted(stl_files)

def get_file_hash(file_path: Path) -> str:
    """
    Calculate MD5 hash of a file
    
    Args:
        file_path: Path to file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_file_info(file_path: Path) -> Dict[str, Any]:
    """
    Get comprehensive file information
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    if not file_path.exists():
        return {}
        
    stat = file_path.stat()
    
    return {
        "filename": file_path.name,
        "stem": file_path.stem,
        "suffix": file_path.suffix,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "created": datetime.fromtimestamp(stat.st_ctime),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "hash": get_file_hash(file_path),
        "absolute_path": str(file_path.absolute()),
        "relative_path": str(file_path)
    }

def create_asset_id(file_path: Path) -> str:
    """
    Create unique asset ID from file path and hash
    
    Args:
        file_path: Path to asset file
        
    Returns:
        Unique asset ID
    """
    file_hash = get_file_hash(file_path)
    return f"{file_path.stem}_{file_hash[:8]}"

def safe_filename(filename: str) -> str:
    """
    Create safe filename by removing/replacing problematic characters
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Replace problematic characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    safe_name = "".join(c if c in safe_chars else "_" for c in filename)
    
    # Remove multiple underscores
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")
        
    return safe_name.strip("_")

def ensure_directory(directory: Path) -> Path:
    """
    Ensure directory exists, create if necessary
    
    Args:
        directory: Directory path
        
    Returns:
        Directory path
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def copy_file_safe(source: Path, destination: Path) -> bool:
    """
    Safely copy file with error handling
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_directory(destination.parent)
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        print(f"❌ Error copying {source} to {destination}: {e}")
        return False

def save_json(data: Dict[str, Any], file_path: Path) -> bool:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        file_path: JSON file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_directory(file_path.parent)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        print(f"❌ Error saving JSON to {file_path}: {e}")
        return False

def load_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load data from JSON file
    
    Args:
        file_path: JSON file path
        
    Returns:
        Loaded data or None if error
    """
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON from {file_path}: {e}")
    return None

def get_directory_size(directory: Path) -> Dict[str, Any]:
    """
    Calculate directory size and file count
    
    Args:
        directory: Directory to analyze
        
    Returns:
        Dictionary with size information
    """
    total_size = 0
    file_count = 0
    
    if directory.exists():
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
    
    return {
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "total_size_gb": round(total_size / (1024 * 1024 * 1024), 2),
        "file_count": file_count
    }

def cleanup_empty_directories(directory: Path) -> int:
    """
    Remove empty directories recursively
    
    Args:
        directory: Root directory to clean
        
    Returns:
        Number of directories removed
    """
    removed_count = 0
    
    if not directory.exists():
        return removed_count
        
    for subdir in directory.rglob("*"):
        if subdir.is_dir():
            try:
                subdir.rmdir()  # Only removes if empty
                removed_count += 1
            except OSError:
                pass  # Directory not empty
                
    return removed_count
