"""
Image utilities for rendering and processing
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageEnhance, ImageFilter
import cv2

def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Load image as numpy array
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (H, W, C) or None if error
    """
    try:
        image = Image.open(image_path)
        return np.array(image)
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return None

def save_image(image: np.ndarray, output_path: Path, quality: int = 95) -> bool:
    """
    Save numpy array as image
    
    Args:
        image: Image as numpy array
        output_path: Output file path
        quality: JPEG quality (if applicable)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        pil_image = Image.fromarray(image)
        
        # Save with appropriate settings
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            pil_image.save(output_path, 'JPEG', quality=quality, optimize=True)
        elif output_path.suffix.lower() == '.png':
            pil_image.save(output_path, 'PNG', optimize=True)
        else:
            pil_image.save(output_path)
            
        return True
        
    except Exception as e:
        print(f"❌ Error saving image to {output_path}: {e}")
        return False

def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    try:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        if maintain_aspect:
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Pad to target size if needed
            if new_w != target_w or new_h != target_h:
                # Create padded image
                padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
                
                # Calculate padding
                pad_x = (target_w - new_w) // 2
                pad_y = (target_h - new_h) // 2
                
                # Place resized image in center
                padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
                
                return padded
            else:
                return resized
        else:
            # Direct resize without maintaining aspect ratio
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
            
    except Exception as e:
        print(f"❌ Error resizing image: {e}")
        return image

def enhance_image_for_analysis(image: np.ndarray) -> np.ndarray:
    """
    Enhance image for better LLM analysis
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    try:
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(image)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # Slight saturation boost for better color analysis
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        return np.array(pil_image)
        
    except Exception as e:
        print(f"❌ Error enhancing image: {e}")
        return image

def create_thumbnail(image: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Create thumbnail from image
    
    Args:
        image: Input image
        size: Thumbnail size (width, height)
        
    Returns:
        Thumbnail image
    """
    return resize_image(image, size, maintain_aspect=True)

def extract_dominant_colors(image: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
    """
    Extract dominant colors from image using K-means
    
    Args:
        image: Input image
        k: Number of dominant colors to extract
        
    Returns:
        List of RGB color tuples
    """
    try:
        # Reshape image to list of pixels
        pixels = image.reshape(-1, 3)
        
        # Apply K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        return [tuple(color) for color in colors]
        
    except Exception as e:
        print(f"❌ Error extracting dominant colors: {e}")
        return [(128, 128, 128)] * k  # Return gray colors as fallback

def analyze_image_properties(image: np.ndarray) -> Dict[str, Any]:
    """
    Analyze image properties for metadata
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image properties
    """
    try:
        h, w, c = image.shape
        
        # Basic properties
        properties = {
            "width": w,
            "height": h,
            "channels": c,
            "total_pixels": w * h,
            "aspect_ratio": round(w / h, 3)
        }
        
        # Color analysis
        properties["dominant_colors"] = extract_dominant_colors(image)
        
        # Brightness analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        properties["mean_brightness"] = float(np.mean(gray))
        properties["brightness_std"] = float(np.std(gray))
        
        # Contrast analysis
        properties["contrast"] = float(np.std(gray))
        
        # Edge density (complexity measure)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        properties["edge_density"] = float(edge_density)
        properties["complexity_score"] = min(edge_density * 10, 1.0)  # Normalize to 0-1
        
        return properties
        
    except Exception as e:
        print(f"❌ Error analyzing image properties: {e}")
        return {}

def create_image_grid(images: List[np.ndarray], grid_size: Tuple[int, int],
                     image_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Create grid of images (for 2x3 view display)
    
    Args:
        images: List of images
        grid_size: (rows, cols)
        image_size: Size of each image in grid
        
    Returns:
        Grid image
    """
    try:
        rows, cols = grid_size
        img_h, img_w = image_size
        
        # Create empty grid
        grid = np.zeros((rows * img_h, cols * img_w, 3), dtype=np.uint8)
        
        # Fill grid with images
        for i, image in enumerate(images[:rows * cols]):
            row = i // cols
            col = i % cols
            
            # Resize image to fit grid cell
            resized = resize_image(image, (img_w, img_h))
            
            # Place in grid
            y_start = row * img_h
            y_end = y_start + img_h
            x_start = col * img_w
            x_end = x_start + img_w
            
            grid[y_start:y_end, x_start:x_end] = resized
            
        return grid
        
    except Exception as e:
        print(f"❌ Error creating image grid: {e}")
        return np.zeros((256, 256, 3), dtype=np.uint8)

def add_image_border(image: np.ndarray, border_width: int = 2, 
                    border_color: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """
    Add border to image
    
    Args:
        image: Input image
        border_width: Border width in pixels
        border_color: RGB border color
        
    Returns:
        Image with border
    """
    try:
        h, w, c = image.shape
        
        # Create bordered image
        bordered = np.full((h + 2*border_width, w + 2*border_width, c), 
                          border_color, dtype=image.dtype)
        
        # Place original image in center
        bordered[border_width:h+border_width, border_width:w+border_width] = image
        
        return bordered
        
    except Exception as e:
        print(f"❌ Error adding border: {e}")
        return image
