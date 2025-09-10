

import torch
import torch.nn as nn
import cv2
import numpy as np
import open3d as o3d
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class ViTDepthEstimator:
    def __init__(self, model_name="Intel/dpt-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)
        
    def estimate_depth(self, image):
        """Estimate depth map using ViT-based DPT model"""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        return prediction.squeeze().cpu().numpy()

class Genesis3D:
    def __init__(self):
        self.depth_estimator = ViTDepthEstimator()
        self.mesh = None
        
    def image_to_3d(self, image_path, camera_intrinsics=None, visualize=True):
        """Convert 2D image to 3D mesh using ViT depth estimation"""
        image = Image.open(image_path).convert("RGB")
        depth_map = self.depth_estimator.estimate_depth(image)
        
        if visualize:
            self.display_image_and_depth(image, depth_map)
        
        # Convert to point cloud
        self.point_cloud = self._depth_to_pointcloud(
            np.array(image), depth_map, camera_intrinsics
        )
        
        # Generate mesh from point cloud
        self.mesh = self._pointcloud_to_mesh(self.point_cloud)
        return self.mesh
    
    def display_image_and_depth(self, image, depth_map):
        """Display original image and depth map side by side"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original image
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Display depth map with colormap
        depth_display = ax2.imshow(depth_map, cmap='plasma')
        ax2.set_title('Depth Map')
        ax2.axis('off')
        
        # Add colorbar for depth
        plt.colorbar(depth_display, ax=ax2, shrink=0.8)
        
        plt.tight_layout()
        plt.show()
    
    def save_depth_visualization(self, image_path, output_path):
        """Save depth visualization to file"""
        image = Image.open(image_path).convert("RGB")
        depth_map = self.depth_estimator.estimate_depth(image)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        depth_display = ax2.imshow(depth_map, cmap='plasma')
        ax2.set_title('Depth Map')
        ax2.axis('off')
        
        plt.colorbar(depth_display, ax=ax2, shrink=0.8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _depth_to_pointcloud(self, rgb_image, depth_map, intrinsics=None):
        """Convert depth map to 3D point cloud"""
        if intrinsics is None:
            # Default camera intrinsics (adjust based on your needs)
            h, w = depth_map.shape
            fx = fy = w  # Approximate focal length
            cx, cy = w // 2, h // 2
        else:
            fx, fy, cx, cy = intrinsics
            
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(depth_map.shape[1]), 
                          np.arange(depth_map.shape[0]))
        
        # Convert to 3D coordinates
        z = depth_map
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack coordinates and colors
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = rgb_image.reshape(-1, 3) / 255.0
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def _pointcloud_to_mesh(self, point_cloud):
        """Convert point cloud to mesh using Poisson reconstruction"""
        # Estimate normals
        point_cloud.estimate_normals()
        
        # Poisson surface reconstruction
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, depth=9
        )
        
        return mesh

if __name__ == "__main__":
    genesis = Genesis3D()
    
    # Example usage
    image_path = "path/to/your/image.jpg"
    mesh = genesis.image_to_3d(image_path, visualize=True)
    
    # Or save visualization without displaying
    genesis.save_depth_visualization(image_path, "depth_comparison.png")
    
    print("Genesis 3D with ViT Depth Estimation")
