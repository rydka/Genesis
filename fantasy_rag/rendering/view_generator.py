#!/usr/bin/env python3
"""
View Generator for Fantasy Assets RAG
Orchestrates 6-view rendering for multiple STL assets
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .blender_renderer import BlenderRenderer, RenderResult
from ..asset_management.stl_loader import STLAsset

logger = logging.getLogger(__name__)

@dataclass
class BatchRenderResult:
    """Result of batch rendering operation"""
    total_assets: int
    successful_renders: int
    failed_renders: int
    total_time: float
    results: Dict[str, RenderResult]  # asset_id -> RenderResult
    errors: List[str]

class ViewGenerator:
    """
    Orchestrates 6-view rendering for multiple STL assets
    """
    
    def __init__(self, 
                 renderer: Optional[BlenderRenderer] = None,
                 max_concurrent: int = 1):
        """
        Initialize View Generator
        
        Args:
            renderer: BlenderRenderer instance
            max_concurrent: Maximum concurrent renders (Blender is resource-intensive)
        """
        self.renderer = renderer or BlenderRenderer()
        self.max_concurrent = max_concurrent
        
        logger.info(f"View Generator initialized with max_concurrent={max_concurrent}")
    
    def generate_views_for_asset(self, asset: STLAsset) -> RenderResult:
        """
        Generate 6 views for a single STL asset
        
        Args:
            asset: STLAsset to render
            
        Returns:
            RenderResult with rendering outcome
        """
        logger.info(f"Generating views for: {asset.file_name}")
        
        if not asset.loaded:
            error_msg = f"Asset {asset.file_name} is not loaded"
            logger.error(error_msg)
            return RenderResult(
                asset_id=asset.asset_id,
                asset_name=asset.file_name,
                views={},
                render_time=0.0,
                success=False,
                error_message=error_msg
            )
        
        return self.renderer.render_asset_views(
            stl_path=asset.file_path,
            asset_id=asset.asset_id,
            asset_name=asset.file_name
        )
    
    def generate_views_batch(self, assets: List[STLAsset]) -> BatchRenderResult:
        """
        Generate 6 views for multiple STL assets
        
        Args:
            assets: List of STLAsset objects to render
            
        Returns:
            BatchRenderResult with overall results
        """
        start_time = time.time()
        
        logger.info(f"Starting batch rendering for {len(assets)} assets")
        logger.info(f"Using {self.max_concurrent} concurrent render(s)")
        
        results = {}
        errors = []
        successful = 0
        failed = 0
        
        if self.max_concurrent == 1:
            # Sequential rendering (safer for Blender)
            for i, asset in enumerate(assets, 1):
                logger.info(f"Rendering asset {i}/{len(assets)}: {asset.file_name}")
                
                try:
                    result = self.generate_views_for_asset(asset)
                    results[asset.asset_id] = result
                    
                    if result.success:
                        successful += 1
                        logger.info(f"✅ {asset.file_name}: 6 views rendered in {result.render_time:.1f}s")
                    else:
                        failed += 1
                        error_msg = f"❌ {asset.file_name}: {result.error_message}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        
                except Exception as e:
                    failed += 1
                    error_msg = f"❌ {asset.file_name}: Unexpected error: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
        
        else:
            # Concurrent rendering (experimental)
            logger.warning("Concurrent rendering is experimental with Blender")
            
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                # Submit all rendering tasks
                future_to_asset = {
                    executor.submit(self.generate_views_for_asset, asset): asset 
                    for asset in assets
                }
                
                # Process completed tasks
                for future in as_completed(future_to_asset):
                    asset = future_to_asset[future]
                    
                    try:
                        result = future.result()
                        results[asset.asset_id] = result
                        
                        if result.success:
                            successful += 1
                            logger.info(f"✅ {asset.file_name}: 6 views rendered in {result.render_time:.1f}s")
                        else:
                            failed += 1
                            error_msg = f"❌ {asset.file_name}: {result.error_message}"
                            logger.error(error_msg)
                            errors.append(error_msg)
                            
                    except Exception as e:
                        failed += 1
                        error_msg = f"❌ {asset.file_name}: Unexpected error: {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)
        
        total_time = time.time() - start_time
        
        logger.info(f"Batch rendering complete: {successful} successful, {failed} failed in {total_time:.1f}s")
        
        return BatchRenderResult(
            total_assets=len(assets),
            successful_renders=successful,
            failed_renders=failed,
            total_time=total_time,
            results=results,
            errors=errors
        )
    
    def get_render_summary(self, batch_result: BatchRenderResult) -> Dict[str, Any]:
        """
        Get summary statistics from batch render result
        
        Args:
            batch_result: BatchRenderResult to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        successful_results = [r for r in batch_result.results.values() if r.success]
        
        if not successful_results:
            return {
                "total_assets": batch_result.total_assets,
                "successful_renders": 0,
                "failed_renders": batch_result.failed_renders,
                "total_time": batch_result.total_time,
                "total_images": 0
            }
        
        total_images = sum(len(r.views) for r in successful_results)
        avg_render_time = sum(r.render_time for r in successful_results) / len(successful_results)
        
        render_times = [r.render_time for r in successful_results]
        min_time = min(render_times)
        max_time = max(render_times)
        
        return {
            "total_assets": batch_result.total_assets,
            "successful_renders": batch_result.successful_renders,
            "failed_renders": batch_result.failed_renders,
            "total_time": batch_result.total_time,
            "total_images": total_images,
            "avg_render_time": avg_render_time,
            "min_render_time": min_time,
            "max_render_time": max_time,
            "images_per_second": total_images / batch_result.total_time if batch_result.total_time > 0 else 0
        }
    
    def verify_rendered_views(self, asset_id: str) -> Dict[str, bool]:
        """
        Verify that all 6 views exist for an asset
        
        Args:
            asset_id: Asset ID to check
            
        Returns:
            Dictionary mapping view names to existence status
        """
        asset_dir = self.renderer.output_dir / asset_id
        view_status = {}
        
        for view_name in self.renderer.RENDER_VIEWS.keys():
            image_path = asset_dir / f"{asset_id}_{view_name}.png"
            view_status[view_name] = image_path.exists()
        
        return view_status
    
    def get_asset_view_paths(self, asset_id: str) -> Dict[str, Optional[Path]]:
        """
        Get paths to all rendered views for an asset
        
        Args:
            asset_id: Asset ID
            
        Returns:
            Dictionary mapping view names to image paths (None if not found)
        """
        asset_dir = self.renderer.output_dir / asset_id
        view_paths = {}
        
        for view_name in self.renderer.RENDER_VIEWS.keys():
            image_path = asset_dir / f"{asset_id}_{view_name}.png"
            view_paths[view_name] = image_path if image_path.exists() else None
        
        return view_paths


def main():
    """Test the View Generator"""
    logging.basicConfig(level=logging.INFO)
    
    print("🎨 View Generator Test")
    print("=" * 50)
    
    # Test renderer verification
    generator = ViewGenerator()
    
    if not generator.renderer.verify_blender():
        print("❌ Blender verification failed")
        return 1
    
    print("✅ Blender verification successful")
    print(f"📁 Render output directory: {generator.renderer.output_dir}")
    print(f"🎯 Ready for 6-view batch rendering")
    
    return 0

if __name__ == "__main__":
    exit(main())
