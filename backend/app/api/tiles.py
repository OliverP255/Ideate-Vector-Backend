"""
Tile generation and retrieval endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
import sys
import os

# Add pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# Tile generation functionality would be implemented here

router = APIRouter()


@router.post("/tiles/generate")
async def generate_tiles(zoom_levels: Optional[List[int]] = None):
    """
    Generate tiles for all zoom levels.
    
    Args:
        zoom_levels: Optional list of zoom levels to generate
        
    Returns:
        Dict containing generation results
    """
    try:
        # Load coordinates from mapping service
        from app.services.mapping import MappingService
        mapping_service = MappingService()
        all_coords = await mapping_service.get_all_coordinates()
        coordinates = all_coords["coordinates"]
        
        if not coordinates:
            raise HTTPException(status_code=404, detail="No coordinates available for tile generation")
        
        # Generate tiles
        tile_generator = TileGenerator()
        results = tile_generator.generate_tiles(coordinates, zoom_levels)
        
        return {
            "status": "success",
            "message": f"Generated tiles for {len(coordinates)} documents",
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tile generation failed: {str(e)}")


@router.get("/tiles/{zoom_level}")
async def get_tile(
    zoom_level: int,
    min_x: float = Query(..., description="Minimum X coordinate"),
    min_y: float = Query(..., description="Minimum Y coordinate"),
    max_x: float = Query(..., description="Maximum X coordinate"),
    max_y: float = Query(..., description="Maximum Y coordinate")
):
    """
    Get tile data for specific bounds and zoom level.
    
    Args:
        zoom_level: Zoom level for the tile
        min_x: Minimum X coordinate
        min_y: Minimum Y coordinate
        max_x: Maximum X coordinate
        max_y: Maximum Y coordinate
        
    Returns:
        Dict containing tile data
    """
    try:
        bounds = (min_x, min_y, max_x, max_y)
        
        tile_generator = TileGenerator()
        tile_data = tile_generator.generate_tile_for_bounds(bounds, zoom_level)
        
        return tile_data
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Tiles not generated yet. Call /tiles/generate first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tile: {str(e)}")


@router.get("/tiles/clusters")
async def get_cluster_labels(min_cluster_size: int = Query(5, description="Minimum cluster size")):
    """
    Get cluster labels for documents.
    
    Args:
        min_cluster_size: Minimum cluster size
        
    Returns:
        Dict containing cluster labels and metadata
    """
    try:
        # Load coordinates
        from app.services.mapping import MappingService
        mapping_service = MappingService()
        all_coords = await mapping_service.get_all_coordinates()
        coordinates = all_coords["coordinates"]
        
        if not coordinates:
            raise HTTPException(status_code=404, detail="No coordinates available for clustering")
        
        # Generate cluster labels
        tile_generator = TileGenerator()
        cluster_data = tile_generator.get_cluster_labels(coordinates, min_cluster_size)
        
        # Save cluster labels
        tile_generator.save_cluster_labels(cluster_data)
        
        return cluster_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate cluster labels: {str(e)}")


@router.get("/tiles/metadata")
async def get_tile_metadata():
    """
    Get tile generation metadata.
    
    Returns:
        Dict containing tile metadata
    """
    try:
        import json
        from pathlib import Path
        
        metadata_file = Path("data/tiles/tile_metadata.json")
        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail="No tile metadata found. Generate tiles first.")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return metadata
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tile metadata: {str(e)}")


@router.get("/tiles/health")
async def get_tile_health():
    """
    Get tile generation system health.
    
    Returns:
        Dict containing system health information
    """
    try:
        from pathlib import Path
        
        tiles_dir = Path("data/tiles")
        metadata_file = tiles_dir / "tile_metadata.json"
        labels_file = tiles_dir / "cluster_labels.json"
        
        health_info = {
            "tiles_directory_exists": tiles_dir.exists(),
            "metadata_file_exists": metadata_file.exists(),
            "cluster_labels_file_exists": labels_file.exists(),
            "tiles_directory_size": sum(f.stat().st_size for f in tiles_dir.rglob('*') if f.is_file()) if tiles_dir.exists() else 0
        }
        
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            health_info["total_documents"] = metadata.get("total_documents", 0)
            health_info["zoom_levels"] = metadata.get("zoom_levels", [])
            health_info["generated_at"] = metadata.get("generated_at")
        
        return health_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tile health: {str(e)}")
