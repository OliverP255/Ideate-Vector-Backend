"""
Map data and visualization endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from app.models.map import MapData, MapBounds, DocumentPoint
from app.services.map import MapService

router = APIRouter()


@router.get("/map", response_model=MapData)
async def get_map_data(
    bounds: Optional[str] = Query(None, description="Map bounds as 'min_lat,min_lon,max_lat,max_lon'"),
    zoom_level: Optional[int] = Query(None, description="Map zoom level"),
    limit: int = Query(1000, description="Maximum number of points to return")
):
    """
    Get map data for visualization.
    
    Args:
        bounds: Optional map bounds for spatial filtering
        zoom_level: Optional zoom level for level-of-detail
        limit: Maximum number of document points to return
        
    Returns:
        MapData: Map visualization data
    """
    try:
        map_service = MapService()
        
        # Parse bounds if provided
        map_bounds = None
        if bounds:
            try:
                coords = [float(x.strip()) for x in bounds.split(',')]
                if len(coords) == 4:
                    map_bounds = MapBounds(
                        min_lat=coords[0],
                        min_lon=coords[1],
                        max_lat=coords[2],
                        max_lon=coords[3]
                    )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid bounds format")
        
        map_data = await map_service.get_map_data(map_bounds, zoom_level, limit)
        return map_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get map data: {str(e)}")


@router.get("/map/tiles/{z}/{x}/{y}")
async def get_map_tile(
    z: int,
    x: int,
    y: int
):
    """
    Get map tile data for a specific zoom level and coordinates.
    
    Args:
        z: Zoom level
        x: Tile X coordinate
        y: Tile Y coordinate
        
    Returns:
        dict: Tile data with aggregated document points
    """
    try:
        map_service = MapService()
        tile_data = await map_service.get_tile_data(z, x, y)
        return tile_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tile data: {str(e)}")


@router.get("/map/clusters", response_model=List[dict])
async def get_clusters(
    bounds: Optional[str] = Query(None, description="Map bounds for clustering"),
    min_cluster_size: int = Query(5, description="Minimum cluster size")
):
    """
    Get cluster information for the map.
    
    Args:
        bounds: Optional map bounds for clustering
        min_cluster_size: Minimum number of documents per cluster
        
    Returns:
        List[dict]: Cluster information with centroids and metadata
    """
    try:
        map_service = MapService()
        
        # Parse bounds if provided
        map_bounds = None
        if bounds:
            try:
                coords = [float(x.strip()) for x in bounds.split(',')]
                if len(coords) == 4:
                    map_bounds = MapBounds(
                        min_lat=coords[0],
                        min_lon=coords[1],
                        max_lat=coords[2],
                        max_lon=coords[3]
                    )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid bounds format")
        
        clusters = await map_service.get_clusters(map_bounds, min_cluster_size)
        return clusters
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get clusters: {str(e)}")
