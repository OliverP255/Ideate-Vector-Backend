"""
Map visualization models.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class MapBounds(BaseModel):
    """Map bounds for spatial filtering."""
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


class DocumentPoint(BaseModel):
    """Document point on the map."""
    document_id: str
    lat: float
    lon: float
    title: str
    author: Optional[str] = None
    year: Optional[int] = None
    source_url: Optional[str] = None
    cluster_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MapData(BaseModel):
    """Map visualization data."""
    documents: List[DocumentPoint]
    bounds: Optional[MapBounds] = None
    zoom_level: Optional[int] = None
    total_count: int
    clusters: Optional[List[Dict[str, Any]]] = None
