"""
User profile and overlay models.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class UserProfile(BaseModel):
    """User profile information."""
    user_id: str
    username: Optional[str] = None
    email: Optional[str] = None
    created_at: datetime
    last_active: datetime
    preferences: Optional[Dict[str, Any]] = None


class UserRead(BaseModel):
    """User read document record."""
    document_id: str
    read_at: datetime
    reading_time_seconds: Optional[int] = None
    rating: Optional[int] = None  # 1-5 scale


class UserOverlay(BaseModel):
    """User overlay data for map visualization."""
    user_id: str
    read_documents: List[UserRead]
    read_points: List[Dict[str, Any]]  # Lat/lon points for visualization
    convex_hull: Optional[Dict[str, Any]] = None  # GeoJSON polygon
    coverage_stats: Dict[str, Any]  # Coverage by cluster
    total_read: int
    last_updated: datetime
