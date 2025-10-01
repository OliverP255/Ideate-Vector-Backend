"""
Map click and document retrieval models.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ClickRequest(BaseModel):
    """Map click request."""
    x: float
    y: float
    radius: float = 0.1
    user_id: Optional[str] = None
    query_text: Optional[str] = None
    limit: int = 10


class DocumentResult(BaseModel):
    """Document search result."""
    document_id: str
    coordinates: List[float]
    similarity_score: Optional[float] = None
    spatial_distance: Optional[float] = None
    rerank_method: str
    metadata: Optional[Dict[str, Any]] = None


class ClickResponse(BaseModel):
    """Map click response."""
    click_id: str
    documents: List[DocumentResult]
    spatial_candidates: int
    message: str
    processed_at: Optional[str] = None