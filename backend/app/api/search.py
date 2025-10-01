"""
Semantic search endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
import asyncio
from app.services.search import SearchService

router = APIRouter()


@router.get("/search/semantic")
async def semantic_search(
    query: str = Query(..., description="Search query text"),
    limit: int = Query(10, description="Maximum number of results per page"),
    score_threshold: float = Query(0.0, description="Minimum similarity score threshold"),
    offset: int = Query(0, description="Number of results to skip (for pagination)")
):
    """
    Perform semantic search using text query with pagination.
    
    Args:
        query: Search query text
        limit: Maximum number of results per page
        score_threshold: Minimum similarity score threshold
        offset: Number of results to skip (for pagination)
        
    Returns:
        dict: Search results with documents, pagination info, and metadata
    """
    try:
        search_service = SearchService()
        # Add timeout for search operations
        search_result = await asyncio.wait_for(
            search_service.semantic_search(query, limit, score_threshold, offset),
            timeout=30.0  # 30 second timeout for search
        )
        
        return {
            "query": query,
            "results": search_result["results"],
            "total_count": search_result["total_count"],
            "limit": search_result["limit"],
            "offset": search_result["offset"],
            "has_more": search_result["has_more"],
            "score_threshold": score_threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@router.get("/search/semantic-with-coordinates")
async def semantic_search_with_coordinates(
    query: str = Query(..., description="Search query text"),
    x: float = Query(0.0, description="X coordinate for spatial filtering"),
    y: float = Query(0.0, description="Y coordinate for spatial filtering"),
    radius: float = Query(5.0, description="Search radius for spatial filtering"),
    limit: int = Query(10, description="Maximum number of results per page"),
    score_threshold: float = Query(0.0, description="Minimum similarity score threshold"),
    offset: int = Query(0, description="Number of results to skip (for pagination)")
):
    """
    Perform semantic search with optional spatial filtering and pagination.
    
    Args:
        query: Search query text
        x: X coordinate for spatial filtering
        y: Y coordinate for spatial filtering
        radius: Search radius for spatial filtering
        limit: Maximum number of results per page
        score_threshold: Minimum similarity score threshold
        offset: Number of results to skip (for pagination)
        
    Returns:
        dict: Search results with documents, pagination info, and metadata
    """
    try:
        search_service = SearchService()
        search_result = await search_service.semantic_search_with_spatial_filter(
            query, x, y, radius, limit, score_threshold, offset
        )
        
        return {
            "query": query,
            "coordinates": {"x": x, "y": y},
            "radius": radius,
            "results": search_result["results"],
            "total_count": search_result["total_count"],
            "limit": search_result["limit"],
            "offset": search_result["offset"],
            "has_more": search_result["has_more"],
            "score_threshold": score_threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search with coordinates failed: {str(e)}")


@router.get("/search/semantic-spatial")
async def semantic_search_with_spatial_area(
    query: str = Query(..., description="Search query text"),
    limit: int = Query(10, description="Maximum number of results per page"),
    score_threshold: float = Query(0.0, description="Minimum similarity score threshold"),
    offset: int = Query(0, description="Number of results to skip (for pagination)"),
    spatial_expansion_factor: float = Query(0.8, description="Factor to expand spatial area around semantic results")
):
    """
    Perform semantic search and find spatial area around semantically similar documents.
    
    Args:
        query: Search query text
        limit: Maximum number of results per page
        score_threshold: Minimum similarity score threshold
        offset: Number of results to skip (for pagination)
        spatial_expansion_factor: Factor to expand spatial area around semantic results
        
    Returns:
        dict: Search results with documents, spatial area info, and pagination info
    """
    try:
        search_service = SearchService()
        search_result = await search_service.semantic_search_with_spatial_area(
            query, limit, score_threshold, offset, spatial_expansion_factor
        )
        
        return {
            "query": query,
            "results": search_result["results"],
            "total_count": search_result["total_count"],
            "limit": search_result["limit"],
            "offset": search_result["offset"],
            "has_more": search_result["has_more"],
            "spatial_area": search_result["spatial_area"],
            "score_threshold": score_threshold,
            "spatial_expansion_factor": spatial_expansion_factor
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search with spatial area failed: {str(e)}")
