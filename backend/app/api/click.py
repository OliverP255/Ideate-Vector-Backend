"""
Document click endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.models.click import ClickRequest, ClickResponse
from app.services.click import ClickService

router = APIRouter()


@router.post("/click", response_model=ClickResponse)
async def handle_click(request: ClickRequest):
    """
    Handle a map click and return relevant documents.
    
    Args:
        request: Click request with coordinates and optional parameters
        
    Returns:
        ClickResponse: Click results with relevant documents
    """
    try:
        click_service = ClickService()
        
        result = await click_service.handle_map_click(
            x=request.x,
            y=request.y,
            radius=request.radius,
            user_id=request.user_id,
            query_text=request.query_text,
            limit=request.limit
        )
        
        return ClickResponse(
            click_id=f"click_{request.x}_{request.y}_{request.radius}",
            documents=result["documents"],
            spatial_candidates=result["spatial_candidates"],
            message=f"Found {len(result['documents'])} documents"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Click handling failed: {str(e)}")


@router.get("/click")
async def handle_click_get(
    x: float = Query(..., description="X coordinate of click"),
    y: float = Query(..., description="Y coordinate of click"),
    radius: float = Query(0.1, description="Search radius"),
    user_id: Optional[str] = Query(None, description="User ID for personalized results"),
    query_text: Optional[str] = Query(None, description="Query text for semantic reranking"),
    limit: int = Query(5, description="Maximum number of results")
):
    """
    Handle a map click via GET request.
    
    Args:
        x: X coordinate of click
        y: Y coordinate of click
        radius: Search radius
        user_id: Optional user ID
        query_text: Optional query text
        limit: Maximum number of results
        
    Returns:
        dict: Click results
    """
    try:
        click_service = ClickService()
        
        result = await click_service.handle_map_click(
            x=x,
            y=y,
            radius=radius,
            user_id=user_id,
            query_text=query_text,
            limit=limit
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Click handling failed: {str(e)}")


@router.get("/click/document/{document_id}")
async def get_document_details(document_id: str):
    """
    Get full document details for a clicked document.
    
    Args:
        document_id: Document identifier
        
    Returns:
        dict: Document details
    """
    try:
        click_service = ClickService()
        document_details = await click_service.get_document_details(document_id)
        
        if document_details is None:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        return document_details
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document details: {str(e)}")


@router.get("/click/statistics")
async def get_click_statistics():
    """
    Get click statistics and system status.
    
    Returns:
        dict: Click statistics
    """
    try:
        click_service = ClickService()
        stats = await click_service.get_click_statistics()
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get click statistics: {str(e)}")