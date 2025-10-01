"""
Simple gap filling API endpoint using the precision Vec2Text service directly.
This bypasses the complex initialization issues.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Global service instance
_precision_service = None


class SimpleGapFillingRequest(BaseModel):
    """Request model for simple gap filling."""
    x: float
    y: float
    user_id: str = "system"


def get_precision_service():
    """Get or create the precision Vec2Text service."""
    global _precision_service
    
    if _precision_service is None:
        logger.info("Initializing precision Vec2Text service")
        
        try:
            from ..services.embedding_to_text.text_generation.simple_vec2text_service import SimpleVec2TextService
            
            _precision_service = SimpleVec2TextService()
            logger.info("Simple Vec2Text service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize precision service: {e}")
            raise
    
    return _precision_service


@router.post("/simple-gap-filling/generate")
async def generate_simple_gap_filling(request: SimpleGapFillingRequest):
    """
    Generate text using the precision Vec2Text service.
    This is a simplified version that bypasses complex initialization.
    """
    try:
        logger.info(f"Simple gap filling request: ({request.x}, {request.y})")
        
        service = get_precision_service()
        
        # Generate text using the Vec2Text service
        result = service.generate_text_for_coordinates((request.x, request.y))
        
        if not result or not result.get('generated_text'):
            raise HTTPException(
                status_code=500,
                detail="Text generation failed - no text generated"
            )
        
        return {
            "success": True,
            "generated_text": result['generated_text'],
            "title": result.get('title', 'Generated Content'),
            "target_coordinates": [request.x, request.y],
            "final_coordinates": result.get('predicted_coordinates', [request.x, request.y]),
            "method_used": result.get('method_used', 'precision_vec2text'),
            "processing_time_seconds": 0.0,
            "correction_iterations": result.get('iterations_used', 1),
            "embedding_distance": result.get('embedding_distance', 0.0),
            "coordinate_error": result.get('coordinate_error', 0.0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in simple gap filling: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/simple-gap-filling/generate")
async def generate_simple_gap_filling_get(
    x: float = Query(..., description="X coordinate on the map"),
    y: float = Query(..., description="Y coordinate on the map"),
    user_id: str = Query("system", description="User ID")
):
    """Generate text using GET method."""
    request = SimpleGapFillingRequest(x=x, y=y, user_id=user_id)
    return await generate_simple_gap_filling(request)


@router.get("/simple-gap-filling/status")
async def get_service_status():
    """Check if the precision service is available."""
    try:
        service = get_precision_service()
        if service:
            status = service.get_service_status()
            return {
                "success": True,
                "service_available": True,
                "precision_vec2text_loaded": True,
                "parametric_umap_loaded": status.get('parametric_umap_loaded', False),
                "training_data_loaded": status.get('training_data_loaded', False)
            }
        else:
            return {
                "success": False,
                "service_available": False,
                "precision_vec2text_loaded": False,
                "parametric_umap_loaded": False,
                "training_data_loaded": False
            }
    except Exception as e:
        return {
            "success": False,
            "service_available": False,
            "error": str(e)
        }
