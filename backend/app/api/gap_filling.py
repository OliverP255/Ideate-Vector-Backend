"""
Gap filling API endpoints for generating contextual text in empty map areas.
"""

from fastapi import APIRouter, HTTPException, Query, Path
from typing import Dict, Any, Optional
from pydantic import BaseModel

from app.services.gap_filling import GapFillingService

router = APIRouter()


class GapAnalysisRequest(BaseModel):
    x: float
    y: float
    radius: float = 1.0


class GapFillingRequest(BaseModel):
    x: float
    y: float
    user_id: str = "system"
    radius: float = 1.0
    force_gap_filling: bool = False


@router.get("/gap-filling/analyze")
async def analyze_gap_area(
    x: float = Query(..., description="X coordinate of gap center"),
    y: float = Query(..., description="Y coordinate of gap center"),
    radius: float = Query(1.0, description="Analysis radius around gap")
):
    """
    Analyze an empty area to understand its semantic context.
    
    Args:
        x: X coordinate of the gap center
        y: Y coordinate of the gap center
        radius: Analysis radius around the gap
        
    Returns:
        Gap analysis results including nearby documents and semantic context
    """
    try:
        gap_service = GapFillingService()
        analysis = await gap_service.analyze_gap_area(x, y, radius)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gap analysis failed: {str(e)}")


@router.post("/gap-filling/generate")
async def generate_gap_filling_document(request: GapFillingRequest):
    """
    Generate contextual text to fill an empty map area.
    
    Args:
        request: Gap filling request with coordinates and user info
        
    Returns:
        Generated document ready for map display
    """
    try:
        gap_service = GapFillingService()
        document = await gap_service.create_gap_filling_document(
            request.x, 
            request.y, 
            request.user_id,
            request.force_gap_filling
        )
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gap filling generation failed: {str(e)}")


@router.get("/gap-filling/quick-generate")
async def quick_generate_gap_filling(
    x: float = Query(..., description="X coordinate of gap center"),
    y: float = Query(..., description="Y coordinate of gap center"),
    user_id: str = Query("system", description="User ID requesting generation"),
    force: bool = Query(False, description="Force gap filling even in dense areas")
):
    """
    Quick endpoint to generate gap-filling text with minimal parameters.
    
    Args:
        x: X coordinate of the gap center
        y: Y coordinate of the gap center
        user_id: User requesting the generation
        
    Returns:
        Generated document ready for map display
    """
    try:
        gap_service = GapFillingService()
        document = await gap_service.create_gap_filling_document(x, y, user_id, force)
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick gap filling failed: {str(e)}")


@router.get("/gap-filling/status")
async def get_gap_filling_status():
    """
    Get status of the gap filling system.
    
    Returns:
        System status and capabilities
    """
    try:
        return {
            "status": "operational",
            "capabilities": [
                "semantic_context_analysis",
                "contextual_text_generation", 
                "coordinate_prediction",
                "gap_density_analysis"
            ],
            "supported_clusters": [
                "computer_science",
                "mathematics", 
                "astronomy",
                "quantum_physics",
                "condensed_matter",
                "interdisciplinary"
            ],
            "max_iterations": 5,
            "coordinate_tolerance": 0.5
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
