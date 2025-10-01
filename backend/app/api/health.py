"""
Health check endpoints for monitoring and load balancers.
"""

from fastapi import APIRouter, HTTPException
from app.services.health import HealthService
from app.models.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        HealthResponse: Service health status
    """
    try:
        health_service = HealthService()
        health_status = await health_service.check_health()
        return health_status
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness check for Kubernetes.
    
    Returns:
        dict: Ready status
    """
    try:
        health_service = HealthService()
        is_ready = await health_service.check_readiness()
        if is_ready:
            return {"status": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Readiness check failed: {str(e)}")


@router.get("/health/live")
async def liveness_check():
    """
    Liveness check for Kubernetes.
    
    Returns:
        dict: Live status
    """
    return {"status": "alive"}
