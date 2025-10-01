"""
Health check models.
"""

from pydantic import BaseModel
from typing import Dict, Optional
from enum import Enum


class ServiceStatus(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthResponse(BaseModel):
    """Health check response model."""
    status: ServiceStatus
    services: Dict[str, ServiceStatus]
    version: str
    environment: str
    error: Optional[str] = None
