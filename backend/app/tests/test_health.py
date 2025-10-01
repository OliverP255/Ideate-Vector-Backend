"""
Health check tests.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "services" in data
    assert "version" in data
    assert "environment" in data


def test_readiness_check():
    """Test readiness check endpoint."""
    response = client.get("/api/health/ready")
    assert response.status_code in [200, 503]  # May be unhealthy in test environment


def test_liveness_check():
    """Test liveness check endpoint."""
    response = client.get("/api/health/live")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "alive"
