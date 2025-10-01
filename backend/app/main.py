"""
Knowledge Map Backend Service

FastAPI application for document ingestion, embedding, and map visualization.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging
import os

from app.api import health, ingest, map_api, click, user, embedding, mapping, vector_db, tiles, search, upload, gap_filling, advanced_gap_filling, simple_gap_filling, true_vec2text_gap_filling
from app.db.database import init_db
from app.services.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Knowledge Map API",
    description="Interactive document visualization and retrieval system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Get settings
settings = get_settings()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for production
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["knowledge-map.com", "*.knowledge-map.com"]
    )

# Include API routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(ingest.router, prefix="/api", tags=["ingestion"])
app.include_router(embedding.router, prefix="/api", tags=["embedding"])
app.include_router(mapping.router, prefix="/api", tags=["mapping"])
app.include_router(vector_db.router, prefix="/api", tags=["vector-db"])
app.include_router(map_api.router, prefix="/api", tags=["map"])
app.include_router(click.router, prefix="/api", tags=["click"])
app.include_router(user.router, prefix="/api", tags=["user"])
app.include_router(tiles.router, prefix="/api", tags=["tiles"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(gap_filling.router, prefix="/api", tags=["gap-filling"])

# Include advanced gap filling router with debug logging
try:
    logger.info("Including advanced gap filling router...")
    app.include_router(advanced_gap_filling.router, prefix="/api", tags=["advanced-gap-filling"])
    logger.info("Advanced gap filling router included successfully")
except Exception as e:
    logger.error(f"Failed to include advanced gap filling router: {e}")

# Include simple gap filling router
try:
    logger.info("Including simple gap filling router...")
    app.include_router(simple_gap_filling.router, prefix="/api", tags=["simple-gap-filling"])
    logger.info("Simple gap filling router included successfully")
except Exception as e:
    logger.error(f"Failed to include simple gap filling router: {e}")

# Include true Vec2Text gap filling router
try:
    logger.info("Including true Vec2Text gap filling router...")
    app.include_router(true_vec2text_gap_filling.router, prefix="/api/true-vec2text", tags=["true-vec2text"])
    logger.info("True Vec2Text gap filling router included successfully")
except Exception as e:
    logger.error(f"Failed to include true Vec2Text gap filling router: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup."""
    logger.info("Starting Knowledge Map API...")
    await init_db()
    logger.info("Database initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Knowledge Map API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
