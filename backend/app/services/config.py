"""
Configuration management for the Knowledge Map service.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database Configuration (local SQLite)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "knowledge_map"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres_password"
    
    # Vector Database (local ChromaDB)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "documents"
    
    # Redis (disabled for local development)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    
    # Object Storage
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "knowledge-map"
    
    # GROBID Service
    grobid_host: str = "grobid"
    grobid_port: int = 8070
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "text-embedding-3-large"
    
    # Local Embedding Model
    sentence_transformers_model: str = "all-MiniLM-L12-v2"
    
    # Application Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_port: int = 3000
    
    # Environment
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    # Security
    secret_key: str = "your-secret-key-here"
    jwt_secret: str = "your-jwt-secret-here"
    
    # Monitoring
    sentry_dsn: Optional[str] = None
    prometheus_port: int = 9090
    
    # Feature Flags
    use_openai_embeddings: bool = False
    use_parametric_umap: bool = True
    enable_user_overlay: bool = True
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
