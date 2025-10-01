"""
Document ingestion models.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any
from enum import Enum
from datetime import datetime


class IngestStatus(str, Enum):
    """Document ingestion status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestRequest(BaseModel):
    """Document ingestion request."""
    file_path: str
    file_type: str  # pdf, html, txt
    metadata: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None


class IngestResponse(BaseModel):
    """Document ingestion response."""
    document_id: str
    status: IngestStatus
    message: str
    created_at: datetime = datetime.now()
    metadata: Optional[Dict[str, Any]] = None
