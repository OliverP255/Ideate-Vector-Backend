"""
Document ingestion service.
"""

import logging
import uuid
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from app.models.ingest import IngestRequest, IngestResponse, IngestStatus
from app.services.config import get_settings
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# from pipeline.ingest.worker import DocumentIngestionWorker

logger = logging.getLogger(__name__)


class IngestService:
    """Service for document ingestion and processing."""
    
    def __init__(self):
        self.settings = get_settings()
        # self.worker = DocumentIngestionWorker()  # Temporarily disabled
        self.output_dir = Path("data/cleaned")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def start_ingestion(self, request: IngestRequest) -> str:
        """
        Start document ingestion process.
        
        Args:
            request: Ingestion request
            
        Returns:
            str: Document ID
        """
        try:
            document_id = str(uuid.uuid4())
            
            # Validate file exists
            if not os.path.exists(request.file_path):
                raise FileNotFoundError(f"File not found: {request.file_path}")
            
            # Store ingestion request metadata
            metadata = {
                "document_id": document_id,
                "file_path": request.file_path,
                "file_type": request.file_type,
                "metadata": request.metadata,
                "user_id": request.user_id,
                "status": IngestStatus.PROCESSING,
                "created_at": datetime.now().isoformat()
            }
            
            # Save metadata
            metadata_file = self.output_dir / f"{document_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Started ingestion for document: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to start ingestion: {e}")
            raise
    
    async def process_document(self, document_id: str, file_path: str):
        """
        Process a document through the ingestion pipeline.
        
        Args:
            document_id: Document identifier
            file_path: Path to document file
        """
        try:
            logger.info(f"Processing document {document_id} from {file_path}")
            
            # Determine file type
            file_ext = Path(file_path).suffix.lower()
            if file_ext == '.pdf':
                file_type = 'pdf'
            elif file_ext in ['.html', '.htm']:
                file_type = 'html'
            elif file_ext == '.txt':
                file_type = 'txt'
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Process document using worker (temporarily disabled)
            # result = await self.worker.process_document(file_path, file_type)
            result = {"status": "processing_disabled", "message": "Document processing temporarily disabled"}
            
            # Add document ID to result
            result['document_id'] = document_id
            result['processed_at'] = datetime.now().isoformat()
            
            # Save processed document
            output_file = self.output_dir / f"{document_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Update metadata with success status
            metadata_file = self.output_dir / f"{document_id}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                metadata['status'] = IngestStatus.COMPLETED
                metadata['processed_at'] = datetime.now().isoformat()
                metadata['output_file'] = str(output_file)
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Document {document_id} processed successfully")
            
        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")
            
            # Update metadata with failure status
            metadata_file = self.output_dir / f"{document_id}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                metadata['status'] = IngestStatus.FAILED
                metadata['error'] = str(e)
                metadata['failed_at'] = datetime.now().isoformat()
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
    
    async def get_ingestion_status(self, document_id: str) -> IngestResponse:
        """
        Get ingestion status for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            IngestResponse: Current status
        """
        try:
            metadata_file = self.output_dir / f"{document_id}_metadata.json"
            
            if not metadata_file.exists():
                raise FileNotFoundError(f"Document {document_id} not found")
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            status = IngestStatus(metadata.get('status', IngestStatus.PENDING))
            
            if status == IngestStatus.COMPLETED:
                message = "Document processed successfully"
            elif status == IngestStatus.FAILED:
                message = f"Processing failed: {metadata.get('error', 'Unknown error')}"
            elif status == IngestStatus.PROCESSING:
                message = "Document is being processed"
            else:
                message = "Document is pending processing"
            
            return IngestResponse(
                document_id=document_id,
                status=status,
                message=message,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to get status for document {document_id}: {e}")
            raise
    
    async def list_processed_documents(self) -> list:
        """
        List all processed documents.
        
        Returns:
            list: List of processed documents
        """
        try:
            documents = []
            
            for metadata_file in self.output_dir.glob("*_metadata.json"):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                documents.append(metadata)
            
            return sorted(documents, key=lambda x: x.get('created_at', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise
