"""
Document ingestion endpoints.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
from app.models.ingest import IngestRequest, IngestResponse, IngestStatus
from app.services.ingest import IngestService

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest a new document into the knowledge map.
    
    Args:
        request: Document ingestion request
        background_tasks: FastAPI background tasks
        
    Returns:
        IngestResponse: Ingestion status and document ID
    """
    try:
        ingest_service = IngestService()
        
        # Start ingestion in background
        document_id = await ingest_service.start_ingestion(request)
        
        # Add background task for processing
        background_tasks.add_task(
            ingest_service.process_document,
            document_id,
            request.file_path
        )
        
        return IngestResponse(
            document_id=document_id,
            status=IngestStatus.PROCESSING,
            message="Document ingestion started"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {str(e)}")


@router.get("/ingest/{document_id}/status", response_model=IngestResponse)
async def get_ingestion_status(document_id: str):
    """
    Get the status of a document ingestion.
    
    Args:
        document_id: Unique document identifier
        
    Returns:
        IngestResponse: Current ingestion status
    """
    try:
        ingest_service = IngestService()
        status = await ingest_service.get_ingestion_status(document_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Document not found: {str(e)}")


@router.post("/ingest/batch", response_model=List[IngestResponse])
async def ingest_batch(
    requests: List[IngestRequest],
    background_tasks: BackgroundTasks
):
    """
    Ingest multiple documents in batch.
    
    Args:
        requests: List of document ingestion requests
        background_tasks: FastAPI background tasks
        
    Returns:
        List[IngestResponse]: Ingestion status for each document
    """
    try:
        ingest_service = IngestService()
        responses = []
        
        for request in requests:
            document_id = await ingest_service.start_ingestion(request)
            background_tasks.add_task(
                ingest_service.process_document,
                document_id,
                request.file_path
            )
            responses.append(IngestResponse(
                document_id=document_id,
                status=IngestStatus.PROCESSING,
                message="Document ingestion started"
            ))
        
        return responses
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch ingestion failed: {str(e)}")


@router.get("/ingest/documents")
async def list_processed_documents():
    """
    List all processed documents.
    
    Returns:
        List[dict]: List of processed documents with metadata
    """
    try:
        ingest_service = IngestService()
        documents = await ingest_service.list_processed_documents()
        return {"documents": documents, "total": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")
