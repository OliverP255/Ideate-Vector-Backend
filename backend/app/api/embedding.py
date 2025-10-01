"""
Document embedding endpoints.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
from app.services.embedding import EmbeddingService
from app.services.ingest import IngestService

router = APIRouter()


@router.post("/embedding/generate/{document_id}")
async def generate_embedding(
    document_id: str,
    background_tasks: BackgroundTasks
):
    """
    Generate embedding for a document.
    
    Args:
        document_id: Document identifier
        background_tasks: FastAPI background tasks
        
    Returns:
        dict: Generation status
    """
    try:
        # Get document data from ingestion service
        ingest_service = IngestService()
        document_file = ingest_service.output_dir / f"{document_id}.json"
        
        if not document_file.exists():
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Load document data
        import json
        with open(document_file, 'r') as f:
            document_data = json.load(f)
        
        # Generate embedding
        embedding_service = EmbeddingService()
        result = await embedding_service.generate_embedding(document_id, document_data)
        
        return {
            "document_id": document_id,
            "status": "completed",
            "embedding_dimension": result["embedding_dimension"],
            "embedding_method": result["embedding_method"],
            "chunk_count": result["chunk_count"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@router.get("/embedding/{document_id}")
async def get_embedding(document_id: str):
    """
    Get embedding for a document.
    
    Args:
        document_id: Document identifier
        
    Returns:
        dict: Embedding data
    """
    try:
        embedding_service = EmbeddingService()
        embedding_data = await embedding_service.get_embedding(document_id)
        
        if embedding_data is None:
            raise HTTPException(status_code=404, detail=f"Embedding for document {document_id} not found")
        
        return embedding_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get embedding: {str(e)}")


@router.post("/embedding/similar")
async def find_similar_documents(
    query_text: str,
    limit: int = 10,
    exclude_document_ids: Optional[List[str]] = None
):
    """
    Find similar documents based on text query.
    
    Args:
        query_text: Text to find similar documents for
        limit: Maximum number of results
        exclude_document_ids: Document IDs to exclude
        
    Returns:
        dict: List of similar documents
    """
    try:
        embedding_service = EmbeddingService()
        
        # Generate embedding for query text
        query_data = {"text": query_text, "title": "", "document_id": "query"}
        query_result = embedding_service.generator.generate_embedding(query_data)
        query_embedding = query_result["embedding"]
        
        # Find similar documents
        similar_docs = await embedding_service.find_similar_documents(
            query_embedding, 
            limit, 
            exclude_document_ids
        )
        
        return {
            "query_text": query_text,
            "similar_documents": similar_docs,
            "total_found": len(similar_docs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")


@router.get("/embedding/list")
async def list_embeddings():
    """
    List all generated embeddings.
    
    Returns:
        dict: List of embedding metadata
    """
    try:
        embedding_service = EmbeddingService()
        embeddings = await embedding_service.list_embeddings()
        
        return {
            "embeddings": embeddings,
            "total": len(embeddings)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list embeddings: {str(e)}")


@router.post("/embedding/batch")
async def batch_generate_embeddings(
    document_ids: List[str],
    background_tasks: BackgroundTasks
):
    """
    Generate embeddings for multiple documents.
    
    Args:
        document_ids: List of document identifiers
        background_tasks: FastAPI background tasks
        
    Returns:
        dict: Batch generation status
    """
    try:
        ingest_service = IngestService()
        embedding_service = EmbeddingService()
        
        documents = []
        for document_id in document_ids:
            document_file = ingest_service.output_dir / f"{document_id}.json"
            if document_file.exists():
                import json
                with open(document_file, 'r') as f:
                    document_data = json.load(f)
                documents.append(document_data)
        
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found")
        
        # Generate embeddings
        results = await embedding_service.batch_generate_embeddings(documents)
        
        return {
            "status": "completed",
            "processed_documents": len(results),
            "successful": len([r for r in results if 'error' not in r]),
            "failed": len([r for r in results if 'error' in r])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch embedding generation failed: {str(e)}")
