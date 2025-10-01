"""
Vector database endpoints.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
from app.services.vector_db import VectorDatabaseService

router = APIRouter()


@router.post("/vector-db/create-collection")
async def create_collection():
    """
    Create the documents collection in Qdrant.
    
    Returns:
        dict: Collection creation status
    """
    try:
        vector_db = VectorDatabaseService()
        success = await vector_db.create_collection()
        
        return {
            "success": success,
            "message": "Collection created successfully" if success else "Failed to create collection"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Collection creation failed: {str(e)}")


@router.post("/vector-db/upsert/{document_id}")
async def upsert_document(
    document_id: str,
    embedding: List[float],
    metadata: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """
    Upsert a document with its embedding into the vector database.
    
    Args:
        document_id: Document identifier
        embedding: Document embedding vector
        metadata: Document metadata
        background_tasks: FastAPI background tasks
        
    Returns:
        dict: Upsert status
    """
    try:
        vector_db = VectorDatabaseService()
        success = await vector_db.upsert_document(document_id, embedding, metadata)
        
        return {
            "success": success,
            "document_id": document_id,
            "message": "Document upserted successfully" if success else "Failed to upsert document"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upsert failed: {str(e)}")


@router.post("/vector-db/search")
async def search_similar_documents(
    query_embedding: List[float],
    limit: int = 10,
    score_threshold: float = 0.0,
    filter_conditions: Optional[Dict[str, Any]] = None
):
    """
    Search for similar documents using vector similarity.
    
    Args:
        query_embedding: Query embedding vector
        limit: Maximum number of results
        score_threshold: Minimum similarity score threshold
        filter_conditions: Optional filter conditions
        
    Returns:
        dict: Search results
    """
    try:
        vector_db = VectorDatabaseService()
        results = await vector_db.search_similar_documents(
            query_embedding, limit, score_threshold, filter_conditions
        )
        
        return {
            "results": results,
            "total_found": len(results),
            "query_dimension": len(query_embedding)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")


@router.get("/vector-db/document/{document_id}")
async def get_document_by_id(document_id: str):
    """
    Get a document by its ID from the vector database.
    
    Args:
        document_id: Document identifier
        
    Returns:
        dict: Document data
    """
    try:
        vector_db = VectorDatabaseService()
        document = await vector_db.get_document_by_id(document_id)
        
        if document is None:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        return document
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@router.delete("/vector-db/document/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the vector database.
    
    Args:
        document_id: Document identifier
        
    Returns:
        dict: Deletion status
    """
    try:
        vector_db = VectorDatabaseService()
        success = await vector_db.delete_document(document_id)
        
        return {
            "success": success,
            "document_id": document_id,
            "message": "Document deleted successfully" if success else "Failed to delete document"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document deletion failed: {str(e)}")


@router.get("/vector-db/collection-info")
async def get_collection_info():
    """
    Get information about the collection.
    
    Returns:
        dict: Collection information
    """
    try:
        vector_db = VectorDatabaseService()
        info = await vector_db.get_collection_info()
        
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")


@router.post("/vector-db/batch-upsert")
async def batch_upsert_documents(
    documents: List[Dict[str, Any]],
    background_tasks: BackgroundTasks
):
    """
    Batch upsert multiple documents.
    
    Args:
        documents: List of documents with embeddings
        background_tasks: FastAPI background tasks
        
    Returns:
        dict: Batch operation results
    """
    try:
        vector_db = VectorDatabaseService()
        result = await vector_db.batch_upsert_documents(documents)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch upsert failed: {str(e)}")
