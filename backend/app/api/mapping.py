"""
Document mapping endpoints for dimensionality reduction.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
from app.services.mapping import MappingService

router = APIRouter()


@router.post("/mapping/create")
async def create_initial_mapping(
    document_ids: List[str],
    background_tasks: BackgroundTasks
):
    """
    Create initial 2D mapping for a set of documents.
    
    Args:
        document_ids: List of document IDs to map
        background_tasks: FastAPI background tasks
        
    Returns:
        dict: Mapping creation status
    """
    try:
        mapping_service = MappingService()
        result = await mapping_service.create_initial_mapping(document_ids)
        
        return {
            "status": "completed",
            "document_count": len(result['document_ids']),
            "pca_components": result['mapping_metadata']['pca_components'],
            "pca_explained_variance": result['mapping_metadata']['pca_explained_variance'],
            "training_mse": result['mapping_metadata']['training_metrics']['mse'],
            "training_mae": result['mapping_metadata']['training_metrics']['mae']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mapping creation failed: {str(e)}")


@router.post("/mapping/project")
async def project_new_documents(
    document_ids: List[str],
    background_tasks: BackgroundTasks
):
    """
    Project new documents to 2D using the parametric mapper.
    
    Args:
        document_ids: List of new document IDs to project
        background_tasks: FastAPI background tasks
        
    Returns:
        dict: Projection status
    """
    try:
        mapping_service = MappingService()
        result = await mapping_service.project_new_documents(document_ids)
        
        return {
            "status": "completed",
            "document_count": len(result['document_ids']),
            "method": result['projection_metadata']['method'],
            "projected_at": result['projection_metadata']['projected_at']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document projection failed: {str(e)}")


@router.get("/mapping/coordinates/{document_id}")
async def get_document_coordinates(document_id: str):
    """
    Get 2D coordinates for a specific document.
    
    Args:
        document_id: Document identifier
        
    Returns:
        dict: Document coordinates
    """
    try:
        mapping_service = MappingService()
        coordinates = await mapping_service.get_document_coordinates(document_id)
        
        if coordinates is None:
            raise HTTPException(status_code=404, detail=f"Coordinates for document {document_id} not found")
        
        return coordinates
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get coordinates: {str(e)}")


@router.get("/mapping/coordinates")
async def get_all_coordinates():
    """
    Get all document coordinates.
    
    Returns:
        dict: All document coordinates
    """
    try:
        mapping_service = MappingService()
        result = await mapping_service.get_all_coordinates()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get all coordinates: {str(e)}")


@router.post("/mapping/validate")
async def validate_mapping_accuracy(test_document_ids: List[str]):
    """
    Validate mapping accuracy on held-out documents.
    
    Args:
        test_document_ids: List of document IDs to test
        
    Returns:
        dict: Validation results
    """
    try:
        mapping_service = MappingService()
        result = await mapping_service.validate_mapping_accuracy(test_document_ids)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mapping validation failed: {str(e)}")


@router.get("/mapping/status")
async def get_mapping_status():
    """
    Get current mapping status and metadata.
    
    Returns:
        dict: Mapping status
    """
    try:
        mapping_service = MappingService()
        
        # Check if initial mapping exists
        initial_mapping_file = mapping_service.output_dir / "initial_mapping.json"
        has_initial_mapping = initial_mapping_file.exists()
        
        # Check if model exists
        model_dir = mapping_service.model_dir
        has_model = (model_dir / "metadata.json").exists()
        
        # Count projection files
        projection_files = list(mapping_service.output_dir.glob("projection_*.json"))
        
        return {
            "has_initial_mapping": has_initial_mapping,
            "has_trained_model": has_model,
            "projection_count": len(projection_files),
            "model_directory": str(model_dir),
            "mapping_directory": str(mapping_service.output_dir)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get mapping status: {str(e)}")
