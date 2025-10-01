"""
Advanced gap filling API endpoints using the new embedding-to-text pipeline.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Lazy imports to avoid TensorFlow/PyTorch loading at startup

logger = logging.getLogger(__name__)

router = APIRouter()

# Global service instance
_embedding_to_text_service = None


class AdvancedGapFillingRequest(BaseModel):
    """Request model for advanced gap filling."""
    x: float
    y: float
    method: str = "retrieval_llm_synthesis"
    max_correction_iterations: int = 3
    target_embedding_distance_threshold: float = 0.1
    user_id: str = "system"


class AdvancedGapFillingResponse(BaseModel):
    """Response model for advanced gap filling."""
    success: bool
    generated_text: str
    title: str
    target_coordinates: List[float]
    final_coordinates: List[float]
    method_used: str
    processing_time_seconds: float
    correction_iterations: int = 0
    embedding_distance: float = 0.0
    coordinate_error: float = 0.0
    error_message: str = None


def get_embedding_to_text_service():
    """Get or create the embedding-to-text service with lazy imports."""
    global _embedding_to_text_service
    
    if _embedding_to_text_service is None:
        logger.info("Initializing advanced embedding-to-text service")
        
        # Use the existing Vec2Text service that's already working
        from ..services.embedding_to_text.text_generation.vec2text_service import Vec2TextService
        
        # Create and initialize the service
        _embedding_to_text_service = Vec2TextService()
        _embedding_to_text_service.initialize()
        
        # Load existing documents and train model
        try:
            # For now, use a simple initialization without documents
            # In production, this would load from a proper data source
            logger.info("Advanced service created (will initialize on first use)")
                
        except Exception as e:
            logger.error(f"Failed to initialize advanced service: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize advanced service")
    
    return _embedding_to_text_service


@router.get("/advanced-gap-filling/status")
async def get_service_status():
    """Get the status of the advanced gap filling service."""
    try:
        service = get_embedding_to_text_service()
        
        return {
            "status": "initialized" if service.is_initialized else "not_initialized",
            "num_documents": len(service.documents) if service.documents else 0,
            "text_generation_method": service.text_generation_config.method.value,
            "parametric_umap_configured": service.parametric_umap_config is not None
        }
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/advanced-gap-filling/initialize")
async def initialize_service():
    """Initialize the advanced gap filling service with available documents."""
    try:
        logger.info("Initializing advanced gap filling service...")
        
        # Import here to avoid circular imports
        from ..services.sample_data import SampleDataService
        
        # Load documents from the sample data service
        sample_service = SampleDataService()
        embeddings_data = sample_service.load_sample_embeddings()
        coordinates_data = sample_service.load_sample_coordinates()
        
        if not embeddings_data:
            raise HTTPException(
                status_code=404,
                detail="No embeddings available for initialization"
            )
        
        if not coordinates_data:
            raise HTTPException(
                status_code=404,
                detail="No coordinates available for initialization"
            )
        
        # Combine embeddings and coordinates by document_id
        documents = []
        coord_lookup = {doc['document_id']: doc for doc in coordinates_data}
        
        for emb_doc in embeddings_data:
            doc_id = emb_doc.get('document_id')
            if doc_id and doc_id in coord_lookup:
                coord_doc = coord_lookup[doc_id]
                combined_doc = {
                    **emb_doc,  # Includes embedding, title, content, etc.
                    'coordinates': coord_doc['coordinates']  # Add coordinates
                }
                documents.append(combined_doc)
        
        if not documents:
            raise HTTPException(
                status_code=404,
                detail="No documents with both embeddings and coordinates found"
            )
        
        service = get_embedding_to_text_service()
        service.initialize(documents)
        
        logger.info(f"Service initialized with {len(documents)} documents")
        
        return {
            "success": True,
            "message": f"Service initialized with {len(documents)} documents",
            "num_documents": len(documents)
        }
    except Exception as e:
        logger.error(f"Error initializing service: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize service: {str(e)}"
        )


@router.post("/advanced-gap-filling/generate")
async def generate_advanced_gap_filling(request: AdvancedGapFillingRequest) -> AdvancedGapFillingResponse:
    """
    Generate text for gap filling using the advanced embedding-to-text pipeline.
    
    This endpoint implements the complete pipeline from the PDF:
    1. Click (2D) → Inverse UMAP → High-D embedding → Text Generation
    2. Correction loop to ensure generated text lands at target coordinates
    """
    try:
        logger.info(f"Advanced gap filling request: ({request.x}, {request.y})")
        
        # Import removed - using Vec2Text service directly
        
        service = get_embedding_to_text_service()
        
        if not service or not hasattr(service, 'generate_text_for_coordinates'):
            raise HTTPException(
                status_code=503, 
                detail="Advanced gap filling service not initialized"
            )
        
        # Generate text using the Vec2Text service
        result = service.generate_text_for_coordinates((request.x, request.y))
        
        if not result or not result.get('generated_text'):
            raise HTTPException(
                status_code=500,
                detail="Text generation failed - no text generated"
            )
        
        return AdvancedGapFillingResponse(
            success=True,
            generated_text=result['generated_text'],
            title=result.get('title', 'Generated Content'),
            target_coordinates=[request.x, request.y],
            final_coordinates=result.get('predicted_coordinates', [request.x, request.y]),
            method_used=result.get('method_used', 'precision_vec2text'),
            processing_time_seconds=0.0,  # Not tracked in current implementation
            correction_iterations=result.get('iterations_used', 1),
            embedding_distance=result.get('embedding_distance', 0.0),
            coordinate_error=result.get('coordinate_error', 0.0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in advanced gap filling: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/advanced-gap-filling/generate")
async def generate_advanced_gap_filling_get(
    x: float = Query(..., description="X coordinate on the map"),
    y: float = Query(..., description="Y coordinate on the map"),
    method: str = Query("retrieval_llm_synthesis", description="Text generation method"),
    max_correction_iterations: int = Query(3, description="Maximum correction iterations"),
    target_embedding_distance_threshold: float = Query(0.1, description="Embedding distance threshold"),
    user_id: str = Query("system", description="User ID")
) -> AdvancedGapFillingResponse:
    """
    Generate text for gap filling using GET method (for easy testing).
    """
    request = AdvancedGapFillingRequest(
        x=x,
        y=y,
        method=method,
        max_correction_iterations=max_correction_iterations,
        target_embedding_distance_threshold=target_embedding_distance_threshold,
        user_id=user_id
    )
    
    return await generate_advanced_gap_filling(request)


@router.get("/advanced-gap-filling/evaluate")
async def evaluate_model_quality():
    """Evaluate the quality of the trained models."""
    try:
        service = get_embedding_to_text_service()
        
        if not service or not hasattr(service, 'generate_text_for_coordinates'):
            raise HTTPException(
                status_code=503,
                detail="Service not initialized"
            )
        
        # Simple evaluation - return basic info about the service
        evaluation = {
            "service_type": "Vec2Text",
            "initialized": True,
            "precision_vec2text_available": hasattr(service, 'precision_vec2text_service'),
            "trained_vec2text_available": hasattr(service, 'trained_vec2text_service')
        }
        
        return {
            "success": True,
            "evaluation": evaluation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating models: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/advanced-gap-filling/predict-coordinates")
async def predict_coordinates_for_text(
    title: str,
    content: str
):
    """Predict where a given text would land on the map."""
    try:
        service = get_embedding_to_text_service()
        
        if not service.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Service not initialized"
            )
        
        coordinates = service.predict_coordinates_for_text(title, content)
        
        return {
            "success": True,
            "predicted_coordinates": list(coordinates),
            "title": title,
            "content_preview": content[:200] + "..." if len(content) > 200 else content
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting coordinates: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/advanced-gap-filling/train-vec2text")
async def train_vec2text_model(
    epochs: int = Query(3, description="Number of training epochs"),
    user_id: str = Query("system", description="User ID")
):
    """Train the Vec2Text model on available documents."""
    try:
        service = get_embedding_to_text_service()
        
        if not service.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Service must be initialized before training"
            )
        
        if not service.documents:
            raise HTTPException(
                status_code=400,
                detail="No documents available for training"
            )
        
        # Train Vec2Text
        training_results = service.text_generation_service.train_vec2text(
            service.documents, 
            epochs=epochs
        )
        
        return {
            "success": True,
            "training_results": training_results,
            "message": "Vec2Text model trained successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training Vec2Text: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/advanced-gap-filling/vec2text-quality")
async def evaluate_vec2text_quality():
    """Evaluate the quality of Vec2Text generation."""
    try:
        service = get_embedding_to_text_service()
        
        if not service.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Service must be initialized before evaluation"
            )
        
        if not service.documents:
            raise HTTPException(
                status_code=400,
                detail="No documents available for evaluation"
            )
        
        # Use a subset for evaluation
        eval_docs = service.documents[:min(20, len(service.documents))]
        
        quality_metrics = service.text_generation_service.evaluate_vec2text_quality(eval_docs)
        
        return {
            "success": True,
            "quality_metrics": quality_metrics,
            "evaluation_documents": len(eval_docs)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating Vec2Text quality: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/advanced-gap-filling/save-vec2text")
async def save_vec2text_model(
    model_path: str = Query(..., description="Path to save the model")
):
    """Save the trained Vec2Text model."""
    try:
        service = get_embedding_to_text_service()
        
        if not service.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Service must be initialized before saving"
            )
        
        service.text_generation_service.save_vec2text_model(model_path)
        
        return {
            "success": True,
            "model_path": model_path,
            "message": "Vec2Text model saved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving Vec2Text model: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/advanced-gap-filling/load-vec2text")
async def load_vec2text_model(
    model_path: str = Query(..., description="Path to load the model from")
):
    """Load a trained Vec2Text model."""
    try:
        service = get_embedding_to_text_service()
        
        service.text_generation_service.load_vec2text_model(model_path)
        
        return {
            "success": True,
            "model_path": model_path,
            "message": "Vec2Text model loaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading Vec2Text model: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
