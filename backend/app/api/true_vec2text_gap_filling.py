#!/usr/bin/env python3
"""
True Vec2Text Gap Filling API
This endpoint uses the custom neural network-based Vec2Text service for true embedding-to-text generation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter()

# Global service instance
true_vec2text_service = None

class CoordinateRequest(BaseModel):
    x: float
    y: float

class CoordinateResponse(BaseModel):
    generated_text: str
    title: str
    method_used: str
    coordinate_error: float
    predicted_coordinates: List[float]
    target_coordinates: List[float]
    embedding_distance: float
    iterations_used: int
    converged: bool
    correction_history: List[Dict[str, Any]]
    precision_score: float
    processing_time_seconds: float

class BatchCoordinateRequest(BaseModel):
    coordinates: List[Tuple[float, float]]

class ServiceStatusResponse(BaseModel):
    service_available: bool
    parametric_umap_loaded: bool
    device: str
    training_data_loaded: bool
    training_samples: int

def get_true_vec2text_service():
    """Get or initialize the True Vec2Text service."""
    global true_vec2text_service
    
    if true_vec2text_service is None:
        try:
            from app.services.embedding_to_text.text_generation.custom_vec2text_service import CustomVec2TextService
            logger.info("Initializing True Vec2Text service...")
            true_vec2text_service = CustomVec2TextService()
            logger.info("True Vec2Text service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize True Vec2Text service: {e}")
            raise HTTPException(status_code=500, detail=f"Service initialization failed: {str(e)}")
    
    return true_vec2text_service

@router.post("/generate", response_model=CoordinateResponse)
async def generate_true_vec2text_gap_filling(request: CoordinateRequest) -> CoordinateResponse:
    """
    Generate text for coordinates using true Vec2Text neural network.
    
    Args:
        request: Coordinate request with x, y coordinates
        
    Returns:
        Generated text and metadata
    """
    start_time = time.time()
    
    try:
        service = get_true_vec2text_service()
        coordinates = (request.x, request.y)
        
        logger.info(f"Generating true Vec2Text for coordinates: {coordinates}")
        
        # Generate text using true Vec2Text
        result = service.generate_text_for_coordinates(coordinates)
        
        # Add processing time
        processing_time = time.time() - start_time
        result['processing_time_seconds'] = processing_time
        
        logger.info(f"True Vec2Text generation completed in {processing_time:.2f}s")
        logger.info(f"Generated text length: {len(result['generated_text'])} characters")
        logger.info(f"Coordinate error: {result['coordinate_error']:.2f}")
        
        return CoordinateResponse(**result)
        
    except Exception as e:
        logger.error(f"True Vec2Text generation failed: {e}")
        processing_time = time.time() - start_time
        
        # Return error response
        return CoordinateResponse(
            generated_text=f"True Vec2Text generation failed: {str(e)}",
            title="Generation Failed",
            method_used="true_vec2text_error",
            coordinate_error=999999.0,  # Use large number instead of inf
            predicted_coordinates=[request.x, request.y],
            target_coordinates=[request.x, request.y],
            embedding_distance=999999.0,  # Use large number instead of inf
            iterations_used=0,
            converged=False,
            correction_history=[],
            precision_score=0.0,
            processing_time_seconds=processing_time
        )

@router.post("/generate-batch")
async def generate_batch_true_vec2text_gap_filling(request: BatchCoordinateRequest) -> List[CoordinateResponse]:
    """
    Generate text for multiple coordinates using true Vec2Text.
    
    Args:
        request: Batch coordinate request
        
    Returns:
        List of generated text results
    """
    start_time = time.time()
    
    try:
        service = get_true_vec2text_service()
        
        logger.info(f"Generating true Vec2Text for {len(request.coordinates)} coordinates")
        
        results = []
        for coordinates in request.coordinates:
            try:
                result = service.generate_text_for_coordinates(coordinates)
                processing_time = time.time() - start_time
                result['processing_time_seconds'] = processing_time
                results.append(CoordinateResponse(**result))
            except Exception as e:
                logger.error(f"Batch generation failed for {coordinates}: {e}")
                processing_time = time.time() - start_time
                error_result = CoordinateResponse(
                    generated_text=f"Generation failed: {str(e)}",
                    title="Generation Failed",
                    method_used="true_vec2text_error",
                    coordinate_error=999999.0,  # Use large number instead of inf
                    predicted_coordinates=list(coordinates),
                    target_coordinates=list(coordinates),
                    embedding_distance=999999.0,  # Use large number instead of inf
                    iterations_used=0,
                    converged=False,
                    correction_history=[],
                    precision_score=0.0,
                    processing_time_seconds=processing_time
                )
                results.append(error_result)
        
        logger.info(f"Batch generation completed for {len(results)} coordinates")
        return results
        
    except Exception as e:
        logger.error(f"Batch True Vec2Text generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

@router.get("/status", response_model=ServiceStatusResponse)
async def get_true_vec2text_status() -> ServiceStatusResponse:
    """
    Get the status of the True Vec2Text service.
    
    Returns:
        Service status information
    """
    try:
        service = get_true_vec2text_service()
        status = service.get_service_status()
        
        return ServiceStatusResponse(
            service_available=status['custom_vec2text_available'],
            parametric_umap_loaded=status['parametric_umap_loaded'],
            device=status['device'],
            training_data_loaded=status['training_data_loaded'],
            training_samples=status['training_samples']
        )
        
    except Exception as e:
        logger.error(f"Failed to get service status: {e}")
        return ServiceStatusResponse(
            service_available=False,
            parametric_umap_loaded=False,
            device="unknown",
            training_data_loaded=False,
            training_samples=0
        )

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    try:
        service = get_true_vec2text_service()
        status = service.get_service_status()
        
        if status['custom_vec2text_available']:
            return {"status": "healthy", "service": "true_vec2text"}
        else:
            return {"status": "unhealthy", "service": "true_vec2text", "error": "Service not available"}
            
    except Exception as e:
        return {"status": "unhealthy", "service": "true_vec2text", "error": str(e)}

@router.post("/train")
async def train_true_vec2text_model(
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4
) -> Dict[str, Any]:
    """
    Train the True Vec2Text neural network.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        
    Returns:
        Training results
    """
    try:
        service = get_true_vec2text_service()
        
        logger.info(f"Starting True Vec2Text training: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        # Train the network
        service.train_network(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        
        # Save the trained model
        model_path = "data/models/true_vec2text"
        service.save_model(model_path)
        
        return {
            "status": "success",
            "message": "True Vec2Text model trained successfully",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_saved_to": model_path
        }
        
    except Exception as e:
        logger.error(f"True Vec2Text training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/test-precision")
async def test_coordinate_precision() -> Dict[str, Any]:
    """
    Test the coordinate precision of the True Vec2Text service.
    
    Returns:
        Precision test results
    """
    try:
        service = get_true_vec2text_service()
        
        # Test multiple coordinates
        test_coordinates = [
            (0.0, 0.0),
            (10.5, -5.2),
            (-15.3, 8.7),
            (25.1, -12.4),
            (5.0, 5.0)
        ]
        
        results = []
        total_error = 0
        converged_count = 0
        
        for coords in test_coordinates:
            result = service.generate_text_for_coordinates(coords)
            results.append({
                'coordinates': coords,
                'error': result['coordinate_error'],
                'converged': result['converged'],
                'method': result['method_used']
            })
            
            total_error += result['coordinate_error']
            if result['converged']:
                converged_count += 1
        
        avg_error = total_error / len(test_coordinates)
        precision_rate = converged_count / len(test_coordinates)
        
        return {
            "status": "success",
            "test_coordinates": len(test_coordinates),
            "converged_count": converged_count,
            "precision_rate": precision_rate,
            "average_error": avg_error,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Precision test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Precision test failed: {str(e)}")
