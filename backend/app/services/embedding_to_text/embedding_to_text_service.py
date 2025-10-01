"""
Main embedding-to-text service implementing the full pipeline from the PDF.

This service orchestrates the complete pipeline:
1. Click (2D) → Inverse UMAP → High-D embedding → Text Generation
2. Correction loop to ensure generated text lands at target coordinates
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .models.base import (
    EmbeddingToTextRequest,
    EmbeddingToTextResponse,
    TextGenerationMethod
)
from .models.parametric_umap import ParametricUMAPConfig
from .models.text_generation import TextGenerationConfig
from .parametric_umap.parametric_umap_service import ParametricUMAPService
from .text_generation.text_generation_service import TextGenerationService
from .correction_loop.correction_loop_service import CorrectionLoopService

logger = logging.getLogger(__name__)


class EmbeddingToTextService:
    """
    Main service for the embedding-to-text pipeline.
    
    This implements the complete pipeline described in the PDF:
    1. User clicks a location (x, y) on the 2D map
    2. Parametric UMAP decoder maps (x, y) → high-dim embedding vector E_target
    3. Text generation step (MVP: Find nearest papers to E_target and prompt LLM)
    4. Re-embed generated text, compare with E_target, optionally refine
    5. Display generated text to user
    """
    
    def __init__(
        self,
        parametric_umap_config: Optional[ParametricUMAPConfig] = None,
        text_generation_config: Optional[TextGenerationConfig] = None
    ):
        self.parametric_umap_config = parametric_umap_config or ParametricUMAPConfig()
        self.text_generation_config = text_generation_config or TextGenerationConfig()
        
        # Initialize services
        self.parametric_umap_service = ParametricUMAPService(self.parametric_umap_config)
        self.text_generation_service = TextGenerationService(self.text_generation_config)
        self.correction_loop_service = CorrectionLoopService(self.parametric_umap_service)
        
        # State
        self.is_initialized = False
        self.documents: List[Dict[str, Any]] = []
        
    def initialize(self, documents: List[Dict[str, Any]]) -> None:
        """
        Initialize the service with documents and train models.
        
        Args:
            documents: List of documents with embeddings and metadata
        """
        logger.info(f"Initializing EmbeddingToTextService with {len(documents)} documents")
        
        self.documents = documents
        
        # Extract embeddings and coordinates
        embeddings = []
        coordinates = []
        
        for doc in documents:
            if 'embedding' in doc and 'coordinates' in doc:
                embeddings.append(doc['embedding'])
                coordinates.append(doc['coordinates'])
            else:
                logger.warning(f"Document {doc.get('document_id', 'unknown')} missing embedding or coordinates")
        
        if not embeddings:
            raise ValueError("No valid documents with embeddings and coordinates found")
        
        embeddings_array = np.array(embeddings)
        coordinates_array = np.array(coordinates)
        
        logger.info(f"Training Parametric UMAP with {len(embeddings_array)} samples")
        
        # Train Parametric UMAP
        model_path = Path("data/models/parametric_umap")
        self.parametric_umap_service.train_model(
            embeddings_array, 
            coordinates_array,
            save_path=model_path
        )
        
        # Initialize text generation service
        self.text_generation_service.load_documents(documents)
        
        # Train Vec2Text model if documents are available
        if len(documents) > 10:  # Only train if we have enough documents
            logger.info("Training Vec2Text model on available documents")
            try:
                self.text_generation_service.train_vec2text(documents, epochs=2)
                logger.info("Vec2Text model trained successfully")
            except Exception as e:
                logger.warning(f"Vec2Text training failed: {e}")
        
        self.is_initialized = True
        logger.info("EmbeddingToTextService initialized successfully")
    
    def load_trained_model(self, model_path: Path) -> None:
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the saved model directory
        """
        logger.info(f"Loading trained model from {model_path}")
        
        self.parametric_umap_service.load_model(model_path)
        
        # Load documents for text generation
        # Note: In practice, you'd load documents from the same source as training
        if self.documents:
            self.text_generation_service.load_documents(self.documents)
        
        self.is_initialized = True
        logger.info("Trained model loaded successfully")
    
    def generate_text_from_coordinates(
        self, 
        request: EmbeddingToTextRequest
    ) -> EmbeddingToTextResponse:
        """
        Generate text from 2D coordinates using the full pipeline.
        
        Args:
            request: Embedding-to-text request with coordinates and parameters
            
        Returns:
            Complete response with generated text and metadata
        """
        if not self.is_initialized:
            raise ValueError("Service must be initialized before generating text")
        
        start_time = time.time()
        
        try:
            logger.info(f"Generating text from coordinates ({request.x}, {request.y})")
            
            # Step 1: Parametric UMAP inverse transform (2D → embedding)
            target_embedding = self.parametric_umap_service.get_target_embedding(
                request.x, request.y
            )
            
            logger.info(f"Target embedding computed: shape {target_embedding.shape}")
            
            # Step 2: Text generation (embedding → text)
            text_result = self.text_generation_service.generate_text(
                target_embedding, request.method, target_coordinates=(request.x, request.y)
            )
            
            logger.info(f"Text generated: {text_result.title}")
            
            # Step 3: Correction loop (optional)
            correction_result = None
            if request.max_correction_iterations > 0:
                logger.info("Running correction loop")
                
                correction_result = self.correction_loop_service.correct_text_placement(
                    text_result=text_result,
                    target_embedding=target_embedding,
                    target_coordinates=(request.x, request.y),
                    max_iterations=request.max_correction_iterations,
                    distance_threshold=request.target_embedding_distance_threshold
                )
                
                logger.info(f"Correction completed: {correction_result.iterations_used} iterations")
            
            # Create response
            processing_time = time.time() - start_time
            
            response = EmbeddingToTextResponse(
                request=request,
                text_generation_result=text_result,
                correction_result=correction_result,
                success=True,
                processing_time_seconds=processing_time
            )
            
            logger.info(f"Text generation completed in {processing_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            
            processing_time = time.time() - start_time
            
            return EmbeddingToTextResponse(
                request=request,
                text_generation_result=None,
                correction_result=None,
                success=False,
                error_message=str(e),
                processing_time_seconds=processing_time
            )
    
    def get_model_evaluation(self) -> Dict[str, Any]:
        """
        Get evaluation metrics for the trained models.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_initialized:
            raise ValueError("Service must be initialized before evaluation")
        
        # Get Parametric UMAP evaluation
        umap_metrics = self.parametric_umap_service.evaluate_model()
        
        return {
            'parametric_umap': umap_metrics,
            'num_documents': len(self.documents),
            'text_generation_method': self.text_generation_config.method.value
        }
    
    def predict_coordinates_for_text(self, title: str, content: str) -> Tuple[float, float]:
        """
        Predict where a given text would land on the map.
        
        Args:
            title: Document title
            content: Document content
            
        Returns:
            Predicted (x, y) coordinates
        """
        if not self.is_initialized:
            raise ValueError("Service must be initialized before prediction")
        
        # Embed the text
        embedding = self.correction_loop_service._embed_combined_text(title, content)
        
        # Transform to coordinates
        coordinates = self.parametric_umap_service.transform(embedding.reshape(1, -1))[0]
        
        return tuple(coordinates)
