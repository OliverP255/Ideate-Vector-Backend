"""
Text generation service implementing the MVP approach from the PDF.

This service implements the "Retrieval + LLM Synthesis" method:
1. Find nearest-neighbor papers in embedding space
2. Prompt an LLM to synthesize a new abstract that fills the semantic gap
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from ..models.base import TextGenerationMethod, TextGenerationResult
from ..models.text_generation import TextGenerationConfig
from .vec2text_service import Vec2TextService
from .simple_coordinate_service import SimpleCoordinateTextService

logger = logging.getLogger(__name__)


class TextGenerationService:
    """
    Service for generating text from embeddings using various methods.
    
    Currently implements MVP approach: Retrieval + LLM Synthesis
    Future: Will support embedding-conditioned generators and Vec2Text
    """
    
    def __init__(self, config: Optional[TextGenerationConfig] = None):
        self.config = config or TextGenerationConfig()
        self.vec2text = Vec2TextService()
        self.simple_coordinate_service = SimpleCoordinateTextService()
        self._document_cache: Optional[Dict[str, Any]] = None
    
    def load_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Load documents and their embeddings for nearest neighbor search.
        
        Args:
            documents: List of documents with embeddings and metadata
        """
        logger.info(f"Loading {len(documents)} documents for text generation")
        
        self._document_cache = {doc['document_id']: doc for doc in documents}
        
        # Extract embeddings and filter documents
        embeddings = []
        valid_documents = []
        for doc in documents:
            if 'embedding' in doc and doc['embedding']:
                embeddings.append(doc['embedding'])
                valid_documents.append(doc)
            else:
                logger.warning(f"Document {doc['document_id']} missing embedding")
        
        if not embeddings:
            raise ValueError("No valid embeddings found in documents")
        
        # Update document cache to only include documents with embeddings
        self._document_cache = {doc['document_id']: doc for doc in valid_documents}
        
        logger.info(f"Loaded {len(valid_documents)} documents for Vec2Text")
    
    def generate_text(
        self, 
        target_embedding: np.ndarray,
        method: Optional[TextGenerationMethod] = None,
        target_coordinates: Optional[Tuple[float, float]] = None
    ) -> TextGenerationResult:
        """
        Generate text from a target embedding.
        
        Args:
            target_embedding: Target embedding vector
            method: Text generation method (defaults to config method)
            target_coordinates: Target coordinates (x, y) for coordinate-aware generation
            
        Returns:
            TextGenerationResult with generated text and metadata
        """
        if method is None:
            method = self.config.method
        
        if not self._document_cache:
            raise ValueError("Documents must be loaded before text generation")
        
        # Handle string method inputs
        if isinstance(method, str):
            try:
                method = TextGenerationMethod(method)
            except ValueError:
                raise ValueError(f"Unsupported text generation method: {method}")
        
        logger.info(f"Generating text using method: {method.value}")
        
        # Use Vec2Text for unique text generation that embeds to target coordinates
        return self._vec2text_generation(target_embedding, target_coordinates)
    
    def _simple_coordinate_generation(self, x: float, y: float) -> TextGenerationResult:
        """
        Generate text using simple coordinate-based approach.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            
        Returns:
            TextGenerationResult with generated text and metadata
        """
        logger.info(f"Generating text using simple coordinate-based approach for ({x:.2f}, {y:.2f})")
        
        # Generate coordinate-specific text
        result = self.simple_coordinate_service.generate_text_from_coordinates(x, y)
        
        return TextGenerationResult(
            generated_text=result['full_text'],
            title=result['title'],
            target_embedding=[],  # Empty since we're not using embeddings
            predicted_coordinates=tuple(result['target_coordinates']),
            method_used=TextGenerationMethod.VEC2TEXT,  # Keep same method for compatibility
            nearest_neighbors=[],
            generation_metadata={
                'simple_coordinate_based': True,
                'target_coordinates': result['target_coordinates'],
                'region': result['region']
            }
        )
    
    
    
    def _vec2text_generation(self, target_embedding: np.ndarray, target_coordinates: Optional[Tuple[float, float]] = None) -> TextGenerationResult:
        """
        Generate text using Vec2Text.
        
        Args:
            target_embedding: Target embedding vector
            target_coordinates: Target coordinates (x, y) for coordinate-aware generation
            
        Returns:
            TextGenerationResult
        """
        logger.info("Generating text using Vec2Text method")
        
        # Generate text using Vec2Text with target coordinates
        vec2text_result = self.vec2text.generate_text_from_embedding(
            target_embedding, 
            target_coordinates=target_coordinates
        )
        
        # Create result
        result = TextGenerationResult(
            generated_text=vec2text_result['generated_text'],
            title=vec2text_result['title'],
            target_embedding=target_embedding.tolist(),
            predicted_coordinates=(0.0, 0.0),  # Will be updated by correction loop
            method_used=TextGenerationMethod.VEC2TEXT,
            nearest_neighbors=[],  # Vec2Text doesn't use nearest neighbors
            generation_metadata={
                'vec2text_iterations': vec2text_result['iterations_used'],
                'embedding_distance': vec2text_result['embedding_distance'],
                'converged': vec2text_result['converged'],
                'correction_history': vec2text_result['correction_history']
            }
        )
        
        logger.info(f"Vec2Text generated text: {result.title[:50]}...")
        return result
    
    def train_vec2text(self, documents: List[Dict[str, Any]], epochs: int = 3) -> Dict[str, Any]:
        """
        Train the Vec2Text model on document-embedding pairs.
        
        Args:
            documents: List of documents with embeddings
            epochs: Number of training epochs
            
        Returns:
            Training results
        """
        logger.info(f"Training Vec2Text on {len(documents)} documents")
        
        # Filter documents with embeddings
        training_docs = [
            doc for doc in documents 
            if 'embedding' in doc and doc['embedding']
        ]
        
        if not training_docs:
            raise ValueError("No documents with embeddings found for training")
        
        # Train Vec2Text
        training_results = self.vec2text.train_model(training_docs, epochs=epochs)
        
        logger.info("Vec2Text training completed")
        return training_results
    
    def save_vec2text_model(self, model_path: str) -> None:
        """Save the trained Vec2Text model."""
        path = Path(model_path)
        self.vec2text.save_model(path)
        logger.info(f"Vec2Text model saved to {model_path}")
    
    def load_vec2text_model(self, model_path: str) -> None:
        """Load a trained Vec2Text model."""
        path = Path(model_path)
        self.vec2text.load_model(path)
        logger.info(f"Vec2Text model loaded from {model_path}")
    
    def evaluate_vec2text_quality(self, test_documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate the quality of Vec2Text generation."""
        return self.vec2text.evaluate_quality(test_documents)
