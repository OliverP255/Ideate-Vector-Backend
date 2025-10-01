"""
Embedding-guided text generation service.

This service implements a more direct approach to generate text that will embed
to coordinates close to the target coordinates.
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class EmbeddingGuidedTextService:
    """
    Service that generates text by finding similar embeddings and adapting their text.
    """
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.reference_embeddings = None
        self.reference_texts = None
        self.neighbors_model = None
        self._is_initialized = False
    
    def initialize(self, reference_embeddings: np.ndarray, reference_texts: List[str]):
        """
        Initialize with reference embeddings and texts.
        
        Args:
            reference_embeddings: Array of reference embeddings
            reference_texts: List of corresponding texts
        """
        logger.info("Initializing EmbeddingGuidedTextService...")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Store reference data
        self.reference_embeddings = reference_embeddings
        self.reference_texts = reference_texts
        
        # Build nearest neighbors model
        self.neighbors_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        # Ensure embeddings are 2D
        if len(reference_embeddings.shape) > 2:
            reference_embeddings = reference_embeddings.reshape(reference_embeddings.shape[0], -1)
        self.neighbors_model.fit(reference_embeddings)
        
        self._is_initialized = True
        logger.info(f"Initialized with {len(reference_texts)} reference texts")
    
    def generate_text_from_embedding(
        self, 
        target_embedding: np.ndarray,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Generate text that should embed close to the target embedding.
        
        Args:
            target_embedding: Target embedding vector
            max_iterations: Maximum refinement iterations
            
        Returns:
            Dictionary with generated text and metadata
        """
        if not self._is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        # Find similar reference texts
        distances, indices = self.neighbors_model.kneighbors([target_embedding])
        similar_indices = indices[0]
        similar_distances = distances[0]
        
        # Get similar texts
        similar_texts = [self.reference_texts[i] for i in similar_indices]
        
        # Generate adapted text
        adapted_text = self._adapt_text_to_target(similar_texts, target_embedding)
        
        # Iteratively refine to get closer to target
        current_text = adapted_text
        for iteration in range(max_iterations):
            # Get embedding of current text
            current_embedding = self.embedding_model.encode([current_text])[0]
            
            # Calculate similarity to target
            similarity = np.dot(target_embedding, current_embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(current_embedding)
            )
            
            logger.info(f"Iteration {iteration + 1}: similarity = {similarity:.4f}")
            
            # If similarity is high enough, stop
            if similarity > 0.95:
                break
            
            # Refine text based on embedding difference
            current_text = self._refine_text_for_embedding(
                current_text, target_embedding, current_embedding
            )
        
        return {
            'generated_text': current_text,
            'title': self._extract_title(current_text),
            'final_embedding': current_embedding.tolist(),
            'similarity_to_target': float(similarity),
            'method_used': 'embedding_guided',
            'iterations_used': iteration + 1
        }
    
    def _adapt_text_to_target(
        self, 
        similar_texts: List[str], 
        target_embedding: np.ndarray
    ) -> str:
        """Adapt similar texts to better match the target embedding."""
        
        # Analyze target embedding characteristics
        target_mean = np.mean(target_embedding)
        target_std = np.std(target_embedding)
        target_norm = np.linalg.norm(target_embedding)
        
        # Select the most similar text as base
        base_text = similar_texts[0]
        
        # Create adaptations based on target characteristics
        adaptations = []
        
        # Add topic-specific content based on embedding statistics
        if target_mean > 0.1:
            adaptations.append("computational algorithms and machine learning methodologies")
        elif target_mean > 0:
            adaptations.append("mathematical foundations and theoretical frameworks")
        elif target_mean > -0.1:
            adaptations.append("experimental validation and empirical analysis")
        else:
            adaptations.append("fundamental principles and basic research")
        
        # Add complexity based on embedding norm
        if target_norm > 10:
            adaptations.append("advanced computational techniques and sophisticated analytical methods")
        elif target_norm > 8:
            adaptations.append("rigorous mathematical analysis and comprehensive experimental design")
        else:
            adaptations.append("basic computational approaches and fundamental theoretical concepts")
        
        # Add variability based on embedding standard deviation
        if target_std > 0.5:
            adaptations.append("diverse applications across multiple domains and interdisciplinary research")
        elif target_std > 0.3:
            adaptations.append("focused research with specific applications and targeted methodologies")
        else:
            adaptations.append("specialized research with precise theoretical foundations")
        
        # Combine base text with adaptations
        adapted_text = f"{base_text} This research extends the methodology by incorporating {', '.join(adaptations[:2])}. The approach demonstrates significant improvements in computational efficiency and theoretical rigor."
        
        return adapted_text
    
    def _refine_text_for_embedding(
        self, 
        current_text: str, 
        target_embedding: np.ndarray, 
        current_embedding: np.ndarray
    ) -> str:
        """Refine text to move its embedding closer to the target."""
        
        # Calculate embedding difference
        embedding_diff = target_embedding - current_embedding
        
        # Analyze the difference to determine refinement direction
        diff_mean = np.mean(embedding_diff)
        diff_std = np.std(embedding_diff)
        
        # Add refinements based on embedding difference
        refinements = []
        
        if diff_mean > 0.1:
            refinements.append("advanced computational methods and sophisticated algorithms")
        elif diff_mean < -0.1:
            refinements.append("fundamental theoretical principles and basic computational approaches")
        
        if diff_std > 0.5:
            refinements.append("comprehensive experimental validation and rigorous analytical techniques")
        elif diff_std < 0.3:
            refinements.append("focused theoretical analysis and precise methodological approaches")
        
        # Add refinements to text
        if refinements:
            refined_text = f"{current_text} The methodology incorporates {', '.join(refinements)} to achieve enhanced performance and accuracy."
        else:
            refined_text = current_text
        
        return refined_text
    
    def _extract_title(self, text: str) -> str:
        """Extract a title from the generated text."""
        # Find the first sentence or use a default
        sentences = text.split('.')
        if sentences:
            title = sentences[0].strip()
            if len(title) > 100:
                title = title[:97] + "..."
            return title
        return "Generated Research Paper"


class EmbeddingInterpolationService:
    """
    Service that generates text by interpolating between similar embeddings and texts.
    """
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.reference_embeddings = None
        self.reference_texts = None
        self.neighbors_model = None
        self._is_initialized = False
    
    def initialize(self, reference_embeddings: np.ndarray, reference_texts: List[str]):
        """Initialize with reference data."""
        logger.info("Initializing EmbeddingInterpolationService...")
        
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.reference_embeddings = reference_embeddings
        self.reference_texts = reference_texts
        
        self.neighbors_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        # Ensure embeddings are 2D
        if len(reference_embeddings.shape) > 2:
            reference_embeddings = reference_embeddings.reshape(reference_embeddings.shape[0], -1)
        self.neighbors_model.fit(reference_embeddings)
        
        self._is_initialized = True
        logger.info(f"Initialized with {len(reference_texts)} reference texts")
    
    def generate_text_from_embedding(self, target_embedding: np.ndarray) -> Dict[str, Any]:
        """Generate text by interpolating between similar embeddings."""
        if not self._is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        # Find similar embeddings
        distances, indices = self.neighbors_model.kneighbors([target_embedding])
        similar_indices = indices[0]
        similar_distances = distances[0]
        
        # Calculate weights based on similarity (closer = higher weight)
        weights = 1 / (1 + similar_distances)
        weights = weights / np.sum(weights)  # Normalize
        
        # Interpolate between similar texts
        interpolated_text = self._interpolate_texts(
            [self.reference_texts[i] for i in similar_indices],
            weights,
            target_embedding
        )
        
        # Get embedding of interpolated text
        interpolated_embedding = self.embedding_model.encode([interpolated_text])[0]
        
        # Calculate similarity to target
        similarity = np.dot(target_embedding, interpolated_embedding) / (
            np.linalg.norm(target_embedding) * np.linalg.norm(interpolated_embedding)
        )
        
        return {
            'generated_text': interpolated_text,
            'title': self._extract_title(interpolated_text),
            'final_embedding': interpolated_embedding.tolist(),
            'similarity_to_target': float(similarity),
            'method_used': 'embedding_interpolation'
        }
    
    def _interpolate_texts(
        self, 
        texts: List[str], 
        weights: np.ndarray, 
        target_embedding: np.ndarray
    ) -> str:
        """Interpolate between texts based on weights."""
        
        # Use the most similar text as base
        base_text = texts[0]
        
        # Extract key concepts from other texts
        concepts = []
        for i, text in enumerate(texts[1:4]):  # Use top 3 additional texts
            # Extract key phrases (simplified)
            sentences = text.split('.')
            if sentences:
                key_sentence = sentences[0].strip()
                concepts.append(key_sentence)
        
        # Create interpolated text
        interpolated_text = f"{base_text}"
        
        if concepts:
            interpolated_text += f" This approach combines elements from related research including {', '.join(concepts[:2])}. The methodology demonstrates significant advances in computational efficiency and theoretical understanding."
        
        return interpolated_text
    
    def _extract_title(self, text: str) -> str:
        """Extract a title from the text."""
        sentences = text.split('.')
        if sentences:
            title = sentences[0].strip()
            if len(title) > 100:
                title = title[:97] + "..."
            return title
        return "Interpolated Research Paper"
