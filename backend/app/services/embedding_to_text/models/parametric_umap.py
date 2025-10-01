"""
Models for Parametric UMAP configuration and management.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any
import numpy as np


@dataclass
class ParametricUMAPConfig:
    """Configuration for Parametric UMAP model."""
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = 'cosine'
    random_state: int = 42
    encoder_dim: int = 384  # Should match embedding dimension
    decoder_dim: int = 384  # Should match embedding dimension
    encoder_layers: Tuple[int, ...] = (256, 128, 64)
    decoder_layers: Tuple[int, ...] = (64, 128, 256)
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001


@dataclass
class ParametricUMAPModel:
    """Container for trained Parametric UMAP model."""
    config: ParametricUMAPConfig
    encoder_model: Optional[Any] = None  # TensorFlow/Keras model
    decoder_model: Optional[Any] = None  # TensorFlow/Keras model
    embedding_model: Optional[Any] = None  # UMAP embedding model
    training_embeddings: Optional[np.ndarray] = None
    training_coordinates: Optional[np.ndarray] = None
    is_trained: bool = False
    
    def inverse_transform(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Transform 2D coordinates back to high-dimensional embeddings.
        
        Args:
            coordinates: 2D coordinates array of shape (n_samples, 2)
            
        Returns:
            High-dimensional embeddings of shape (n_samples, embedding_dim)
        """
        if not self.is_trained or self.decoder_model is None:
            raise ValueError("Model must be trained before inverse transformation")
        
        return self.decoder_model.predict(coordinates, verbose=0)
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform high-dimensional embeddings to 2D coordinates.
        
        Args:
            embeddings: High-dimensional embeddings array of shape (n_samples, embedding_dim)
            
        Returns:
            2D coordinates of shape (n_samples, 2)
        """
        if not self.is_trained or self.encoder_model is None:
            raise ValueError("Model must be trained before transformation")
        
        return self.encoder_model.predict(embeddings, verbose=0)
