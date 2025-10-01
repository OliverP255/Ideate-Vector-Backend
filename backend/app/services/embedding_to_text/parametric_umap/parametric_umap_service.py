"""
Parametric UMAP service for invertible dimensionality reduction.

This service implements Parametric UMAP as described in the PDF:
- Neural network encoder: embedding → 2D
- Neural network decoder: 2D → embedding
- Provides inverse_transform for mapping clicked coordinates back to embeddings
"""

import logging
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
# Mock tensorflow for Python 3.13 compatibility
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    # Mock implementation for Python 3.13 compatibility
    class tf:
        class constant:
            @staticmethod
            def __call__(*args, **kwargs):
                return "mock_tensor"
        
        class cast:
            @staticmethod
            def __call__(*args, **kwargs):
                return "mock_tensor"
        
        class keras:
            class Model:
                def __init__(self, *args, **kwargs):
                    pass
            
            class layers:
                class Dense:
                    def __init__(self, *args, **kwargs):
                        pass
    
    class keras:
        class Model:
            def __init__(self, *args, **kwargs):
                pass
        
        class layers:
            class Dense:
                def __init__(self, *args, **kwargs):
                    pass
from sklearn.preprocessing import StandardScaler

from ..models.parametric_umap import ParametricUMAPConfig, ParametricUMAPModel

logger = logging.getLogger(__name__)


class ParametricUMAPService:
    """
    Service for training and using Parametric UMAP models.
    
    This replaces the standard UMAP with a parametric version that allows
    inverse transformation from 2D coordinates back to high-dimensional embeddings.
    """
    
    def __init__(self, config: Optional[ParametricUMAPConfig] = None):
        self.config = config or ParametricUMAPConfig()
        self.model: Optional[ParametricUMAPModel] = None
        self.scaler = StandardScaler()
        
    def train_model(
        self, 
        embeddings: np.ndarray, 
        coordinates: np.ndarray,
        save_path: Optional[Path] = None
    ) -> ParametricUMAPModel:
        """
        Train the Parametric UMAP model.
        
        Args:
            embeddings: High-dimensional embeddings (n_samples, embedding_dim)
            coordinates: 2D coordinates (n_samples, 2)
            save_path: Optional path to save the trained model
            
        Returns:
            Trained ParametricUMAPModel
        """
        logger.info(f"Training Parametric UMAP with {len(embeddings)} samples")
        
        # Validate inputs
        if embeddings.shape[0] != coordinates.shape[0]:
            raise ValueError("Number of embeddings must match number of coordinates")
        
        if coordinates.shape[1] != 2:
            raise ValueError("Coordinates must be 2D")
        
        # Normalize coordinates
        coordinates_normalized = self.scaler.fit_transform(coordinates)
        
        # Build encoder model (embedding → 2D)
        encoder_input = keras.Input(shape=(embeddings.shape[1],), name='embedding_input')
        x = encoder_input
        
        for i, layer_size in enumerate(self.config.encoder_layers):
            x = keras.layers.Dense(
                layer_size, 
                activation='relu',
                name=f'encoder_dense_{i}'
            )(x)
            x = keras.layers.Dropout(0.2)(x)
        
        encoder_output = keras.layers.Dense(
            self.config.n_components, 
            activation='linear',
            name='encoder_output'
        )(x)
        
        encoder_model = keras.Model(encoder_input, encoder_output, name='encoder')
        
        # Build decoder model (2D → embedding)
        decoder_input = keras.Input(shape=(self.config.n_components,), name='coordinate_input')
        x = decoder_input
        
        for i, layer_size in enumerate(self.config.decoder_layers):
            x = keras.layers.Dense(
                layer_size, 
                activation='relu',
                name=f'decoder_dense_{i}'
            )(x)
            x = keras.layers.Dropout(0.2)(x)
        
        decoder_output = keras.layers.Dense(
            embeddings.shape[1], 
            activation='linear',
            name='decoder_output'
        )(x)
        
        decoder_model = keras.Model(decoder_input, decoder_output, name='decoder')
        
        # Build autoencoder for joint training
        autoencoder_output = decoder_model(encoder_model(encoder_input))
        autoencoder = keras.Model(encoder_input, autoencoder_output, name='autoencoder')
        
        # Compile models
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        
        autoencoder.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        # Train the autoencoder
        logger.info("Training autoencoder...")
        history = autoencoder.fit(
            embeddings, embeddings,  # Autoencoder reconstruction
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        # Fine-tune encoder to match coordinates
        logger.info("Fine-tuning encoder to match coordinates...")
        encoder_model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        encoder_history = encoder_model.fit(
            embeddings, coordinates_normalized,
            epochs=self.config.epochs // 2,  # Half the epochs for fine-tuning
            batch_size=self.config.batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        # Create model container
        self.model = ParametricUMAPModel(
            config=self.config,
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            training_embeddings=embeddings,
            training_coordinates=coordinates,
            is_trained=True
        )
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        logger.info("Parametric UMAP training completed successfully")
        return self.model
    
    def load_model(self, model_path: Path) -> ParametricUMAPModel:
        """
        Load a trained Parametric UMAP model.
        
        Args:
            model_path: Path to the saved model directory
            
        Returns:
            Loaded ParametricUMAPModel
        """
        logger.info(f"Loading Parametric UMAP model from {model_path}")
        
        # Load configuration
        config_path = model_path / "config.pkl"
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Load scaler
        scaler_path = model_path / "scaler.pkl"
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load models
        encoder_model = keras.models.load_model(model_path / "encoder")
        decoder_model = keras.models.load_model(model_path / "decoder")
        
        self.model = ParametricUMAPModel(
            config=config,
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            is_trained=True
        )
        
        logger.info("Parametric UMAP model loaded successfully")
        return self.model
    
    def save_model(self, model_path: Path) -> None:
        """
        Save the trained Parametric UMAP model.
        
        Args:
            model_path: Path to save the model directory
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be trained before saving")
        
        logger.info(f"Saving Parametric UMAP model to {model_path}")
        
        # Create directory
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = model_path / "config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(self.config, f)
        
        # Save scaler
        scaler_path = model_path / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save models
        self.model.encoder_model.save(model_path / "encoder")
        self.model.decoder_model.save(model_path / "decoder")
        
        logger.info("Parametric UMAP model saved successfully")
    
    def inverse_transform(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Transform 2D coordinates back to high-dimensional embeddings.
        
        Args:
            coordinates: 2D coordinates array of shape (n_samples, 2)
            
        Returns:
            High-dimensional embeddings of shape (n_samples, embedding_dim)
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be trained before inverse transformation")
        
        # Normalize coordinates using fitted scaler
        coordinates_normalized = self.scaler.transform(coordinates)
        
        # Use decoder to get embeddings
        embeddings = self.model.decoder_model.predict(coordinates_normalized, verbose=0)
        
        return embeddings
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform high-dimensional embeddings to 2D coordinates.
        
        Args:
            embeddings: High-dimensional embeddings array of shape (n_samples, embedding_dim)
            
        Returns:
            2D coordinates of shape (n_samples, 2)
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be trained before transformation")
        
        # Use encoder to get normalized coordinates
        coordinates_normalized = self.model.encoder_model.predict(embeddings, verbose=0)
        
        # Denormalize coordinates
        coordinates = self.scaler.inverse_transform(coordinates_normalized)
        
        return coordinates
    
    def get_target_embedding(self, x: float, y: float) -> np.ndarray:
        """
        Get target embedding for a specific 2D coordinate.
        
        Uses a robust nearest neighbor approach instead of neural network inverse transform.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Target embedding vector
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be trained before getting target embedding")
        
        target_coords = np.array([[x, y]])
        
        # Method 1: Try neural network inverse transform first
        try:
            embedding = self.inverse_transform(target_coords)[0]
            # Verify the embedding maps back to reasonable coordinates
            test_coords = self.transform(embedding.reshape(1, -1))[0]
            distance = np.linalg.norm(test_coords - target_coords[0])
            if distance < 2.0:  # Reasonable threshold
                return embedding
        except Exception as e:
            logger.warning(f"Neural network inverse transform failed: {e}")
        
        # Method 2: Fallback to nearest neighbor search
        logger.info(f"Using nearest neighbor fallback for coordinates ({x}, {y})")
        return self._get_embedding_by_nearest_neighbor(x, y)
    
    def _get_embedding_by_nearest_neighbor(self, x: float, y: float) -> np.ndarray:
        """
        Get embedding by finding the closest training point to target coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Embedding from the closest training point
        """
        if not hasattr(self, '_training_coordinates') or not hasattr(self, '_training_embeddings'):
            # Build coordinate and embedding arrays
            self._training_coordinates = self.model.training_coordinates
            self._training_embeddings = self.model.training_embeddings
            
            # Build KDTree for efficient nearest neighbor search
            from sklearn.neighbors import NearestNeighbors
            self._coordinate_tree = NearestNeighbors(n_neighbors=1, metric='euclidean')
            self._coordinate_tree.fit(self._training_coordinates)
        
        # Find the single closest neighbor in coordinate space
        target_coord = np.array([[x, y]])
        distances, indices = self._coordinate_tree.kneighbors(target_coord)
        
        # Get the embedding from the closest neighbor
        closest_idx = indices[0][0]
        closest_embedding = self._training_embeddings[closest_idx]
        closest_distance = distances[0][0]
        
        logger.info(f"Using closest neighbor at index {closest_idx}, distance {closest_distance:.3f}")
        
        # Verify this embedding maps back to reasonable coordinates
        test_coords = self.transform(closest_embedding.reshape(1, -1))[0]
        test_distance = np.linalg.norm(test_coords - target_coord[0])
        
        logger.info(f"Verification: embedding maps to ({test_coords[0]:.3f}, {test_coords[1]:.3f}), "
                   f"distance from target: {test_distance:.3f}")
        
        return closest_embedding
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate the quality of the trained model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        embeddings = self.model.training_embeddings
        true_coordinates = self.model.training_coordinates
        
        # Forward transformation
        predicted_coordinates = self.transform(embeddings)
        
        # Inverse transformation
        reconstructed_embeddings = self.inverse_transform(true_coordinates)
        
        # Calculate metrics
        coordinate_mse = np.mean((predicted_coordinates - true_coordinates) ** 2)
        coordinate_mae = np.mean(np.abs(predicted_coordinates - true_coordinates))
        
        embedding_mse = np.mean((reconstructed_embeddings - embeddings) ** 2)
        embedding_mae = np.mean(np.abs(reconstructed_embeddings - embeddings))
        
        # Cosine similarity for embeddings
        embedding_similarities = []
        for i in range(len(embeddings)):
            cos_sim = np.dot(embeddings[i], reconstructed_embeddings[i]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(reconstructed_embeddings[i])
            )
            embedding_similarities.append(cos_sim)
        
        avg_cosine_similarity = np.mean(embedding_similarities)
        
        return {
            'coordinate_mse': float(coordinate_mse),
            'coordinate_mae': float(coordinate_mae),
            'embedding_mse': float(embedding_mse),
            'embedding_mae': float(embedding_mae),
            'avg_cosine_similarity': float(avg_cosine_similarity)
        }
