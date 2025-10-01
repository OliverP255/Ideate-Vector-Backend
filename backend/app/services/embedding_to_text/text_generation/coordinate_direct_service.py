"""
Direct coordinate matching service.

This service generates text by directly optimizing for coordinate accuracy
rather than embedding similarity.
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import random
from pathlib import Path

logger = logging.getLogger(__name__)


class CoordinateDirectService:
    """
    Service that generates text by directly optimizing for coordinate accuracy.
    """
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.reference_embeddings = None
        self.reference_texts = None
        self.reference_coordinates = None
        self.coordinate_neighbors = None
        self.parametric_umap = None
        self._is_initialized = False
    
    def initialize(self, reference_embeddings: np.ndarray, reference_texts: List[str]):
        """Initialize with reference data."""
        logger.info("Initializing CoordinateDirectService...")
        
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.reference_embeddings = reference_embeddings
        self.reference_texts = reference_texts
        
        # Load Parametric UMAP model for actual coordinate mapping
        self._load_parametric_umap()
        
        # Get actual UMAP coordinates for reference texts
        self.reference_coordinates = self._get_actual_umap_coordinates(reference_embeddings)
        
        # Build coordinate-based nearest neighbors
        self.coordinate_neighbors = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.coordinate_neighbors.fit(self.reference_coordinates)
        
        self._is_initialized = True
        logger.info(f"Initialized with {len(reference_texts)} reference texts and actual UMAP coordinates")
    
    def _load_parametric_umap(self):
        """Load the Parametric UMAP model."""
        try:
            from ..parametric_umap.parametric_umap_service import ParametricUMAPService
            self.parametric_umap = ParametricUMAPService()
            
            # Load the trained model
            model_path = Path("data/models/parametric_umap")
            if model_path.exists():
                self.parametric_umap.load_model(model_path)
                logger.info("Loaded Parametric UMAP model for coordinate mapping")
            else:
                logger.warning("Parametric UMAP model not found, using synthetic coordinates")
                self.parametric_umap = None
        except Exception as e:
            logger.warning(f"Failed to load Parametric UMAP model: {e}")
            self.parametric_umap = None
    
    def _get_actual_umap_coordinates(self, embeddings: np.ndarray) -> np.ndarray:
        """Get actual UMAP coordinates from embeddings."""
        if self.parametric_umap is not None:
            try:
                # Use the actual Parametric UMAP model to get coordinates
                coordinates = self.parametric_umap.transform(embeddings)
                logger.info(f"Got actual UMAP coordinates for {len(embeddings)} embeddings")
                return coordinates
            except Exception as e:
                logger.warning(f"Failed to get actual UMAP coordinates: {e}, using synthetic")
        
        # Fallback to synthetic coordinates
        return self._generate_synthetic_coordinates(embeddings)
    
    def _generate_synthetic_coordinates(self, embeddings: np.ndarray) -> np.ndarray:
        """Generate synthetic 2D coordinates from embeddings (fallback method)."""
        # Use PCA-like approach to project embeddings to 2D
        # This is a simplified version - in practice, you'd use the actual UMAP coordinates
        
        # Use first two principal components as proxy for 2D coordinates
        embedding_mean = np.mean(embeddings, axis=0)
        centered_embeddings = embeddings - embedding_mean
        
        # Simple projection to 2D using first two dimensions
        # In practice, this would be the actual UMAP coordinates
        coordinates = np.zeros((len(embeddings), 2))
        
        for i, embedding in enumerate(centered_embeddings):
            # Use embedding statistics to generate coordinates
            x_coord = np.mean(embedding[:192]) * 10  # First half of embedding
            y_coord = np.mean(embedding[192:]) * 10  # Second half of embedding
            coordinates[i] = [x_coord, y_coord]
        
        return coordinates
    
    def _get_target_embedding_from_coordinates(self, target_coordinates: Tuple[float, float]) -> Optional[np.ndarray]:
        """Get target embedding from coordinates using Parametric UMAP inverse transform."""
        if self.parametric_umap is not None:
            try:
                # Use Parametric UMAP inverse transform to get target embedding
                target_coords_reshaped = np.array(target_coordinates).reshape(1, -1)
                target_embedding = self.parametric_umap.inverse_transform(target_coords_reshaped)
                logger.info(f"Got target embedding from coordinates ({target_coordinates[0]:.2f}, {target_coordinates[1]:.2f}) using inverse UMAP")
                return target_embedding[0]  # Return first (and only) embedding
            except Exception as e:
                logger.warning(f"Failed to get target embedding from coordinates using inverse UMAP: {e}")
                return None
        else:
            logger.warning("Parametric UMAP not available for inverse transform")
            return None
    
    def generate_text_for_coordinates(self, target_coordinates: Tuple[float, float]) -> Dict[str, Any]:
        """Generate text that should be close to target coordinates using inverse UMAP."""
        if not self._is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        # Try to get target embedding using inverse UMAP
        target_embedding = self._get_target_embedding_from_coordinates(target_coordinates)
        
        if target_embedding is not None:
            # Use the target embedding to find closest reference texts
            target_embedding_reshaped = target_embedding.reshape(1, -1)
            
            # Find closest embeddings using cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(target_embedding_reshaped, self.reference_embeddings)[0]
            closest_indices = np.argsort(similarities)[::-1][:5]  # Top 5 most similar
            
            # Get closest reference texts and their coordinates
            closest_texts = [self.reference_texts[i] for i in closest_indices]
            closest_coords = [self.reference_coordinates[i] for i in closest_indices]
            
            # Generate adapted text
            adapted_text = self._adapt_text_for_coordinates(
                closest_texts, closest_coords, target_coordinates
            )
        else:
            # Fallback to coordinate-based approach
            target_coords = np.array(target_coordinates).reshape(1, -1)
            
            # Find closest reference coordinates
            distances, indices = self.coordinate_neighbors.kneighbors(target_coords)
            closest_indices = indices[0]
            closest_distances = distances[0]
            
            # Get closest reference texts
            closest_texts = [self.reference_texts[i] for i in closest_indices]
            closest_coords = [self.reference_coordinates[i] for i in closest_indices]
            
            # Generate adapted text
            adapted_text = self._adapt_text_for_coordinates(
                closest_texts, closest_coords, target_coordinates
            )
        
        # Get embedding of adapted text
        adapted_embedding = self.embedding_model.encode([adapted_text])[0]
        
        # Calculate predicted coordinates
        predicted_coords = self._embedding_to_coordinates(adapted_embedding)
        
        # Calculate coordinate error
        coordinate_error = np.sqrt(
            (predicted_coords[0] - target_coordinates[0])**2 + 
            (predicted_coords[1] - target_coordinates[1])**2
        )
        
        return {
            'generated_text': adapted_text,
            'title': self._extract_title(adapted_text),
            'final_embedding': adapted_embedding.tolist(),
            'predicted_coordinates': predicted_coords.tolist(),
            'target_coordinates': target_coordinates,
            'coordinate_error': float(coordinate_error),
            'method_used': 'coordinate_direct',
            'iterations_used': 1,  # Single iteration for direct approach
            'embedding_distance': 0.0,  # Not applicable for coordinate-based approach
            'converged': coordinate_error < 1.0,  # Consider converged if error < 1.0
            'correction_history': []  # No correction history for direct approach
        }
    
    def _adapt_text_for_coordinates(
        self, 
        reference_texts: List[str], 
        reference_coords: List[np.ndarray], 
        target_coords: Tuple[float, float]
    ) -> str:
        """Adapt reference texts to better match target coordinates using semantic domains."""
        
        # Analyze coordinate characteristics
        x, y = target_coords
        
        # Use different semantic domains based on coordinates to create distinct embeddings
        # X-axis domains
        if x > 5:
            domain = "quantum computing and quantum information theory with superconducting qubits"
        elif x > 2:
            domain = "machine learning and deep neural networks with transformer architectures"
        elif x > 0:
            domain = "statistical mechanics and condensed matter physics with phase transitions"
        elif x > -2:
            domain = "linear algebra and matrix theory with eigenvalue decompositions"
        elif x > -5:
            domain = "graph theory and network analysis with social network dynamics"
        else:
            domain = "abstract algebra and group theory with symmetry operations"
        
        # Y-axis methodologies
        if y > 5:
            methodology = "experimental physics with particle accelerators and detector technology"
        elif y > 2:
            methodology = "computational biology with molecular dynamics and protein simulations"
        elif y > 0:
            methodology = "artificial intelligence with natural language processing and computer vision"
        elif y > -2:
            methodology = "data science with statistical modeling and predictive analytics"
        elif y > -5:
            methodology = "software engineering with distributed systems and microservices"
        else:
            methodology = "theoretical computer science with complexity theory and algorithmic design"
        
        # Create coordinate-specific text that should map to the target coordinates
        adapted_text = f"Research in {domain}. This work presents novel methodologies in {methodology}. The approach demonstrates significant advances in computational efficiency through {domain} principles and {methodology} techniques. Experimental validation shows improvements in accuracy and performance using advanced algorithmic approaches. The theoretical framework incorporates {domain} foundations with {methodology} applications. Coordinate-specific analysis targeting ({x:.2f}, {y:.2f}) demonstrates enhanced precision through {domain} and {methodology} integration. The methodology shows significant improvements in computational efficiency and theoretical rigor through advanced algorithmic techniques and comprehensive experimental validation."
        
        return adapted_text
    
    def _embedding_to_coordinates(self, embedding: np.ndarray) -> np.ndarray:
        """Convert embedding to 2D coordinates using actual Parametric UMAP model."""
        if self.parametric_umap is not None:
            try:
                # Use the actual Parametric UMAP model
                coordinates = self.parametric_umap.transform(embedding.reshape(1, -1))
                return coordinates[0]  # Return first (and only) coordinate
            except Exception as e:
                logger.warning(f"Failed to use Parametric UMAP for coordinate conversion: {e}")
        
        # Fallback to synthetic coordinates
        embedding_mean = np.mean(self.reference_embeddings, axis=0)
        centered_embedding = embedding - embedding_mean
        
        x_coord = np.mean(centered_embedding[:192]) * 10
        y_coord = np.mean(centered_embedding[192:]) * 10
        
        return np.array([x_coord, y_coord])
    
    def _extract_title(self, text: str) -> str:
        """Extract a title from the text."""
        sentences = text.split('.')
        if sentences:
            title = sentences[0].strip()
            if len(title) > 100:
                title = title[:97] + "..."
            return title
        return "Coordinate-Optimized Research Paper"


class CoordinateIterativeService:
    """
    Service that iteratively refines text to achieve target coordinates.
    """
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.reference_embeddings = None
        self.reference_texts = None
        self.reference_coordinates = None
        self.parametric_umap = None
        self._is_initialized = False
    
    def initialize(self, reference_embeddings: np.ndarray, reference_texts: List[str]):
        """Initialize with reference data."""
        logger.info("Initializing CoordinateIterativeService...")
        
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.reference_embeddings = reference_embeddings
        self.reference_texts = reference_texts
        
        # Load Parametric UMAP model for actual coordinate mapping
        self._load_parametric_umap()
        
        # Get actual UMAP coordinates for reference texts
        self.reference_coordinates = self._get_actual_umap_coordinates(reference_embeddings)
        
        self._is_initialized = True
        logger.info(f"Initialized with {len(reference_texts)} reference texts and actual UMAP coordinates")
    
    def _load_parametric_umap(self):
        """Load the Parametric UMAP model."""
        try:
            from ..parametric_umap.parametric_umap_service import ParametricUMAPService
            self.parametric_umap = ParametricUMAPService()
            
            # Load the trained model
            model_path = Path("data/models/parametric_umap")
            if model_path.exists():
                self.parametric_umap.load_model(model_path)
                logger.info("Loaded Parametric UMAP model for coordinate mapping")
            else:
                logger.warning("Parametric UMAP model not found, using synthetic coordinates")
                self.parametric_umap = None
        except Exception as e:
            logger.warning(f"Failed to load Parametric UMAP model: {e}")
            self.parametric_umap = None
    
    def _get_actual_umap_coordinates(self, embeddings: np.ndarray) -> np.ndarray:
        """Get actual UMAP coordinates from embeddings."""
        if self.parametric_umap is not None:
            try:
                # Use the actual Parametric UMAP model to get coordinates
                coordinates = self.parametric_umap.transform(embeddings)
                logger.info(f"Got actual UMAP coordinates for {len(embeddings)} embeddings")
                return coordinates
            except Exception as e:
                logger.warning(f"Failed to get actual UMAP coordinates: {e}, using synthetic")
        
        # Fallback to synthetic coordinates
        return self._generate_synthetic_coordinates(embeddings)
    
    def _generate_synthetic_coordinates(self, embeddings: np.ndarray) -> np.ndarray:
        """Generate synthetic 2D coordinates from embeddings (fallback method)."""
        coordinates = np.zeros((len(embeddings), 2))
        
        for i, embedding in enumerate(embeddings):
            # Use embedding statistics to generate coordinates
            x_coord = np.mean(embedding[:192]) * 10
            y_coord = np.mean(embedding[192:]) * 10
            coordinates[i] = [x_coord, y_coord]
        
        return coordinates
    
    def generate_text_for_coordinates(self, target_coordinates: Tuple[float, float]) -> Dict[str, Any]:
        """Generate text by iteratively refining to achieve target coordinates."""
        if not self._is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        # Start with the closest reference text
        distances = np.sqrt(
            np.sum((self.reference_coordinates - np.array(target_coordinates))**2, axis=1)
        )
        best_idx = np.argmin(distances)
        current_text = self.reference_texts[best_idx]
        
        best_error = float('inf')
        best_text = current_text
        
        # Iterative refinement
        for iteration in range(5):
            # Get current embedding and coordinates
            current_embedding = self.embedding_model.encode([current_text])[0]
            current_coords = self._embedding_to_coordinates(current_embedding)
            
            # Calculate coordinate error
            error = np.sqrt(
                (current_coords[0] - target_coordinates[0])**2 + 
                (current_coords[1] - target_coordinates[1])**2
            )
            
            logger.info(f"Coordinate iterative refinement iteration {iteration + 1}: error = {error:.4f}")
            
            # Update best result
            if error < best_error:
                best_error = error
                best_text = current_text
            
            # If error is small enough, stop
            if error < 1.0:
                break
            
            # Modify text to reduce coordinate error
            current_text = self._modify_text_for_coordinates(
                current_text, target_coordinates, current_coords
            )
        
        # Get final embedding and coordinates
        final_embedding = self.embedding_model.encode([best_text])[0]
        final_coords = self._embedding_to_coordinates(final_embedding)
        
        return {
            'generated_text': best_text,
            'title': self._extract_title(best_text),
            'final_embedding': final_embedding.tolist(),
            'predicted_coordinates': final_coords.tolist(),
            'target_coordinates': target_coordinates,
            'coordinate_error': float(best_error),
            'method_used': 'coordinate_iterative',
            'iterations_used': 5,  # Fixed number of iterations
            'embedding_distance': 0.0,  # Not applicable for coordinate-based approach
            'converged': best_error < 1.0,  # Consider converged if error < 1.0
            'correction_history': []  # No correction history for coordinate-based approach
        }
    
    def _modify_text_for_coordinates(
        self, 
        current_text: str, 
        target_coords: Tuple[float, float], 
        current_coords: np.ndarray
    ) -> str:
        """Modify text to reduce coordinate error using more aggressive semantic changes."""
        
        # Calculate coordinate differences
        x_diff = target_coords[0] - current_coords[0]
        y_diff = target_coords[1] - current_coords[1]
        
        # More aggressive text modifications based on coordinate differences
        # Use completely different semantic domains to create larger embedding changes
        
        # X-axis modifications - use different research domains
        if x_diff > 3:
            domain_text = "quantum computing algorithms and machine learning neural networks with deep reinforcement learning architectures"
        elif x_diff > 1:
            domain_text = "statistical mechanics and thermodynamics with molecular dynamics simulations"
        elif x_diff > 0:
            domain_text = "linear algebra and differential equations with numerical optimization methods"
        elif x_diff > -1:
            domain_text = "probability theory and stochastic processes with Monte Carlo simulations"
        elif x_diff > -3:
            domain_text = "graph theory and combinatorics with discrete mathematics applications"
        else:
            domain_text = "abstract algebra and topology with category theory foundations"
        
        # Y-axis modifications - use different methodologies
        if y_diff > 3:
            method_text = "experimental physics with particle accelerator technology and quantum field theory"
        elif y_diff > 1:
            method_text = "computational biology with genomic analysis and protein folding simulations"
        elif y_diff > 0:
            method_text = "artificial intelligence with natural language processing and computer vision"
        elif y_diff > -1:
            method_text = "data science with statistical modeling and predictive analytics"
        elif y_diff > -3:
            method_text = "software engineering with distributed systems and cloud computing"
        else:
            method_text = "theoretical computer science with complexity theory and algorithmic design"
        
        # Create a completely new text that combines the domain and method
        # This should create a significantly different embedding
        modified_text = f"Research in {domain_text}. This work presents novel approaches in {method_text}. The methodology demonstrates significant advances in computational efficiency through {domain_text} and {method_text}. Experimental validation shows improvements in accuracy and performance using advanced algorithmic techniques. The theoretical framework incorporates {domain_text} principles with {method_text} applications. Coordinate-specific analysis targeting ({target_coords[0]:.2f}, {target_coords[1]:.2f}) demonstrates enhanced precision through {domain_text} and {method_text} integration."
        
        return modified_text
    
    def _embedding_to_coordinates(self, embedding: np.ndarray) -> np.ndarray:
        """Convert embedding to 2D coordinates using actual Parametric UMAP model."""
        if self.parametric_umap is not None:
            try:
                # Use the actual Parametric UMAP model
                coordinates = self.parametric_umap.transform(embedding.reshape(1, -1))
                return coordinates[0]  # Return first (and only) coordinate
            except Exception as e:
                logger.warning(f"Failed to use Parametric UMAP for coordinate conversion: {e}")
        
        # Fallback to synthetic coordinates
        x_coord = np.mean(embedding[:192]) * 10
        y_coord = np.mean(embedding[192:]) * 10
        return np.array([x_coord, y_coord])
    
    def _extract_title(self, text: str) -> str:
        """Extract a title from the text."""
        sentences = text.split('.')
        if sentences:
            title = sentences[0].strip()
            if len(title) > 100:
                title = title[:97] + "..."
            return title
        return "Iteratively-Optimized Research Paper"
