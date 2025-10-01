"""
Precision coordinate matching service.

This service uses multiple strategies to achieve high-precision coordinate matching:
1. Embedding interpolation to get closer to target coordinates
2. Direct embedding optimization that modifies text to minimize coordinate distance
3. Hybrid approach combining inverse UMAP + embedding interpolation + direct optimization
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import random

logger = logging.getLogger(__name__)


class PrecisionCoordinateService:
    """
    Service that achieves high-precision coordinate matching using multiple strategies.
    """
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.reference_embeddings = None
        self.reference_texts = None
        self.reference_coordinates = None
        self.parametric_umap = None
        self.embedding_neighbors = None
        self.coordinate_neighbors = None
        self._is_initialized = False
    
    def initialize(self, reference_embeddings: np.ndarray, reference_texts: List[str]):
        """Initialize with reference data."""
        logger.info("Initializing PrecisionCoordinateService...")
        
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.reference_embeddings = reference_embeddings
        self.reference_texts = reference_texts
        
        # Load Parametric UMAP model
        self._load_parametric_umap()
        
        # Get actual UMAP coordinates
        self.reference_coordinates = self._get_actual_umap_coordinates(reference_embeddings)
        
        # Build nearest neighbors for both embeddings and coordinates
        self.embedding_neighbors = NearestNeighbors(n_neighbors=10, metric='cosine')
        self.embedding_neighbors.fit(reference_embeddings)
        
        self.coordinate_neighbors = NearestNeighbors(n_neighbors=10, metric='euclidean')
        self.coordinate_neighbors.fit(self.reference_coordinates)
        
        self._is_initialized = True
        logger.info(f"Initialized with {len(reference_texts)} reference texts and coordinates")
    
    def _load_parametric_umap(self):
        """Load the Parametric UMAP model."""
        try:
            from ..parametric_umap.parametric_umap_service import ParametricUMAPService
            self.parametric_umap = ParametricUMAPService()
            
            model_path = Path("data/models/parametric_umap")
            if model_path.exists():
                self.parametric_umap.load_model(model_path)
                logger.info("Loaded Parametric UMAP model for precision coordinate mapping")
            else:
                logger.warning("Parametric UMAP model not found")
                self.parametric_umap = None
        except Exception as e:
            logger.warning(f"Failed to load Parametric UMAP model: {e}")
            self.parametric_umap = None
    
    def _get_actual_umap_coordinates(self, embeddings: np.ndarray) -> np.ndarray:
        """Get actual UMAP coordinates from embeddings."""
        if self.parametric_umap is not None:
            try:
                coordinates = self.parametric_umap.transform(embeddings)
                logger.info(f"Got actual UMAP coordinates for {len(embeddings)} embeddings")
                return coordinates
            except Exception as e:
                logger.warning(f"Failed to get actual UMAP coordinates: {e}")
        
        # Fallback to synthetic coordinates
        return self._generate_synthetic_coordinates(embeddings)
    
    def _generate_synthetic_coordinates(self, embeddings: np.ndarray) -> np.ndarray:
        """Generate synthetic 2D coordinates from embeddings (fallback method)."""
        coordinates = np.zeros((len(embeddings), 2))
        
        for i, embedding in enumerate(embeddings):
            x_coord = np.mean(embedding[:192]) * 10
            y_coord = np.mean(embedding[192:]) * 10
            coordinates[i] = [x_coord, y_coord]
        
        return coordinates
    
    def generate_text_for_coordinates(self, target_coordinates: Tuple[float, float]) -> Dict[str, Any]:
        """Generate text with high precision for target coordinates."""
        if not self._is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        logger.info(f"Generating precision text for coordinates ({target_coordinates[0]:.2f}, {target_coordinates[1]:.2f})")
        
        # Strategy 1: Try inverse UMAP + embedding interpolation
        result = self._try_inverse_umap_strategy(target_coordinates)
        if result['coordinate_error'] < 1.0:
            logger.info(f"Inverse UMAP strategy achieved target precision: {result['coordinate_error']:.4f}")
            return result
        
        # Strategy 2: Try embedding interpolation
        result = self._try_embedding_interpolation_strategy(target_coordinates)
        if result['coordinate_error'] < 1.0:
            logger.info(f"Embedding interpolation strategy achieved target precision: {result['coordinate_error']:.4f}")
            return result
        
        # Strategy 3: Try direct optimization
        result = self._try_direct_optimization_strategy(target_coordinates)
        if result['coordinate_error'] < 1.0:
            logger.info(f"Direct optimization strategy achieved target precision: {result['coordinate_error']:.4f}")
            return result
        
        # Strategy 4: Try hybrid approach
        result = self._try_hybrid_strategy(target_coordinates)
        
        logger.info(f"Final precision result: error = {result['coordinate_error']:.4f}")
        return result
    
    def _try_inverse_umap_strategy(self, target_coordinates: Tuple[float, float]) -> Dict[str, Any]:
        """Try inverse UMAP strategy to get target embedding."""
        if self.parametric_umap is None:
            return self._create_failure_result(target_coordinates, "inverse_umap", "Parametric UMAP not available")
        
        try:
            # Get target embedding using inverse UMAP
            target_coords_reshaped = np.array(target_coordinates).reshape(1, -1)
            target_embedding = self.parametric_umap.inverse_transform(target_coords_reshaped)[0]
            
            # Find closest embeddings to target
            similarities = cosine_similarity(target_embedding.reshape(1, -1), self.reference_embeddings)[0]
            closest_indices = np.argsort(similarities)[::-1][:5]
            
            # Get closest texts and interpolate
            closest_texts = [self.reference_texts[i] for i in closest_indices]
            closest_embeddings = [self.reference_embeddings[i] for i in closest_indices]
            
            # Create interpolated text
            interpolated_text = self._create_interpolated_text(closest_texts, closest_embeddings, target_embedding)
            
            # Get final coordinates
            final_embedding = self.embedding_model.encode([interpolated_text])[0]
            final_coords = self._embedding_to_coordinates(final_embedding)
            
            coordinate_error = np.sqrt(
                (final_coords[0] - target_coordinates[0])**2 + 
                (final_coords[1] - target_coordinates[1])**2
            )
            
            return {
                'generated_text': interpolated_text,
                'title': self._extract_title(interpolated_text),
                'final_embedding': final_embedding.tolist(),
                'predicted_coordinates': final_coords.tolist(),
                'target_coordinates': target_coordinates,
                'coordinate_error': float(coordinate_error),
                'method_used': 'precision_inverse_umap',
                'iterations_used': 1,
                'embedding_distance': 0.0,
                'converged': coordinate_error < 1.0,
                'correction_history': []
            }
            
        except Exception as e:
            logger.warning(f"Inverse UMAP strategy failed: {e}")
            return self._create_failure_result(target_coordinates, "inverse_umap", str(e))
    
    def _try_embedding_interpolation_strategy(self, target_coordinates: Tuple[float, float]) -> Dict[str, Any]:
        """Try embedding interpolation strategy."""
        try:
            # Find closest coordinate neighbors
            target_coords_reshaped = np.array(target_coordinates).reshape(1, -1)
            distances, indices = self.coordinate_neighbors.kneighbors(target_coords_reshaped)
            
            closest_indices = indices[0][:5]  # Top 5 closest
            closest_distances = distances[0][:5]
            
            # Get closest texts and embeddings
            closest_texts = [self.reference_texts[i] for i in closest_indices]
            closest_embeddings = [self.reference_embeddings[i] for i in closest_indices]
            closest_coords = [self.reference_coordinates[i] for i in closest_indices]
            
            # Create weighted interpolation
            weights = 1.0 / (closest_distances + 1e-6)  # Avoid division by zero
            weights = weights / np.sum(weights)  # Normalize
            
            # Interpolate embeddings
            interpolated_embedding = np.average(closest_embeddings, axis=0, weights=weights)
            
            # Create interpolated text
            interpolated_text = self._create_interpolated_text(closest_texts, closest_embeddings, interpolated_embedding)
            
            # Get final coordinates
            final_embedding = self.embedding_model.encode([interpolated_text])[0]
            final_coords = self._embedding_to_coordinates(final_embedding)
            
            coordinate_error = np.sqrt(
                (final_coords[0] - target_coordinates[0])**2 + 
                (final_coords[1] - target_coordinates[1])**2
            )
            
            return {
                'generated_text': interpolated_text,
                'title': self._extract_title(interpolated_text),
                'final_embedding': final_embedding.tolist(),
                'predicted_coordinates': final_coords.tolist(),
                'target_coordinates': target_coordinates,
                'coordinate_error': float(coordinate_error),
                'method_used': 'precision_embedding_interpolation',
                'iterations_used': 1,
                'embedding_distance': 0.0,
                'converged': coordinate_error < 1.0,
                'correction_history': []
            }
            
        except Exception as e:
            logger.warning(f"Embedding interpolation strategy failed: {e}")
            return self._create_failure_result(target_coordinates, "embedding_interpolation", str(e))
    
    def _try_direct_optimization_strategy(self, target_coordinates: Tuple[float, float]) -> Dict[str, Any]:
        """Try direct optimization strategy."""
        try:
            # Start with closest coordinate text
            target_coords_reshaped = np.array(target_coordinates).reshape(1, -1)
            distances, indices = self.coordinate_neighbors.kneighbors(target_coords_reshaped)
            
            best_idx = indices[0][0]
            current_text = self.reference_texts[best_idx]
            best_error = float('inf')
            best_text = current_text
            
            # Iterative optimization
            for iteration in range(10):  # More iterations for precision
                current_embedding = self.embedding_model.encode([current_text])[0]
                current_coords = self._embedding_to_coordinates(current_embedding)
                
                error = np.sqrt(
                    (current_coords[0] - target_coordinates[0])**2 + 
                    (current_coords[1] - target_coordinates[1])**2
                )
                
                if error < best_error:
                    best_error = error
                    best_text = current_text
                
                if error < 1.0:
                    break
                
                # Optimize text to reduce coordinate error
                current_text = self._optimize_text_for_coordinates(
                    current_text, target_coordinates, current_coords, iteration
                )
            
            # Get final coordinates
            final_embedding = self.embedding_model.encode([best_text])[0]
            final_coords = self._embedding_to_coordinates(final_embedding)
            
            return {
                'generated_text': best_text,
                'title': self._extract_title(best_text),
                'final_embedding': final_embedding.tolist(),
                'predicted_coordinates': final_coords.tolist(),
                'target_coordinates': target_coordinates,
                'coordinate_error': float(best_error),
                'method_used': 'precision_direct_optimization',
                'iterations_used': 10,
                'embedding_distance': 0.0,
                'converged': best_error < 1.0,
                'correction_history': []
            }
            
        except Exception as e:
            logger.warning(f"Direct optimization strategy failed: {e}")
            return self._create_failure_result(target_coordinates, "direct_optimization", str(e))
    
    def _try_hybrid_strategy(self, target_coordinates: Tuple[float, float]) -> Dict[str, Any]:
        """Try hybrid strategy combining multiple approaches."""
        try:
            # Combine inverse UMAP + embedding interpolation + direct optimization
            
            # Step 1: Get target embedding using inverse UMAP
            target_embedding = None
            if self.parametric_umap is not None:
                try:
                    target_coords_reshaped = np.array(target_coordinates).reshape(1, -1)
                    target_embedding = self.parametric_umap.inverse_transform(target_coords_reshaped)[0]
                except:
                    pass
            
            # Step 2: Find best reference texts
            if target_embedding is not None:
                similarities = cosine_similarity(target_embedding.reshape(1, -1), self.reference_embeddings)[0]
                closest_indices = np.argsort(similarities)[::-1][:3]
            else:
                target_coords_reshaped = np.array(target_coordinates).reshape(1, -1)
                distances, indices = self.coordinate_neighbors.kneighbors(target_coords_reshaped)
                closest_indices = indices[0][:3]
            
            # Step 3: Create hybrid text
            closest_texts = [self.reference_texts[i] for i in closest_indices]
            hybrid_text = self._create_hybrid_text(closest_texts, target_coordinates)
            
            # Step 4: Optimize the hybrid text
            best_text = hybrid_text
            best_error = float('inf')
            
            for iteration in range(5):
                current_embedding = self.embedding_model.encode([best_text])[0]
                current_coords = self._embedding_to_coordinates(current_embedding)
                
                error = np.sqrt(
                    (current_coords[0] - target_coordinates[0])**2 + 
                    (current_coords[1] - target_coordinates[1])**2
                )
                
                if error < best_error:
                    best_error = error
                    best_text = best_text
                
                if error < 1.0:
                    break
                
                # Optimize text
                best_text = self._optimize_text_for_coordinates(
                    best_text, target_coordinates, current_coords, iteration
                )
            
            # Get final coordinates
            final_embedding = self.embedding_model.encode([best_text])[0]
            final_coords = self._embedding_to_coordinates(final_embedding)
            
            return {
                'generated_text': best_text,
                'title': self._extract_title(best_text),
                'final_embedding': final_embedding.tolist(),
                'predicted_coordinates': final_coords.tolist(),
                'target_coordinates': target_coordinates,
                'coordinate_error': float(best_error),
                'method_used': 'precision_hybrid',
                'iterations_used': 5,
                'embedding_distance': 0.0,
                'converged': best_error < 1.0,
                'correction_history': []
            }
            
        except Exception as e:
            logger.warning(f"Hybrid strategy failed: {e}")
            return self._create_failure_result(target_coordinates, "hybrid", str(e))
    
    def _create_interpolated_text(self, texts: List[str], embeddings: List[np.ndarray], target_embedding: np.ndarray) -> str:
        """Create interpolated text based on embedding similarities."""
        # Calculate similarities to target embedding
        similarities = []
        for embedding in embeddings:
            sim = cosine_similarity(target_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
            similarities.append(sim)
        
        # Normalize similarities to weights
        weights = np.array(similarities)
        weights = weights / np.sum(weights)
        
        # Create weighted combination of texts
        combined_text = ""
        for i, (text, weight) in enumerate(zip(texts, weights)):
            # Extract key phrases from each text
            sentences = text.split('.')
            key_sentences = sentences[:2] if len(sentences) >= 2 else sentences
            
            # Weight the contribution
            if weight > 0.3:  # Only include significant contributions
                combined_text += f" {'. '.join(key_sentences)}."
        
        # Add coordinate-specific content
        if not combined_text.strip():
            combined_text = texts[0]  # Fallback to first text
        
        return combined_text.strip()
    
    def _optimize_text_for_coordinates(
        self, 
        current_text: str, 
        target_coords: Tuple[float, float], 
        current_coords: np.ndarray,
        iteration: int
    ) -> str:
        """Optimize text to reduce coordinate error."""
        
        # Calculate coordinate differences
        x_diff = target_coords[0] - current_coords[0]
        y_diff = target_coords[1] - current_coords[1]
        
        # More aggressive modifications for precision
        modifications = []
        
        # X-axis modifications - use different research domains
        if x_diff > 2:
            modifications.append("quantum computing algorithms and superconducting quantum devices")
        elif x_diff > 1:
            modifications.append("machine learning neural networks and deep learning architectures")
        elif x_diff > 0:
            modifications.append("statistical mechanics and condensed matter physics")
        elif x_diff > -1:
            modifications.append("linear algebra and matrix theory with eigenvalue analysis")
        elif x_diff > -2:
            modifications.append("graph theory and network analysis with social dynamics")
        else:
            modifications.append("abstract algebra and group theory with symmetry operations")
        
        # Y-axis modifications - use different methodologies
        if y_diff > 2:
            modifications.append("experimental physics with particle accelerators and detector technology")
        elif y_diff > 1:
            modifications.append("computational biology with molecular dynamics and protein folding")
        elif y_diff > 0:
            modifications.append("artificial intelligence with natural language processing and computer vision")
        elif y_diff > -1:
            modifications.append("data science with statistical modeling and predictive analytics")
        elif y_diff > -2:
            modifications.append("software engineering with distributed systems and microservices")
        else:
            modifications.append("theoretical computer science with complexity theory and algorithmic design")
        
        # Create precision-optimized text
        precision_text = f"Research in {modifications[0]}. This work presents novel methodologies in {modifications[1]}. The approach demonstrates significant advances in computational efficiency through {modifications[0]} principles and {modifications[1]} techniques. Experimental validation shows improvements in accuracy and performance using advanced algorithmic approaches. The theoretical framework incorporates {modifications[0]} foundations with {modifications[1]} applications. Precision coordinate analysis targeting ({target_coords[0]:.2f}, {target_coords[1]:.2f}) demonstrates enhanced accuracy through {modifications[0]} and {modifications[1]} integration. Iteration {iteration + 1} optimization shows improved coordinate precision and theoretical rigor."
        
        return precision_text
    
    def _create_hybrid_text(self, texts: List[str], target_coordinates: Tuple[float, float]) -> str:
        """Create hybrid text combining multiple approaches."""
        # Combine key elements from reference texts
        combined_elements = []
        
        for text in texts:
            sentences = text.split('.')
            if sentences:
                combined_elements.append(sentences[0].strip())
        
        # Create hybrid text
        hybrid_text = f"Hybrid research combining {', '.join(combined_elements[:2])}. This work presents novel methodologies targeting coordinates ({target_coordinates[0]:.2f}, {target_coordinates[1]:.2f}) through advanced computational techniques and theoretical frameworks. The approach demonstrates significant improvements in precision and accuracy through multi-strategy optimization and comprehensive experimental validation."
        
        return hybrid_text
    
    def _embedding_to_coordinates(self, embedding: np.ndarray) -> np.ndarray:
        """Convert embedding to 2D coordinates using actual Parametric UMAP model."""
        if self.parametric_umap is not None:
            try:
                coordinates = self.parametric_umap.transform(embedding.reshape(1, -1))
                return coordinates[0]
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
        return "Precision Coordinate Research Paper"
    
    def _create_failure_result(self, target_coordinates: Tuple[float, float], method: str, error_msg: str) -> Dict[str, Any]:
        """Create a failure result."""
        return {
            'generated_text': f"Research in precision coordinate analysis targeting ({target_coordinates[0]:.2f}, {target_coordinates[1]:.2f}). This work presents novel methodologies in computational analysis. The approach demonstrates significant improvements in accuracy and efficiency through advanced algorithmic techniques.",
            'title': f"Precision Coordinate Research ({method})",
            'final_embedding': [0.0] * 384,  # Dummy embedding
            'predicted_coordinates': [0.0, 0.0],
            'target_coordinates': target_coordinates,
            'coordinate_error': 100.0,  # High error to indicate failure
            'method_used': f'precision_{method}_failed',
            'iterations_used': 0,
            'embedding_distance': 0.0,
            'converged': False,
            'correction_history': []
        }
