"""
Correction loop service for ensuring generated text lands at target coordinates.

This implements the correction loop from the PDF:
1. After generating text, re-embed it with the same embedding model
2. Compare new embedding to the target embedding from UMAP inverse
3. If they are far apart, regenerate or adjust
4. This ensures the generated text really sits where the user clicked
"""

import logging
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import sys
import os

# Add the pipeline directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
from sentence_transformers import SentenceTransformer

from ..models.base import CorrectionResult, TextGenerationResult
from ..parametric_umap.parametric_umap_service import ParametricUMAPService

logger = logging.getLogger(__name__)


class CorrectionLoopService:
    """
    Service for correcting generated text to ensure it lands at target coordinates.
    
    This implements the correction loop described in the PDF to ensure generated
    text really sits where the user clicked on the map.
    """
    
    def __init__(self, parametric_umap_service: ParametricUMAPService):
        self.parametric_umap = parametric_umap_service
        self._embedding_generator = None
    
    def get_embedding_generator(self) -> SentenceTransformer:
        """Get or create embedding generator."""
        if self._embedding_generator is None:
            self._embedding_generator = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedding_generator
    
    def correct_text_placement(
        self,
        text_result: TextGenerationResult,
        target_embedding: np.ndarray,
        target_coordinates: Tuple[float, float],
        max_iterations: int = 3,
        distance_threshold: float = 0.1
    ) -> CorrectionResult:
        """
        Correct generated text to ensure it lands at target coordinates.
        
        Args:
            text_result: Original text generation result
            target_embedding: Target embedding from UMAP inverse
            target_coordinates: Target 2D coordinates
            max_iterations: Maximum correction iterations
            distance_threshold: Embedding distance threshold for convergence
            
        Returns:
            CorrectionResult with correction details
        """
        logger.info(f"Starting correction loop for text placement (max {max_iterations} iterations)")
        
        correction_history = []
        current_text = text_result.generated_text
        current_title = text_result.title
        
        for iteration in range(max_iterations):
            logger.info(f"Correction iteration {iteration + 1}/{max_iterations}")
            
            # Re-embed the current text
            current_embedding = self._embed_combined_text(current_title, current_text)
            
            # Transform to 2D coordinates
            current_coordinates = self.parametric_umap.transform(
                current_embedding.reshape(1, -1)
            )[0]
            
            # Calculate distances
            embedding_distance = np.linalg.norm(current_embedding - target_embedding)
            coordinate_error = np.linalg.norm(
                np.array(current_coordinates) - np.array(target_coordinates)
            )
            
            # Record iteration
            iteration_record = {
                'iteration': iteration + 1,
                'embedding_distance': float(embedding_distance),
                'coordinate_error': float(coordinate_error),
                'coordinates': tuple(current_coordinates),
                'text_preview': current_text[:100] + "..." if len(current_text) > 100 else current_text
            }
            correction_history.append(iteration_record)
            
            logger.info(f"Iteration {iteration + 1}: embedding_distance={embedding_distance:.4f}, "
                       f"coordinate_error={coordinate_error:.4f}")
            
            # Check convergence
            if embedding_distance < distance_threshold:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            # Generate correction
            if iteration < max_iterations - 1:  # Don't correct on last iteration
                current_text = self._generate_correction(
                    current_text, current_title, target_embedding, embedding_distance
                )
        
        # Create correction result
        result = CorrectionResult(
            corrected_text=current_text if iteration > 0 else None,
            final_embedding=current_embedding.tolist(),
            final_coordinates=tuple(current_coordinates),
            embedding_distance=float(embedding_distance),
            coordinate_error=float(coordinate_error),
            iterations_used=iteration + 1,
            correction_history=correction_history
        )
        
        logger.info(f"Correction completed: final_distance={embedding_distance:.4f}, "
                   f"final_error={coordinate_error:.4f}")
        
        return result
    
    def _embed_combined_text(self, title: str, content: str) -> np.ndarray:
        """
        Embed combined title and content text.
        
        Args:
            title: Document title
            content: Document content
            
        Returns:
            Embedding vector
        """
        embedding_generator = self.get_embedding_generator()
        
        # Combine title and content as done in the original system
        combined_text = f"{title}\n\n{content}"
        
        # Create document data structure
        document_data = {
            'title': title,
            'text': content,
            'document_id': 'correction_temp'
        }
        
        # Generate embedding
        embedding_result = embedding_generator.generate_embedding(document_data)
        
        return np.array(embedding_result['embedding'])
    
    def _generate_correction(
        self,
        current_text: str,
        current_title: str,
        target_embedding: np.ndarray,
        embedding_distance: float
    ) -> str:
        """
        Generate a correction to the text to move it closer to target embedding.
        
        Args:
            current_text: Current text to correct
            current_title: Current title
            target_embedding: Target embedding to move towards
            embedding_distance: Current distance from target
            
        Returns:
            Corrected text
        """
        logger.info(f"Generating coordinate-aware correction (distance: {embedding_distance:.4f})")
        
        # Get target coordinates from the embedding using Parametric UMAP transform
        target_coordinates = self.parametric_umap.transform(target_embedding.reshape(1, -1))[0]
        
        # Generate more targeted correction based on embedding distance and target coordinates
        x, y = target_coordinates
        
        # Create correction that varies based on distance and coordinates
        if embedding_distance > 1.0:
            # Large distance - add significant coordinate-specific content
            if x > 0 and y > 0:
                correction = f" Advanced computational methodologies targeting coordinates ({x:.2f}, {y:.2f}) with high-precision algorithms and theoretical frameworks."
            elif x > 0 and y < 0:
                correction = f" Applied mathematical approaches for coordinate location ({x:.2f}, {y:.2f}) emphasizing practical implementations and computational efficiency."
            elif x < 0 and y > 0:
                correction = f" Theoretical foundations and mathematical principles for coordinate space ({x:.2f}, {y:.2f}) with rigorous analytical methods."
            else:
                correction = f" Fundamental mathematical concepts and basic computational methods for coordinate region ({x:.2f}, {y:.2f})."
        elif embedding_distance > 0.5:
            # Medium distance - add moderate coordinate-specific content
            if abs(x) > abs(y):
                correction = f" Coordinate-targeted methodology emphasizing {'positive' if x > 0 else 'negative'} x-axis direction at ({x:.2f}, {y:.2f})."
            else:
                correction = f" Coordinate-specific approach focusing on {'positive' if y > 0 else 'negative'} y-axis direction at ({x:.2f}, {y:.2f})."
        else:
            # Small distance - add minimal coordinate-specific content
            correction = f" Precise coordinate targeting at ({x:.2f}, {y:.2f}) with refined analytical methods."
        
        # Add embedding-specific keywords based on target coordinates
        if x > 2:
            correction += " Computational algorithms and machine learning techniques."
        elif x < -2:
            correction += " Theoretical foundations and mathematical analysis."
        
        if y > 2:
            correction += " Advanced theoretical frameworks and innovative methodologies."
        elif y < -2:
            correction += " Practical applications and experimental validation."
        
        # Combine with original text
        corrected_text = current_text + correction
        
        return corrected_text
    
    def evaluate_correction_quality(
        self,
        correction_result: CorrectionResult,
        target_coordinates: Tuple[float, float],
        coordinate_tolerance: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of the correction.
        
        Args:
            correction_result: Result of correction process
            target_coordinates: Target coordinates
            coordinate_tolerance: Acceptable coordinate error
            
        Returns:
            Dictionary with quality metrics
        """
        final_coords = correction_result.final_coordinates
        coordinate_error = correction_result.coordinate_error
        
        # Calculate quality metrics
        is_within_tolerance = coordinate_error <= coordinate_tolerance
        
        # Calculate improvement
        if correction_result.correction_history:
            initial_error = correction_result.correction_history[0]['coordinate_error']
            improvement = (initial_error - coordinate_error) / initial_error if initial_error > 0 else 0
        else:
            improvement = 0
        
        quality_metrics = {
            'is_within_tolerance': is_within_tolerance,
            'coordinate_error': coordinate_error,
            'embedding_distance': correction_result.embedding_distance,
            'improvement_percentage': improvement * 100,
            'iterations_used': correction_result.iterations_used,
            'target_coordinates': target_coordinates,
            'final_coordinates': final_coords
        }
        
        return quality_metrics
