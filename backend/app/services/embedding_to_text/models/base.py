"""
Base models and data structures for the embedding-to-text pipeline.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum


class TextGenerationMethod(Enum):
    """Methods for generating text from embeddings."""
    RETRIEVAL_LLM_SYNTHESIS = "retrieval_llm_synthesis"
    EMBEDDING_CONDITIONED_GENERATOR = "embedding_conditioned_generator"
    VEC2TEXT = "vec2text"


@dataclass
class EmbeddingToTextRequest:
    """Request for generating text from 2D coordinates."""
    x: float
    y: float
    method: TextGenerationMethod = TextGenerationMethod.RETRIEVAL_LLM_SYNTHESIS
    max_correction_iterations: int = 3
    target_embedding_distance_threshold: float = 0.1
    user_id: str = "system"
    context_radius: float = 1.0


@dataclass
class TextGenerationResult:
    """Result of text generation process."""
    generated_text: str
    title: str
    target_embedding: List[float]
    predicted_coordinates: Tuple[float, float]
    method_used: TextGenerationMethod
    nearest_neighbors: List[Dict[str, Any]]
    generation_metadata: Dict[str, Any]


@dataclass
class CorrectionResult:
    """Result of embedding correction process."""
    corrected_text: Optional[str] = None
    final_embedding: Optional[List[float]] = None
    final_coordinates: Optional[Tuple[float, float]] = None
    embedding_distance: Optional[float] = None
    coordinate_error: Optional[float] = None
    iterations_used: int = 0
    correction_history: List[Dict[str, Any]] = None


@dataclass
class EmbeddingToTextResponse:
    """Complete response from embedding-to-text pipeline."""
    request: EmbeddingToTextRequest
    text_generation_result: TextGenerationResult
    correction_result: Optional[CorrectionResult] = None
    success: bool = True
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    
    @property
    def final_text(self) -> str:
        """Get the final corrected text or original generated text."""
        if self.correction_result and self.correction_result.corrected_text:
            return self.correction_result.corrected_text
        return self.text_generation_result.generated_text
    
    @property
    def final_coordinates(self) -> Tuple[float, float]:
        """Get the final coordinates after correction."""
        if self.correction_result and self.correction_result.final_coordinates:
            return self.correction_result.final_coordinates
        return self.text_generation_result.predicted_coordinates
    
    @property
    def final_embedding(self) -> List[float]:
        """Get the final embedding after correction."""
        if self.correction_result and self.correction_result.final_embedding:
            return self.correction_result.final_embedding
        return self.text_generation_result.target_embedding


@dataclass
class NearestNeighborResult:
    """Result of nearest neighbor search."""
    document_id: str
    title: str
    content: str
    embedding: List[float]
    coordinates: Tuple[float, float]
    distance: float
    categories: List[str]
