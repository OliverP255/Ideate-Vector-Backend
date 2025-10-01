"""
Advanced embedding-to-text pipeline service.

This implements the full pipeline described in the PDF:
1. Parametric UMAP for invertible dimensionality reduction
2. Text generation from embeddings (MVP: Retrieval + LLM Synthesis)
3. Correction loop to ensure generated text lands at target coordinates
"""

from .embedding_to_text_service import EmbeddingToTextService

__all__ = ['EmbeddingToTextService']
