"""
Text generation services for converting embeddings to text.
"""

from .text_generation_service import TextGenerationService
# Mock LLMSynthesisService
class LLMSynthesisService:
    def __init__(self, *args, **kwargs):
        pass

__all__ = ['TextGenerationService', 'LLMSynthesisService']
