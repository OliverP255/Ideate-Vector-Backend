"""
Models for text generation configuration.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from .base import TextGenerationMethod


@dataclass
class TextGenerationConfig:
    """Configuration for text generation methods."""
    method: TextGenerationMethod = TextGenerationMethod.RETRIEVAL_LLM_SYNTHESIS
    
    # Retrieval + LLM Synthesis settings
    num_nearest_neighbors: int = 5
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    max_tokens: int = 500
    
    # Context settings
    include_title: bool = True
    include_abstract: bool = True
    include_categories: bool = True
    
    # Synthesis prompt template
    synthesis_prompt_template: str = """
You are a research paper synthesis expert. Based on the following research papers, generate a new research paper abstract that would semantically bridge the gap between them.

Nearest neighbor papers:
{nearest_neighbors}

Generate a new research paper abstract that:
1. Fills the semantic gap between these papers
2. Is scientifically coherent and realistic
3. Uses appropriate academic language
4. Is approximately 150-300 words
5. Includes a clear research question and methodology

Title: [Generate a concise, descriptive title]

Abstract: [Generate the abstract here]
"""

    # Embedding-conditioned generator settings (for future use)
    generator_model_path: Optional[str] = None
    prefix_length: int = 10
    generation_length: int = 300
    
    # Vec2Text settings (for future use)
    vec2text_model_path: Optional[str] = None
    num_beams: int = 4
    max_length: int = 400


@dataclass
class NearestNeighborResult:
    """Result of nearest neighbor search for text generation."""
    document_id: str
    title: str
    content: str
    embedding: List[float]
    coordinates: Tuple[float, float]
    distance: float
    categories: List[str]
    
    def to_prompt_context(self) -> str:
        """Convert to formatted string for LLM prompt."""
        categories_str = ", ".join(self.categories) if self.categories else "Unknown"
        return f"""
Paper {self.document_id}:
Title: {self.title}
Categories: {categories_str}
Distance: {self.distance:.4f}
Content: {self.content[:200]}...
"""
