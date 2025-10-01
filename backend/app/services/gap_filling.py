"""
Gap filling service for generating contextual text in empty map areas.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from app.services.config import get_settings
from app.services.search import SearchService
from app.services.mapping import MappingService
from app.services.embedding_to_text.text_generation.vec2text_service import Vec2TextService
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class GapFillingService:
    """Service for generating contextual text to fill empty map areas."""
    
    _embedding_generator = None  # Class-level cache for the embedding generator
    
    def __init__(self):
        self.settings = get_settings()
        self.search_service = SearchService()
        self.mapping_service = MappingService()
        
        # Initialize Vec2Text service for high-precision text generation
        self.vec2text_service = Vec2TextService()
        self.vec2text_service.initialize()
        
        # Use cached embedding generator or create new one
        if GapFillingService._embedding_generator is None:
            logger.info("Creating new embedding generator instance for GapFillingService")
            GapFillingService._embedding_generator = DocumentEmbeddingGenerator()
        self.embedding_generator = GapFillingService._embedding_generator
    
    async def analyze_gap_area(self, x: float, y: float, radius: float = 1.0) -> Dict[str, Any]:
        """
        Analyze an empty area to understand its semantic context.
        
        Args:
            x: X coordinate of the gap center
            y: Y coordinate of the gap center
            radius: Analysis radius around the gap
            
        Returns:
            Dict containing gap analysis results
        """
        try:
            logger.info(f"Analyzing gap area at ({x}, {y}) with radius {radius}")
            
            # Get all documents and find nearby ones
            all_coords = await self.mapping_service.get_all_coordinates()
            nearby_docs = []
            
            for coord_data in all_coords["coordinates"]:
                doc_x, doc_y = coord_data["coordinates"]
                distance = np.sqrt((doc_x - x) ** 2 + (doc_y - y) ** 2)
                
                if distance <= radius * 2:  # Look at a wider area for context
                    nearby_docs.append({
                        "document_id": coord_data["document_id"],
                        "coordinates": coord_data["coordinates"],
                        "title": coord_data.get("title", ""),
                        "categories": coord_data.get("categories", []),
                        "distance": distance
                    })
            
            # Sort by distance
            nearby_docs.sort(key=lambda doc: doc["distance"])
            
            # Analyze the semantic context
            context_analysis = self._analyze_semantic_context(nearby_docs)
            
            # Determine if this is actually a gap (low density area)
            gap_score = self._calculate_gap_score(x, y, radius, all_coords["coordinates"])
            
            return {
                "gap_coordinates": {"x": x, "y": y},
                "analysis_radius": radius,
                "nearby_documents": nearby_docs[:10],  # Top 10 closest
                "context_analysis": context_analysis,
                "gap_score": gap_score,
                "is_valid_gap": gap_score > 0.3,  # Lowered threshold for more flexible gap filling
                "analyzed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze gap area: {e}")
            raise
    
    def _analyze_semantic_context(self, nearby_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the semantic context of nearby documents."""
        if not nearby_docs:
            return {"primary_topics": [], "secondary_topics": [], "semantic_cluster": "unknown"}
        
        # Extract and analyze categories
        all_categories = []
        for doc in nearby_docs:
            for cat in doc.get("categories", []):
                main_cat = cat.split('.')[0] if '.' in cat else cat
                all_categories.append(main_cat)
        
        # Count category frequencies
        category_counts = {}
        for cat in all_categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Sort by frequency
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        primary_topics = [cat for cat, count in sorted_categories[:3] if count > 1]
        secondary_topics = [cat for cat, count in sorted_categories[3:6] if count > 0]
        
        # Determine semantic cluster
        if primary_topics:
            if 'cs' in primary_topics:
                semantic_cluster = "computer_science"
            elif 'math' in primary_topics:
                semantic_cluster = "mathematics"
            elif any(topic.startswith('astro') for topic in primary_topics):
                semantic_cluster = "astronomy"
            elif any(topic.startswith('quant') for topic in primary_topics):
                semantic_cluster = "quantum_physics"
            elif any(topic.startswith('cond-mat') for topic in primary_topics):
                semantic_cluster = "condensed_matter"
            else:
                semantic_cluster = "interdisciplinary"
        else:
            semantic_cluster = "unknown"
        
        return {
            "primary_topics": primary_topics,
            "secondary_topics": secondary_topics,
            "semantic_cluster": semantic_cluster,
            "topic_diversity": len(set(all_categories)),
            "document_count": len(nearby_docs)
        }
    
    def _calculate_gap_score(self, x: float, y: float, radius: float, all_coords: List[Dict]) -> float:
        """Calculate how much of a gap this area represents (0-1 scale)."""
        docs_in_radius = 0
        
        for coord_data in all_coords:
            doc_x, doc_y = coord_data["coordinates"]
            distance = np.sqrt((doc_x - x) ** 2 + (doc_y - y) ** 2)
            if distance <= radius:
                docs_in_radius += 1
        
        # Normalize based on expected density
        # If there are 400 docs in an 8x5 area, expected density is ~10 docs per unit area
        expected_docs = np.pi * radius * radius * (400 / (8 * 5))  # Rough estimate
        actual_ratio = docs_in_radius / max(expected_docs, 1)
        
        # Convert to gap score (higher = more of a gap)
        gap_score = max(0, 1 - actual_ratio)
        return min(gap_score, 1.0)
    
    async def generate_gap_filling_text(
        self, 
        target_x: float, 
        target_y: float, 
        context_analysis: Dict[str, Any],
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Generate text that would embed to the target coordinates.
        
        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            context_analysis: Semantic context from gap analysis
            max_iterations: Maximum iterations for coordinate refinement
            
        Returns:
            Dict containing generated text and predicted coordinates
        """
        try:
            logger.info(f"Generating gap-filling text for coordinates ({target_x}, {target_y})")
            
            # Try Vec2Text service first for high-precision generation
            try:
                vec2text_result = self.vec2text_service.generate_text_for_coordinates(
                    (target_x, target_y)
                )
                
                if vec2text_result and vec2text_result.get('generated_text'):
                    logger.info(f"Vec2Text service generated text with method: {vec2text_result.get('method_used', 'unknown')}")
                    
                    # Use Vec2Text result if it's available
                    generated_text = {
                        "title": vec2text_result.get('title', 'Generated Content'),
                        "content": vec2text_result['generated_text'],
                        "semantic_cluster": context_analysis.get('semantic_cluster', 'unknown'),
                        "primary_topics": context_analysis.get('primary_topics', [])
                    }
                    
                    return {
                        "generated_text": generated_text,
                        "predicted_coordinates": vec2text_result.get('predicted_coordinates', [target_x, target_y]),
                        "target_coordinates": [target_x, target_y],
                        "coordinate_error": vec2text_result.get('coordinate_error', 0.0),
                        "iterations_used": vec2text_result.get('iterations_used', 1),
                        "generated_at": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                logger.warning(f"Vec2Text service failed: {e}, falling back to contextual generation")
            
            # Fallback to original contextual generation
            generated_text = self._generate_contextual_text(context_analysis)
            
            # Iteratively refine to hit target coordinates
            best_text = generated_text
            best_coords = None
            best_distance = float('inf')
            
            for iteration in range(max_iterations):
                # Generate embedding for current text
                text_data = {
                    "title": generated_text["title"],
                    "text": generated_text["content"],
                    "document_id": f"generated_gap_{iteration}"
                }
                
                embedding_result = self.embedding_generator.generate_embedding(text_data)
                predicted_coords = self._predict_coordinates(embedding_result["embedding"])
                
                # Calculate distance to target
                distance = np.sqrt((predicted_coords[0] - target_x) ** 2 + (predicted_coords[1] - target_y) ** 2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_text = generated_text
                    best_coords = predicted_coords
                
                # If we're close enough, stop
                if distance < 0.5:  # Within 0.5 units of target
                    break
                
                # Refine text based on coordinate error
                generated_text = self._refine_text_for_coordinates(
                    generated_text, 
                    target_x, 
                    target_y, 
                    predicted_coords
                )
            
            return {
                "generated_text": best_text,
                "predicted_coordinates": best_coords,
                "target_coordinates": [target_x, target_y],
                "coordinate_error": best_distance,
                "iterations_used": iteration + 1,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate gap-filling text: {e}")
            raise
    
    def _generate_contextual_text(self, context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate initial text based on semantic context."""
        cluster = context_analysis.get("semantic_cluster", "unknown")
        primary_topics = context_analysis.get("primary_topics", [])
        
        # Define text templates for different clusters
        templates = {
            "computer_science": {
                "title": f"Advances in {primary_topics[0] if primary_topics else 'Computer Science'}",
                "content": f"This paper presents novel approaches to fundamental problems in {primary_topics[0] if primary_topics else 'computer science'}. We explore theoretical foundations and practical applications, bridging the gap between existing methodologies and emerging paradigms. The work contributes to the growing body of research in computational methods and algorithmic efficiency."
            },
            "mathematics": {
                "title": f"Mathematical Foundations in {primary_topics[0] if primary_topics else 'Mathematics'}",
                "content": f"We develop new mathematical frameworks that extend classical results in {primary_topics[0] if primary_topics else 'mathematical analysis'}. The theoretical contributions provide deeper insights into fundamental mathematical structures, with implications for both pure and applied mathematics. This work establishes connections between disparate areas of mathematical research."
            },
            "astronomy": {
                "title": f"Observational Studies in {primary_topics[0] if primary_topics else 'Astrophysics'}",
                "content": f"This study presents comprehensive observational analysis of {primary_topics[0] if primary_topics else 'astronomical phenomena'}. We combine multi-wavelength observations with theoretical modeling to advance our understanding of cosmic processes. The findings contribute to our knowledge of stellar evolution, galactic dynamics, and cosmological structure formation."
            },
            "quantum_physics": {
                "title": f"Quantum Mechanical Approaches to {primary_topics[0] if primary_topics else 'Quantum Systems'}",
                "content": f"We investigate quantum mechanical properties of {primary_topics[0] if primary_topics else 'quantum systems'} using advanced theoretical and computational methods. The work explores quantum entanglement, coherence, and information processing in complex quantum environments. These results have implications for quantum computing and fundamental physics."
            },
            "condensed_matter": {
                "title": f"Materials Properties in {primary_topics[0] if primary_topics else 'Condensed Matter'}",
                "content": f"This research examines electronic and structural properties of {primary_topics[0] if primary_topics else 'condensed matter systems'}. We employ first-principles calculations and experimental characterization to understand material behavior at the atomic scale. The findings contribute to the design of novel materials with tailored properties."
            },
            "interdisciplinary": {
                "title": f"Interdisciplinary Research in {primary_topics[0] if primary_topics else 'Cross-Disciplinary Studies'}",
                "content": f"This work bridges multiple disciplines, combining insights from {', '.join(primary_topics[:3]) if primary_topics else 'various fields'}. We develop integrated approaches that leverage methodologies from different domains to address complex research questions. The interdisciplinary nature of this work opens new avenues for scientific discovery."
            }
        }
        
        # Get template or default
        template = templates.get(cluster, templates["interdisciplinary"])
        
        return {
            "title": template["title"],
            "content": template["content"],
            "semantic_cluster": cluster,
            "primary_topics": primary_topics
        }
    
    def _predict_coordinates(self, embedding: List[float]) -> Tuple[float, float]:
        """
        Predict 2D coordinates from embedding using the existing UMAP model.
        This is a simplified version - in practice, you'd load the actual UMAP model.
        """
        # For now, use a simple projection based on the embedding
        # In a real implementation, you'd load the trained UMAP model
        embedding_array = np.array(embedding)
        
        # Simple linear projection (this should be replaced with actual UMAP model)
        x = np.dot(embedding_array[:10], np.random.randn(10)) * 2 + 8  # Center around 8
        y = np.dot(embedding_array[10:20], np.random.randn(10)) * 1.5  # Center around 0
        
        return (float(x), float(y))
    
    def _refine_text_for_coordinates(
        self, 
        current_text: Dict[str, Any], 
        target_x: float, 
        target_y: float, 
        current_coords: Tuple[float, float]
    ) -> Dict[str, Any]:
        """Refine text to better match target coordinates."""
        # Simple refinement strategy - modify content based on coordinate error
        x_error = target_x - current_coords[0]
        y_error = target_y - current_coords[1]
        
        # Add content that might shift the embedding in the right direction
        additions = []
        if abs(x_error) > 0.5:
            if x_error > 0:
                additions.append("This research emphasizes computational methods and algorithmic approaches.")
            else:
                additions.append("We focus on theoretical foundations and mathematical rigor.")
        
        if abs(y_error) > 0.5:
            if y_error > 0:
                additions.append("The work has broad implications for interdisciplinary research.")
            else:
                additions.append("This study provides detailed experimental validation and practical applications.")
        
        if additions:
            current_text["content"] += " " + " ".join(additions)
        
        return current_text
    
    async def create_gap_filling_document(
        self, 
        x: float, 
        y: float, 
        user_id: str = "system",
        force_gap_filling: bool = False
    ) -> Dict[str, Any]:
        """
        Complete workflow to create a gap-filling document.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            user_id: User who requested the generation
            force_gap_filling: If True, skip gap validation and fill anywhere
            
        Returns:
            Complete document ready for map display
        """
        try:
            # Step 1: Analyze the gap area
            gap_analysis = await self.analyze_gap_area(x, y)
            
            # Always allow gap filling - no restrictions based on gap score
            # The advanced Vec2Text and Parametric UMAP inverse will handle coordinate targeting
            
            # Step 2: Generate contextual text
            generation_result = await self.generate_gap_filling_text(
                x, y, gap_analysis["context_analysis"]
            )
            
            # Step 3: Create complete document
            document_id = f"gap_filled_{user_id}_{int(datetime.now().timestamp())}"
            
            document = {
                "document_id": document_id,
                "title": generation_result["generated_text"]["title"],
                "content": generation_result["generated_text"]["content"],
                "coordinates": generation_result["predicted_coordinates"],
                "source": "gap_filling",
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "gap_analysis": gap_analysis,
                "generation_metadata": generation_result,
                "semantic_cluster": generation_result["generated_text"]["semantic_cluster"]
            }
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to create gap-filling document: {e}")
            raise
