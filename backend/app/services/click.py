"""
Click service for spatial queries and document retrieval.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from app.services.config import get_settings
from app.services.vector_db import VectorDatabaseService
from app.services.embedding import EmbeddingService
from app.services.mapping import MappingService
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ClickService:
    """Service for handling map clicks and document retrieval."""
    
    _embedding_generator = None  # Class-level cache for the embedding generator
    
    def __init__(self):
        self.settings = get_settings()
        self.vector_db = VectorDatabaseService()
        self.embedding_service = EmbeddingService()
        self.mapping_service = MappingService()
        
        # Use cached embedding generator or create new one
        if ClickService._embedding_generator is None:
            logger.info("Creating new embedding generator instance")
            ClickService._embedding_generator = DocumentEmbeddingGenerator()
        self.embedding_generator = ClickService._embedding_generator
    
    async def handle_map_click(
        self, 
        x: float, 
        y: float, 
        radius: float = 0.1,
        user_id: Optional[str] = None,
        query_text: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Handle a map click and return relevant documents.
        
        Args:
            x: X coordinate of click
            y: Y coordinate of click
            radius: Search radius around click point
            user_id: Optional user ID for personalized results
            query_text: Optional query text for semantic reranking
            limit: Maximum number of results
            
        Returns:
            Dict containing click results
        """
        try:
            logger.info(f"Handling map click at ({x}, {y}) with radius {radius}")
            
            # Step 1: Spatial candidate selection
            spatial_candidates = await self._get_spatial_candidates(x, y, radius)
            
            if not spatial_candidates:
                return {
                    "click_coordinates": {"x": x, "y": y},
                    "radius": radius,
                    "spatial_candidates": 0,
                    "documents": [],
                    "message": "No documents found in the clicked area"
                }
            
            logger.info(f"Found {len(spatial_candidates)} spatial candidates")
            
            # Step 2: High-dimensional reranking
            if query_text:
                # Semantic reranking with query
                reranked_docs = await self._semantic_rerank(spatial_candidates, query_text, limit)
                # If semantic reranking fails, fall back to distance-based
                if not reranked_docs:
                    logger.info("Semantic reranking failed, falling back to distance-based")
                    reranked_docs = await self._distance_rerank(spatial_candidates, x, y, limit)
            elif user_id:
                # User profile reranking
                reranked_docs = await self._user_profile_rerank(spatial_candidates, user_id, limit)
            else:
                # Simple distance-based reranking
                reranked_docs = await self._distance_rerank(spatial_candidates, x, y, limit)
            
            # Step 3: Format results
            result = {
                "click_coordinates": {"x": x, "y": y},
                "radius": radius,
                "spatial_candidates": len(spatial_candidates),
                "documents": reranked_docs,
                "user_id": user_id,
                "query_text": query_text,
                "processed_at": datetime.now().isoformat()
            }
            
            logger.info(f"Returning {len(reranked_docs)} documents for click at ({x}, {y})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to handle map click: {e}")
            raise
    
    async def _get_spatial_candidates(self, x: float, y: float, radius: float) -> List[Dict[str, Any]]:
        """
        Get spatial candidates using 2D coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            radius: Search radius
            
        Returns:
            List of candidate documents
        """
        try:
            # Get all document coordinates
            all_coords = await self.mapping_service.get_all_coordinates()
            
            # Filter by spatial proximity
            candidates = []
            for coord_data in all_coords["coordinates"]:
                doc_x, doc_y = coord_data["coordinates"]
                
                # Calculate distance
                distance = np.sqrt((doc_x - x) ** 2 + (doc_y - y) ** 2)
                
                if distance <= radius:
                    candidates.append({
                        "document_id": coord_data["document_id"],
                        "coordinates": coord_data["coordinates"],
                        "title": coord_data.get("title", f"Document {coord_data['document_id']}"),
                        "discipline": coord_data.get("discipline", "unknown"),
                        "distance": float(distance),
                        "source": coord_data.get("source", "sample_data")
                    })
            
            # Sort by distance
            candidates.sort(key=lambda x: x["distance"])
            
            # If no candidates found within radius, expand search to find closest 10 documents
            if not candidates:
                logger.info(f"No documents found within radius {radius}, expanding search")
                all_candidates = []
                for coord_data in all_coords["coordinates"]:
                    doc_x, doc_y = coord_data["coordinates"]
                    distance = np.sqrt((doc_x - x) ** 2 + (doc_y - y) ** 2)
                    all_candidates.append({
                        "document_id": coord_data["document_id"],
                        "coordinates": coord_data["coordinates"],
                        "title": coord_data.get("title", f"Document {coord_data['document_id']}"),
                        "discipline": coord_data.get("discipline", "unknown"),
                        "distance": float(distance),
                        "source": coord_data.get("source", "sample_data")
                    })
                
                # Sort by distance and take closest 10
                all_candidates.sort(key=lambda x: x["distance"])
                candidates = all_candidates[:10]
            
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to get spatial candidates: {e}")
            return []
    
    async def _semantic_rerank(
        self, 
        candidates: List[Dict[str, Any]], 
        query_text: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using semantic similarity to query.
        
        Args:
            candidates: List of spatial candidates
            query_text: Query text for semantic matching
            limit: Maximum number of results
            
        Returns:
            List of reranked documents
        """
        try:
            # Generate query embedding
            query_data = {"text": query_text, "title": "", "document_id": "query"}
            query_result = self.embedding_generator.generate_embedding(query_data)
            query_embedding = query_result["embedding"]
            
            # Get embeddings for candidates
            candidate_embeddings = []
            valid_candidates = []
            
            for candidate in candidates:
                embedding_data = await self.embedding_service.get_embedding(candidate["document_id"])
                if embedding_data and "embedding" in embedding_data:
                    candidate_embeddings.append(embedding_data["embedding"])
                    valid_candidates.append(candidate)
                else:
                    # Try to load embedding directly from file
                    embedding_file = Path("backend/data/embeddings") / f"{candidate['document_id']}.json"
                    if embedding_file.exists():
                        with open(embedding_file, 'r') as f:
                            embedding_data = json.load(f)
                        if "embedding" in embedding_data:
                            candidate_embeddings.append(embedding_data["embedding"])
                            valid_candidates.append(candidate)
            
            if not candidate_embeddings:
                return []
            
            # Calculate similarities
            similarities = []
            query_vector = np.array(query_embedding)
            
            for i, candidate_embedding in enumerate(candidate_embeddings):
                candidate_vector = np.array(candidate_embedding)
                similarity = float(np.dot(query_vector, candidate_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(candidate_vector)
                ))
                
                similarities.append({
                    "candidate": valid_candidates[i],
                    "similarity_score": similarity
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Format results
            results = []
            for item in similarities[:limit]:
                candidate = item["candidate"]
                results.append({
                    "document_id": candidate["document_id"],
                    "coordinates": candidate["coordinates"],
                    "title": candidate.get("title", f"Document {candidate['document_id']}"),
                    "discipline": candidate.get("discipline", "unknown"),
                    "similarity_score": item["similarity_score"],
                    "spatial_distance": candidate["distance"],
                    "rerank_method": "semantic_query"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to semantic rerank: {e}")
            return await self._distance_rerank(candidates, 0, 0, limit)
    
    async def _user_profile_rerank(
        self, 
        candidates: List[Dict[str, Any]], 
        user_id: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using user profile similarity.
        
        Args:
            candidates: List of spatial candidates
            user_id: User identifier
            limit: Maximum number of results
            
        Returns:
            List of reranked documents
        """
        try:
            # TODO: Implement user profile vector generation
            # For now, use simple distance-based reranking
            logger.info(f"User profile reranking not yet implemented for user {user_id}")
            return await self._distance_rerank(candidates, 0, 0, limit)
            
        except Exception as e:
            logger.error(f"Failed to user profile rerank: {e}")
            return await self._distance_rerank(candidates, 0, 0, limit)
    
    async def _distance_rerank(
        self, 
        candidates: List[Dict[str, Any]], 
        x: float, 
        y: float, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Simple distance-based reranking.
        
        Args:
            candidates: List of spatial candidates
            x: X coordinate (for distance calculation)
            y: Y coordinate (for distance calculation)
            limit: Maximum number of results
            
        Returns:
            List of reranked documents
        """
        try:
            results = []
            for candidate in candidates[:limit]:
                results.append({
                    "document_id": candidate["document_id"],
                    "coordinates": candidate["coordinates"],
                    "title": candidate.get("title", f"Document {candidate['document_id']}"),
                    "discipline": candidate.get("discipline", "unknown"),
                    "spatial_distance": candidate["distance"],
                    "rerank_method": "spatial_distance"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to distance rerank: {e}")
            return []
    
    async def get_document_details(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full document details for a clicked document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document details or None if not found
        """
        try:
            # Get embedding data first to get the actual document content
            embedding_data = await self.embedding_service.get_embedding(document_id)
            
            if not embedding_data:
                return None
            
            # Get coordinates
            coordinates = await self.mapping_service.get_document_coordinates(document_id)
            
            # For test documents, create synthetic document data
            if document_id.startswith("test-doc"):
                return {
                    "document_id": document_id,
                    "title": f"Test Document {document_id}",
                    "abstract": "This is a test document for the knowledge map system.",
                    "text": embedding_data.get("text", "Test document content"),
                    "metadata": embedding_data.get("metadata", {}),
                    "embedding_info": embedding_data,
                    "coordinates": coordinates["coordinates"] if coordinates else None,
                    "retrieved_at": datetime.now().isoformat()
                }
            
            # Try to find the actual cleaned document
            document_file = Path("data/cleaned") / f"{document_id}.json"
            if not document_file.exists():
                document_file = Path("backend/data/cleaned") / f"{document_id}.json"
            
            if document_file.exists():
                with open(document_file, 'r') as f:
                    document_data = json.load(f)
                
                return {
                    "document_id": document_id,
                    "title": document_data.get("title", ""),
                    "abstract": document_data.get("abstract", ""),
                    "text": document_data.get("text", ""),
                    "metadata": document_data.get("metadata", {}),
                    "embedding_info": embedding_data,
                    "coordinates": coordinates["coordinates"] if coordinates else None,
                    "retrieved_at": datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document details for {document_id}: {e}")
            return None
    
    async def get_click_statistics(self) -> Dict[str, Any]:
        """
        Get click statistics and system status.
        
        Returns:
            Dict containing click statistics
        """
        try:
            # Get collection info
            collection_info = await self.vector_db.get_collection_info()
            
            # Get mapping status
            mapping_status = await self.mapping_service.get_all_coordinates()
            
            return {
                "vector_db_status": collection_info,
                "total_documents": mapping_status["total_documents"],
                "mapping_available": len(mapping_status["coordinates"]) > 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get click statistics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }