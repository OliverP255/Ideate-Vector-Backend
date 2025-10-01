"""
Search service for semantic search functionality.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from app.services.sample_data import SampleDataService
from app.services.knowledge_database import KnowledgeDatabase
from app.services.vector_db import VectorDatabaseService

logger = logging.getLogger(__name__)


class SearchService:
    """Service for semantic search operations."""
    
    _embedding_model = None  # Class-level cache for the embedding model
    
    def __init__(self):
        self.sample_data_service = SampleDataService()
        self.knowledge_db = KnowledgeDatabase()
        self.vector_db = VectorDatabaseService()
        
        # Use cached model or create new one
        if SearchService._embedding_model is None:
            logger.info("Creating new embedding model instance for SearchService")
            SearchService._embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
        self.embedding_model = SearchService._embedding_model
        logger.info("SearchService initialized with cached embedding model")
    
    async def semantic_search(self, query: str, limit: int = 10, score_threshold: float = 0.0, offset: int = 0) -> Dict[str, Any]:
        """
        Perform semantic search using text query with pagination.
        
        Args:
            query: Search query text
            limit: Maximum number of results per page
            score_threshold: Minimum similarity score threshold
            offset: Number of results to skip (for pagination)
            
        Returns:
            Dict containing results, total count, and pagination info
        """
        try:
            logger.info(f"Performing semantic search for query: '{query}'")
            
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(query)
            
            # Search using vector database
            search_results = await self.vector_db.search_similar(query_embedding, limit=limit + offset)
            
            if not search_results:
                logger.warning("No documents found in vector database")
                # Fallback to sample data if available
                embeddings_data = self.sample_data_service.load_sample_embeddings()
                if not embeddings_data:
                    logger.warning("No embeddings found for search")
                    return {
                        "results": [],
                        "total_count": 0,
                        "limit": limit,
                        "offset": offset,
                        "has_more": False
                    }
                
                # Calculate similarities with sample data
                similarities = []
                for doc_data in embeddings_data:
                    doc_embedding = np.array(doc_data['embedding'])
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    
                    if similarity >= score_threshold:
                        discipline = "unknown"
                        if doc_data.get('categories') and len(doc_data['categories']) > 0:
                            discipline = doc_data['categories'][0].split('.')[0]
                        
                        similarities.append({
                            'document_id': doc_data['document_id'],
                            'title': doc_data['title'],
                            'discipline': discipline,
                            'similarity_score': float(similarity),
                            'content_preview': doc_data['content'][:200] + "..." if len(doc_data['content']) > 200 else doc_data['content']
                        })
                
                similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
                total_results = len(similarities)
                start_idx = offset
                end_idx = min(offset + limit, total_results)
                results = similarities[start_idx:end_idx]
            else:
                # Convert vector database results to expected format
                results = []
                for doc in search_results:
                    results.append({
                        'document_id': doc['document_id'],
                        'title': doc['title'],
                        'discipline': doc.get('source', 'unknown'),
                        'similarity_score': doc['similarity_score'],
                        'content_preview': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                    })
                
                # Apply pagination
                total_results = len(results)
                start_idx = offset
                end_idx = min(offset + limit, total_results)
                results = results[start_idx:end_idx]
            
            logger.info(f"Semantic search completed: {len(results)} results found (total: {total_results}, offset: {offset})")
            return {
                "results": results,
                "total_count": total_results,
                "limit": limit,
                "offset": offset,
                "has_more": end_idx < total_results
            }
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    async def semantic_search_with_spatial_filter(
        self, 
        query: str, 
        x: float, 
        y: float, 
        radius: float, 
        limit: int = 10, 
        score_threshold: float = 0.0,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Perform semantic search with spatial filtering and pagination.
        
        Args:
            query: Search query text
            x: X coordinate for spatial filtering
            y: Y coordinate for spatial filtering
            radius: Search radius for spatial filtering
            limit: Maximum number of results per page
            score_threshold: Minimum similarity score threshold
            offset: Number of results to skip (for pagination)
            
        Returns:
            Dict containing results, total count, and pagination info
        """
        try:
            logger.info(f"Performing semantic search with spatial filter: '{query}' at ({x}, {y}) with radius {radius}")
            
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(query)
            
            # Load document embeddings and coordinates
            embeddings_data = self.sample_data_service.load_sample_embeddings()
            coordinates_data = self.sample_data_service.load_sample_coordinates()
            
            if not embeddings_data or not coordinates_data:
                logger.warning("No embeddings or coordinates found for search")
                return []
            
            # Create coordinate lookup
            coord_lookup = {coord['document_id']: coord['coordinates'] for coord in coordinates_data}
            
            # Calculate similarities and spatial distances
            results = []
            for doc_data in embeddings_data:
                doc_id = doc_data['document_id']
                
                # Check if document has coordinates
                if doc_id not in coord_lookup:
                    continue
                
                doc_coords = coord_lookup[doc_id]
                doc_x, doc_y = doc_coords[0], doc_coords[1]
                
                # Calculate spatial distance
                spatial_distance = np.sqrt((doc_x - x) ** 2 + (doc_y - y) ** 2)
                
                # Skip if outside spatial radius
                if spatial_distance > radius:
                    continue
                
                # Calculate semantic similarity
                doc_embedding = np.array(doc_data['embedding'])
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                
                if similarity >= score_threshold:
                    # Get discipline from categories (first category)
                    discipline = "unknown"
                    if doc_data.get('categories') and len(doc_data['categories']) > 0:
                        discipline = doc_data['categories'][0].split('.')[0]
                    
                    results.append({
                        'document_id': doc_id,
                        'title': doc_data['title'],
                        'discipline': discipline,
                        'coordinates': doc_coords,
                        'similarity_score': float(similarity),
                        'spatial_distance': float(spatial_distance),
                        'content_preview': doc_data['content'][:200] + "..." if len(doc_data['content']) > 200 else doc_data['content']
                    })
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Apply pagination
            total_results = len(results)
            start_idx = offset
            end_idx = min(offset + limit, total_results)
            final_results = results[start_idx:end_idx]
            
            logger.info(f"Semantic search with spatial filter completed: {len(final_results)} results found (total: {total_results}, offset: {offset})")
            return {
                "results": final_results,
                "total_count": total_results,
                "limit": limit,
                "offset": offset,
                "has_more": end_idx < total_results
            }
            
        except Exception as e:
            logger.error(f"Semantic search with spatial filter failed: {e}")
            raise
    
    async def semantic_search_with_spatial_area(
        self, 
        query: str, 
        limit: int = 10, 
        score_threshold: float = 0.0,
        offset: int = 0,
        spatial_expansion_factor: float = 0.8
    ) -> Dict[str, Any]:
        """
        Perform semantic search and find spatial area around semantically similar documents.
        
        Args:
            query: Search query text
            limit: Maximum number of results per page
            score_threshold: Minimum similarity score threshold
            offset: Number of results to skip (for pagination)
            spatial_expansion_factor: Factor to expand spatial area around semantic results
            
        Returns:
            Dict containing results, spatial area info, and pagination info
        """
        try:
            logger.info(f"Performing semantic search with spatial area for query: '{query}'")
            
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(query)
            
            # Search using vector database
            search_results = await self.vector_db.search_similar(query_embedding, limit=limit + offset)
            
            if not search_results:
                logger.warning("No documents found in vector database")
                return {
                    "results": [],
                    "total_count": 0,
                    "limit": limit,
                    "offset": offset,
                    "has_more": False,
                    "spatial_area": None
                }
            
            # Process search results
            results = []
            for doc in search_results:
                if doc['similarity_score'] >= score_threshold:
                    results.append({
                        'document_id': doc['document_id'],
                        'title': doc['title'],
                        'coordinates': doc['coordinates'],
                        'similarity_score': float(doc['similarity_score']),
                        'content_preview': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                        'source': doc.get('source', 'unknown')
                    })
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Calculate spatial area around semantic results first
            spatial_area = None
            if results:
                # Get coordinates of top semantic results (first 20 for area calculation)
                top_coords = [doc['coordinates'] for doc in results[:20]]
                if top_coords:
                    coords_array = np.array(top_coords)
                    
                    # Calculate centroid
                    centroid_x = np.mean(coords_array[:, 0])
                    centroid_y = np.mean(coords_array[:, 1])
                    
                    # Calculate radius as max distance from centroid * expansion factor
                    distances = np.sqrt((coords_array[:, 0] - centroid_x)**2 + (coords_array[:, 1] - centroid_y)**2)
                    max_distance = np.max(distances)
                    radius = max_distance * spatial_expansion_factor
                    
                    # Ensure minimum radius
                    radius = max(radius, 1.0)
                    
                    spatial_area = {
                        'center': [float(centroid_x), float(centroid_y)],
                        'radius': float(radius)
                    }
                    
                    # Now find the 10 closest documents to the circle center
                    circle_center = [centroid_x, centroid_y]
                    for doc in results:
                        doc_coords = doc['coordinates']
                        distance_to_center = np.sqrt((doc_coords[0] - circle_center[0])**2 + (doc_coords[1] - circle_center[1])**2)
                        doc['distance_to_center'] = float(distance_to_center)
                    
                    # Sort by distance to center (closest first)
                    results.sort(key=lambda x: x['distance_to_center'])
            
            # Apply pagination
            total_results = len(results)
            start_idx = offset
            end_idx = min(offset + limit, total_results)
            paginated_results = results[start_idx:end_idx]
            
            logger.info(f"Semantic search with spatial area completed: {len(results)} results found (total: {total_results}, offset: {offset})")
            return {
                "results": paginated_results,
                "total_count": total_results,
                "limit": limit,
                "offset": offset,
                "has_more": end_idx < total_results,
                "spatial_area": spatial_area
            }
            
        except Exception as e:
            logger.error(f"Semantic search with spatial area failed: {e}")
            raise

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
