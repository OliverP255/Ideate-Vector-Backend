"""
Document mapping service for dimensionality reduction.
Simplified to work with UMAP coordinates from sample data.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.services.config import get_settings
from app.services.sample_data import SampleDataService

logger = logging.getLogger(__name__)


class MappingService:
    """Service for document mapping and 2D projection using UMAP coordinates."""
    
    def __init__(self):
        self.settings = get_settings()
        self.sample_data_service = SampleDataService()
    
    async def create_initial_mapping(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        Create initial 2D mapping for a set of documents using UMAP.
        
        Args:
            document_ids: List of document IDs to map
            
        Returns:
            Dict containing mapping results
        """
        try:
            logger.info(f"Creating initial UMAP mapping for {len(document_ids)} documents")
            
            # Load embeddings from database or files
            embeddings = []
            valid_document_ids = []
            
            # Try to get embeddings from database first
            try:
                from app.services.knowledge_database import KnowledgeDatabase
                db = KnowledgeDatabase()
                
                # Get embeddings from ChromaDB
                if db.chroma_collection is not None:
                    # Get documents from ChromaDB with embeddings explicitly included
                    results = db.chroma_collection.get(ids=document_ids, include=['embeddings'])
                    logger.info(f"ChromaDB query returned {len(results.get('ids', []))} documents")
                    logger.info(f"ChromaDB embeddings type: {type(results.get('embeddings'))}")
                    
                    if results and results.get('embeddings') is not None:
                        embeddings_list = results['embeddings']
                        logger.info(f"Found {len(embeddings_list)} embeddings in ChromaDB results")
                        
                        for i, doc_id in enumerate(document_ids):
                            if i < len(embeddings_list) and embeddings_list[i] is not None:
                                # Convert numpy array to list for consistency
                                embedding = embeddings_list[i]
                                if hasattr(embedding, 'tolist'):
                                    embedding = embedding.tolist()
                                embeddings.append(embedding)
                                valid_document_ids.append(doc_id)
                        
                        logger.info(f"Loaded {len(embeddings)} embeddings from database")
                    else:
                        logger.warning("No embeddings found in ChromaDB results")
                
            except Exception as e:
                logger.warning(f"Failed to load embeddings from database: {e}")
            
            # Fallback to file-based embeddings if database method failed
            if len(embeddings) < 2:
                logger.info("Falling back to file-based embedding loading")
                embeddings = []
                valid_document_ids = []
                
                for doc_id in document_ids:
                    embedding_file = Path("data/embeddings") / f"{doc_id}.json"
                    if not embedding_file.exists():
                        embedding_file = Path("backend/data/embeddings") / f"{doc_id}.json"
                    
                    if embedding_file.exists():
                        with open(embedding_file, 'r') as f:
                            embedding_data = json.load(f)
                        
                        if 'embedding' in embedding_data:
                            embeddings.append(embedding_data['embedding'])
                            valid_document_ids.append(doc_id)
            
            if len(embeddings) < 2:
                raise ValueError("Need at least 2 documents with embeddings to create mapping")
            
            embeddings_array = np.array(embeddings)
            
            # Apply UMAP directly
            import umap
            reducer = umap.UMAP(
                n_neighbors=min(5, len(embeddings) - 1),
                min_dist=0.1,
                n_components=2,
                metric='cosine',
                random_state=42
            )
            
            coordinates_2d = reducer.fit_transform(embeddings_array)
            
            # Create mapping data
            mapping_data = {
                'document_ids': valid_document_ids,
                'coordinates_2d': coordinates_2d.tolist(),
                'mapping_metadata': {
                    'method': 'umap_direct',
                    'umap_params': {
                        'n_neighbors': reducer.n_neighbors,
                        'min_dist': reducer.min_dist,
                        'n_components': reducer.n_components,
                        'metric': reducer.metric
                    },
                    'random_seed': 42,
                    'created_at': datetime.now().isoformat()
                }
            }
            
            logger.info(f"Initial UMAP mapping created for {len(valid_document_ids)} documents")
            return mapping_data
            
        except Exception as e:
            logger.error(f"Failed to create initial mapping: {e}")
            raise
    
    async def project_new_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        Project new documents to 2D using UMAP.
        For simplicity, this just creates a new UMAP mapping with all documents.
        
        Args:
            document_ids: List of new document IDs to project
            
        Returns:
            Dict containing projection results
        """
        try:
            logger.info(f"Projecting {len(document_ids)} new documents using UMAP")
            
            # For simplicity, we'll just create a new mapping with all documents
            # In a production system, you might want to implement incremental UMAP
            return await self.create_initial_mapping(document_ids)
            
        except Exception as e:
            logger.error(f"Failed to project new documents: {e}")
            raise
    
    async def get_document_coordinates(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get 2D coordinates for a specific document from sample data.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dict containing coordinates and metadata
        """
        try:
            # Get all coordinates and find the specific document
            all_coords = await self.get_all_coordinates()
            
            for coord in all_coords['coordinates']:
                if coord['document_id'] == document_id:
                    return {
                        'document_id': document_id,
                        'coordinates': coord['coordinates'],
                        'source': 'sample_data',
                        'metadata': {
                            'title': coord.get('title', ''),
                            'discipline': coord.get('discipline', ''),
                            'retrieved_at': all_coords['retrieved_at']
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get coordinates for document {document_id}: {e}")
            raise
    
    async def get_all_coordinates(self) -> Dict[str, Any]:
        """
        Get all document coordinates from REAL database.
        
        Returns:
            Dict containing all coordinates and metadata
        """
        try:
            # Import KnowledgeDatabase to get real coordinates
            from app.services.knowledge_database import KnowledgeDatabase
            
            # Get real coordinates from database
            db = KnowledgeDatabase()
            all_coords = db.get_all_coordinates()
            
            if all_coords:
                logger.info(f"Using {len(all_coords)} REAL coordinates from database")
                
                # Convert to the format expected by frontend
                coordinates = []
                for coord_data in all_coords:
                    coordinates.append({
                        'document_id': coord_data['document_id'],
                        'coordinates': coord_data['coordinates'],
                        'title': coord_data['title'],
                        'categories': [coord_data.get('category', 'general')],
                        'source': coord_data['source']
                    })
                
                return {
                    'coordinates': coordinates,
                    'total_documents': len(coordinates),
                    'retrieved_at': datetime.now().isoformat(),
                    'source': 'real_database'
                }
            
            # Fallback to sample data if no real coordinates
            logger.warning("No real coordinates found, falling back to sample data")
            sample_coordinates = self.sample_data_service.load_sample_coordinates()
            if sample_coordinates:
                logger.info(f"Using {len(sample_coordinates)} UMAP coordinates from sample data")
                return {
                    'coordinates': sample_coordinates,
                    'total_documents': len(sample_coordinates),
                    'retrieved_at': datetime.now().isoformat(),
                    'source': 'sample_data_umap'
                }
            
            # Final fallback: return empty result
            logger.warning("No coordinates found anywhere")
            return {
                'coordinates': [],
                'total_documents': 0,
                'retrieved_at': datetime.now().isoformat(),
                'source': 'none'
            }
            
        except Exception as e:
            logger.error(f"Failed to get all coordinates: {e}")
            return {
                'coordinates': [],
                'total_documents': 0,
                'retrieved_at': datetime.now().isoformat(),
                'source': 'error'
            }
    
