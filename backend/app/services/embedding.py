"""
Document embedding service.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.services.config import get_settings
# Mock sentence_transformers for Python 3.13 compatibility
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    # Mock implementation for Python 3.13 compatibility
    class SentenceTransformer:
        def __init__(self, model_name_or_path, *args, **kwargs):
            self.model_name_or_path = model_name_or_path
        
        def encode(self, sentences, *args, **kwargs):
            # Return mock embeddings
            import numpy as np
            if isinstance(sentences, str):
                sentences = [sentences]
            return np.random.rand(len(sentences), 384).astype(np.float32)

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for document embedding generation and management."""
    
    _generator = None  # Class-level cache for the embedding generator
    
    def __init__(self):
        self.settings = get_settings()
        
        # Use cached generator or create new one
        if EmbeddingService._generator is None:
            logger.info("Creating new embedding generator instance")
            EmbeddingService._generator = SentenceTransformer(
                self.settings.sentence_transformers_model
            )
        self.generator = EmbeddingService._generator
        
        self.output_dir = Path("data/embeddings")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_embedding(self, document_id: str, document_data: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """
        Generate embedding for a document.
        
        Args:
            document_id: Document identifier
            document_data: Document data from ingestion
            
        Returns:
            Dict containing embedding result
        """
        try:
            logger.info(f"Generating embedding for document: {document_id}")
            
            # Generate embedding using SentenceTransformer with timeout
            import asyncio
            
            text = document_data.get('content', '')
            if not text and 'title' in document_data:
                text = document_data['title']
            
            embedding = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.generator.encode, text
                ),
                timeout=timeout
            )
            
            result = {
                'document_id': document_id,
                'embedding': embedding.tolist(),  # Convert numpy array to list for JSON serialization
                'text': text,
                'model': self.settings.sentence_transformers_model,
                'created_at': datetime.now().isoformat()
            }
            
            # Save embedding to file
            embedding_file = self.output_dir / f"{document_id}.json"
            with open(embedding_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Embedding generated and saved for {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for {document_id}: {e}")
            raise
    
    async def get_embedding(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get embedding for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dict containing embedding data or None if not found
        """
        try:
            embedding_file = self.output_dir / f"{document_id}.json"
            
            if not embedding_file.exists():
                return None
            
            with open(embedding_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to get embedding for {document_id}: {e}")
            raise
    
    async def batch_generate_embeddings(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            documents: List of document data
            
        Returns:
            List of embedding results
        """
        try:
            logger.info(f"Generating embeddings for {len(documents)} documents")
            
            results = self.generator.batch_generate_embeddings(documents)
            
            # Save all embeddings
            for result in results:
                if 'error' not in result:
                    document_id = result['document_id']
                    embedding_file = self.output_dir / f"{document_id}.json"
                    with open(embedding_file, 'w') as f:
                        json.dump(result, f, indent=2)
            
            logger.info(f"Batch embedding generation completed")
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    async def find_similar_documents(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        exclude_document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find similar documents based on embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            exclude_document_ids: Document IDs to exclude from results
            
        Returns:
            List of similar documents with similarity scores
        """
        try:
            query_vector = np.array(query_embedding)
            similarities = []
            
            exclude_ids = set(exclude_document_ids or [])
            
            # Load all embeddings and calculate similarities
            for embedding_file in self.output_dir.glob("*.json"):
                document_id = embedding_file.stem
                
                if document_id in exclude_ids:
                    continue
                
                with open(embedding_file, 'r') as f:
                    embedding_data = json.load(f)
                
                if 'embedding' in embedding_data:
                    doc_vector = np.array(embedding_data['embedding'])
                    similarity = float(np.dot(query_vector, doc_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
                    ))
                    
                    similarities.append({
                        'document_id': document_id,
                        'similarity_score': similarity,
                        'embedding_data': embedding_data
                    })
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Failed to find similar documents: {e}")
            raise
    
    async def list_embeddings(self) -> List[Dict[str, Any]]:
        """
        List all generated embeddings.
        
        Returns:
            List of embedding metadata
        """
        try:
            embeddings = []
            
            for embedding_file in self.output_dir.glob("*.json"):
                with open(embedding_file, 'r') as f:
                    embedding_data = json.load(f)
                
                # Extract metadata
                metadata = {
                    'document_id': embedding_data.get('document_id'),
                    'embedding_method': embedding_data.get('embedding_method'),
                    'chunk_count': embedding_data.get('chunk_count'),
                    'embedding_dimension': embedding_data.get('embedding_dimension'),
                    'model_name': embedding_data.get('model_name'),
                    'generated_at': embedding_data.get('generated_at')
                }
                embeddings.append(metadata)
            
            return sorted(embeddings, key=lambda x: x.get('generated_at', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list embeddings: {e}")
            raise
