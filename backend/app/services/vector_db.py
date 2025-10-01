"""
Vector database service for document indexing and retrieval using ChromaDB.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import chromadb
from chromadb.config import Settings

from app.services.config import get_settings

logger = logging.getLogger(__name__)


class VectorDatabaseService:
    """Service for vector database operations using ChromaDB."""
    
    def __init__(self):
        self.settings = get_settings()
        self.collection_name = "documents"
        self.vector_size = 384  # all-MiniLM-L12-v2 dimension
        
        # Initialize ChromaDB client
        try:
            self.chroma_client = chromadb.PersistentClient(
                path="data/chroma_db"
            )
            self.chroma_available = True
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Connected to ChromaDB at data/chroma_db")
        except Exception as e:
            logger.warning(f"ChromaDB not available, using fallback mode: {e}")
            self.chroma_client = None
            self.chroma_available = False
            self.collection = None
    
    async def create_collection(self) -> bool:
        """
        Create the documents collection in ChromaDB.
        
        Returns:
            bool: True if collection was created successfully
        """
        try:
            if not self.chroma_available:
                logger.warning("ChromaDB not available, cannot create collection")
                return False
            
            logger.info(f"Collection {self.collection_name} is ready")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of documents with embeddings and metadata
            
        Returns:
            bool: True if documents were added successfully
        """
        try:
            if not self.chroma_available:
                logger.warning("ChromaDB not available, cannot add documents")
                return False
            
            if not documents:
                return True
            
            # Prepare data for ChromaDB
            ids = [doc.get("document_id", f"doc_{i}") for i, doc in enumerate(documents)]
            embeddings = [doc.get("embedding", [0.0] * self.vector_size) for doc in documents]
            metadatas = []
            texts = []
            
            for doc in documents:
                # Get metadata from the document or from the metadata field
                doc_metadata = doc.get("metadata", {})
                metadata = {
                    "title": doc_metadata.get("title", doc.get("title", "")),
                    "source": doc_metadata.get("source", doc.get("source", "")),
                    "coordinates": json.dumps(doc_metadata.get("coordinates", doc.get("coordinates", [0, 0]))),
                    "created_at": doc_metadata.get("created_at", doc.get("created_at", datetime.now().isoformat()))
                }
                metadatas.append(metadata)
                # Get content from metadata or document
                content = doc_metadata.get("content", doc.get("content", ""))
                if not content:
                    content = doc_metadata.get("title", doc.get("title", ""))
                texts.append(content)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
            logger.info(f"Added {len(documents)} documents to collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    async def search_similar(self, query_embedding: List[float], limit: int = 10, 
                           filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            limit: Maximum number of results
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar documents with scores
        """
        try:
            if not self.chroma_available:
                logger.warning("ChromaDB not available, returning empty results")
                return []
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=filter_metadata
            )
            
            # Convert results to expected format
            documents = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    doc = {
                        "document_id": doc_id,
                        "title": results['metadatas'][0][i].get("title", ""),
                        "content": results['documents'][0][i] if results['documents'] and results['documents'][0] else "",
                        "coordinates": json.loads(results['metadatas'][0][i].get("coordinates", "[0,0]")),
                        "similarity_score": 1.0 - results['distances'][0][i],  # Convert distance to similarity
                        "source": results['metadatas'][0][i].get("source", "")
                    }
                    documents.append(doc)
            
            logger.info(f"Found {len(documents)} similar documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            return []
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document data or None if not found
        """
        try:
            if not self.chroma_available:
                logger.warning("ChromaDB not available")
                return None
            
            results = self.collection.get(ids=[document_id])
            
            if not results['ids']:
                return None
            
            doc = {
                "document_id": document_id,
                "title": results['metadatas'][0].get("title", ""),
                "content": results['documents'][0] if results['documents'] else "",
                "coordinates": json.loads(results['metadatas'][0].get("coordinates", "[0,0]")),
                "source": results['metadatas'][0].get("source", "")
            }
            
            return doc
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector database.
        
        Args:
            document_id: Document identifier
            
        Returns:
            bool: True if document was deleted successfully
        """
        try:
            if not self.chroma_available:
                logger.warning("ChromaDB not available")
                return False
            
            self.collection.delete(ids=[document_id])
            logger.info(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            if not self.chroma_available:
                return {"status": "unavailable", "count": 0}
            
            count = self.collection.count()
            return {
                "status": "healthy",
                "count": count,
                "collection_name": self.collection_name,
                "vector_size": self.vector_size
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"status": "error", "count": 0, "error": str(e)}
    
    async def batch_upsert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Batch upsert documents to the vector database.
        
        Args:
            documents: List of documents to upsert
            
        Returns:
            bool: True if documents were upserted successfully
        """
        try:
            if not self.chroma_available:
                logger.warning("ChromaDB not available, cannot upsert documents")
                return False
            
            if not documents:
                return True
            
            # Prepare data for ChromaDB
            ids = [doc.get("document_id", f"doc_{i}") for i, doc in enumerate(documents)]
            embeddings = [doc.get("embedding", [0.0] * self.vector_size) for doc in documents]
            metadatas = []
            texts = []
            
            for doc in documents:
                metadata = {
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "coordinates": json.dumps(doc.get("coordinates", [0, 0])),
                    "created_at": doc.get("created_at", datetime.now().isoformat())
                }
                metadatas.append(metadata)
                texts.append(doc.get("content", doc.get("title", "")))
            
            # Upsert to collection
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
            logger.info(f"Upserted {len(documents)} documents to collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
            return False