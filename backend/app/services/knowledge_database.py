"""
High-performance knowledge database for 1M+ documents.
Combines SQLite for metadata and ChromaDB for vector search.
"""

import sqlite3
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import chromadb
from chromadb.config import Settings
import hashlib

logger = logging.getLogger(__name__)


class KnowledgeDatabase:
    """High-performance database for knowledge map with 1M+ documents."""
    
    def __init__(self, db_path: str = "../scripts/data/knowledge_map"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # SQLite for metadata
        self.sqlite_path = self.db_path / "metadata.db"
        
        # ChromaDB for vector search
        self.chroma_path = self.db_path / "chroma_db"
        
        # Initialize databases
        self._init_sqlite()
        self._init_chromadb()
        
        logger.info(f"Knowledge database initialized at {self.db_path}")
    
    def _init_sqlite(self):
        """Initialize SQLite database for metadata."""
        self.sqlite_conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self.sqlite_conn.execute("PRAGMA journal_mode=WAL")  # Better performance
        self.sqlite_conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        
        # Create tables
        self.sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                quality_score REAL,
                category TEXT,
                language TEXT DEFAULT 'en',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                document_id TEXT PRIMARY KEY,
                embedding BLOB,
                embedding_dimension INTEGER,
                model_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """)
        
        self.sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS coordinates (
                document_id TEXT PRIMARY KEY,
                x REAL NOT NULL,
                y REAL NOT NULL,
                umap_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """)
        
        # Create indexes for performance
        self.sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON documents (source)")
        self.sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_quality ON documents (quality_score)")
        self.sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON documents (category)")
        
        self.sqlite_conn.commit()
        logger.info("SQLite database initialized")
    
    def _init_chromadb(self):
        """Initialize ChromaDB for vector search."""
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        try:
            self.chroma_collection = self.chroma_client.get_collection("documents")
            logger.info("Loaded existing ChromaDB collection")
        except:
            self.chroma_collection = self.chroma_client.create_collection(
                name="documents",
                metadata={"description": "Knowledge map embeddings"}
            )
            logger.info("Created new ChromaDB collection")
    
    def add_document(self, document: Dict[str, Any]) -> str:
        """Add a document to the database."""
        try:
            doc_id = document['id']
            
            # Add to SQLite
            self.sqlite_conn.execute("""
                INSERT OR REPLACE INTO documents 
                (id, title, content, source, quality_score, category, language)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                document['title'],
                document['content'],
                document['source'],
                document.get('quality_score', 0.0),
                document.get('category', ''),
                document.get('language', 'en')
            ))
            
            self.sqlite_conn.commit()
            
            # Add embedding to ChromaDB if available
            if 'embedding' in document:
                embedding = document['embedding']
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                
                self.chroma_collection.add(
                    ids=[doc_id],
                    embeddings=[embedding.tolist()],
                    metadatas=[{
                        'title': document['title'],
                        'source': document['source'],
                        'quality_score': document.get('quality_score', 0.0),
                        'category': document.get('category', '')
                    }],
                    documents=[document['content'][:1000]]  # First 1000 chars for search
                )
            
            logger.debug(f"Added document {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            raise
    
    def batch_add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add multiple documents efficiently."""
        try:
            if not documents:
                logger.warning("No documents to add")
                return []
                
            logger.info(f"Processing {len(documents)} documents for batch insertion")
            
            doc_ids = []
            embeddings_data = []
            metadatas = []
            ids = []
            documents_text = []
            
            # Prepare SQLite batch
            sqlite_data = []
            for i, doc in enumerate(documents):
                try:
                    # Debug: Log document structure
                    logger.debug(f"Processing document {i}: {type(doc)}")
                    
                    if doc is None:
                        logger.warning(f"Document {i} is None, skipping")
                        continue
                        
                    if not isinstance(doc, dict):
                        logger.warning(f"Document {i} is not a dict: {type(doc)}, skipping")
                        continue
                    
                    # Check for required 'id' field
                    if 'id' not in doc:
                        logger.warning(f"Document {i} missing 'id' field: {list(doc.keys()) if doc else 'None'}, skipping")
                        continue
                    
                    doc_id = doc['id']
                    if doc_id is None:
                        logger.warning(f"Document {i} has None id, skipping")
                        continue
                        
                    doc_ids.append(doc_id)
                    
                    # Extract fields with safe defaults
                    title = doc.get('title', '') or ''
                    content = doc.get('content', '') or ''
                    source = doc.get('source', '') or 'unknown'
                    quality_score = doc.get('quality_score', 0.0)
                    category = doc.get('category', '') or 'general'
                    language = doc.get('language', 'en') or 'en'
                    
                    # Ensure quality_score is a number
                    try:
                        quality_score = float(quality_score)
                    except (ValueError, TypeError):
                        quality_score = 0.0
                    
                    sqlite_data.append((
                        doc_id,
                        title,
                        content,
                        source,
                        quality_score,
                        category,
                        language
                    ))
                    
                    # Prepare ChromaDB data
                    if 'embedding' in doc and doc['embedding'] is not None:
                        embedding = doc['embedding']
                        
                        # Convert embedding to list if needed
                        if isinstance(embedding, list):
                            embedding = np.array(embedding)
                        elif hasattr(embedding, 'tolist'):
                            embedding = embedding.tolist()
                        
                        if embedding is not None:
                            ids.append(doc_id)
                            embeddings_data.append(embedding.tolist() if hasattr(embedding, 'tolist') else embedding)
                            metadatas.append({
                                'title': title,
                                'source': source,
                                'quality_score': quality_score,
                                'category': category
                            })
                            documents_text.append(content[:1000] if content else title[:1000])
                        else:
                            logger.warning(f"Document {doc_id} has None embedding after processing")
                    else:
                        logger.warning(f"Document {doc_id} missing embedding")
                        
                except Exception as e:
                    logger.error(f"Error processing document {i}: {e}")
                    logger.error(f"Document content: {doc}")
                    continue
            
            logger.info(f"Prepared {len(sqlite_data)} documents for SQLite, {len(embeddings_data)} for ChromaDB")
            
            # Batch insert to SQLite
            if sqlite_data:
                try:
                    self.sqlite_conn.executemany("""
                        INSERT OR REPLACE INTO documents 
                        (id, title, content, source, quality_score, category, language)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, sqlite_data)
                    
                    self.sqlite_conn.commit()
                    logger.info(f"Successfully added {len(sqlite_data)} documents to SQLite")
                except Exception as e:
                    logger.error(f"Failed to insert to SQLite: {e}")
                    raise
            else:
                logger.warning("No valid documents for SQLite insertion")
            
            # Batch insert to ChromaDB
            if embeddings_data and len(embeddings_data) > 0:
                if self.chroma_collection is not None:
                    try:
                        self.chroma_collection.add(
                            ids=ids,
                            embeddings=embeddings_data,
                            metadatas=metadatas,
                            documents=documents_text
                        )
                        logger.info(f"Successfully added {len(embeddings_data)} embeddings to ChromaDB")
                    except Exception as e:
                        logger.error(f"Failed to add embeddings to ChromaDB: {e}")
                        # Don't raise - ChromaDB failure shouldn't stop SQLite
                else:
                    logger.warning("ChromaDB collection is None, skipping vector storage")
            else:
                logger.warning("No valid embeddings for ChromaDB insertion")
            
            logger.info(f"Batch processing complete: {len(sqlite_data)} documents processed")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to batch add documents: {e}")
            logger.error(f"Documents type: {type(documents)}")
            logger.error(f"Documents length: {len(documents) if documents else 'None'}")
            if documents:
                logger.error(f"First document type: {type(documents[0])}")
                logger.error(f"First document keys: {list(documents[0].keys()) if isinstance(documents[0], dict) else 'Not a dict'}")
            raise
    
    def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        try:
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=['metadatas', 'documents', 'distances']
            )
            
            similar_docs = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    similar_docs.append({
                        'document_id': doc_id,
                        'title': results['metadatas'][0][i]['title'],
                        'source': results['metadatas'][0][i]['source'],
                        'quality_score': results['metadatas'][0][i]['quality_score'],
                        'category': results['metadatas'][0][i]['category'],
                        'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'content_preview': results['documents'][0][i]
                    })
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        try:
            cursor = self.sqlite_conn.execute("""
                SELECT * FROM documents WHERE id = ?
            """, (doc_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'source': row[3],
                    'quality_score': row[4],
                    'category': row[5],
                    'language': row[6],
                    'created_at': row[7],
                    'updated_at': row[8]
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    def add_coordinates(self, document_id: str, x: float, y: float, umap_model: str = "default") -> bool:
        """Add UMAP coordinates for a document."""
        try:
            self.sqlite_conn.execute("""
                INSERT OR REPLACE INTO coordinates (document_id, x, y, umap_model)
                VALUES (?, ?, ?, ?)
            """, (document_id, x, y, umap_model))
            
            self.sqlite_conn.commit()
            logger.debug(f"Added coordinates for document {document_id}: ({x:.3f}, {y:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add coordinates for {document_id}: {e}")
            return False
    
    def get_coordinates(self, document_id: str) -> Optional[Dict[str, float]]:
        """Get UMAP coordinates for a document."""
        try:
            cursor = self.sqlite_conn.execute("""
                SELECT x, y FROM coordinates WHERE document_id = ?
            """, (document_id,))
            
            row = cursor.fetchone()
            if row:
                return {'x': row[0], 'y': row[1]}
            return None
            
        except Exception as e:
            logger.error(f"Failed to get coordinates for {document_id}: {e}")
            return None
    
    def get_all_coordinates(self) -> List[Dict[str, Any]]:
        """Get all documents with their coordinates."""
        try:
            cursor = self.sqlite_conn.execute("""
                SELECT d.id, d.title, d.source, d.category, c.x, c.y
                FROM documents d
                LEFT JOIN coordinates c ON d.id = c.document_id
                WHERE c.x IS NOT NULL AND c.y IS NOT NULL
            """)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'document_id': row[0],
                    'title': row[1],
                    'source': row[2],
                    'category': row[3],
                    'coordinates': [row[4], row[5]]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get all coordinates: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            # Document count by source
            cursor = self.sqlite_conn.execute("""
                SELECT source, COUNT(*) as count 
                FROM documents 
                GROUP BY source 
                ORDER BY count DESC
            """)
            source_counts = dict(cursor.fetchall())
            
            # Total documents
            cursor = self.sqlite_conn.execute("SELECT COUNT(*) FROM documents")
            total_documents = cursor.fetchone()[0]
            
            # Mapped documents count
            cursor = self.sqlite_conn.execute("SELECT COUNT(*) FROM coordinates")
            mapped_documents = cursor.fetchone()[0]
            
            # Quality distribution
            cursor = self.sqlite_conn.execute("""
                SELECT 
                    CASE 
                        WHEN quality_score >= 0.8 THEN 'High'
                        WHEN quality_score >= 0.6 THEN 'Medium'
                        ELSE 'Low'
                    END as quality_level,
                    COUNT(*) as count
                FROM documents 
                GROUP BY quality_level
            """)
            quality_distribution = dict(cursor.fetchall())
            
            return {
                'total_documents': total_documents,
                'mapped_documents': mapped_documents,
                'source_counts': source_counts,
                'quality_distribution': quality_distribution,
                'database_size_mb': self._get_database_size(),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def _get_database_size(self) -> float:
        """Get database size in MB."""
        try:
            sqlite_size = self.sqlite_path.stat().st_size / (1024 * 1024)
            chroma_size = sum(f.stat().st_size for f in self.chroma_path.rglob('*') if f.is_file()) / (1024 * 1024)
            return sqlite_size + chroma_size
        except:
            return 0.0
    
    def close(self):
        """Close database connections."""
        if hasattr(self, 'sqlite_conn'):
            self.sqlite_conn.close()
        logger.info("Database connections closed")


def main():
    """Test the knowledge database."""
    db = KnowledgeDatabase()
    
    # Test document
    test_doc = {
        'id': 'test_001',
        'title': 'Test Document',
        'content': 'This is a test document for the knowledge database.',
        'source': 'test',
        'quality_score': 0.9,
        'category': 'test',
        'embedding': np.random.rand(384).tolist()
    }
    
    # Add document
    doc_id = db.add_document(test_doc)
    print(f"Added document: {doc_id}")
    
    # Get stats
    stats = db.get_stats()
    print(f"Database stats: {stats}")
    
    # Search
    results = db.search_similar(test_doc['embedding'], limit=5)
    print(f"Search results: {results}")
    
    db.close()


if __name__ == "__main__":
    main()
