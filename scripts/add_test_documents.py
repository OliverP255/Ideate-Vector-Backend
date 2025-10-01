#!/usr/bin/env python3
"""
Script to add test documents to the knowledge database for testing search functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.knowledge_database import KnowledgeDatabase
from app.services.vector_db import VectorDatabaseService
from app.services.embedding import EmbeddingService
import asyncio
import json

async def add_test_documents():
    """Add test documents to the database."""
    
    # Initialize services
    db = KnowledgeDatabase()
    vector_db = VectorDatabaseService()
    embedding_service = EmbeddingService()
    
    # Test documents
    test_documents = [
        {
            "document_id": "test_ai_001",
            "title": "Introduction to Artificial Intelligence",
            "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. AI systems can learn, reason, and make decisions similar to humans. Machine learning is a subset of AI that focuses on algorithms that can learn from data.",
            "source": "test_data",
            "author": "Test Author",
            "year": 2024,
            "topic": "Artificial Intelligence",
            "coordinates": [1.0, 2.0]
        },
        {
            "document_id": "test_ml_002", 
            "title": "Machine Learning Fundamentals",
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
            "source": "test_data",
            "author": "Test Author",
            "year": 2024,
            "topic": "Machine Learning",
            "coordinates": [2.0, 3.0]
        },
        {
            "document_id": "test_nlp_003",
            "title": "Natural Language Processing",
            "content": "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. NLP techniques include text analysis, sentiment analysis, and language translation.",
            "source": "test_data", 
            "author": "Test Author",
            "year": 2024,
            "topic": "Natural Language Processing",
            "coordinates": [3.0, 4.0]
        },
        {
            "document_id": "test_cv_004",
            "title": "Computer Vision Basics",
            "content": "Computer vision is a field of artificial intelligence that trains computers to interpret and understand visual information from the world. It includes image recognition, object detection, and image segmentation techniques.",
            "source": "test_data",
            "author": "Test Author", 
            "year": 2024,
            "topic": "Computer Vision",
            "coordinates": [4.0, 5.0]
        },
        {
            "document_id": "test_ds_005",
            "title": "Data Science Overview",
            "content": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It combines statistics, machine learning, and domain expertise.",
            "source": "test_data",
            "author": "Test Author",
            "year": 2024,
            "topic": "Data Science", 
            "coordinates": [5.0, 6.0]
        }
    ]
    
    print("Adding test documents to the database...")
    
    for doc in test_documents:
        try:
            # Generate embedding for the document
            embedding_result = await embedding_service.generate_embedding(
                document_id=doc['document_id'],
                document_data={
                    'title': doc['title'],
                    'content': doc['content'],
                    'source': doc['source']
                }
            )
            embedding = embedding_result['embedding']
            
            # Add to knowledge database
            document_dict = {
                'id': doc['document_id'],
                'title': doc['title'],
                'content': doc['content'],
                'source': doc['source'],
                'quality_score': 0.8,
                'category': doc['topic'],
                'language': 'en',
                'coordinates': doc['coordinates'],
                'metadata': {
                    'author': doc['author'],
                    'year': doc['year'],
                    'topic': doc['topic']
                }
            }
            db.add_document(document_dict)
            
            # Add to vector database
            vector_doc = {
                'id': doc['document_id'],
                'embedding': embedding,
                'metadata': {
                    'title': doc['title'],
                    'content': doc['content'],
                    'source': doc['source'],
                    'coordinates': doc['coordinates'],
                    'author': doc['author'],
                    'year': doc['year'],
                    'topic': doc['topic']
                }
            }
            await vector_db.add_documents([vector_doc])
            
            print(f"✓ Added document: {doc['title']}")
            
        except Exception as e:
            print(f"✗ Failed to add document {doc['document_id']}: {e}")
    
    # Get database stats
    stats = db.get_stats()
    print(f"\nDatabase stats after adding test documents:")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Database size: {stats['database_size_mb']:.2f} MB")
    
    # Close database connections
    db.close()

if __name__ == "__main__":
    asyncio.run(add_test_documents())
