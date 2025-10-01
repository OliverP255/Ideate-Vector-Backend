#!/usr/bin/env python3
"""
Test script to verify the mapping service fix.
This will attempt to map some unmapped papers using the corrected collection name.
"""

import sys
import os
import asyncio
import sqlite3

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.services.knowledge_database import KnowledgeDatabase
from app.services.mapping import MappingService

async def test_mapping_fix():
    """Test if the mapping service can now access embeddings properly."""
    
    print("🔧 Testing Mapping Service Fix...")
    print("=" * 50)
    
    # Initialize database
    db = KnowledgeDatabase()
    
    # Get some unmapped papers
    conn = sqlite3.connect('data/knowledge_map/metadata.db')
    cursor = conn.cursor()
    
    # Get unmapped papers (papers without coordinates)
    cursor.execute("""
        SELECT d.id, d.title 
        FROM documents d 
        LEFT JOIN coordinates c ON d.id = c.document_id 
        WHERE c.x IS NULL OR c.y IS NULL 
        LIMIT 10
    """)
    unmapped_papers = cursor.fetchall()
    
    print(f"📊 Found {len(unmapped_papers)} unmapped papers")
    
    if len(unmapped_papers) < 2:
        print("❌ Need at least 2 unmapped papers to test mapping")
        conn.close()
        return
    
    # Get the document IDs
    doc_ids = [paper[0] for paper in unmapped_papers[:5]]  # Test with 5 papers
    print(f"🎯 Testing mapping for {len(doc_ids)} papers:")
    for doc_id in doc_ids:
        print(f"  - {doc_id}")
    
    # Test ChromaDB access
    print(f"\n🔍 Testing ChromaDB access...")
    try:
        if db.chroma_collection is not None:
            results = db.chroma_collection.get(ids=doc_ids, include=['embeddings'])
            print(f"✅ ChromaDB query successful: {len(results.get('ids', []))} documents found")
            
            if results.get('embeddings') is not None:
                embeddings_count = len([e for e in results['embeddings'] if e is not None])
                print(f"✅ Found {embeddings_count} valid embeddings")
            else:
                print("❌ No embeddings found in results")
        else:
            print("❌ ChromaDB collection is None")
    except Exception as e:
        print(f"❌ ChromaDB access failed: {e}")
        conn.close()
        return
    
    # Test mapping service
    print(f"\n🗺️  Testing MappingService...")
    try:
        mapping_service = MappingService()
        mapping_data = await mapping_service.create_initial_mapping(doc_ids)
        
        print(f"✅ Mapping service successful!")
        print(f"   - Mapped {len(mapping_data['document_ids'])} documents")
        print(f"   - Method: {mapping_data['mapping_metadata']['method']}")
        
        # Add coordinates to database
        print(f"\n💾 Adding coordinates to database...")
        for i, doc_id in enumerate(mapping_data['document_ids']):
            coords = mapping_data['coordinates_2d'][i]
            db.add_coordinates(doc_id, coords[0], coords[1])
        
        print(f"✅ Successfully added coordinates for {len(mapping_data['document_ids'])} papers")
        
    except Exception as e:
        print(f"❌ Mapping service failed: {e}")
    
    conn.close()

if __name__ == "__main__":
    asyncio.run(test_mapping_fix())
