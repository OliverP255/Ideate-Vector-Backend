#!/usr/bin/env python3
"""
Show Documents Script
Display the documents that have been processed in the knowledge map.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.services.knowledge_database import KnowledgeDatabase

def show_documents():
    """Show all documents in the knowledge database."""
    db = KnowledgeDatabase()
    
    try:
        # Get database stats
        stats = db.get_stats()
        print(f"=== KNOWLEDGE MAP DATABASE ===")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Database Size: {stats['database_size_mb']:.2f} MB")
        print(f"Last Updated: {stats['last_updated']}")
        print()
        
        # Get documents from SQLite
        cursor = db.sqlite_conn.execute("""
            SELECT id, title, source, quality_score, category, language, created_at
            FROM documents 
            ORDER BY quality_score DESC, created_at DESC
        """)
        
        documents = cursor.fetchall()
        
        if not documents:
            print("No documents found in the database.")
            return
        
        print(f"=== DOCUMENTS ({len(documents)} total) ===")
        print()
        
        for i, doc in enumerate(documents, 1):
            doc_id, title, source, quality_score, category, language, created_at = doc
            
            # Quality indicator
            if quality_score >= 0.8:
                quality_indicator = "🟢 HIGH"
            elif quality_score >= 0.6:
                quality_indicator = "🟡 MEDIUM"
            else:
                quality_indicator = "🔴 LOW"
            
            print(f"{i:2d}. {title}")
            print(f"    Source: {source.upper()}")
            print(f"    Quality: {quality_indicator} ({quality_score:.2f})")
            print(f"    Category: {category}")
            print(f"    Language: {language}")
            print(f"    Added: {created_at}")
            print(f"    ID: {doc_id}")
            print()
        
        # Show source breakdown
        print("=== SOURCE BREAKDOWN ===")
        for source, count in stats['source_counts'].items():
            print(f"{source.upper()}: {count} documents")
        print()
        
        # Show quality distribution
        print("=== QUALITY DISTRIBUTION ===")
        for quality, count in stats['quality_distribution'].items():
            print(f"{quality}: {count} documents")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    show_documents()
