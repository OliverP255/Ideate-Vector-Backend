"""
Service for loading and managing sample data.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SampleDataService:
    """Service for managing sample data."""
    
    def __init__(self):
        self.data_dir = Path("data/sample_docs")
        self.coordinates_file = self.data_dir / "coordinates.json"
        self.embeddings_file = Path("data/embeddings/all_real_embeddings.json")
    
    def load_sample_coordinates(self) -> List[Dict[str, Any]]:
        """Load sample document coordinates."""
        try:
            if not self.coordinates_file.exists():
                logger.warning("Sample coordinates file not found, using default data")
                return self._get_default_coordinates()
            
            with open(self.coordinates_file, 'r') as f:
                coordinates = json.load(f)
            
            logger.info(f"Loaded {len(coordinates)} sample coordinates")
            return coordinates
            
        except Exception as e:
            logger.error(f"Failed to load sample coordinates: {e}")
            return self._get_default_coordinates()
    
    def _get_default_coordinates(self) -> List[Dict[str, Any]]:
        """Get default coordinates if sample data is not available."""
        return [
            {"document_id": "test-doc-id", "coordinates": [1.9094771146774292, -12.717576026916504], "title": "Test Document 1", "author": "Test Author", "year": 2024, "topic": "Machine Learning"},
            {"document_id": "test-doc-2", "coordinates": [2.852782726287842, -13.04093074798584], "title": "Test Document 2", "author": "Test Author", "year": 2024, "topic": "Deep Learning"},
            {"document_id": "test-doc-3", "coordinates": [3.265191078186035, -12.215641021728516], "title": "Test Document 3", "author": "Test Author", "year": 2024, "topic": "NLP"},
            {"document_id": "test-doc-4", "coordinates": [2.5669329166412354, -11.660048484802246], "title": "Test Document 4", "author": "Test Author", "year": 2024, "topic": "Computer Vision"},
            {"document_id": "test-doc-5", "coordinates": [1.4680349826812744, -11.844719886779785], "title": "Test Document 5", "author": "Test Author", "year": 2024, "topic": "Data Science"}
        ]
    
    def get_document_by_id(self, document_id: str) -> Dict[str, Any]:
        """Get a specific document by ID."""
        try:
            doc_file = self.data_dir / f"{document_id}.json"
            if not doc_file.exists():
                return None
            
            with open(doc_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load document {document_id}: {e}")
            return None
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all sample documents."""
        documents = []
        
        try:
            for doc_file in self.data_dir.glob("doc_*.json"):
                with open(doc_file, 'r') as f:
                    doc = json.load(f)
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} sample documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load sample documents: {e}")
            return []
    
    def load_sample_embeddings(self) -> List[Dict[str, Any]]:
        """Load sample document embeddings."""
        try:
            if not self.embeddings_file.exists():
                logger.warning("Sample embeddings file not found")
                return []
            
            with open(self.embeddings_file, 'r') as f:
                embeddings = json.load(f)
            
            logger.info(f"Loaded {len(embeddings)} sample embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to load sample embeddings: {e}")
            return []
