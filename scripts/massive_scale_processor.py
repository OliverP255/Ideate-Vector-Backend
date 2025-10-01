#!/usr/bin/env python3
"""
Massive Scale Knowledge Processor
Optimized for processing 1M documents with efficient batching, quality control, and resume capability.
"""

import os
import sys
import json
import time
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing as mp
import pickle
import hashlib

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sentence_transformers import SentenceTransformer
from backend.app.services.knowledge_database import KnowledgeDatabase
from scripts.breadth_focused_processor import BreadthFocusedProcessor
from scripts.enhanced_arxiv_processor import EnhancedArxivProcessor

# Configure logging
# Create logs directory first
logs_dir = Path('../data/logs')
logs_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'massive_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MassiveScaleProcessor:
    """Optimized processor for scaling to 1M documents."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.db = KnowledgeDatabase()
        
        # Initialize embedding model (shared across processes)
        self.embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
        
        # Processing targets for 1M documents
        self.targets = {
            'wikipedia_breadth': 500_000,  # 50% - Breadth-focused Wikipedia
            'arxiv': 200_000,              # 20% - Research papers
            'pubmed': 100_000,             # 10% - Medical papers
            'books': 50_000,               # 5% - Project Gutenberg books
            'government': 50_000,          # 5% - Government documents
            'wikipedia_featured': 100_000  # 10% - Featured Wikipedia articles
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'wikipedia_breadth': 0.4,  # Lower threshold for breadth
            'arxiv': 0.6,              # Higher threshold for research
            'pubmed': 0.7,             # Very high for medical
            'books': 0.5,              # Medium for literature
            'government': 0.6,         # High for official docs
            'wikipedia_featured': 0.8  # Very high for featured
        }
        
        # Batch sizes for efficient processing
        self.batch_sizes = {
            'wikipedia_breadth': 1000,
            'arxiv': 500,
            'pubmed': 300,
            'books': 200,
            'government': 300,
            'wikipedia_featured': 1000
        }
        
        # Progress tracking
        self.progress_file = Path("../data/processing_progress.json")
        self.duplicate_cache = set()
        
        # Create necessary directories
        os.makedirs("../data/logs", exist_ok=True)
        os.makedirs("../data/checkpoints", exist_ok=True)
        
        logger.info(f"Massive Scale Processor initialized - Target: 1,000,000 documents")
        logger.info(f"Max workers: {self.max_workers}")
    
    def load_progress(self) -> Dict[str, Any]:
        """Load processing progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load progress file: {e}")
        return {
            'completed': {source: 0 for source in self.targets.keys()},
            'last_updated': None,
            'checkpoints': {}
        }
    
    def save_progress(self, progress: Dict[str, Any]):
        """Save processing progress to file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def create_document_hash(self, title: str, content: str) -> str:
        """Create a hash for duplicate detection."""
        combined = f"{title.lower().strip()}|{content[:1000].lower().strip()}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def is_duplicate(self, doc_hash: str) -> bool:
        """Check if document is a duplicate."""
        return doc_hash in self.duplicate_cache
    
    def add_to_duplicate_cache(self, doc_hash: str):
        """Add document hash to duplicate cache."""
        self.duplicate_cache.add(doc_hash)
    
    def process_batch_with_quality_control(self, documents: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        """Process a batch of documents with quality control and duplicate detection."""
        processed_docs = []
        quality_threshold = self.quality_thresholds[source]
        
        for doc in documents:
            try:
                # Check for duplicates
                doc_hash = self.create_document_hash(doc.get('title', ''), doc.get('content', ''))
                if self.is_duplicate(doc_hash):
                    continue
                
                # Quality filtering
                if doc.get('quality_score', 0) < quality_threshold:
                    continue
                
                # Generate embedding
                combined_text = f"{doc.get('title', '')}\n\n{doc.get('content', '')}"
                embedding = self.embedding_model.encode([combined_text])[0]
                
                # Create final document
                final_doc = {
                    'id': f"{source}_{doc_hash[:12]}",
                    'title': doc.get('title', ''),
                    'content': doc.get('content', ''),
                    'source': source,
                    'quality_score': doc.get('quality_score', 0),
                    'category': doc.get('category', 'general'),
                    'language': doc.get('language', 'en'),
                    'embedding': embedding.tolist(),
                    'metadata': {
                        'doc_hash': doc_hash,
                        'processed_at': datetime.now().isoformat(),
                        **doc.get('metadata', {})
                    }
                }
                
                processed_docs.append(final_doc)
                self.add_to_duplicate_cache(doc_hash)
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                continue
        
        return processed_docs
    
    def process_wikipedia_breadth_at_scale(self, target_count: int = None) -> Dict[str, Any]:
        """Process Wikipedia breadth articles at massive scale."""
        if target_count is None:
            target_count = self.targets['wikipedia_breadth']
        
        logger.info(f"Starting massive Wikipedia breadth processing - Target: {target_count:,} articles")
        
        # Load progress
        progress = self.load_progress()
        completed = progress['completed'].get('wikipedia_breadth', 0)
        
        if completed >= target_count:
            logger.info(f"Wikipedia breadth processing already completed: {completed:,}/{target_count:,}")
            return {'total_processed': completed, 'skipped': True}
        
        remaining = target_count - completed
        batch_size = self.batch_sizes['wikipedia_breadth']
        
        # Initialize breadth processor
        breadth_processor = BreadthFocusedProcessor()
        
        # Process in batches
        total_processed = completed
        batch_num = 0
        
        while total_processed < target_count:
            batch_num += 1
            current_batch_size = min(batch_size, target_count - total_processed)
            
            logger.info(f"Processing Wikipedia breadth batch {batch_num} - Target: {current_batch_size} documents")
            
            try:
                # Get documents from breadth processor
                batch_docs = breadth_processor.run_breadth_processing(max_documents=current_batch_size)
                
                if batch_docs.get('total_processed', 0) == 0:
                    logger.warning("No more documents available from breadth processor")
                    break
                
                # Process with quality control
                processed_docs = self.process_batch_with_quality_control(
                    batch_docs.get('documents', []), 'wikipedia_breadth'
                )
                
                if processed_docs:
                    # Add to database
                    self.db.batch_add_documents(processed_docs)
                    total_processed += len(processed_docs)
                    
                    logger.info(f"Batch {batch_num} completed: {len(processed_docs)} documents added")
                    logger.info(f"Total Wikipedia breadth: {total_processed:,}/{target_count:,}")
                
                # Save progress
                progress['completed']['wikipedia_breadth'] = total_processed
                progress['last_updated'] = datetime.now().isoformat()
                self.save_progress(progress)
                
                # Checkpoint every 10 batches
                if batch_num % 10 == 0:
                    self.save_checkpoint('wikipedia_breadth', total_processed)
                
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                continue
        
        logger.info(f"Wikipedia breadth processing completed: {total_processed:,} documents")
        return {'total_processed': total_processed}
    
    def process_arxiv_at_scale(self, target_count: int = None) -> Dict[str, Any]:
        """Process arXiv papers at massive scale."""
        if target_count is None:
            target_count = self.targets['arxiv']
        
        logger.info(f"Starting massive arXiv processing - Target: {target_count:,} papers")
        
        # Load progress
        progress = self.load_progress()
        completed = progress['completed'].get('arxiv', 0)
        
        if completed >= target_count:
            logger.info(f"arXiv processing already completed: {completed:,}/{target_count:,}")
            return {'total_processed': completed, 'skipped': True}
        
        # Initialize arXiv processor
        arxiv_processor = EnhancedArxivProcessor(self.db)
        
        # Process in batches
        batch_size = self.batch_sizes['arxiv']
        total_processed = completed
        
        while total_processed < target_count:
            current_batch_size = min(batch_size, target_count - total_processed)
            
            logger.info(f"Processing arXiv batch - Target: {current_batch_size} papers")
            
            try:
                # Process papers
                results = arxiv_processor.process_papers(
                    max_papers=current_batch_size,
                    min_quality_score=self.quality_thresholds['arxiv']
                )
                
                batch_processed = results.get('total_processed', 0)
                if batch_processed == 0:
                    logger.warning("No more arXiv papers available")
                    break
                
                total_processed += batch_processed
                logger.info(f"arXiv batch completed: {batch_processed} papers")
                logger.info(f"Total arXiv: {total_processed:,}/{target_count:,}")
                
                # Save progress
                progress['completed']['arxiv'] = total_processed
                progress['last_updated'] = datetime.now().isoformat()
                self.save_progress(progress)
                
            except Exception as e:
                logger.error(f"Error in arXiv batch: {e}")
                continue
        
        logger.info(f"arXiv processing completed: {total_processed:,} papers")
        return {'total_processed': total_processed}
    
    def save_checkpoint(self, source: str, count: int):
        """Save processing checkpoint."""
        checkpoint_file = Path(f"../data/checkpoints/{source}_checkpoint.json")
        checkpoint_data = {
            'source': source,
            'count': count,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.info(f"Checkpoint saved for {source}: {count:,} documents")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """Get overall processing progress."""
        progress = self.load_progress()
        total_target = sum(self.targets.values())
        total_completed = sum(progress['completed'].values())
        
        return {
            'total_documents': total_completed,
            'total_target': total_target,
            'progress_percentage': (total_completed / total_target) * 100,
            'source_breakdown': progress['completed'],
            'targets': self.targets,
            'last_updated': progress['last_updated']
        }
    
    def run_massive_processing(self, sources: List[str] = None, resume: bool = True) -> Dict[str, Any]:
        """Run massive processing for all sources."""
        if sources is None:
            sources = ['wikipedia_breadth', 'arxiv', 'pubmed', 'books', 'government', 'wikipedia_featured']
        
        logger.info(f"Starting massive processing - Sources: {sources}")
        start_time = time.time()
        
        results = {}
        
        try:
            # Phase 1: Wikipedia Breadth (Priority - 50% of data)
            if 'wikipedia_breadth' in sources:
                logger.info("=== PHASE 1: MASSIVE WIKIPEDIA BREADTH PROCESSING ===")
                results['wikipedia_breadth'] = self.process_wikipedia_breadth_at_scale()
                
                # Status update
                overall_progress = self.get_overall_progress()
                logger.info(f"Overall progress: {overall_progress['progress_percentage']:.1f}%")
            
            # Phase 2: arXiv (20% of data)
            if 'arxiv' in sources:
                logger.info("=== PHASE 2: MASSIVE ARXIV PROCESSING ===")
                results['arxiv'] = self.process_arxiv_at_scale()
                
                # Status update
                overall_progress = self.get_overall_progress()
                logger.info(f"Overall progress: {overall_progress['progress_percentage']:.1f}%")
            
            # Phase 3: PubMed (10% of data)
            if 'pubmed' in sources:
                logger.info("=== PHASE 3: MASSIVE PUBMED PROCESSING ===")
                # TODO: Implement PubMed processor
                logger.info("PubMed processor not yet implemented - skipping")
                results['pubmed'] = {'total_processed': 0, 'skipped': True}
            
            # Phase 4: Books (5% of data)
            if 'books' in sources:
                logger.info("=== PHASE 4: MASSIVE BOOKS PROCESSING ===")
                # TODO: Implement books processor
                logger.info("Books processor not yet implemented - skipping")
                results['books'] = {'total_processed': 0, 'skipped': True}
            
            # Phase 5: Government (5% of data)
            if 'government' in sources:
                logger.info("=== PHASE 5: MASSIVE GOVERNMENT PROCESSING ===")
                # TODO: Implement government processor
                logger.info("Government processor not yet implemented - skipping")
                results['government'] = {'total_processed': 0, 'skipped': True}
            
            # Phase 6: Wikipedia Featured (10% of data)
            if 'wikipedia_featured' in sources:
                logger.info("=== PHASE 6: MASSIVE WIKIPEDIA FEATURED PROCESSING ===")
                # TODO: Implement featured Wikipedia processor
                logger.info("Featured Wikipedia processor not yet implemented - skipping")
                results['wikipedia_featured'] = {'total_processed': 0, 'skipped': True}
            
            # Final status
            processing_time = time.time() - start_time
            final_progress = self.get_overall_progress()
            
            logger.info(f"Massive processing completed in {processing_time/3600:.1f} hours")
            logger.info(f"Final progress: {final_progress['progress_percentage']:.1f}%")
            
            return {
                'results': results,
                'final_progress': final_progress,
                'processing_time_hours': processing_time / 3600
            }
            
        except Exception as e:
            logger.error(f"Massive processing failed: {e}")
            return {'error': str(e)}


def main():
    """Main function for testing."""
    processor = MassiveScaleProcessor()
    
    # Test with small batch first
    logger.info("Testing massive scale processor...")
    
    # Get current progress
    progress = processor.get_overall_progress()
    logger.info(f"Current progress: {progress}")
    
    # Run processing for available sources
    results = processor.run_massive_processing(sources=['wikipedia_breadth', 'arxiv'])
    logger.info(f"Processing results: {results}")


if __name__ == "__main__":
    main()
