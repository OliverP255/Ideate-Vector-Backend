#!/usr/bin/env python3
"""
Knowledge Map Processor - Main Pipeline
Orchestrates the processing of 1M high-quality documents for the knowledge map.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import argparse

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.services.knowledge_database import KnowledgeDatabase
from scripts.wikipedia_processor import WikipediaQualityProcessor
from scripts.enhanced_arxiv_processor import EnhancedArxivProcessor
from scripts.breadth_focused_processor import BreadthFocusedProcessor
from scripts.massive_scale_processor import MassiveScaleProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('knowledge_map_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class KnowledgeMapProcessor:
    """Main processor for building the knowledge map with 1M documents."""
    
    def __init__(self):
        self.db = KnowledgeDatabase()
        self.wiki_processor = WikipediaQualityProcessor()
        self.arxiv_processor = EnhancedArxivProcessor()
        self.breadth_processor = BreadthFocusedProcessor()
        self.massive_processor = MassiveScaleProcessor()
        
        # Processing targets - Updated for breadth and factual accuracy
        self.targets = {
            'wikipedia_breadth': 500_000,  # 500K breadth-focused Wikipedia articles
            'wikipedia_featured': 100_000, # 100K Featured/Good Wikipedia articles
            'arxiv': 200_000,              # 200K high-quality arXiv papers (reduced)
            'pubmed': 100_000,             # 100K PubMed papers (reduced)
            'books': 50_000,               # 50K Project Gutenberg books
            'government': 50_000           # 50K government documents
        }
        
        self.total_target = sum(self.targets.values())
        
        logger.info(f"Knowledge Map Processor initialized - Target: {self.total_target:,} documents")
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        try:
            stats = self.db.get_stats()
            
            return {
                'total_documents': stats.get('total_documents', 0),
                'source_breakdown': stats.get('source_counts', {}),
                'quality_distribution': stats.get('quality_distribution', {}),
                'database_size_mb': stats.get('database_size_mb', 0),
                'target_progress': {
                    source: {
                        'target': count,
                        'current': stats.get('source_counts', {}).get(source, 0),
                        'percentage': (stats.get('source_counts', {}).get(source, 0) / count) * 100
                    }
                    for source, count in self.targets.items()
                },
                'overall_progress': (stats.get('total_documents', 0) / self.total_target) * 100,
                'last_updated': stats.get('last_updated', '')
            }
        except Exception as e:
            logger.error(f"Failed to get processing status: {e}")
            return {}
    
    def process_wikipedia(self, max_articles: int = None) -> Dict[str, Any]:
        """Process Wikipedia articles."""
        if max_articles is None:
            max_articles = self.targets['wikipedia']
        
        logger.info(f"Starting Wikipedia processing - Target: {max_articles:,} articles")
        start_time = time.time()
        
        try:
            results = self.wiki_processor.run_processing(max_articles=max_articles)
            
            processing_time = time.time() - start_time
            results['processing_time_minutes'] = processing_time / 60
            results['articles_per_minute'] = results['total_processed'] / (processing_time / 60)
            
            logger.info(f"Wikipedia processing completed in {processing_time/60:.1f} minutes")
            return results
            
        except Exception as e:
            logger.error(f"Wikipedia processing failed: {e}")
            return {'error': str(e)}
    
    def process_arxiv(self, max_papers: int = None) -> Dict[str, Any]:
        """Process arXiv papers."""
        if max_papers is None:
            max_papers = self.targets['arxiv']
        
        logger.info(f"Starting arXiv processing - Target: {max_papers:,} papers")
        start_time = time.time()
        
        try:
            results = self.arxiv_processor.run_processing(max_papers=max_papers)
            
            processing_time = time.time() - start_time
            results['processing_time_minutes'] = processing_time / 60
            results['papers_per_minute'] = results['total_processed'] / (processing_time / 60)
            
            logger.info(f"arXiv processing completed in {processing_time/60:.1f} minutes")
            return results
            
        except Exception as e:
            logger.error(f"arXiv processing failed: {e}")
            return {'error': str(e)}
    
    def process_breadth_wikipedia(self, max_articles: int = None) -> Dict[str, Any]:
        """Process Wikipedia articles with breadth focus."""
        if max_articles is None:
            max_articles = self.targets['wikipedia_breadth']
        
        logger.info(f"Starting breadth-focused Wikipedia processing - Target: {max_articles:,} articles")
        start_time = time.time()
        
        try:
            results = self.breadth_processor.run_breadth_processing(max_documents=max_articles)
            
            processing_time = time.time() - start_time
            results['processing_time_minutes'] = processing_time / 60
            results['articles_per_minute'] = results['total_processed'] / (processing_time / 60)
            
            logger.info(f"Breadth Wikipedia processing completed in {processing_time/60:.1f} minutes")
            return results
            
        except Exception as e:
            logger.error(f"Breadth Wikipedia processing failed: {e}")
            return {'error': str(e)}
    
    def run_massive_scale_processing(self, sources: List[str] = None) -> Dict[str, Any]:
        """Run massive scale processing to reach 1M documents."""
        logger.info("Starting massive scale processing to reach 1M documents")
        
        try:
            # Use the massive scale processor
            results = self.massive_processor.run_massive_processing(sources=sources)
            
            # Get final status
            final_status = self.get_processing_status()
            logger.info(f"Massive scale processing completed: {final_status['total_documents']:,} documents")
            
            return results
            
        except Exception as e:
            logger.error(f"Massive scale processing failed: {e}")
            return {'error': str(e)}
    
    def run_full_processing(self, phases: List[str] = None) -> Dict[str, Any]:
        """Run the complete processing pipeline."""
        if phases is None:
            phases = ['breadth_wikipedia', 'wikipedia', 'arxiv']
        
        logger.info(f"Starting full processing pipeline - Phases: {phases}")
        overall_start_time = time.time()
        
        results = {}
        
        try:
            # Phase 1: Breadth-focused Wikipedia (priority)
            if 'breadth_wikipedia' in phases:
                logger.info("=== PHASE 1: BREADTH-FOCUSED WIKIPEDIA PROCESSING ===")
                breadth_results = self.process_breadth_wikipedia()
                results['breadth_wikipedia'] = breadth_results
                
                # Status update
                status = self.get_processing_status()
                logger.info(f"Progress: {status['overall_progress']:.1f}% complete")
            
            # Phase 2: Featured Wikipedia
            if 'wikipedia' in phases:
                logger.info("=== PHASE 2: FEATURED WIKIPEDIA PROCESSING ===")
                wiki_results = self.process_wikipedia()
                results['wikipedia'] = wiki_results
                
                # Status update
                status = self.get_processing_status()
                logger.info(f"Progress: {status['overall_progress']:.1f}% complete")
            
            # Phase 3: arXiv
            if 'arxiv' in phases:
                logger.info("=== PHASE 3: ARXIV PROCESSING ===")
                arxiv_results = self.process_arxiv()
                results['arxiv'] = arxiv_results
                
                # Status update
                status = self.get_processing_status()
                logger.info(f"Progress: {status['overall_progress']:.1f}% complete")
            
            # Final status
            overall_time = time.time() - overall_start_time
            final_status = self.get_processing_status()
            
            results['summary'] = {
                'total_processing_time_hours': overall_time / 3600,
                'total_documents_processed': final_status['total_documents'],
                'overall_progress_percentage': final_status['overall_progress'],
                'source_breakdown': final_status['source_breakdown'],
                'database_size_mb': final_status['database_size_mb']
            }
            
            logger.info(f"Full processing completed in {overall_time/3600:.1f} hours")
            logger.info(f"Total documents: {final_status['total_documents']:,}")
            logger.info(f"Database size: {final_status['database_size_mb']:.1f} MB")
            
            return results
            
        except Exception as e:
            logger.error(f"Full processing failed: {e}")
            return {'error': str(e)}
    
    def run_incremental_processing(self, batch_size: int = 1000) -> Dict[str, Any]:
        """Run incremental processing in small batches."""
        logger.info(f"Starting incremental processing - Batch size: {batch_size}")
        
        # Get current status
        status = self.get_processing_status()
        current_total = status['total_documents']
        
        logger.info(f"Current total: {current_total:,} documents")
        
        # Determine what to process next
        if status['source_breakdown'].get('wikipedia', 0) < self.targets['wikipedia']:
            # Process Wikipedia batch
            remaining_wiki = self.targets['wikipedia'] - status['source_breakdown'].get('wikipedia', 0)
            batch_size = min(batch_size, remaining_wiki)
            
            logger.info(f"Processing {batch_size} Wikipedia articles")
            results = self.wiki_processor.run_processing(max_articles=batch_size)
            
        elif status['source_breakdown'].get('arxiv', 0) < self.targets['arxiv']:
            # Process arXiv batch
            remaining_arxiv = self.targets['arxiv'] - status['source_breakdown'].get('arxiv', 0)
            batch_size = min(batch_size, remaining_arxiv)
            
            logger.info(f"Processing {batch_size} arXiv papers")
            results = self.arxiv_processor.run_processing(max_papers=batch_size)
            
        else:
            logger.info("All targets reached!")
            results = {'message': 'All processing targets reached'}
        
        # Return updated status
        new_status = self.get_processing_status()
        results['updated_status'] = new_status
        
        return results
    
    def export_processing_report(self, output_file: str = None) -> str:
        """Export a detailed processing report."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"processing_report_{timestamp}.json"
        
        status = self.get_processing_status()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'processing_status': status,
            'targets': self.targets,
            'database_info': {
                'path': str(self.db.db_path),
                'sqlite_path': str(self.db.sqlite_path),
                'chroma_path': str(self.db.chroma_path)
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Processing report exported to {output_file}")
        return output_file


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Knowledge Map Processor')
    parser.add_argument('--phase', choices=['wikipedia', 'arxiv', 'full', 'massive'], 
                       default='full', help='Processing phase to run')
    parser.add_argument('--max-docs', type=int, default=1000,
                       help='Maximum documents to process (for testing)')
    parser.add_argument('--incremental', action='store_true',
                       help='Run incremental processing')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for incremental processing')
    parser.add_argument('--status', action='store_true',
                       help='Show current processing status')
    parser.add_argument('--report', action='store_true',
                       help='Export processing report')
    parser.add_argument('--sources', nargs='+', 
                       choices=['wikipedia_breadth', 'arxiv', 'pubmed', 'books', 'government', 'wikipedia_featured'],
                       help='Specific sources to process for massive scale')
    
    args = parser.parse_args()
    
    processor = KnowledgeMapProcessor()
    
    if args.status:
        # Show status
        status = processor.get_processing_status()
        print(f"\n=== KNOWLEDGE MAP PROCESSING STATUS ===")
        print(f"Total Documents: {status['total_documents']:,}")
        print(f"Overall Progress: {status['overall_progress']:.1f}%")
        print(f"Database Size: {status['database_size_mb']:.1f} MB")
        print(f"\nSource Breakdown:")
        for source, info in status['target_progress'].items():
            print(f"  {source}: {info['current']:,}/{info['target']:,} ({info['percentage']:.1f}%)")
        
    elif args.report:
        # Export report
        report_file = processor.export_processing_report()
        print(f"Processing report exported to: {report_file}")
        
    elif args.incremental:
        # Incremental processing
        results = processor.run_incremental_processing(batch_size=args.batch_size)
        print(f"Incremental processing results: {results}")
        
    else:
        # Full processing
        if args.phase == 'wikipedia':
            results = processor.process_wikipedia(max_articles=args.max_docs)
        elif args.phase == 'arxiv':
            results = processor.process_arxiv(max_papers=args.max_docs)
        elif args.phase == 'massive':
            # Massive scale processing to 1M documents
            sources = args.sources or ['wikipedia_breadth', 'arxiv', 'pubmed', 'books', 'government', 'wikipedia_featured']
            logger.info(f"Starting massive scale processing with sources: {sources}")
            results = processor.run_massive_scale_processing(sources=sources)
            logger.info(f"Massive processing results: {results}")
        else:  # full
            results = processor.run_full_processing()
        
        print(f"Processing results: {results}")
    
    # Close database connection
    processor.db.close()


if __name__ == "__main__":
    main()
