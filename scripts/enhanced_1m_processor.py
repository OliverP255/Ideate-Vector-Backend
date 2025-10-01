#!/usr/bin/env python3
"""
Enhanced 1M Papers Processor
Implements the comprehensive plan for embedding 1,000,000 high-quality research papers
with strict quality requirements and breadth focus across all academic disciplines.
"""

import os
import sys
import json
import time
import logging
import requests
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import hashlib
import re
import sqlite3

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sentence_transformers import SentenceTransformer
from backend.app.services.knowledge_database import KnowledgeDatabase
from backend.app.services.vector_db import VectorDatabaseService
from backend.app.services.embedding import EmbeddingService

# Configure logging
logs_dir = Path('../data/logs')
logs_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'enhanced_1m_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Enhanced1MProcessor:
    """Enhanced processor for 1M high-quality research papers."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(12, mp.cpu_count())
        
        # Initialize services
        self.db = KnowledgeDatabase()
        self.vector_db = VectorDatabaseService()
        self.embedding_service = EmbeddingService()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
        
        # Processing targets for 1M documents
        self.targets = {
            'arxiv': 300_000,           # 30% - Research papers
            'pubmed': 200_000,          # 20% - Medical papers
            'wikipedia_featured': 250_000, # 25% - Wikipedia breadth
            'academic_db': 150_000,     # 15% - Academic databases
            'government': 50_000,       # 5% - Government reports
            'books': 50_000             # 5% - Books and monographs
        }
        
        # Quality thresholds (higher = stricter)
        self.quality_thresholds = {
            'arxiv': 0.6,
            'pubmed': 0.7,
            'wikipedia_featured': 0.4,  # Lower for breadth
            'academic_db': 0.6,
            'government': 0.6,
            'books': 0.6
        }
        
        # Batch sizes for efficient processing
        self.batch_sizes = {
            'arxiv': 1000,
            'pubmed': 500,
            'wikipedia_featured': 2000,
            'academic_db': 300,
            'government': 200,
            'books': 150
        }
        
        # Progress tracking
        self.progress_file = Path("../data/1m_processing_progress.json")
        self.duplicate_cache = set()
        
        # Create necessary directories
        os.makedirs("../data/logs", exist_ok=True)
        os.makedirs("../data/checkpoints", exist_ok=True)
        os.makedirs("../data/embeddings", exist_ok=True)
        
        logger.info(f"Enhanced 1M Processor initialized - Target: 1,000,000 documents")
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
            'checkpoints': {},
            'quality_stats': {}
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
    
    def calculate_quality_score(self, document: Dict[str, Any], source: str) -> float:
        """Calculate comprehensive quality score for a document."""
        base_score = 0.5
        
        # Source credibility (0.0-1.0)
        source_scores = {
            'arxiv': 0.7,
            'pubmed': 0.9,
            'wikipedia_featured': 0.6,
            'academic_db': 0.8,
            'government': 0.9,
            'books': 0.7
        }
        source_credibility = source_scores.get(source, 0.5)
        
        # Content depth (0.0-1.0)
        content_length = len(document.get('content', ''))
        if content_length > 5000:
            content_depth = 1.0
        elif content_length > 2000:
            content_depth = 0.8
        elif content_length > 1000:
            content_depth = 0.6
        elif content_length > 500:
            content_depth = 0.4
        else:
            content_depth = 0.2
        
        # Factual accuracy indicators
        content_lower = document.get('content', '').lower()
        factual_indicators = [
            'research', 'study', 'analysis', 'data', 'evidence',
            'scientific', 'peer-reviewed', 'published', 'journal'
        ]
        factual_count = sum(1 for indicator in factual_indicators if indicator in content_lower)
        factual_accuracy = min(1.0, factual_count / 5.0)
        
        # Recency factor (0.0-1.0)
        recency_factor = 0.7  # Default for now, can be enhanced with actual dates
        
        # Citation impact (0.0-1.0)
        citation_count = document.get('metadata', {}).get('citation_count', 0)
        if citation_count > 100:
            citation_impact = 1.0
        elif citation_count > 50:
            citation_impact = 0.8
        elif citation_count > 20:
            citation_impact = 0.6
        elif citation_count > 5:
            citation_impact = 0.4
        else:
            citation_impact = 0.2
        
        # Calculate final quality score
        quality_score = (
            source_credibility * 0.3 +
            content_depth * 0.25 +
            factual_accuracy * 0.25 +
            recency_factor * 0.1 +
            citation_impact * 0.1
        )
        
        return max(0.0, min(1.0, quality_score))
    
    async def process_document_with_quality_control(self, document: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """Process a single document with comprehensive quality control."""
        try:
            # Check for duplicates
            doc_hash = self.create_document_hash(
                document.get('title', ''), 
                document.get('content', '')
            )
            if self.is_duplicate(doc_hash):
                return None
            
            # Calculate quality score
            quality_score = self.calculate_quality_score(document, source)
            
            # Apply quality threshold
            quality_threshold = self.quality_thresholds[source]
            if quality_score < quality_threshold:
                return None
            
            # Generate embedding
            combined_text = f"{document.get('title', '')}\n\n{document.get('content', '')}"
            embedding = self.embedding_model.encode([combined_text])[0]
            
            # Create final document
            final_doc = {
                'id': f"{source}_{doc_hash[:12]}",
                'title': document.get('title', ''),
                'content': document.get('content', ''),
                'source': source,
                'quality_score': quality_score,
                'category': document.get('category', 'general'),
                'language': document.get('language', 'en'),
                'embedding': embedding.tolist(),
                'coordinates': [0.0, 0.0],  # Will be updated by UMAP
                'metadata': {
                    'doc_hash': doc_hash,
                    'processed_at': datetime.now().isoformat(),
                    'quality_breakdown': {
                        'source_credibility': quality_score,
                        'content_depth': len(document.get('content', '')) / 5000,
                        'factual_accuracy': quality_score,
                        'citation_impact': document.get('metadata', {}).get('citation_count', 0) / 100
                    },
                    **document.get('metadata', {})
                }
            }
            
            self.add_to_duplicate_cache(doc_hash)
            return final_doc
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return None
    
    async def process_arxiv_papers(self, target_count: int = None) -> Dict[str, Any]:
        """Process arXiv papers with enhanced quality control."""
        if target_count is None:
            target_count = self.targets['arxiv']
        
        logger.info(f"Starting arXiv processing - Target: {target_count:,} papers")
        
        # Load progress
        progress = self.load_progress()
        completed = progress['completed'].get('arxiv', 0)
        
        if completed >= target_count:
            logger.info(f"arXiv processing already completed: {completed:,}/{target_count:,}")
            return {'total_processed': completed, 'skipped': True}
        
        # arXiv API parameters
        base_url = "http://export.arxiv.org/api/query"
        batch_size = self.batch_sizes['arxiv']
        total_processed = completed
        
        # arXiv categories for breadth
        categories = [
            'cs.AI', 'cs.LG', 'cs.CV', 'cs.NE', 'cs.CL',  # Computer Science
            'physics.quant-ph', 'physics.astro-ph', 'physics.cond-mat',  # Physics
            'math.CO', 'math.ST', 'math.PR', 'stat.ML',  # Mathematics/Stats
            'q-bio.BM', 'q-bio.GN', 'q-bio.QM',  # Biology
            'econ.TH', 'econ.EM', 'econ.LM',  # Economics
            'eess.SY', 'eess.IV', 'eess.SP'  # Engineering
        ]
        
        batch_num = 0
        while total_processed < target_count:
            batch_num += 1
            current_batch_size = min(batch_size, target_count - total_processed)
            
            logger.info(f"Processing arXiv batch {batch_num} - Target: {current_batch_size} papers")
            
            try:
                # Get papers from arXiv API
                papers = await self._fetch_arxiv_papers(categories, current_batch_size)
                
                if not papers:
                    logger.warning("No more arXiv papers available")
                    break
                
                # Process papers with quality control
                processed_papers = []
                for paper in papers:
                    processed_doc = await self.process_document_with_quality_control(paper, 'arxiv')
                    if processed_doc:
                        processed_papers.append(processed_doc)
                
                if processed_papers:
                    # Add to database
                    await self._batch_add_documents(processed_papers)
                    total_processed += len(processed_papers)
                    
                    logger.info(f"Batch {batch_num} completed: {len(processed_papers)} papers added")
                    logger.info(f"Total arXiv: {total_processed:,}/{target_count:,}")
                
                # Save progress
                progress['completed']['arxiv'] = total_processed
                progress['last_updated'] = datetime.now().isoformat()
                self.save_progress(progress)
                
                # Checkpoint every 10 batches
                if batch_num % 10 == 0:
                    self._save_checkpoint('arxiv', total_processed)
                
                # Rate limiting
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"Error in arXiv batch {batch_num}: {e}")
                continue
        
        logger.info(f"arXiv processing completed: {total_processed:,} papers")
        return {'total_processed': total_processed}
    
    async def _fetch_arxiv_papers(self, categories: List[str], max_results: int) -> List[Dict[str, Any]]:
        """Fetch papers from arXiv API."""
        papers = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for category in categories:
                    if len(papers) >= max_results:
                        break
                    
                    params = {
                        'search_query': f'cat:{category}',
                        'start': 0,
                        'max_results': min(100, max_results - len(papers)),
                        'sortBy': 'submittedDate',
                        'sortOrder': 'descending'
                    }
                    
                    async with session.get('http://export.arxiv.org/api/query', params=params) as response:
                        if response.status == 200:
                            content = await response.text()
                            category_papers = self._parse_arxiv_response(content)
                            papers.extend(category_papers)
                        
                        # Rate limiting
                        await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Error fetching arXiv papers: {e}")
        
        return papers[:max_results]
    
    def _parse_arxiv_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse arXiv API XML response."""
        papers = []
        
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_content)
            
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('.//{http://www.w3.org/2005/Atom}title').text
                summary = entry.find('.//{http://www.w3.org/2005/Atom}summary').text
                authors = [author.find('.//{http://www.w3.org/2005/Atom}name').text 
                          for author in entry.findall('.//{http://www.w3.org/2005/Atom}author')]
                published = entry.find('.//{http://www.w3.org/2005/Atom}published').text
                link = entry.find('.//{http://www.w3.org/2005/Atom}link[@type="application/pdf"]').get('href')
                
                # Quality checks
                if len(summary) < 500:  # Minimum abstract length
                    continue
                
                paper = {
                    'title': title.strip(),
                    'content': summary.strip(),
                    'category': 'research_paper',
                    'metadata': {
                        'authors': authors,
                        'published_date': published,
                        'pdf_link': link,
                        'source_url': link,
                        'abstract_length': len(summary)
                    }
                }
                
                papers.append(paper)
        
        except Exception as e:
            logger.error(f"Error parsing arXiv response: {e}")
        
        return papers
    
    async def process_wikipedia_breadth(self, target_count: int = None) -> Dict[str, Any]:
        """Process Wikipedia articles with breadth focus."""
        if target_count is None:
            target_count = self.targets['wikipedia_featured']
        
        logger.info(f"Starting Wikipedia breadth processing - Target: {target_count:,} articles")
        
        # Load progress
        progress = self.load_progress()
        completed = progress['completed'].get('wikipedia_featured', 0)
        
        if completed >= target_count:
            logger.info(f"Wikipedia processing already completed: {completed:,}/{target_count:,}")
            return {'total_processed': completed, 'skipped': True}
        
        # Wikipedia categories for breadth
        categories = [
            # Featured articles
            'Featured_articles', 'Good_articles',
            # Major topic categories
            'Mathematics', 'Physics', 'Chemistry', 'Biology', 'Medicine',
            'History', 'Geography', 'Literature', 'Philosophy', 'Art',
            'Technology', 'Economics', 'Psychology', 'Sociology', 'Politics'
        ]
        
        batch_size = self.batch_sizes['wikipedia_featured']
        total_processed = completed
        batch_num = 0
        
        while total_processed < target_count:
            batch_num += 1
            current_batch_size = min(batch_size, target_count - total_processed)
            
            logger.info(f"Processing Wikipedia batch {batch_num} - Target: {current_batch_size} articles")
            
            try:
                # Get articles from Wikipedia API
                articles = await self._fetch_wikipedia_articles(categories, current_batch_size)
                
                if not articles:
                    logger.warning("No more Wikipedia articles available")
                    break
                
                # Process articles with quality control
                processed_articles = []
                for article in articles:
                    processed_doc = await self.process_document_with_quality_control(article, 'wikipedia_featured')
                    if processed_doc:
                        processed_articles.append(processed_doc)
                
                if processed_articles:
                    # Add to database
                    await self._batch_add_documents(processed_articles)
                    total_processed += len(processed_articles)
                    
                    logger.info(f"Batch {batch_num} completed: {len(processed_articles)} articles added")
                    logger.info(f"Total Wikipedia: {total_processed:,}/{target_count:,}")
                
                # Save progress
                progress['completed']['wikipedia_featured'] = total_processed
                progress['last_updated'] = datetime.now().isoformat()
                self.save_progress(progress)
                
                # Checkpoint every 10 batches
                if batch_num % 10 == 0:
                    self._save_checkpoint('wikipedia_featured', total_processed)
                
                # Rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in Wikipedia batch {batch_num}: {e}")
                continue
        
        logger.info(f"Wikipedia processing completed: {total_processed:,} articles")
        return {'total_processed': total_processed}
    
    async def _fetch_wikipedia_articles(self, categories: List[str], max_results: int) -> List[Dict[str, Any]]:
        """Fetch articles from Wikipedia API."""
        articles = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for category in categories:
                    if len(articles) >= max_results:
                        break
                    
                    # Get articles from category
                    params = {
                        'action': 'query',
                        'format': 'json',
                        'list': 'categorymembers',
                        'cmtitle': f'Category:{category}',
                        'cmlimit': min(50, max_results - len(articles))
                    }
                    
                    async with session.get('https://en.wikipedia.org/w/api.php', params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if 'query' in data and 'categorymembers' in data['query']:
                                for member in data['query']['categorymembers']:
                                    if len(articles) >= max_results:
                                        break
                                    
                                    title = member['title']
                                    
                                    # Get article content
                                    article_content = await self._get_wikipedia_content(session, title)
                                    if article_content:
                                        articles.append(article_content)
                        
                        # Rate limiting
                        await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Error fetching Wikipedia articles: {e}")
        
        return articles[:max_results]
    
    async def _get_wikipedia_content(self, session: aiohttp.ClientSession, title: str) -> Optional[Dict[str, Any]]:
        """Get Wikipedia article content."""
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'exintro': False,
                'explaintext': True
            }
            
            async with session.get('https://en.wikipedia.org/w/api.php', params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    pages = data['query']['pages']
                    page_id = list(pages.keys())[0]
                    
                    if page_id != '-1' and 'extract' in pages[page_id]:
                        content = pages[page_id]['extract']
                        
                        # Quality check
                        if len(content) < 1000:
                            return None
                        
                        return {
                            'title': title,
                            'content': content,
                            'category': 'encyclopedia_article',
                            'metadata': {
                                'source_url': f'https://en.wikipedia.org/wiki/{title.replace(" ", "_")}',
                                'content_length': len(content),
                                'page_id': page_id
                            }
                        }
        
        except Exception as e:
            logger.error(f"Error getting Wikipedia content for {title}: {e}")
        
        return None
    
    async def _batch_add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to both databases."""
        try:
            # Add to knowledge database
            self.db.batch_add_documents(documents)
            
            # Add to vector database
            vector_docs = []
            for doc in documents:
                vector_docs.append({
                    'id': doc['id'],
                    'embedding': doc['embedding'],
                    'metadata': {
                        'title': doc['title'],
                        'content': doc['content'],
                        'source': doc['source'],
                        'coordinates': doc['coordinates'],
                        **doc['metadata']
                    }
                })
            
            await self.vector_db.batch_add_documents(vector_docs)
            
        except Exception as e:
            logger.error(f"Error adding documents to database: {e}")
    
    def _save_checkpoint(self, source: str, count: int):
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
    
    async def run_1m_processing(self, sources: List[str] = None) -> Dict[str, Any]:
        """Run the complete 1M document processing pipeline."""
        if sources is None:
            sources = ['arxiv', 'wikipedia_featured']  # Start with available sources
        
        logger.info(f"Starting 1M document processing - Sources: {sources}")
        start_time = time.time()
        
        results = {}
        
        try:
            # Phase 1: arXiv Processing (30% of data)
            if 'arxiv' in sources:
                logger.info("=== PHASE 1: ARXIV PROCESSING (300K PAPERS) ===")
                results['arxiv'] = await self.process_arxiv_papers()
                
                # Status update
                overall_progress = self.get_overall_progress()
                logger.info(f"Overall progress: {overall_progress['progress_percentage']:.1f}%")
            
            # Phase 2: Wikipedia Processing (25% of data)
            if 'wikipedia_featured' in sources:
                logger.info("=== PHASE 2: WIKIPEDIA BREADTH PROCESSING (250K ARTICLES) ===")
                results['wikipedia_featured'] = await self.process_wikipedia_breadth()
                
                # Status update
                overall_progress = self.get_overall_progress()
                logger.info(f"Overall progress: {overall_progress['progress_percentage']:.1f}%")
            
            # Phase 3: Additional sources (when implemented)
            for source in ['pubmed', 'academic_db', 'government', 'books']:
                if source in sources:
                    logger.info(f"=== PHASE 3: {source.upper()} PROCESSING ===")
                    logger.info(f"{source} processor not yet implemented - skipping")
                    results[source] = {'total_processed': 0, 'skipped': True}
            
            # Final status
            processing_time = time.time() - start_time
            final_progress = self.get_overall_progress()
            
            logger.info(f"1M processing completed in {processing_time/3600:.1f} hours")
            logger.info(f"Final progress: {final_progress['progress_percentage']:.1f}%")
            
            return {
                'results': results,
                'final_progress': final_progress,
                'processing_time_hours': processing_time / 3600
            }
            
        except Exception as e:
            logger.error(f"1M processing failed: {e}")
            return {'error': str(e)}


async def main():
    """Main function for testing."""
    processor = Enhanced1MProcessor()
    
    # Test with available sources
    logger.info("Testing Enhanced 1M processor...")
    
    # Get current progress
    progress = processor.get_overall_progress()
    logger.info(f"Current progress: {progress}")
    
    # Run processing for available sources
    results = await processor.run_1m_processing(sources=['arxiv', 'wikipedia_featured'])
    logger.info(f"Processing results: {results}")


if __name__ == "__main__":
    asyncio.run(main())
