#!/usr/bin/env python3
"""
Enhanced arXiv Quality Processor
Downloads and processes high-quality arXiv papers with advanced filtering.
"""

import os
import sys
import json
import time
import requests
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import re

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sentence_transformers import SentenceTransformer
from backend.app.services.knowledge_database import KnowledgeDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedArxivProcessor:
    """Enhanced arXiv processor with quality filtering for the knowledge map."""
    
    def __init__(self, output_dir: str = "data/arxiv"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
        
        # Initialize database
        self.db = KnowledgeDatabase()
        
        # Quality criteria
        self.min_abstract_length = 500    # Minimum abstract length
        self.min_title_length = 20        # Minimum title length
        self.max_title_length = 200       # Maximum title length
        self.quality_categories = {
            'cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.NE',  # AI/ML
            'cs.SE', 'cs.PL', 'cs.DC', 'cs.AR', 'cs.OS',  # Systems
            'math.ST', 'stat.ML', 'stat.AP', 'stat.CO',    # Statistics
            'physics.comp-ph', 'cond-mat.mtrl-sci',        # Physics
            'q-bio.BM', 'q-bio.QM', 'q-bio.GN',            # Biology
            'econ.EM', 'econ.TH', 'econ.GN'                # Economics
        }
        
        # Rate limiting
        self.request_delay = 0.5  # 500ms between requests
        
        logger.info("Enhanced arXiv Processor initialized")
    
    def _rate_limit(self):
        """Implement rate limiting for arXiv API requests."""
        time.sleep(self.request_delay)
    
    def fetch_arxiv_metadata(self, max_results: int = 50000) -> List[Dict[str, Any]]:
        """Fetch arXiv metadata with quality filtering."""
        logger.info(f"Fetching arXiv metadata (max {max_results} results)")
        
        papers = []
        start = 0
        batch_size = 1000
        
        while start < max_results:
            self._rate_limit()
            
            # arXiv API query for recent high-quality papers
            query_url = "http://export.arxiv.org/api/query"
            
            # Query for recent papers in quality categories
            category_query = ' OR '.join([f'cat:{cat}' for cat in self.quality_categories])
            
            params = {
                'search_query': f'({category_query}) AND submittedDate:[20200101 TO 20241231]',
                'start': start,
                'max_results': min(batch_size, max_results - start),
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            try:
                response = requests.get(query_url, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                papers_batch = self._parse_arxiv_response(root)
                
                if not papers_batch:
                    logger.info("No more papers found, stopping")
                    break
                
                papers.extend(papers_batch)
                logger.info(f"Fetched {len(papers)} papers so far...")
                
                start += batch_size
                
            except Exception as e:
                logger.error(f"Error fetching arXiv metadata: {e}")
                break
        
        logger.info(f"Total papers fetched: {len(papers)}")
        return papers
    
    def _parse_arxiv_response(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Parse arXiv API response."""
        papers = []
        
        # Define namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', ns):
            try:
                # Extract paper ID
                paper_id = entry.find('atom:id', ns).text.split('/')[-1]
                
                # Extract title
                title = entry.find('atom:title', ns).text.strip()
                
                # Extract abstract
                abstract = entry.find('atom:summary', ns).text.strip()
                
                # Extract authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns).text
                    authors.append(name)
                
                # Extract categories
                categories = []
                for category in entry.findall('atom:category', ns):
                    categories.append(category.get('term'))
                
                # Extract publication date
                published = entry.find('atom:published', ns).text
                
                # Extract PDF URL
                pdf_url = None
                for link in entry.findall('atom:link', ns):
                    if link.get('type') == 'application/pdf':
                        pdf_url = link.get('href')
                        break
                
                papers.append({
                    'paper_id': paper_id,
                    'title': title,
                    'abstract': abstract,
                    'authors': authors,
                    'categories': categories,
                    'published': published,
                    'pdf_url': pdf_url
                })
                
            except Exception as e:
                logger.error(f"Error parsing arXiv entry: {e}")
                continue
        
        return papers
    
    def calculate_quality_score(self, paper: Dict[str, Any]) -> float:
        """Calculate quality score for a paper."""
        score = 0.0
        
        # Title quality
        title = paper['title']
        if len(title) >= self.min_title_length and len(title) <= self.max_title_length:
            score += 0.1
            
            # Bonus for specific keywords
            quality_keywords = [
                'novel', 'new', 'improved', 'advanced', 'efficient', 'robust',
                'deep learning', 'machine learning', 'neural network', 'algorithm',
                'framework', 'method', 'approach', 'technique'
            ]
            
            title_lower = title.lower()
            keyword_matches = sum(1 for keyword in quality_keywords if keyword in title_lower)
            score += min(keyword_matches * 0.02, 0.1)
        
        # Abstract quality
        abstract = paper['abstract']
        if len(abstract) >= self.min_abstract_length:
            score += 0.2
            
            # Abstract structure bonus
            if 'introduction' in abstract.lower() or 'background' in abstract.lower():
                score += 0.05
            if 'method' in abstract.lower() or 'approach' in abstract.lower():
                score += 0.05
            if 'result' in abstract.lower() or 'experiment' in abstract.lower():
                score += 0.05
            if 'conclusion' in abstract.lower() or 'summary' in abstract.lower():
                score += 0.05
        
        # Category bonus
        categories = paper['categories']
        high_quality_cats = {'cs.AI', 'cs.LG', 'cs.CV', 'cs.CL', 'stat.ML', 'math.ST'}
        if any(cat in high_quality_cats for cat in categories):
            score += 0.2
        
        # Author count bonus (more authors often means more collaborative work)
        author_count = len(paper['authors'])
        if author_count >= 3:
            score += 0.1
        elif author_count >= 2:
            score += 0.05
        
        # Recent publication bonus
        try:
            pub_date = datetime.fromisoformat(paper['published'].replace('Z', '+00:00'))
            days_ago = (datetime.now(pub_date.tzinfo) - pub_date).days
            
            if days_ago < 365:  # Last year
                score += 0.1
            elif days_ago < 730:  # Last 2 years
                score += 0.05
        except:
            pass
        
        # Length bonus
        total_length = len(title) + len(abstract)
        if total_length > 2000:
            score += 0.1
        elif total_length > 1000:
            score += 0.05
        
        return min(score, 1.0)  # Cap at 1.0
    
    def filter_high_quality_papers(self, papers: List[Dict[str, Any]], min_quality: float = 0.4) -> List[Dict[str, Any]]:
        """Filter papers by quality score."""
        logger.info(f"Filtering papers with quality >= {min_quality}")
        
        filtered_papers = []
        for paper in papers:
            quality_score = self.calculate_quality_score(paper)
            paper['quality_score'] = quality_score
            
            if quality_score >= min_quality:
                filtered_papers.append(paper)
        
        logger.info(f"Filtered to {len(filtered_papers)} high-quality papers from {len(papers)} total")
        return filtered_papers
    
    def process_paper(self, paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single paper."""
        try:
            # Combine title and abstract for embedding
            combined_text = f"{paper['title']}\n\n{paper['abstract']}"
            
            # Generate embedding
            embedding = self.embedding_model.encode([combined_text])[0]
            
            # Determine category
            category = self._determine_category(paper['categories'])
            
            # Create document
            document = {
                'id': f"arxiv_{paper['paper_id']}",
                'title': paper['title'],
                'content': combined_text,
                'source': 'arxiv',
                'quality_score': paper['quality_score'],
                'category': category,
                'language': 'en',
                'embedding': embedding.tolist(),
                'metadata': {
                    'paper_id': paper['paper_id'],
                    'authors': paper['authors'],
                    'categories': paper['categories'],
                    'published': paper['published'],
                    'pdf_url': paper['pdf_url'],
                    'abstract_length': len(paper['abstract']),
                    'title_length': len(paper['title'])
                }
            }
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to process paper {paper['paper_id']}: {e}")
            return None
    
    def _determine_category(self, categories: List[str]) -> str:
        """Determine the main category for a paper."""
        category_mapping = {
            'ai': ['cs.AI', 'cs.LG', 'cs.NE', 'cs.CV', 'cs.CL'],
            'machine_learning': ['stat.ML', 'cs.LG', 'cs.NE'],
            'computer_science': ['cs.SE', 'cs.PL', 'cs.DC', 'cs.AR', 'cs.OS'],
            'mathematics': ['math.ST', 'math.CO', 'math.NA', 'math.OC'],
            'physics': ['physics.comp-ph', 'cond-mat.mtrl-sci', 'cond-mat.dis-nn'],
            'biology': ['q-bio.BM', 'q-bio.QM', 'q-bio.GN', 'q-bio.PE'],
            'economics': ['econ.EM', 'econ.TH', 'econ.GN', 'econ.QM'],
            'statistics': ['stat.ML', 'stat.AP', 'stat.CO', 'stat.TH']
        }
        
        for main_category, subcategories in category_mapping.items():
            if any(cat in subcategories for cat in categories):
                return main_category
        
        return 'general'
    
    def process_papers_batch(self, papers: List[Dict[str, Any]], batch_size: int = 100) -> List[Dict[str, Any]]:
        """Process papers in batches."""
        logger.info(f"Processing {len(papers)} papers in batches of {batch_size}")
        
        all_documents = []
        processed_count = 0
        
        for i in range(0, len(papers), batch_size):
            batch_papers = papers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(papers) + batch_size - 1)//batch_size}")
            
            # Process batch with threading
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_paper = {executor.submit(self.process_paper, paper): paper 
                                 for paper in batch_papers}
                
                batch_documents = []
                for future in as_completed(future_to_paper):
                    paper = future_to_paper[future]
                    try:
                        document = future.result()
                        if document:
                            batch_documents.append(document)
                            processed_count += 1
                    except Exception as e:
                        logger.error(f"Error processing paper {paper['paper_id']}: {e}")
                
                # Add batch to database
                if batch_documents:
                    self.db.batch_add_documents(batch_documents)
                    all_documents.extend(batch_documents)
                    logger.info(f"Added {len(batch_documents)} documents to database")
                
                # Progress update
                logger.info(f"Processed {processed_count}/{len(papers)} papers")
        
        return all_documents
    
    def run_processing(self, max_papers: int = 20000) -> Dict[str, Any]:
        """Run the complete arXiv processing pipeline."""
        logger.info(f"Starting arXiv processing (max {max_papers} papers)")
        
        # Fetch metadata
        papers = self.fetch_arxiv_metadata(max_results=max_papers * 2)  # Fetch more to filter
        
        # Filter for quality
        high_quality_papers = self.filter_high_quality_papers(papers, min_quality=0.4)
        
        # Limit to max_papers
        if len(high_quality_papers) > max_papers:
            high_quality_papers = high_quality_papers[:max_papers]
            logger.info(f"Limited to {max_papers} papers")
        
        # Process papers
        documents = self.process_papers_batch(high_quality_papers, batch_size=50)
        
        # Get final stats
        stats = self.db.get_stats()
        
        logger.info(f"arXiv processing completed: {len(documents)} documents processed")
        return {
            'total_processed': len(documents),
            'total_fetched': len(papers),
            'high_quality_count': len(high_quality_papers),
            'database_stats': stats
        }


def main():
    """Main function for testing."""
    processor = EnhancedArxivProcessor()
    
    # Test with small batch
    results = processor.run_processing(max_papers=1000)
    print(f"Processing results: {results}")
    
    # Get database stats
    stats = processor.db.get_stats()
    print(f"Database stats: {stats}")


if __name__ == "__main__":
    main()
