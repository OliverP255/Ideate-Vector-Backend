#!/usr/bin/env python3
"""
High-Quality Wikipedia Article Processor
Downloads and processes Featured Articles and Good Articles for the knowledge map.
"""

import os
import sys
import json
import time
import requests
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from bs4 import BeautifulSoup

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sentence_transformers import SentenceTransformer
from backend.app.services.knowledge_database import KnowledgeDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WikipediaQualityProcessor:
    """Process high-quality Wikipedia articles for the knowledge map."""
    
    def __init__(self, output_dir: str = "data/wikipedia"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.articles_dir = self.output_dir / "articles"
        self.embeddings_dir = self.output_dir / "embeddings"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.articles_dir, self.embeddings_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
        
        # Initialize database
        self.db = KnowledgeDatabase()
        
        # Quality criteria
        self.min_content_length = 2000  # Minimum characters
        self.min_references = 5         # Minimum references
        self.quality_categories = {
            'Featured Articles',
            'Good Articles',
            'A-Class Articles'
        }
        
        # Rate limiting
        self.request_delay = 0.1  # 100ms between requests
        
        logger.info("Wikipedia Quality Processor initialized")
    
    def get_featured_articles(self) -> List[str]:
        """Get list of Featured Articles from Wikipedia."""
        try:
            logger.info("Fetching Featured Articles list...")
            
            # Wikipedia API for Featured Articles
            url = "https://en.wikipedia.org/w/api.php"
            headers = {
                'User-Agent': 'KnowledgeMapProcessor/1.0 (https://github.com/your-repo)'
            }
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': 'Category:Featured articles',
                'cmlimit': 'max'
            }
            
            featured_articles = []
            continue_param = None
            
            while True:
                if continue_param:
                    params.update(continue_param)
                
                response = requests.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Extract article titles
                for member in data['query']['categorymembers']:
                    title = member['title']
                    if not title.startswith('Category:'):
                        featured_articles.append(title)
                
                # Check for continuation
                if 'continue' in data:
                    continue_param = data['continue']
                else:
                    break
            
            logger.info(f"Found {len(featured_articles)} Featured Articles")
            return featured_articles
            
        except Exception as e:
            logger.error(f"Failed to get Featured Articles: {e}")
            return []
    
    def get_good_articles(self) -> List[str]:
        """Get list of Good Articles from Wikipedia."""
        try:
            logger.info("Fetching Good Articles list...")
            
            url = "https://en.wikipedia.org/w/api.php"
            headers = {
                'User-Agent': 'KnowledgeMapProcessor/1.0 (https://github.com/your-repo)'
            }
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': 'Category:Good articles',
                'cmlimit': 'max'
            }
            
            good_articles = []
            continue_param = None
            
            while True:
                if continue_param:
                    params.update(continue_param)
                
                response = requests.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Extract article titles
                for member in data['query']['categorymembers']:
                    title = member['title']
                    if not title.startswith('Category:'):
                        good_articles.append(title)
                
                # Check for continuation
                if 'continue' in data:
                    continue_param = data['continue']
                else:
                    break
            
            logger.info(f"Found {len(good_articles)} Good Articles")
            return good_articles
            
        except Exception as e:
            logger.error(f"Failed to get Good Articles: {e}")
            return []
    
    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        """Get article content and metadata from Wikipedia."""
        try:
            time.sleep(self.request_delay)  # Rate limiting
            
            url = "https://en.wikipedia.org/w/api.php"
            headers = {
                'User-Agent': 'KnowledgeMapProcessor/1.0 (https://github.com/your-repo)'
            }
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts|info|categories',
                'exintro': False,
                'explaintext': True,
                'inprop': 'url|timestamp',
                'cllimit': 'max'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            pages = data['query']['pages']
            page_id = list(pages.keys())[0]
            
            if page_id == '-1':  # Page not found
                return None
            
            page_data = pages[page_id]
            
            # Extract content
            content = page_data.get('extract', '')
            if not content or len(content) < self.min_content_length:
                return None
            
            # Get categories
            categories = []
            if 'categories' in page_data:
                categories = [cat['title'] for cat in page_data['categories']]
            
            # Determine quality level
            quality_score = self._calculate_quality_score(page_data, categories, content)
            
            # Get references count (simplified)
            references_count = self._estimate_references_count(content)
            
            return {
                'title': title,
                'content': content,
                'categories': categories,
                'quality_score': quality_score,
                'references_count': references_count,
                'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                'last_modified': page_data.get('touched', ''),
                'page_id': page_id
            }
            
        except Exception as e:
            logger.error(f"Failed to get article content for '{title}': {e}")
            return None
    
    def _calculate_quality_score(self, page_data: Dict, categories: List[str], content: str) -> float:
        """Calculate quality score for an article."""
        score = 0.0
        
        # Base score for Featured/Good articles
        category_titles = ' '.join(categories)
        if 'Featured articles' in category_titles:
            score += 0.4
        elif 'Good articles' in category_titles:
            score += 0.3
        elif 'A-Class articles' in category_titles:
            score += 0.25
        
        # Content length bonus
        content_length = len(content)
        if content_length > 10000:
            score += 0.2
        elif content_length > 5000:
            score += 0.15
        elif content_length > 2000:
            score += 0.1
        
        # References bonus (estimated)
        ref_count = self._estimate_references_count(content)
        if ref_count > 50:
            score += 0.2
        elif ref_count > 20:
            score += 0.15
        elif ref_count > 10:
            score += 0.1
        elif ref_count >= self.min_references:
            score += 0.05
        
        # Academic/scientific topics bonus
        academic_keywords = [
            'science', 'research', 'study', 'theory', 'method', 'analysis',
            'mathematics', 'physics', 'chemistry', 'biology', 'medicine',
            'history', 'philosophy', 'literature', 'art', 'culture'
        ]
        
        content_lower = content.lower()
        academic_matches = sum(1 for keyword in academic_keywords if keyword in content_lower)
        if academic_matches > 10:
            score += 0.1
        elif academic_matches > 5:
            score += 0.05
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _estimate_references_count(self, content: str) -> int:
        """Estimate the number of references in the content."""
        # Look for reference patterns
        ref_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\[citation needed\]',
            r'\[who\]',
            r'\[when\]'
        ]
        
        total_refs = 0
        for pattern in ref_patterns:
            matches = re.findall(pattern, content)
            total_refs += len(matches)
        
        return total_refs
    
    def process_article(self, title: str) -> Optional[Dict[str, Any]]:
        """Process a single article."""
        try:
            # Get article content
            article_data = self.get_article_content(title)
            if not article_data:
                return None
            
            # Skip if quality is too low
            if article_data['quality_score'] < 0.3:
                return None
            
            # Generate embedding
            combined_text = f"{article_data['title']}\n\n{article_data['content']}"
            embedding = self.embedding_model.encode([combined_text])[0]
            
            # Create document
            document = {
                'id': f"wikipedia_{article_data['page_id']}",
                'title': article_data['title'],
                'content': article_data['content'],
                'source': 'wikipedia',
                'quality_score': article_data['quality_score'],
                'category': self._determine_category(article_data['categories']),
                'language': 'en',
                'embedding': embedding.tolist(),
                'metadata': {
                    'url': article_data['url'],
                    'references_count': article_data['references_count'],
                    'categories': article_data['categories'],
                    'last_modified': article_data['last_modified'],
                    'content_length': len(article_data['content'])
                }
            }
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to process article '{title}': {e}")
            return None
    
    def _determine_category(self, categories: List[str]) -> str:
        """Determine the main category for an article."""
        category_mapping = {
            'science': ['science', 'physics', 'chemistry', 'biology', 'mathematics'],
            'history': ['history', 'historical'],
            'culture': ['culture', 'art', 'music', 'literature', 'film'],
            'geography': ['geography', 'countries', 'cities', 'places'],
            'technology': ['technology', 'computing', 'engineering'],
            'medicine': ['medicine', 'health', 'medical'],
            'philosophy': ['philosophy', 'religion', 'ethics'],
            'sports': ['sports', 'football', 'basketball', 'tennis'],
            'politics': ['politics', 'government', 'law', 'legal'],
            'economics': ['economics', 'business', 'finance']
        }
        
        categories_text = ' '.join(categories).lower()
        
        for main_category, keywords in category_mapping.items():
            if any(keyword in categories_text for keyword in keywords):
                return main_category
        
        return 'general'
    
    def process_articles_batch(self, titles: List[str], batch_size: int = 100) -> List[Dict[str, Any]]:
        """Process articles in batches."""
        logger.info(f"Processing {len(titles)} articles in batches of {batch_size}")
        
        all_documents = []
        processed_count = 0
        
        for i in range(0, len(titles), batch_size):
            batch_titles = titles[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(titles) + batch_size - 1)//batch_size}")
            
            # Process batch with threading
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_title = {executor.submit(self.process_article, title): title 
                                 for title in batch_titles}
                
                batch_documents = []
                for future in as_completed(future_to_title):
                    title = future_to_title[future]
                    try:
                        document = future.result()
                        if document:
                            batch_documents.append(document)
                            processed_count += 1
                    except Exception as e:
                        logger.error(f"Error processing {title}: {e}")
                
                # Add batch to database
                if batch_documents:
                    self.db.batch_add_documents(batch_documents)
                    all_documents.extend(batch_documents)
                    logger.info(f"Added {len(batch_documents)} documents to database")
                
                # Progress update
                logger.info(f"Processed {processed_count}/{len(titles)} articles")
        
        return all_documents
    
    def run_processing(self, max_articles: int = 10000) -> Dict[str, Any]:
        """Run the complete Wikipedia processing pipeline."""
        logger.info(f"Starting Wikipedia processing (max {max_articles} articles)")
        
        # Get high-quality articles
        featured_articles = self.get_featured_articles()
        good_articles = self.get_good_articles()
        
        # Combine and deduplicate
        all_articles = list(set(featured_articles + good_articles))
        logger.info(f"Total unique high-quality articles: {len(all_articles)}")
        
        # Limit to max_articles
        if len(all_articles) > max_articles:
            all_articles = all_articles[:max_articles]
            logger.info(f"Limited to {max_articles} articles")
        
        # Process articles
        documents = self.process_articles_batch(all_articles, batch_size=50)
        
        # Get final stats
        stats = self.db.get_stats()
        
        logger.info(f"Wikipedia processing completed: {len(documents)} documents processed")
        return {
            'total_processed': len(documents),
            'featured_articles': len(featured_articles),
            'good_articles': len(good_articles),
            'database_stats': stats
        }


def main():
    """Main function for testing."""
    processor = WikipediaQualityProcessor()
    
    # Test with small batch
    results = processor.run_processing(max_articles=100)
    print(f"Processing results: {results}")
    
    # Get database stats
    stats = processor.db.get_stats()
    print(f"Database stats: {stats}")


if __name__ == "__main__":
    main()
