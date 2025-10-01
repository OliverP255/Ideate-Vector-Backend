#!/usr/bin/env python3
"""
Simple Wikipedia Processor
Uses a predefined list of high-quality Wikipedia articles to avoid API issues.
"""

import os
import sys
import json
import time
import requests
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sentence_transformers import SentenceTransformer
from backend.app.services.knowledge_database import KnowledgeDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleWikipediaProcessor:
    """Simple Wikipedia processor using predefined high-quality articles."""
    
    def __init__(self, output_dir: str = "data/wikipedia"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
        
        # Initialize database
        self.db = KnowledgeDatabase()
        
        # Predefined list of high-quality Wikipedia articles
        self.high_quality_articles = [
            # Science & Technology
            "Artificial intelligence", "Machine learning", "Deep learning", "Neural network",
            "Quantum computing", "Quantum mechanics", "Relativity", "Evolution",
            "DNA", "Genetics", "Climate change", "Global warming",
            "Renewable energy", "Solar energy", "Nuclear power", "Biotechnology",
            
            # Mathematics
            "Mathematics", "Calculus", "Linear algebra", "Statistics",
            "Probability theory", "Geometry", "Number theory", "Topology",
            "Game theory", "Optimization", "Algorithm", "Cryptography",
            
            # Computer Science
            "Computer science", "Programming language", "Software engineering",
            "Database", "Operating system", "Computer network", "Cybersecurity",
            "Data structure", "Algorithm complexity", "Distributed computing",
            
            # Medicine & Biology
            "Medicine", "Anatomy", "Physiology", "Pathology", "Pharmacology",
            "Immunology", "Oncology", "Cardiology", "Neurology", "Psychology",
            "Neuroscience", "Biochemistry", "Molecular biology", "Cell biology",
            
            # History
            "World War II", "World War I", "Ancient Rome", "Ancient Greece",
            "Middle Ages", "Renaissance", "Industrial Revolution", "American Revolution",
            "French Revolution", "Cold War", "History of science", "History of technology",
            
            # Philosophy
            "Philosophy", "Ethics", "Logic", "Metaphysics", "Epistemology",
            "Political philosophy", "Aesthetics", "Existentialism", "Stoicism",
            "Confucianism", "Buddhism", "Christianity", "Islam", "Judaism",
            
            # Economics
            "Economics", "Capitalism", "Socialism", "Market economy",
            "Supply and demand", "Inflation", "Unemployment", "Monetary policy",
            "International trade", "Development economics", "Behavioral economics",
            
            # Arts & Culture
            "Art", "Painting", "Sculpture", "Architecture", "Music",
            "Literature", "Poetry", "Drama", "Film", "Photography",
            "Renaissance art", "Classical music", "Jazz", "Rock music",
            
            # Geography
            "Earth", "Geography", "Geology", "Oceanography", "Meteorology",
            "Continents", "Mountains", "Rivers", "Oceans", "Climate",
            "Ecosystem", "Biodiversity", "Conservation biology", "Environmental science",
            
            # Social Sciences
            "Sociology", "Anthropology", "Political science", "International relations",
            "Democracy", "Government", "Law", "Human rights", "Social justice",
            "Education", "Psychology", "Cognitive science", "Linguistics"
        ]
        
        # Rate limiting
        self.request_delay = 0.2  # 200ms between requests
        
        logger.info(f"Simple Wikipedia Processor initialized with {len(self.high_quality_articles)} articles")
    
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
                'prop': 'extracts|info',
                'exintro': False,
                'explaintext': True,
                'inprop': 'url|timestamp'
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
            if not content or len(content) < 1000:  # Minimum content length
                return None
            
            # Calculate quality score (simplified)
            quality_score = self._calculate_quality_score(page_data, content)
            
            return {
                'title': title,
                'content': content,
                'quality_score': quality_score,
                'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                'last_modified': page_data.get('touched', ''),
                'page_id': page_id,
                'content_length': len(content)
            }
            
        except Exception as e:
            logger.error(f"Failed to get article content for '{title}': {e}")
            return None
    
    def _calculate_quality_score(self, page_data: Dict, content: str) -> float:
        """Calculate quality score for an article."""
        score = 0.5  # Base score for being in our curated list
        
        # Content length bonus
        content_length = len(content)
        if content_length > 10000:
            score += 0.3
        elif content_length > 5000:
            score += 0.2
        elif content_length > 2000:
            score += 0.1
        
        # Academic/scientific topics bonus
        academic_keywords = [
            'research', 'study', 'theory', 'method', 'analysis', 'experiment',
            'science', 'mathematics', 'physics', 'chemistry', 'biology',
            'history', 'philosophy', 'literature', 'art', 'culture'
        ]
        
        content_lower = content.lower()
        academic_matches = sum(1 for keyword in academic_keywords if keyword in content_lower)
        if academic_matches > 10:
            score += 0.1
        elif academic_matches > 5:
            score += 0.05
        
        return min(score, 1.0)  # Cap at 1.0
    
    def process_article(self, title: str) -> Optional[Dict[str, Any]]:
        """Process a single article."""
        try:
            # Get article content
            article_data = self.get_article_content(title)
            if not article_data:
                return None
            
            # Skip if quality is too low
            if article_data['quality_score'] < 0.4:
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
                'category': self._determine_category(article_data['title'], article_data['content']),
                'language': 'en',
                'embedding': embedding.tolist(),
                'metadata': {
                    'url': article_data['url'],
                    'content_length': article_data['content_length'],
                    'last_modified': article_data['last_modified']
                }
            }
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to process article '{title}': {e}")
            return None
    
    def _determine_category(self, title: str, content: str) -> str:
        """Determine the main category for an article."""
        text = f"{title} {content}".lower()
        
        if any(word in text for word in ['science', 'physics', 'chemistry', 'biology', 'mathematics']):
            return 'science'
        elif any(word in text for word in ['computer', 'programming', 'software', 'algorithm']):
            return 'technology'
        elif any(word in text for word in ['history', 'war', 'ancient', 'medieval']):
            return 'history'
        elif any(word in text for word in ['philosophy', 'ethics', 'logic', 'metaphysics']):
            return 'philosophy'
        elif any(word in text for word in ['art', 'music', 'literature', 'painting']):
            return 'culture'
        elif any(word in text for word in ['medicine', 'health', 'medical', 'disease']):
            return 'medicine'
        elif any(word in text for word in ['economics', 'market', 'trade', 'business']):
            return 'economics'
        else:
            return 'general'
    
    def process_articles_batch(self, titles: List[str], batch_size: int = 10) -> List[Dict[str, Any]]:
        """Process articles in batches."""
        logger.info(f"Processing {len(titles)} articles in batches of {batch_size}")
        
        all_documents = []
        processed_count = 0
        
        for i in range(0, len(titles), batch_size):
            batch_titles = titles[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(titles) + batch_size - 1)//batch_size}")
            
            # Process batch with threading
            with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced workers for API limits
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
    
    def run_processing(self, max_articles: int = None) -> Dict[str, Any]:
        """Run the Wikipedia processing pipeline."""
        if max_articles is None:
            max_articles = len(self.high_quality_articles)
        
        logger.info(f"Starting Wikipedia processing (max {max_articles} articles)")
        
        # Use predefined high-quality articles
        articles_to_process = self.high_quality_articles[:max_articles]
        logger.info(f"Processing {len(articles_to_process)} high-quality articles")
        
        # Process articles
        documents = self.process_articles_batch(articles_to_process, batch_size=5)
        
        # Get final stats
        stats = self.db.get_stats()
        
        logger.info(f"Wikipedia processing completed: {len(documents)} documents processed")
        return {
            'total_processed': len(documents),
            'articles_attempted': len(articles_to_process),
            'database_stats': stats
        }


def main():
    """Main function for testing."""
    processor = SimpleWikipediaProcessor()
    
    # Test with small batch
    results = processor.run_processing(max_articles=20)
    print(f"Processing results: {results}")
    
    # Get database stats
    stats = processor.db.get_stats()
    print(f"Database stats: {stats}")


if __name__ == "__main__":
    main()
