#!/usr/bin/env python3
"""
Breadth-Focused Knowledge Processor
Focuses on breadth and factual accuracy rather than academic depth.
Prioritizes well-established, reliable knowledge across all domains.
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


class BreadthFocusedProcessor:
    """Processor focused on breadth and factual accuracy across all knowledge domains."""
    
    def __init__(self, output_dir: str = "data/knowledge_map"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
        
        # Initialize database
        self.db = KnowledgeDatabase()
        
        # Comprehensive knowledge domains for breadth
        self.knowledge_domains = {
            # Natural Sciences
            'physics': [
                'Physics', 'Mechanics', 'Thermodynamics', 'Electromagnetism', 
                'Optics', 'Quantum physics', 'Relativity', 'Astrophysics',
                'Nuclear physics', 'Particle physics', 'Solid state physics'
            ],
            'chemistry': [
                'Chemistry', 'Organic chemistry', 'Inorganic chemistry', 'Physical chemistry',
                'Biochemistry', 'Analytical chemistry', 'Chemical bonding', 'Periodic table',
                'Chemical reactions', 'Chemical equilibrium'
            ],
            'biology': [
                'Biology', 'Cell biology', 'Genetics', 'Evolution', 'Ecology',
                'Botany', 'Zoology', 'Microbiology', 'Anatomy', 'Physiology',
                'Biochemistry', 'Molecular biology', 'Immunology', 'Neuroscience'
            ],
            'earth_sciences': [
                'Geology', 'Geography', 'Meteorology', 'Oceanography', 'Climatology',
                'Environmental science', 'Ecology', 'Conservation biology',
                'Natural disasters', 'Climate change', 'Weather patterns'
            ],
            
            # Mathematics
            'mathematics': [
                'Mathematics', 'Algebra', 'Geometry', 'Calculus', 'Statistics',
                'Probability', 'Number theory', 'Topology', 'Linear algebra',
                'Differential equations', 'Mathematical analysis'
            ],
            
            # Technology & Engineering
            'technology': [
                'Technology', 'Computer science', 'Artificial intelligence', 'Machine learning',
                'Programming', 'Software engineering', 'Hardware', 'Networking',
                'Cybersecurity', 'Data science', 'Robotics', 'Automation'
            ],
            'engineering': [
                'Engineering', 'Civil engineering', 'Mechanical engineering', 'Electrical engineering',
                'Chemical engineering', 'Aerospace engineering', 'Biomedical engineering',
                'Materials science', 'Manufacturing', 'Design', 'Innovation'
            ],
            
            # Social Sciences
            'psychology': [
                'Psychology', 'Cognitive psychology', 'Behavioral psychology', 'Social psychology',
                'Developmental psychology', 'Clinical psychology', 'Neuroscience',
                'Human behavior', 'Mental health', 'Learning', 'Memory'
            ],
            'economics': [
                'Economics', 'Microeconomics', 'Macroeconomics', 'International trade',
                'Finance', 'Banking', 'Investment', 'Market economy', 'Supply and demand',
                'Economic policy', 'Development economics', 'Behavioral economics'
            ],
            'sociology': [
                'Sociology', 'Social structure', 'Culture', 'Society', 'Social change',
                'Social psychology', 'Demographics', 'Social inequality', 'Social movements',
                'Community', 'Social networks'
            ],
            'political_science': [
                'Political science', 'Government', 'Democracy', 'Politics', 'Political theory',
                'International relations', 'Public policy', 'Law', 'Constitution',
                'Human rights', 'Political systems'
            ],
            
            # Humanities
            'history': [
                'History', 'World history', 'Ancient history', 'Medieval history',
                'Modern history', 'Historical events', 'Historical figures', 'Civilization',
                'Historical periods', 'Historical analysis'
            ],
            'philosophy': [
                'Philosophy', 'Ethics', 'Logic', 'Metaphysics', 'Epistemology',
                'Political philosophy', 'Aesthetics', 'Philosophy of science',
                'Moral philosophy', 'Philosophical thinking'
            ],
            'literature': [
                'Literature', 'Poetry', 'Fiction', 'Non-fiction', 'Literary analysis',
                'Authors', 'Literary movements', 'Literary criticism', 'Creative writing',
                'Literary genres'
            ],
            'arts': [
                'Art', 'Visual arts', 'Painting', 'Sculpture', 'Photography',
                'Music', 'Dance', 'Theater', 'Film', 'Architecture',
                'Art history', 'Artistic movements', 'Creativity'
            ],
            
            # Medicine & Health
            'medicine': [
                'Medicine', 'Medical science', 'Anatomy', 'Physiology', 'Pathology',
                'Pharmacology', 'Surgery', 'Diagnosis', 'Treatment', 'Public health',
                'Medical research', 'Healthcare', 'Diseases', 'Therapeutics'
            ],
            'nutrition': [
                'Nutrition', 'Diet', 'Food science', 'Vitamins', 'Minerals',
                'Healthy eating', 'Metabolism', 'Nutritional science', 'Food safety',
                'Dietary guidelines'
            ],
            
            # Practical Knowledge
            'agriculture': [
                'Agriculture', 'Farming', 'Crop science', 'Animal husbandry',
                'Agricultural technology', 'Food production', 'Sustainable agriculture',
                'Agricultural economics', 'Rural development'
            ],
            'business': [
                'Business', 'Management', 'Entrepreneurship', 'Marketing', 'Finance',
                'Operations', 'Human resources', 'Strategy', 'Leadership', 'Organizations'
            ],
            'education': [
                'Education', 'Learning', 'Teaching', 'Educational psychology', 'Curriculum',
                'Educational technology', 'Learning theories', 'Educational policy',
                'Academic achievement'
            ]
        }
        
        # Factual accuracy indicators (higher score = more reliable)
        self.factual_accuracy_indicators = {
            'established_facts': 1.0,      # Well-established scientific facts
            'historical_events': 0.9,      # Documented historical events
            'basic_science': 0.9,          # Fundamental scientific principles
            'mathematical_concepts': 0.95, # Mathematical facts and theorems
            'geographical_facts': 0.9,     # Geographical and geological facts
            'medical_basics': 0.8,         # Basic medical knowledge
            'economic_principles': 0.7,    # Economic theories and principles
            'cultural_facts': 0.6,         # Cultural and social facts
            'current_events': 0.3,         # Recent events (less reliable)
            'opinions': 0.2,               # Opinions and subjective content
            'speculation': 0.1             # Speculative or unproven content
        }
        
        # Rate limiting
        self.request_delay = 0.3  # 300ms between requests
        
        logger.info(f"Breadth-Focused Processor initialized with {len(self.knowledge_domains)} knowledge domains")
    
    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        """Get article content and metadata from Wikipedia."""
        try:
            time.sleep(self.request_delay)  # Rate limiting
            
            url = "https://en.wikipedia.org/w/api.php"
            headers = {
                'User-Agent': 'BreadthFocusedProcessor/1.0 (https://github.com/your-repo)'
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
            if not content or len(content) < 500:  # Minimum content length
                return None
            
            # Calculate factual accuracy score
            factual_score = self._calculate_factual_accuracy(content, title)
            
            return {
                'title': title,
                'content': content,
                'factual_score': factual_score,
                'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                'last_modified': page_data.get('touched', ''),
                'page_id': page_id,
                'content_length': len(content)
            }
            
        except Exception as e:
            logger.error(f"Failed to get article content for '{title}': {e}")
            return None
    
    def _calculate_factual_accuracy(self, content: str, title: str) -> float:
        """Calculate factual accuracy score based on content analysis."""
        content_lower = content.lower()
        title_lower = title.lower()
        
        base_score = 0.5  # Start with neutral score
        
        # Check for established facts
        if any(word in content_lower for word in [
            'scientific', 'research', 'study', 'evidence', 'data', 'analysis',
            'theory', 'principle', 'law', 'formula', 'equation', 'measurement'
        ]):
            base_score += 0.2
        
        # Check for historical facts
        if any(word in content_lower for word in [
            'historical', 'history', 'century', 'decade', 'era', 'period',
            'ancient', 'medieval', 'modern', 'traditional', 'established'
        ]):
            base_score += 0.15
        
        # Check for mathematical/scientific facts
        if any(word in title_lower for word in [
            'mathematics', 'physics', 'chemistry', 'biology', 'geometry',
            'algebra', 'calculus', 'statistics', 'probability'
        ]):
            base_score += 0.2
        
        # Check for basic knowledge
        if any(word in content_lower for word in [
            'fundamental', 'basic', 'elementary', 'foundation', 'core',
            'essential', 'primary', 'standard', 'conventional'
        ]):
            base_score += 0.15
        
        # Penalize speculative content
        if any(word in content_lower for word in [
            'speculation', 'hypothesis', 'theory suggests', 'may be', 'could be',
            'possibly', 'perhaps', 'unproven', 'controversial', 'debated'
        ]):
            base_score -= 0.1
        
        # Penalize opinion-based content
        if any(word in content_lower for word in [
            'opinion', 'believe', 'think', 'feel', 'personal', 'subjective',
            'argue', 'claim', 'assert', 'contend'
        ]):
            base_score -= 0.1
        
        # Bonus for well-established topics
        established_topics = [
            'mathematics', 'physics', 'chemistry', 'biology', 'history',
            'geography', 'anatomy', 'physiology', 'geology', 'astronomy'
        ]
        
        if any(topic in title_lower for topic in established_topics):
            base_score += 0.1
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, base_score))
    
    def process_article(self, title: str, domain: str) -> Optional[Dict[str, Any]]:
        """Process a single article."""
        try:
            # Get article content
            article_data = self.get_article_content(title)
            if not article_data:
                return None
            
            # Skip if factual accuracy is too low
            if article_data['factual_score'] < 0.4:
                return None
            
            # Generate embedding
            combined_text = f"{article_data['title']}\n\n{article_data['content']}"
            embedding = self.embedding_model.encode([combined_text])[0]
            
            # Create document
            document = {
                'id': f"breadth_{article_data['page_id']}",
                'title': article_data['title'],
                'content': article_data['content'],
                'source': 'wikipedia_breadth',
                'quality_score': article_data['factual_score'],  # Use factual score as quality
                'category': domain,
                'language': 'en',
                'embedding': embedding.tolist(),
                'metadata': {
                    'url': article_data['url'],
                    'content_length': article_data['content_length'],
                    'last_modified': article_data['last_modified'],
                    'domain': domain,
                    'factual_accuracy': article_data['factual_score']
                }
            }
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to process article '{title}': {e}")
            return None
    
    def process_domain(self, domain: str, topics: List[str], max_per_topic: int = 5) -> List[Dict[str, Any]]:
        """Process all topics in a knowledge domain."""
        logger.info(f"Processing {domain} domain with {len(topics)} topics")
        
        all_documents = []
        processed_count = 0
        
        for topic in topics:
            try:
                document = self.process_article(topic, domain)
                if document:
                    all_documents.append(document)
                    processed_count += 1
                    logger.info(f"Processed {topic} (factual score: {document['quality_score']:.2f})")
                
                # Limit per topic
                if processed_count >= max_per_topic:
                    break
                    
            except Exception as e:
                logger.error(f"Error processing {topic}: {e}")
        
        logger.info(f"Completed {domain}: {len(all_documents)} documents")
        return all_documents
    
    def run_breadth_processing(self, max_documents: int = 1000) -> Dict[str, Any]:
        """Run breadth-focused processing across all knowledge domains."""
        logger.info(f"Starting breadth-focused processing (max {max_documents} documents)")
        
        all_documents = []
        domain_results = {}
        
        # Process each domain
        for domain, topics in self.knowledge_domains.items():
            logger.info(f"=== PROCESSING {domain.upper()} DOMAIN ===")
            
            domain_docs = self.process_domain(domain, topics, max_per_topic=3)
            
            if domain_docs:
                # Add to database
                self.db.batch_add_documents(domain_docs)
                all_documents.extend(domain_docs)
                domain_results[domain] = len(domain_docs)
                
                logger.info(f"Added {len(domain_docs)} documents from {domain}")
            
            # Check if we've reached the limit
            if len(all_documents) >= max_documents:
                logger.info(f"Reached document limit: {max_documents}")
                break
        
        # Get final stats
        stats = self.db.get_stats()
        
        logger.info(f"Breadth processing completed: {len(all_documents)} documents")
        return {
            'total_processed': len(all_documents),
            'domain_breakdown': domain_results,
            'database_stats': stats
        }


def main():
    """Main function for testing."""
    processor = BreadthFocusedProcessor()
    
    # Test with small batch
    results = processor.run_breadth_processing(max_documents=100)
    print(f"Processing results: {results}")
    
    # Get database stats
    stats = processor.db.get_stats()
    print(f"Database stats: {stats}")


if __name__ == "__main__":
    main()
