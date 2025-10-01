#!/usr/bin/env python3
"""
Citation-Driven High-Impact Research Papers Processor
Processes 1M papers sorted by citation count from the last 10 years with maximum breadth.
"""

import os
import sys
import json
import time
import requests
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
from dataclasses import dataclass

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sentence_transformers import SentenceTransformer
from backend.app.services.knowledge_database import KnowledgeDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Paper:
    """Research paper data structure."""
    id: str
    title: str
    abstract: str
    full_text: str
    authors: List[str]
    publication_date: str
    source: str
    citation_count: int
    impact_score: float
    discipline: str
    journal: str
    doi: str
    url: str
    keywords: List[str]

class CitationAggregator:
    """Aggregates citation data from multiple sources."""
    
    def __init__(self):
        self.semantic_scholar_api = "https://api.semanticscholar.org/graph/v1/paper"
        self.crossref_api = "https://api.crossref.org/works"
        self.arxiv_api = "http://export.arxiv.org/api/query"
        
    def get_paper_citations(self, paper_id: str, source: str) -> Dict[str, Any]:
        """Get citation data for a paper from multiple sources."""
        citations = {}
        
        try:
            if source == "arxiv":
                citations.update(self._get_arxiv_citations(paper_id))
            elif source == "pubmed":
                citations.update(self._get_pubmed_citations(paper_id))
            else:
                citations.update(self._get_semantic_scholar_citations(paper_id))
                
        except Exception as e:
            logger.warning(f"Failed to get citations for {paper_id}: {e}")
            
        return citations
    
    def _get_semantic_scholar_citations(self, paper_id: str) -> Dict[str, Any]:
        """Get citations from Semantic Scholar API."""
        try:
            url = f"{self.semantic_scholar_api}/{paper_id}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    'citation_count': data.get('citationCount', 0),
                    'influence_score': data.get('influentialCitationCount', 0),
                    'fields': data.get('fieldsOfStudy', []),
                    'venue': data.get('venue', ''),
                    'year': data.get('year', 0)
                }
        except Exception as e:
            logger.warning(f"Semantic Scholar API error for {paper_id}: {e}")
        return {'citation_count': 0, 'influence_score': 0, 'fields': [], 'venue': '', 'year': 0}
    
    def _get_arxiv_citations(self, paper_id: str) -> Dict[str, Any]:
        """Get citations from arXiv API."""
        try:
            # arXiv papers typically have lower citation counts initially
            # We'll use a simple estimation based on paper age and category
            return {
                'citation_count': 10,  # Placeholder - would need actual arXiv citation API
                'influence_score': 5,
                'fields': ['Computer Science'],  # Default for arXiv
                'venue': 'arXiv',
                'year': 2023
            }
        except Exception as e:
            logger.warning(f"arXiv API error for {paper_id}: {e}")
        return {'citation_count': 0, 'influence_score': 0, 'fields': [], 'venue': 'arXiv', 'year': 0}
    
    def _get_pubmed_citations(self, paper_id: str) -> Dict[str, Any]:
        """Get citations from PubMed API."""
        try:
            # PubMed papers typically have higher citation counts
            return {
                'citation_count': 50,  # Placeholder - would need actual PubMed citation API
                'influence_score': 25,
                'fields': ['Medicine', 'Biology'],
                'venue': 'PubMed',
                'year': 2023
            }
        except Exception as e:
            logger.warning(f"PubMed API error for {paper_id}: {e}")
        return {'citation_count': 0, 'influence_score': 0, 'fields': [], 'venue': 'PubMed', 'year': 0}

class ImpactCalculator:
    """Calculates impact scores for papers."""
    
    def __init__(self):
        self.current_year = 2024
        
    def calculate_impact_score(self, paper: Paper) -> float:
        """Calculate weighted impact score based on multiple factors."""
        try:
            # Base citation count
            base_citations = paper.citation_count
            
            # Recency weight (recent papers get higher weight)
            year = int(paper.publication_date.split('-')[0]) if paper.publication_date else 2020
            recency_weight = self._get_recency_weight(year)
            
            # Field weight (some fields have naturally higher citation rates)
            field_weight = self._get_field_weight(paper.discipline)
            
            # Journal weight (top-tier journals get higher weight)
            journal_weight = self._get_journal_weight(paper.journal)
            
            # Calculate final impact score
            impact_score = (base_citations * recency_weight * field_weight * journal_weight)
            
            return impact_score
            
        except Exception as e:
            logger.warning(f"Error calculating impact score for {paper.id}: {e}")
            return 0.0
    
    def _get_recency_weight(self, year: int) -> float:
        """Get weight based on publication year (recent = higher weight)."""
        if year >= 2024:
            return 1.0
        elif year >= 2023:
            return 0.95
        elif year >= 2022:
            return 0.90
        elif year >= 2021:
            return 0.85
        elif year >= 2020:
            return 0.80
        elif year >= 2019:
            return 0.75
        elif year >= 2018:
            return 0.70
        elif year >= 2017:
            return 0.65
        elif year >= 2016:
            return 0.60
        elif year >= 2015:
            return 0.55
        elif year >= 2014:
            return 0.50
        else:
            return 0.0  # Exclude papers older than 2014
    
    def _get_field_weight(self, discipline: str) -> float:
        """Get weight based on academic field."""
        field_weights = {
            'computer_science': 1.0,
            'medicine': 1.0,
            'biology': 0.9,
            'physics': 0.9,
            'chemistry': 0.8,
            'engineering': 0.8,
            'mathematics': 0.7,
            'economics': 0.7,
            'psychology': 0.6,
            'social_sciences': 0.6,
            'earth_sciences': 0.6,
            'arts': 0.4,
            'humanities': 0.4
        }
        return field_weights.get(discipline.lower(), 0.5)
    
    def _get_journal_weight(self, journal: str) -> float:
        """Get weight based on journal prestige."""
        if not journal:
            return 0.5
            
        # Top-tier journals
        top_journals = ['Nature', 'Science', 'Cell', 'NEJM', 'Lancet', 'JAMA']
        if any(top_journal.lower() in journal.lower() for top_journal in top_journals):
            return 1.0
        
        # High-impact journals
        high_impact = ['Nature Medicine', 'Nature Biotechnology', 'Cell Metabolism', 'PNAS']
        if any(hi_journal.lower() in journal.lower() for hi_journal in high_impact):
            return 0.9
        
        # Mid-tier journals
        mid_tier = ['PLOS', 'Scientific Reports', 'BMC', 'Frontiers']
        if any(mid_journal.lower() in journal.lower() for mid_journal in mid_tier):
            return 0.7
        
        return 0.5

class BreadthEnforcer:
    """Enforces breadth across academic disciplines."""
    
    def __init__(self):
        # Target distribution for 1M papers
        self.discipline_targets = {
            'computer_science': 150000,    # 15%
            'medicine': 150000,            # 15%
            'biology': 100000,             # 10%
            'physics': 100000,             # 10%
            'chemistry': 100000,           # 10%
            'engineering': 100000,         # 10%
            'mathematics': 80000,          # 8%
            'economics': 80000,            # 8%
            'psychology': 60000,           # 6%
            'social_sciences': 60000,      # 6%
            'earth_sciences': 60000,       # 6%
            'arts': 40000,                 # 4%
            'humanities': 40000,           # 4%
            'other': 20000                 # 2%
        }
        
        self.discipline_counts = {k: 0 for k in self.discipline_targets}
        self.total_target = 1000000
    
    def select_papers(self, all_papers: List[Paper]) -> List[Paper]:
        """Select papers maintaining citation rank while enforcing breadth."""
        selected = []
        
        # Sort papers by impact score (highest first)
        sorted_papers = sorted(all_papers, key=lambda x: x.impact_score, reverse=True)
        
        logger.info(f"Starting selection from {len(sorted_papers)} papers")
        
        for paper in sorted_papers:
            discipline = paper.discipline.lower()
            
            # Check if we still need papers from this discipline
            if self.discipline_counts[discipline] < self.discipline_targets[discipline]:
                selected.append(paper)
                self.discipline_counts[discipline] += 1
                
                if len(selected) >= self.total_target:
                    break
                    
                # Log progress every 1000 papers
                if len(selected) % 1000 == 0:
                    logger.info(f"Selected {len(selected)} papers. Discipline counts: {dict(list(self.discipline_counts.items())[:5])}")
        
        logger.info(f"Final selection: {len(selected)} papers")
        logger.info(f"Discipline distribution: {self.discipline_counts}")
        
        return selected

class CitationDrivenProcessor:
    """Main processor for citation-driven paper collection."""
    
    def __init__(self, output_dir: str = "data/knowledge_map"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
        self.db = KnowledgeDatabase()
        self.citation_aggregator = CitationAggregator()
        self.impact_calculator = ImpactCalculator()
        self.breadth_enforcer = BreadthEnforcer()
        
        # Processing targets
        self.target_papers = 1000000
        self.papers_processed = 0
        
        # Discipline categories for breadth
        self.disciplines = {
            'computer_science': [
                'artificial intelligence', 'machine learning', 'deep learning', 'neural networks',
                'computer vision', 'natural language processing', 'robotics', 'cybersecurity',
                'data science', 'software engineering', 'algorithms', 'distributed systems'
            ],
            'medicine': [
                'cancer research', 'immunology', 'cardiology', 'neurology', 'oncology',
                'pharmacology', 'surgery', 'radiology', 'pathology', 'genetics',
                'clinical trials', 'medical imaging', 'drug discovery'
            ],
            'biology': [
                'molecular biology', 'cell biology', 'genetics', 'evolution', 'ecology',
                'microbiology', 'biochemistry', 'biotechnology', 'genomics', 'proteomics',
                'synthetic biology', 'bioinformatics'
            ],
            'physics': [
                'quantum physics', 'particle physics', 'astrophysics', 'condensed matter',
                'optics', 'thermodynamics', 'electromagnetism', 'relativity', 'nuclear physics',
                'plasma physics', 'cosmology'
            ],
            'chemistry': [
                'organic chemistry', 'inorganic chemistry', 'physical chemistry', 'analytical chemistry',
                'biochemistry', 'materials chemistry', 'catalysis', 'polymer chemistry',
                'environmental chemistry', 'medicinal chemistry'
            ],
            'engineering': [
                'civil engineering', 'mechanical engineering', 'electrical engineering',
                'chemical engineering', 'aerospace engineering', 'biomedical engineering',
                'materials engineering', 'environmental engineering', 'computer engineering'
            ],
            'mathematics': [
                'algebra', 'geometry', 'calculus', 'statistics', 'probability', 'number theory',
                'topology', 'differential equations', 'linear algebra', 'mathematical analysis',
                'applied mathematics', 'discrete mathematics'
            ],
            'economics': [
                'microeconomics', 'macroeconomics', 'econometrics', 'behavioral economics',
                'development economics', 'international trade', 'finance', 'game theory',
                'industrial organization', 'labor economics'
            ],
            'psychology': [
                'cognitive psychology', 'social psychology', 'clinical psychology',
                'developmental psychology', 'neuroscience', 'behavioral psychology',
                'experimental psychology', 'educational psychology'
            ],
            'social_sciences': [
                'sociology', 'political science', 'anthropology', 'geography',
                'communication studies', 'international relations', 'public policy',
                'criminology', 'social work'
            ],
            'earth_sciences': [
                'geology', 'meteorology', 'oceanography', 'climatology', 'environmental science',
                'ecology', 'conservation biology', 'geophysics', 'hydrology'
            ],
            'arts': [
                'visual arts', 'performing arts', 'music', 'dance', 'theater',
                'film studies', 'art history', 'design', 'architecture'
            ],
            'humanities': [
                'philosophy', 'literature', 'history', 'linguistics', 'religious studies',
                'classics', 'cultural studies', 'ethics', 'aesthetics'
            ]
        }
    
    def generate_mock_papers(self, discipline: str, topic: str, count: int = 10) -> List[Paper]:
        """Generate mock papers for testing (replace with real API calls)."""
        papers = []
        
        for i in range(count):
            # Generate realistic paper data
            year = 2020 + (i % 5)  # Papers from 2020-2024
            citation_count = max(1, 100 - i * 5)  # Decreasing citation counts
            
            paper = Paper(
                id=f"{discipline}_{topic}_{i}_{year}",
                title=f"{topic.title()} Research: Advances in {discipline.replace('_', ' ').title()}",
                abstract=f"This paper presents novel research in {topic} within the field of {discipline.replace('_', ' ')}. We demonstrate significant advances in methodology and applications.",
                full_text=f"Full research paper content about {topic} in {discipline.replace('_', ' ')}. This paper contains detailed methodology, results, and conclusions.",
                authors=[f"Author {i+1}", f"Co-Author {i+1}"],
                publication_date=f"{year}-01-01",
                source="mock_database",
                citation_count=citation_count,
                impact_score=0.0,  # Will be calculated
                discipline=discipline,
                journal=f"{discipline.replace('_', ' ').title()} Journal",
                doi=f"10.1000/{discipline}_{i}",
                url=f"https://example.com/{discipline}_{i}",
                keywords=[topic, discipline.replace('_', ' '), "research"]
            )
            
            # Calculate impact score
            paper.impact_score = self.impact_calculator.calculate_impact_score(paper)
            papers.append(paper)
        
        return papers
    
    def process_discipline(self, discipline: str, topics: List[str]) -> List[Paper]:
        """Process all topics in a discipline."""
        logger.info(f"Processing {discipline} with {len(topics)} topics")
        
        all_papers = []
        
        for topic in topics:
            try:
                # Generate papers for this topic (replace with real API calls)
                papers = self.generate_mock_papers(discipline, topic, count=50)
                all_papers.extend(papers)
                
                logger.info(f"Generated {len(papers)} papers for {topic}")
                
            except Exception as e:
                logger.error(f"Error processing {topic}: {e}")
        
        logger.info(f"Completed {discipline}: {len(all_papers)} papers")
        return all_papers
    
    def run_citation_driven_processing(self) -> Dict[str, Any]:
        """Run the complete citation-driven processing pipeline."""
        logger.info("Starting citation-driven processing for 1M papers")
        
        all_papers = []
        discipline_results = {}
        
        # Process each discipline
        for discipline, topics in self.disciplines.items():
            logger.info(f"=== PROCESSING {discipline.upper()} ===")
            
            discipline_papers = self.process_discipline(discipline, topics)
            all_papers.extend(discipline_papers)
            discipline_results[discipline] = len(discipline_papers)
            
            logger.info(f"Added {len(discipline_papers)} papers from {discipline}")
            
            # Check if we have enough papers
            if len(all_papers) >= 2000000:  # Generate more than needed for selection
                logger.info(f"Generated sufficient papers ({len(all_papers)}), proceeding to selection")
                break
        
        # Apply breadth enforcement and citation-based selection
        logger.info("Applying breadth enforcement and citation-based selection")
        selected_papers = self.breadth_enforcer.select_papers(all_papers)
        
        # Process selected papers
        logger.info(f"Processing {len(selected_papers)} selected papers")
        processed_count = 0
        
        for paper in selected_papers:
            try:
                # Generate embedding
                combined_text = f"{paper.title}\n\n{paper.abstract}"
                embedding = self.embedding_model.encode([combined_text])[0]
                
                # Create document
                document = {
                    'id': paper.id,
                    'title': paper.title,
                    'content': paper.abstract,
                    'source': paper.source,
                    'quality_score': paper.impact_score / 100.0,  # Normalize to 0-1
                    'category': paper.discipline,
                    'language': 'en',
                    'embedding': embedding.tolist(),
                    'metadata': {
                        'authors': paper.authors,
                        'publication_date': paper.publication_date,
                        'citation_count': paper.citation_count,
                        'impact_score': paper.impact_score,
                        'journal': paper.journal,
                        'doi': paper.doi,
                        'url': paper.url,
                        'keywords': paper.keywords,
                        'full_text': paper.full_text
                    }
                }
                
                # Add to database
                self.db.batch_add_documents([document])
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} papers")
                
            except Exception as e:
                logger.error(f"Error processing paper {paper.id}: {e}")
        
        logger.info(f"Citation-driven processing completed: {processed_count} papers processed")
        
        return {
            'total_papers': processed_count,
            'discipline_results': discipline_results,
            'target_reached': processed_count >= self.target_papers
        }

def main():
    """Main execution function."""
    processor = CitationDrivenProcessor()
    
    try:
        results = processor.run_citation_driven_processing()
        
        logger.info("=== PROCESSING COMPLETE ===")
        logger.info(f"Total papers processed: {results['total_papers']}")
        logger.info(f"Target reached: {results['target_reached']}")
        logger.info(f"Discipline results: {results['discipline_results']}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
