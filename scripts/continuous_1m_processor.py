#!/usr/bin/env python3
"""
Continuous 1M Papers Processor - NEVER STOPS until target is reached.
Processes high-impact papers with citation-driven selection and breadth enforcement.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import signal
import threading

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from citation_driven_processor import CitationDrivenProcessor
from backend.app.services.knowledge_database import KnowledgeDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/continuous_1m_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousProcessor:
    """Continuous processor that never stops until 1M papers are processed."""
    
    def __init__(self):
        self.target_papers = 1000000
        self.batch_size = 1000  # Process in batches
        self.processing_interval = 30  # Seconds between progress reports
        self.last_report_time = time.time()
        
        # Initialize components
        self.processor = CitationDrivenProcessor()
        self.db = KnowledgeDatabase()
        
        # Processing state
        self.is_running = True
        self.start_time = time.time()
        self.total_processed = 0
        self.last_processed_count = 0
        
        # Create logs directory
        os.makedirs("data/logs", exist_ok=True)
        os.makedirs("data/checkpoints", exist_ok=True)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_running = False
    
    def get_current_progress(self) -> Dict[str, Any]:
        """Get current processing progress."""
        try:
            stats = self.db.get_stats()
            current_count = stats["total_documents"]
            
            # Calculate progress metrics
            progress_percent = (current_count / self.target_papers) * 100
            elapsed_time = time.time() - self.start_time
            
            # Calculate processing rate
            if elapsed_time > 0:
                docs_per_minute = (current_count / elapsed_time) * 60
                docs_per_hour = docs_per_minute * 60
                
                # Estimate completion time
                remaining_docs = self.target_papers - current_count
                if docs_per_hour > 0:
                    hours_remaining = remaining_docs / docs_per_hour
                    days_remaining = hours_remaining / 24
                else:
                    hours_remaining = 0
                    days_remaining = 0
            else:
                docs_per_minute = 0
                docs_per_hour = 0
                hours_remaining = 0
                days_remaining = 0
            
            return {
                'current_count': current_count,
                'target': self.target_papers,
                'progress_percent': progress_percent,
                'database_size_mb': stats["database_size_mb"],
                'docs_per_minute': docs_per_minute,
                'docs_per_hour': docs_per_hour,
                'hours_remaining': hours_remaining,
                'days_remaining': days_remaining,
                'elapsed_time': elapsed_time,
                'last_updated': stats["last_updated"]
            }
            
        except Exception as e:
            logger.error(f"Error getting progress: {e}")
            return {
                'current_count': 0,
                'target': self.target_papers,
                'progress_percent': 0,
                'error': str(e)
            }
    
    def print_progress_report(self, progress: Dict[str, Any]):
        """Print detailed progress report."""
        current = progress['current_count']
        target = progress['target']
        progress_percent = progress['progress_percent']
        
        print(f"\n🚀 CONTINUOUS 1M PAPERS PROCESSING - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)
        print(f"📚 Papers processed: {current:,} / {target:,} ({progress_percent:.2f}%)")
        print(f"💾 Database size: {progress.get('database_size_mb', 0):.2f} MB")
        print(f"⚡ Processing rate: {progress.get('docs_per_minute', 0):.1f} docs/min ({progress.get('docs_per_hour', 0):.0f} docs/hour)")
        print(f"⏰ ETA: {progress.get('days_remaining', 0):.1f} days ({progress.get('hours_remaining', 0):.1f} hours)")
        print(f"🕒 Last updated: {progress.get('last_updated', 'Unknown')}")
        
        # Progress bar
        bar_length = 50
        filled_length = int(bar_length * progress_percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"📈 Progress: [{bar}] {progress_percent:.1f}%")
        
        # Processing rate trend
        if hasattr(self, 'last_processed_count') and self.last_processed_count > 0:
            rate_change = current - self.last_processed_count
            if rate_change > 0:
                print(f"📊 Papers added since last report: {rate_change}")
        
        self.last_processed_count = current
        print("=" * 80)
    
    def process_batch(self) -> int:
        """Process a batch of papers."""
        try:
            # Run citation-driven processing for a batch
            results = self.processor.run_citation_driven_processing()
            return results.get('total_papers', 0)
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return 0
    
    def run_continuous_processing(self):
        """Main continuous processing loop - NEVER STOPS until target reached."""
        logger.info("🚀 Starting CONTINUOUS 1M Papers Processing")
        logger.info(f"🎯 Target: {self.target_papers:,} high-impact papers")
        logger.info("⚡ Processing will NEVER STOP until target is reached")
        logger.info("=" * 80)
        
        batch_count = 0
        
        while self.is_running:
            try:
                batch_count += 1
                logger.info(f"🔄 Starting batch {batch_count}")
                
                # Process a batch of papers
                papers_processed = self.process_batch()
                self.total_processed += papers_processed
                
                # Get current progress
                progress = self.get_current_progress()
                current_count = progress['current_count']
                
                # Print progress report
                self.print_progress_report(progress)
                
                # Check if target reached
                if current_count >= self.target_papers:
                    logger.info("🎉 TARGET REACHED! 1M papers processed!")
                    logger.info(f"⏱️  Total processing time: {(time.time() - self.start_time) / 3600:.2f} hours")
                    logger.info("✅ Processing completed successfully!")
                    break
                
                # Wait before next batch (but continue if target not reached)
                logger.info(f"⏳ Waiting 30 seconds before next batch...")
                time.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("⏹️  Processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in processing loop: {e}")
                logger.info("🔄 Continuing processing despite error...")
                time.sleep(60)  # Wait longer on error
        
        # Final report
        final_progress = self.get_current_progress()
        logger.info("=" * 80)
        logger.info("🏁 FINAL PROCESSING REPORT")
        logger.info(f"📚 Final count: {final_progress['current_count']:,} papers")
        logger.info(f"🎯 Target: {self.target_papers:,} papers")
        logger.info(f"⏱️  Total time: {(time.time() - self.start_time) / 3600:.2f} hours")
        logger.info(f"✅ Target reached: {final_progress['current_count'] >= self.target_papers}")
        logger.info("=" * 80)

def main():
    """Main execution function."""
    processor = ContinuousProcessor()
    
    try:
        processor.run_continuous_processing()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
