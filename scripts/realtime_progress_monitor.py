#!/usr/bin/env python3
"""
Real-time Progress Monitor for 1M Papers Processing
Continuously reports progress every 10 seconds until target is reached.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.services.knowledge_database import KnowledgeDatabase

class RealtimeProgressMonitor:
    """Real-time progress monitor that never stops reporting."""
    
    def __init__(self):
        self.target_papers = 1000000
        self.report_interval = 10  # Report every 10 seconds
        self.start_time = time.time()
        self.last_count = 0
        
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress from database."""
        try:
            db = KnowledgeDatabase()
            stats = db.get_stats()
            current_count = stats["total_documents"]
            
            # Calculate metrics
            progress_percent = (current_count / self.target_papers) * 100
            elapsed_time = time.time() - self.start_time
            
            # Calculate rates
            if elapsed_time > 0:
                docs_per_minute = (current_count / elapsed_time) * 60
                docs_per_hour = docs_per_minute * 60
                
                # ETA calculation
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
                'last_updated': stats["last_updated"],
                'papers_added_since_last': current_count - self.last_count
            }
            
        except Exception as e:
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
        
        print(f"\n📊 REAL-TIME PROGRESS - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        print(f"📚 Papers: {current:,} / {target:,} ({progress_percent:.2f}%)")
        print(f"💾 Database: {progress.get('database_size_mb', 0):.2f} MB")
        print(f"⚡ Rate: {progress.get('docs_per_minute', 0):.1f}/min ({progress.get('docs_per_hour', 0):.0f}/hour)")
        print(f"⏰ ETA: {progress.get('days_remaining', 0):.1f} days")
        print(f"📈 Added since last report: {progress.get('papers_added_since_last', 0)}")
        
        # Progress bar
        bar_length = 40
        filled_length = int(bar_length * progress_percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"📊 [{bar}] {progress_percent:.1f}%")
        
        self.last_count = current
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring - NEVER STOPS until target reached."""
        print("🚀 Starting REAL-TIME Progress Monitoring")
        print(f"🎯 Target: {self.target_papers:,} papers")
        print("⚡ Monitoring will NEVER STOP until target is reached")
        print("=" * 60)
        
        while True:
            try:
                progress = self.get_progress()
                self.print_progress_report(progress)
                
                # Check if target reached
                if progress['current_count'] >= self.target_papers:
                    print("\n🎉 TARGET REACHED! 1M papers processed!")
                    print(f"⏱️  Total monitoring time: {(time.time() - self.start_time) / 3600:.2f} hours")
                    print("✅ Monitoring completed successfully!")
                    break
                
                # Wait before next report
                time.sleep(self.report_interval)
                
            except KeyboardInterrupt:
                print("\n⏹️  Monitoring stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Error in monitoring: {e}")
                time.sleep(30)  # Wait longer on error

def main():
    """Main execution function."""
    monitor = RealtimeProgressMonitor()
    
    try:
        monitor.run_continuous_monitoring()
    except Exception as e:
        print(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
