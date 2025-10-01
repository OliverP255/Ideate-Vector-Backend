#!/usr/bin/env python3
"""
Continuous Progress Monitor for 1M Papers Processing
Reports progress every 30 seconds until target is reached.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.services.knowledge_database import KnowledgeDatabase

def format_number(num):
    """Format numbers with commas."""
    return f"{num:,}"

def get_progress_report():
    """Get current progress report."""
    try:
        db = KnowledgeDatabase()
        stats = db.get_stats()
        
        current_count = stats["total_documents"]
        target = 1_000_000
        progress_percent = (current_count / target) * 100
        
        return {
            'current': current_count,
            'target': target,
            'progress_percent': progress_percent,
            'database_size_mb': stats["database_size_mb"],
            'last_updated': stats["last_updated"]
        }
    except Exception as e:
        return {
            'error': str(e),
            'current': 0,
            'target': 1_000_000,
            'progress_percent': 0,
            'database_size_mb': 0,
            'last_updated': 'Unknown'
        }

def main():
    """Main monitoring loop."""
    print("🚀 Starting continuous monitoring of 1M papers processing...")
    print("📊 Target: 1,000,000 documents")
    print("⏱️  Reporting every 30 seconds")
    print("=" * 60)
    
    start_time = time.time()
    
    while True:
        try:
            report = get_progress_report()
            
            if 'error' in report:
                print(f"❌ Error getting progress: {report['error']}")
                time.sleep(30)
                continue
            
            current = report['current']
            target = report['target']
            progress_percent = report['progress_percent']
            database_size = report['database_size_mb']
            last_updated = report['last_updated']
            
            # Calculate rate
            elapsed_time = time.time() - start_time
            if elapsed_time > 0 and current > 0:
                docs_per_second = current / elapsed_time
                docs_per_minute = docs_per_second * 60
                docs_per_hour = docs_per_minute * 60
                
                # Estimate completion time
                remaining_docs = target - current
                if docs_per_hour > 0:
                    hours_remaining = remaining_docs / docs_per_hour
                    days_remaining = hours_remaining / 24
                else:
                    hours_remaining = 0
                    days_remaining = 0
            else:
                docs_per_second = 0
                docs_per_minute = 0
                docs_per_hour = 0
                hours_remaining = 0
                days_remaining = 0
            
            # Print progress report
            print(f"\n📊 PROGRESS REPORT - {datetime.now().strftime('%H:%M:%S')}")
            print(f"📚 Documents processed: {format_number(current)} / {format_number(target)} ({progress_percent:.2f}%)")
            print(f"💾 Database size: {database_size:.2f} MB")
            print(f"⚡ Processing rate: {docs_per_minute:.1f} docs/min ({docs_per_hour:.0f} docs/hour)")
            print(f"⏰ Estimated completion: {days_remaining:.1f} days ({hours_remaining:.1f} hours)")
            print(f"🕒 Last updated: {last_updated}")
            
            # Progress bar
            bar_length = 40
            filled_length = int(bar_length * progress_percent / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"📈 Progress: [{bar}] {progress_percent:.1f}%")
            
            # Check if target reached
            if current >= target:
                print(f"\n🎉 TARGET REACHED! Processed {format_number(current)} documents!")
                print(f"⏱️  Total time: {elapsed_time/3600:.2f} hours")
                break
            
            print("=" * 60)
            time.sleep(30)
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in monitoring loop: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
