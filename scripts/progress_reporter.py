#!/usr/bin/env python3
"""
Simple Progress Reporter for 1M Papers Processing
"""

import os
import sys
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.services.knowledge_database import KnowledgeDatabase

def main():
    """Report progress every 30 seconds."""
    print("🚀 Starting 1M Papers Processing Monitor...")
    print("📊 Target: 1,000,000 documents")
    print("⏱️  Reporting every 30 seconds")
    print("=" * 60)
    
    start_time = time.time()
    
    while True:
        try:
            db = KnowledgeDatabase()
            stats = db.get_stats()
            
            current = stats["total_documents"]
            target = 1_000_000
            progress_percent = (current / target) * 100
            database_size = stats["database_size_mb"]
            last_updated = stats["last_updated"]
            
            # Calculate processing rate
            elapsed_time = time.time() - start_time
            if elapsed_time > 0 and current > 0:
                docs_per_minute = (current / elapsed_time) * 60
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
                docs_per_minute = 0
                docs_per_hour = 0
                hours_remaining = 0
                days_remaining = 0
            
            # Print progress report
            print(f"\n📊 PROGRESS REPORT - {datetime.now().strftime('%H:%M:%S')}")
            print(f"📚 Documents: {current:,} / {target:,} ({progress_percent:.2f}%)")
            print(f"💾 Database: {database_size:.2f} MB")
            print(f"⚡ Rate: {docs_per_minute:.1f} docs/min ({docs_per_hour:.0f} docs/hour)")
            print(f"⏰ ETA: {days_remaining:.1f} days ({hours_remaining:.1f} hours)")
            print(f"🕒 Updated: {last_updated}")
            
            # Progress bar
            bar_length = 40
            filled_length = int(bar_length * progress_percent / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"📈 [{bar}] {progress_percent:.1f}%")
            
            if current >= target:
                print(f"\n🎉 TARGET REACHED! {current:,} documents processed!")
                break
            
            print("=" * 60)
            time.sleep(30)
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
