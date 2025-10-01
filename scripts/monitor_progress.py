#!/usr/bin/env python3
"""
Progress Monitor for Knowledge Map Processing
Monitor the progress of massive scale processing in real-time.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.services.knowledge_database import KnowledgeDatabase

def get_current_status():
    """Get current processing status."""
    try:
        db = KnowledgeDatabase()
        stats = db.get_stats()
        db.close()
        return stats
    except Exception as e:
        return {"error": str(e)}

def get_progress_file_status():
    """Get status from progress file if it exists."""
    progress_file = Path("../data/processing_progress.json")
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to read progress file: {e}"}
    return None

def format_number(num):
    """Format large numbers with commas."""
    return f"{num:,}"

def calculate_percentage(current, target):
    """Calculate percentage."""
    if target == 0:
        return 0.0
    return (current / target) * 100

def display_progress():
    """Display current progress."""
    print("\n" + "="*80)
    print("🚀 KNOWLEDGE MAP MASSIVE SCALE PROCESSING MONITOR")
    print("="*80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get database status
    db_status = get_current_status()
    if "error" in db_status:
        print(f"❌ Error getting database status: {db_status['error']}")
        return
    
    # Get progress file status
    progress_status = get_progress_file_status()
    
    # Display overall stats
    total_docs = db_status.get('total_documents', 0)
    db_size = db_status.get('database_size_mb', 0)
    
    print(f"\n📊 OVERALL STATUS")
    print(f"   Total Documents: {format_number(total_docs)}")
    print(f"   Database Size: {db_size:.1f} MB")
    print(f"   Last Updated: {db_status.get('last_updated', 'Unknown')}")
    
    # Display source breakdown
    print(f"\n📚 SOURCE BREAKDOWN")
    source_counts = db_status.get('source_counts', {})
    
    targets = {
        'wikipedia_breadth': 500_000,
        'wikipedia_featured': 100_000,
        'arxiv': 200_000,
        'pubmed': 100_000,
        'books': 50_000,
        'government': 50_000
    }
    
    total_target = sum(targets.values())
    overall_progress = calculate_percentage(total_docs, total_target)
    
    for source, target in targets.items():
        current = source_counts.get(source, 0)
        percentage = calculate_percentage(current, target)
        
        # Progress bar
        bar_length = 30
        filled_length = int(bar_length * percentage / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"   {source.replace('_', ' ').title():20} {current:>8}/{format_number(target)} ({percentage:5.1f}%) [{bar}]")
    
    print(f"\n🎯 OVERALL PROGRESS: {overall_progress:.1f}% ({format_number(total_docs)}/{format_number(total_target)})")
    
    # Display quality distribution
    print(f"\n⭐ QUALITY DISTRIBUTION")
    quality_dist = db_status.get('quality_distribution', {})
    for quality, count in quality_dist.items():
        print(f"   {quality:8}: {format_number(count)} documents")
    
    # Display processing progress if available
    if progress_status and not progress_status.get('error'):
        print(f"\n⚙️  PROCESSING PROGRESS")
        completed = progress_status.get('completed', {})
        for source, count in completed.items():
            target = targets.get(source, 0)
            if target > 0:
                percentage = calculate_percentage(count, target)
                print(f"   {source.replace('_', ' ').title():20}: {format_number(count)} ({percentage:.1f}%)")
    
    print("="*80)

def main():
    """Main monitoring loop."""
    try:
        while True:
            display_progress()
            print(f"\n⏱️  Next update in 30 seconds... (Press Ctrl+C to stop)")
            time.sleep(30)
    except KeyboardInterrupt:
        print(f"\n\n👋 Monitoring stopped by user.")
        print(f"📊 Final status:")
        display_progress()

if __name__ == "__main__":
    main()
