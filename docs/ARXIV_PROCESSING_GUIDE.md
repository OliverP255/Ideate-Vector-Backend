# arXiv PDF Processing Guide

This guide explains how to process tens of thousands of arXiv PDFs for the Knowledge Map.

## Overview

The arXiv processing pipeline downloads PDFs, extracts text, generates embeddings, and creates UMAP visualizations for large-scale document analysis.

## Prerequisites

### Required Python Packages

```bash
pip install requests sentence-transformers PyMuPDF beautifulsoup4 umap-learn numpy
```

### System Requirements

- **RAM**: At least 8GB (16GB recommended for large batches)
- **Storage**: ~100MB per 1000 papers (PDFs + text + embeddings)
- **Network**: Stable internet connection for downloading PDFs

## Step-by-Step Process

### 1. Download and Process arXiv Papers

#### Basic Usage

```bash
# Process papers from the last 30 days
python scripts/arxiv_processor.py \
    --start-date 2025-08-01 \
    --end-date 2025-08-31 \
    --max-papers 1000 \
    --max-workers 4 \
    --output-dir data/arxiv
```

#### Large-Scale Processing (10,000+ papers)

```bash
# Process papers from the last year
python scripts/arxiv_processor.py \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --max-papers 10000 \
    --max-workers 8 \
    --output-dir data/arxiv_large
```

#### Parameters Explained

- `--start-date`: Start date in YYYY-MM-DD format
- `--end-date`: End date in YYYY-MM-DD format  
- `--max-papers`: Maximum number of papers to process
- `--max-workers`: Number of parallel workers (recommended: 4-8)
- `--output-dir`: Output directory for processed data

### 2. Generate UMAP Coordinates

After processing papers, generate 2D coordinates:

```bash
python scripts/generate_arxiv_umap.py \
    --embeddings-file data/arxiv/all_embeddings.json \
    --output-file data/arxiv/coordinates.json
```

### 3. Update Backend Configuration

#### Update Sample Data Service

Modify `backend/app/services/sample_data.py` to load arXiv data:

```python
def __init__(self):
    self.data_dir = Path("data/sample_docs")
    self.coordinates_file = self.data_dir / "coordinates.json"
    self.embeddings_file = Path("data/arxiv/all_embeddings.json")  # Updated path
```

#### Copy Coordinates to Backend

```bash
cp data/arxiv/coordinates.json data/sample_docs/coordinates.json
cp data/arxiv/coordinates.json backend/data/sample_docs/coordinates.json
```

### 4. Restart Services

```bash
# Restart backend to load new data
cd backend && python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Restart frontend
cd frontend && npm run dev
```

## Processing Statistics

### Typical Processing Times

| Papers | Download Time | Processing Time | Total Time |
|--------|-------------|----------------|------------|
| 1,000  | ~10 minutes | ~5 minutes     | ~15 minutes|
| 5,000  | ~45 minutes | ~25 minutes    | ~70 minutes|
| 10,000 | ~90 minutes | ~50 minutes    | ~140 minutes|

### Storage Requirements

| Papers | PDFs | Text Files | Embeddings | Total |
|--------|------|------------|------------|-------|
| 1,000  | ~500MB| ~50MB     | ~25MB     | ~575MB|
| 5,000  | ~2.5GB| ~250MB    | ~125MB    | ~2.9GB|
| 10,000 | ~5GB  | ~500MB    | ~250MB    | ~5.8GB|

## Advanced Configuration

### Custom Date Ranges

```bash
# Process specific months
python scripts/arxiv_processor.py \
    --start-date 2024-03-01 \
    --end-date 2024-03-31 \
    --max-papers 2000

# Process specific categories (modify script)
python scripts/arxiv_processor.py \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --max-papers 5000 \
    --categories "cs.AI,cs.LG,cs.CV"  # AI, Machine Learning, Computer Vision
```

### Memory Optimization

For large datasets, modify the processor:

```python
# In arxiv_processor.py, reduce text length
def _clean_text(self, text: str) -> str:
    # Limit to 20k characters for memory efficiency
    if len(text) > 20000:
        text = text[:20000] + "..."
    return text
```

### Parallel Processing

Adjust workers based on your system:

```bash
# High-end system (16+ cores, 32GB+ RAM)
--max-workers 12

# Mid-range system (8 cores, 16GB RAM)  
--max-workers 6

# Low-end system (4 cores, 8GB RAM)
--max-workers 2
```

## Troubleshooting

### Common Issues

#### 1. Rate Limiting Errors

```bash
# Reduce request frequency
# Edit arxiv_processor.py:
self.request_delay = 2.0  # Increase delay between requests
```

#### 2. Memory Issues

```bash
# Process in smaller batches
python scripts/arxiv_processor.py \
    --start-date 2024-01-01 \
    --end-date 2024-01-15 \
    --max-papers 2000 \
    --max-workers 2
```

#### 3. PDF Download Failures

```bash
# Check network connectivity
curl -I "http://export.arxiv.org/api/query?search_query=cat:cs.AI&max_results=1"

# Retry failed downloads (implement retry logic in script)
```

#### 4. Text Extraction Failures

```bash
# Install additional PDF libraries
pip install pdfplumber pymupdf4llm

# Modify extraction method in arxiv_processor.py
```

### Performance Monitoring

Monitor processing with:

```bash
# Watch disk usage
watch -n 5 'du -sh data/arxiv/*'

# Monitor memory usage
htop

# Check processing logs
tail -f arxiv_processing.log
```

## Quality Assurance

### Validation Steps

1. **Check Download Success Rate**
   ```bash
   ls data/arxiv/pdfs/ | wc -l  # Should match expected count
   ```

2. **Verify Text Extraction**
   ```bash
   find data/arxiv/texts/ -name "*.txt" -size 0 | wc -l  # Should be 0
   ```

3. **Validate Embeddings**
   ```bash
   python -c "
   import json
   with open('data/arxiv/all_embeddings.json') as f:
       data = json.load(f)
   print(f'Embeddings: {len(data)}')
   print(f'Embedding dimension: {len(data[0][\"embedding\"])}')
   "
   ```

4. **Test UMAP Coordinates**
   ```bash
   python -c "
   import json
   with open('data/arxiv/coordinates.json') as f:
       coords = json.load(f)
   print(f'Coordinates: {len(coords)}')
   print(f'Sample coordinate: {coords[0][\"coordinates\"]}')
   "
   ```

## Scaling to Production

### For 100,000+ Papers

1. **Use Distributed Processing**
   ```bash
   # Split by date ranges
   python scripts/arxiv_processor.py --start-date 2023-01-01 --end-date 2023-06-30 --max-papers 50000
   python scripts/arxiv_processor.py --start-date 2023-07-01 --end-date 2023-12-31 --max-papers 50000
   ```

2. **Use Vector Database**
   ```bash
   # Store embeddings in Qdrant instead of JSON files
   python scripts/upload_to_qdrant.py --embeddings-file data/arxiv/all_embeddings.json
   ```

3. **Implement Incremental Updates**
   ```bash
   # Daily updates
   python scripts/arxiv_processor.py \
       --start-date $(date -d "yesterday" +%Y-%m-%d) \
       --end-date $(date -d "yesterday" +%Y-%m-%d) \
       --max-papers 1000
   ```

## Example Commands

### Quick Start (100 papers)
```bash
python scripts/arxiv_processor.py \
    --start-date 2024-01-01 \
    --end-date 2024-01-07 \
    --max-papers 100 \
    --max-workers 2

python scripts/generate_arxiv_umap.py \
    --embeddings-file data/arxiv/all_embeddings.json \
    --output-file data/arxiv/coordinates.json

cp data/arxiv/coordinates.json data/sample_docs/coordinates.json
cp data/arxiv/coordinates.json backend/data/sample_docs/coordinates.json
```

### Production Scale (10,000 papers)
```bash
python scripts/arxiv_processor.py \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --max-papers 10000 \
    --max-workers 8 \
    --output-dir data/arxiv_production

python scripts/generate_arxiv_umap.py \
    --embeddings-file data/arxiv_production/all_embeddings.json \
    --output-file data/arxiv_production/coordinates.json

cp data/arxiv_production/coordinates.json data/sample_docs/coordinates.json
cp data/arxiv_production/coordinates.json backend/data/sample_docs/coordinates.json
```

## Next Steps

After processing arXiv papers:

1. **Visualize Results**: Open http://localhost:3002 to see the Knowledge Map
2. **Test Search**: Try semantic search queries related to your papers
3. **Analyze Categories**: Check how different arXiv categories cluster in UMAP space
4. **Scale Up**: Process more papers or different date ranges
5. **Production Deploy**: Set up automated daily processing

## Support

For issues or questions:
- Check the troubleshooting section above
- Review processing logs for error messages
- Test with smaller batches first
- Ensure sufficient disk space and memory
