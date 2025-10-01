# Knowledge Map API Documentation

## Overview

The Knowledge Map API provides RESTful endpoints for document processing, embedding generation, vector search, and user interaction. This document describes all available endpoints, request/response formats, and usage examples.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://api.knowledge-map.com`

## Authentication

Currently, the API uses simple API key authentication. Include the API key in the request header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/health
```

## Common Response Formats

### Success Response
```json
{
  "status": "success",
  "message": "Operation completed successfully",
  "data": { ... }
}
```

### Error Response
```json
{
  "status": "error",
  "message": "Error description",
  "error_code": "ERROR_CODE",
  "details": { ... }
}
```

## Endpoints

### Health Check

#### GET /api/health
Check system health and service status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "services": {
    "database": "connected",
    "vector_db": "connected",
    "redis": "connected"
  },
  "version": "1.0.0"
}
```

**Example:**
```bash
curl http://localhost:8000/api/health
```

---

### Document Ingestion

#### POST /api/ingest
Upload and process a document.

**Request Body:**
```json
{
  "file_path": "/path/to/document.pdf",
  "metadata": {
    "title": "Document Title",
    "author": "Author Name",
    "year": 2024,
    "source": "Journal Name"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "document_id": "doc_123456789",
  "message": "Document processed successfully",
  "processing_time": 2.5,
  "cleaned_text_length": 15000
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/data/sample.pdf",
    "metadata": {
      "title": "Sample Document",
      "author": "John Doe",
      "year": 2024
    }
  }'
```

#### GET /api/ingest/status
Get ingestion queue status.

**Response:**
```json
{
  "queue_size": 5,
  "processing": 2,
  "completed_today": 150,
  "failed_today": 3
}
```

---

### Embedding Generation

#### POST /api/embedding/generate
Generate embeddings for a document.

**Request Body:**
```json
{
  "document_id": "doc_123456789",
  "model": "sentence-transformers",
  "chunk_size": 512
}
```

**Response:**
```json
{
  "status": "success",
  "document_id": "doc_123456789",
  "embedding_dimension": 384,
  "chunks_processed": 25,
  "embedding_method": "tfidf_weighted_average"
}
```

#### GET /api/embedding/{document_id}
Get embeddings for a specific document.

**Response:**
```json
{
  "document_id": "doc_123456789",
  "embedding": [0.1, 0.2, 0.3, ...],
  "metadata": {
    "model": "all-MiniLM-L12-v2",
    "generated_at": "2024-01-01T12:00:00Z",
    "dimension": 384
  }
}
```

---

### Dimensionality Reduction & Mapping

#### POST /api/mapping/compute
Compute 2D coordinates for documents.

**Request Body:**
```json
{
  "document_ids": ["doc_1", "doc_2", "doc_3"],
  "method": "pca_umap",
  "n_components_pca": 50,
  "n_neighbors": 15
}
```

**Response:**
```json
{
  "status": "success",
  "coordinates": [
    {"document_id": "doc_1", "coordinates": [0.1, 0.2]},
    {"document_id": "doc_2", "coordinates": [0.3, 0.4]},
    {"document_id": "doc_3", "coordinates": [0.5, 0.6]}
  ],
  "method": "pca_umap",
  "computation_time": 5.2
}
```

#### GET /api/mapping/coordinates
Get all document coordinates.

**Response:**
```json
{
  "coordinates": [
    {"document_id": "doc_1", "coordinates": [0.1, 0.2], "source": "initial_mapping"},
    {"document_id": "doc_2", "coordinates": [0.3, 0.4], "source": "projection"}
  ],
  "total_documents": 2,
  "retrieved_at": "2024-01-01T12:00:00Z"
}
```

#### GET /api/mapping/{document_id}
Get coordinates for a specific document.

**Response:**
```json
{
  "document_id": "doc_1",
  "coordinates": [0.1, 0.2],
  "source": "initial_mapping",
  "metadata": {
    "computed_at": "2024-01-01T12:00:00Z",
    "method": "pca_umap"
  }
}
```

---

### Vector Database Operations

#### POST /api/vector-db/upsert
Upsert document vectors to the vector database.

**Request Body:**
```json
{
  "document_id": "doc_123456789",
  "vector": [0.1, 0.2, 0.3, ...],
  "metadata": {
    "title": "Document Title",
    "author": "Author Name"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "document_id": "doc_123456789",
  "operation": "upserted",
  "vector_dimension": 384
}
```

#### GET /api/vector-db/search
Search for similar documents.

**Query Parameters:**
- `query_vector`: Base64 encoded vector
- `limit`: Number of results (default: 10)
- `threshold`: Similarity threshold (default: 0.7)

**Response:**
```json
{
  "results": [
    {
      "document_id": "doc_123456789",
      "similarity": 0.95,
      "metadata": {
        "title": "Document Title",
        "author": "Author Name"
      }
    }
  ],
  "total_results": 1,
  "search_time": 0.05
}
```

---

### Click Handling

#### POST /api/click
Handle map click and return relevant documents.

**Request Body:**
```json
{
  "x": 0.5,
  "y": 0.3,
  "radius": 0.1,
  "user_id": "user_123",
  "query_text": "machine learning",
  "limit": 10
}
```

**Response:**
```json
{
  "click_coordinates": {"x": 0.5, "y": 0.3},
  "radius": 0.1,
  "spatial_candidates": 15,
  "documents": [
    {
      "document_id": "doc_123456789",
      "coordinates": [0.52, 0.31],
      "similarity_score": 0.95,
      "spatial_distance": 0.02,
      "rerank_method": "semantic",
      "metadata": {
        "title": "Machine Learning Fundamentals",
        "author": "Jane Smith",
        "year": 2024
      }
    }
  ],
  "processing_time": 0.15
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/click \
  -H "Content-Type: application/json" \
  -d '{
    "x": 0.5,
    "y": 0.3,
    "radius": 0.1,
    "query_text": "machine learning",
    "limit": 5
  }'
```

---

### User Management

#### GET /api/user/{user_id}/profile
Get user profile information.

**Response:**
```json
{
  "userId": "user_123",
  "username": "john_doe",
  "email": "john@example.com",
  "createdAt": "2024-01-01T10:00:00Z",
  "lastActive": "2024-01-01T12:00:00Z"
}
```

#### GET /api/user/{user_id}/overlay
Get user overlay data for map visualization.

**Response:**
```json
{
  "userId": "user_123",
  "readDocuments": [
    {
      "document_id": "doc_1",
      "coordinates": [0.1, 0.2]
    }
  ],
  "convexHull": {
    "type": "Feature",
    "geometry": {
      "type": "Polygon",
      "coordinates": [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.1, 0.2]]]
    },
    "properties": {}
  },
  "clusterCoverage": {
    "cluster_1": 0.75,
    "cluster_2": 0.50
  },
  "generatedAt": "2024-01-01T12:00:00Z"
}
```

#### POST /api/user/{user_id}/read
Mark a document as read or unread.

**Request Body:**
```json
{
  "document_id": "doc_123456789",
  "read_status": true
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Document doc_123456789 marked as read"
}
```

#### GET /api/user/{user_id}/read
Get list of documents read by a user.

**Response:**
```json
[
  {
    "document_id": "doc_123456789",
    "title": "Document Title",
    "author": "Author Name",
    "year": 2024,
    "coordinates": [0.1, 0.2]
  }
]
```

#### GET /api/user/{user_id}/coverage
Get user knowledge coverage statistics.

**Response:**
```json
{
  "userId": "user_123",
  "overallCoverage": 25.5,
  "clusterCoverage": {
    "cluster_1": 75.0,
    "cluster_2": 50.0
  },
  "readDocuments": 15,
  "totalDocuments": 100,
  "generatedAt": "2024-01-01T12:00:00Z"
}
```

---

### Tile Generation

#### POST /api/tiles/generate
Generate tiles for all zoom levels.

**Request Body:**
```json
{
  "zoom_levels": [0, 1, 2, 3, 4]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Generated tiles for 100 documents",
  "results": {
    "zoom_0": {
      "clusters": 5,
      "individual_points": 10,
      "total_points": 100
    }
  }
}
```

#### GET /api/tiles/{zoom_level}
Get tile data for specific bounds and zoom level.

**Query Parameters:**
- `min_x`: Minimum X coordinate
- `min_y`: Minimum Y coordinate
- `max_x`: Maximum X coordinate
- `max_y`: Maximum Y coordinate

**Response:**
```json
{
  "bounds": [0.0, 0.0, 1.0, 1.0],
  "zoom_level": 0,
  "clusters": [
    {
      "cluster_id": 1,
      "center": [0.5, 0.5],
      "document_count": 10,
      "convex_hull": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]
    }
  ],
  "individual_points": [
    {
      "document_id": "doc_1",
      "coordinates": [0.1, 0.1]
    }
  ],
  "total_items": 11,
  "generated_at": "2024-01-01T12:00:00Z"
}
```

#### GET /api/tiles/clusters
Get cluster labels for documents.

**Query Parameters:**
- `min_cluster_size`: Minimum cluster size (default: 5)

**Response:**
```json
{
  "clusters": [
    {
      "cluster_id": 1,
      "documents": ["doc_1", "doc_2", "doc_3"],
      "center": [0.5, 0.5],
      "size": 3
    }
  ],
  "labels": [1, 1, 1, -1, 2, 2],
  "cluster_count": 2,
  "noise_count": 1,
  "method": "hdbscan",
  "generated_at": "2024-01-01T12:00:00Z"
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Request format is invalid |
| `DOCUMENT_NOT_FOUND` | Document ID does not exist |
| `PROCESSING_FAILED` | Document processing failed |
| `EMBEDDING_FAILED` | Embedding generation failed |
| `VECTOR_DB_ERROR` | Vector database operation failed |
| `MAPPING_FAILED` | Dimensionality reduction failed |
| `USER_NOT_FOUND` | User ID does not exist |
| `INSUFFICIENT_PERMISSIONS` | User lacks required permissions |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded |
| `SERVICE_UNAVAILABLE` | Required service is unavailable |

## Rate Limits

- **Default**: 1000 requests per hour per API key
- **Click API**: 100 requests per minute per user
- **Ingestion**: 10 documents per minute per user

## Webhooks

The API supports webhooks for real-time notifications:

### Document Processing Complete
```json
{
  "event": "document.processing.complete",
  "document_id": "doc_123456789",
  "status": "success",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### User Action
```json
{
  "event": "user.document.read",
  "user_id": "user_123",
  "document_id": "doc_123456789",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## SDK Examples

### Python
```python
import requests

# Initialize client
api_key = "your-api-key"
base_url = "http://localhost:8000"
headers = {"X-API-Key": api_key}

# Upload document
response = requests.post(
    f"{base_url}/api/ingest",
    json={
        "file_path": "/path/to/document.pdf",
        "metadata": {"title": "Sample Document"}
    },
    headers=headers
)
document_id = response.json()["document_id"]

# Search for similar documents
response = requests.get(
    f"{base_url}/api/vector-db/search",
    params={"query_vector": "base64_encoded_vector", "limit": 10},
    headers=headers
)
results = response.json()["results"]
```

### JavaScript
```javascript
const API_KEY = 'your-api-key';
const BASE_URL = 'http://localhost:8000';

// Click handler
async function handleMapClick(x, y, query) {
  const response = await fetch(`${BASE_URL}/api/click`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY
    },
    body: JSON.stringify({
      x, y,
      radius: 0.1,
      query_text: query,
      limit: 10
    })
  });
  
  const data = await response.json();
  return data.documents;
}
```

## Testing

Use the provided test suite to validate API functionality:

```bash
# Run all tests
python3 tests/test_all_phases.py

# Run specific phase tests
python3 tests/test_phase4_vector_db_click.py
python3 tests/test_phase5_frontend.py
python3 tests/test_phase6_tiles.py
```

## Support

For API support and questions:
- **Email**: api-support@knowledge-map.com
- **Documentation**: https://docs.knowledge-map.com
- **Status Page**: https://status.knowledge-map.com
