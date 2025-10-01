# Backend Service

FastAPI-based backend service for the Knowledge Map project.

## Features

- Document ingestion and processing
- Embedding generation and storage
- Dimensionality reduction and mapping
- Click API for document retrieval
- User overlay management
- Health monitoring

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
pytest app/tests/
```

## API Endpoints

- `GET /health` - Health check
- `POST /ingest` - Ingest new documents
- `GET /map` - Get map data
- `POST /click` - Handle map clicks
- `GET /user/{id}/overlay` - Get user overlay data
- `GET /openapi.json` - OpenAPI specification

## Architecture

- `app/main.py` - FastAPI application entry point
- `app/api/` - API route handlers
- `app/services/` - Business logic services
- `app/models/` - Pydantic models
- `app/db/` - Database models and connections
- `app/tests/` - Test suite
