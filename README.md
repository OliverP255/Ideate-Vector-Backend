# Knowledge Map - Interactive Document Visualization

A sophisticated web application for visualizing and exploring documents in a semantic 2D space. This system allows users to upload documents, search semantically, and interact with a knowledge map that organizes content by meaning rather than traditional categories.

## Features

- **Interactive Knowledge Map**: 2D visualization of documents organized by semantic similarity
- **Semantic Search**: Find documents by meaning, not just keywords
- **Document Upload**: Add PDF documents to the knowledge map
- **Gap Filling**: Generate content for empty areas of the knowledge map
- **Lasso Selection**: Select groups of related documents
- **Real-time Visualization**: Powered by DeckGL and Mapbox

## Architecture

### Backend (FastAPI)
- **Location**: `backend/`
- **Main File**: `backend/app/main.py`
- **APIs**: Document ingestion, embedding generation, semantic search, gap filling
- **Database**: SQLite for metadata + ChromaDB for vector search
- **Services**: Embedding, mapping, search, gap filling, and more

### Frontend (Next.js + React)
- **Location**: `frontend/`
- **Main File**: `frontend/app/page.tsx`
- **Components**: Interactive map, search panel, document upload, gap filling overlay
- **Visualization**: DeckGL with Mapbox integration

### Data Processing Scripts
- **Location**: `scripts/`
- **Main Processor**: `knowledge_map_processor.py` - Orchestrates document processing
- **Data Sources**: Wikipedia, arXiv, and custom document processors
- **Scale**: Designed to handle 1M+ documents

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- pip and npm

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Access the Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8001
- API Docs: http://localhost:8001/docs

## Usage

1. **Upload Documents**: Use the document upload panel to add PDF files
2. **Search**: Enter semantic queries to find related documents
3. **Explore**: Click on the map to discover documents in specific areas
4. **Gap Filling**: Enable gap filling mode and click on empty areas to generate content
5. **Lasso Selection**: Use lasso mode to select groups of related documents

## Data Processing

The system includes powerful data processing capabilities:

- **Wikipedia Processor**: Fetches and processes high-quality Wikipedia articles
- **arXiv Processor**: Processes academic papers from arXiv
- **Massive Scale Processing**: Handles large-scale document ingestion
- **Quality Filtering**: Ensures high-quality, factually accurate content

### Running Data Processing
```bash
cd scripts
python knowledge_map_processor.py --phase massive --sources wikipedia_breadth arxiv
```

## Configuration

- **Environment Variables**: Copy `env.template` to `.env` and configure
- **Mapbox Token**: Add your Mapbox token to the frontend configuration
- **Database Path**: Configure database location in backend settings

## Development

### Project Structure
```
├── backend/           # FastAPI backend
│   ├── app/
│   │   ├── api/      # API endpoints
│   │   ├── models/   # Data models
│   │   ├── services/ # Business logic
│   │   └── main.py   # Application entry point
│   └── requirements.txt
├── frontend/          # Next.js frontend
│   ├── app/          # App router pages
│   ├── components/   # React components
│   └── package.json
├── scripts/          # Data processing scripts
│   ├── knowledge_map_processor.py
│   ├── wikipedia_processor.py
│   └── enhanced_arxiv_processor.py
└── data/             # Processed data and models
    ├── models/       # Trained models
    ├── embeddings/   # Document embeddings
    └── knowledge_map/ # Database files
```

### Key Services
- **EmbeddingService**: Generates document embeddings
- **MappingService**: Creates 2D coordinates from embeddings
- **SearchService**: Performs semantic search
- **GapFillingService**: Generates content for empty map areas
- **KnowledgeDatabase**: Manages document storage and retrieval

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.