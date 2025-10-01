# Knowledge Map Architecture

## Overview

The Knowledge Map is an interactive, scalable visualization system where each point represents a resource (paper, book, article). The system enables users to explore their knowledge landscape, discover related content, and track their reading progress.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Pipeline      │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│   (Processing)  │
│                 │    │                 │    │                 │
│ • deck.gl       │    │ • REST APIs     │    │ • GROBID        │
│ • react-map-gl  │    │ • Authentication│    │ • Embeddings   │
│ • TailwindCSS   │    │ • Rate Limiting │    │ • DR Mapping    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Object Store  │    │   Databases     │    │   Monitoring    │
│   (MinIO/S3)    │    │                 │    │                 │
│                 │    │ • PostgreSQL    │    │ • Prometheus    │
│ • Raw Documents │    │ • Qdrant        │    │ • Grafana       │
│ • Processed     │    │ • Redis         │    │ • Sentry        │
│ • Tiles         │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. Frontend (Next.js + deck.gl)

**Purpose**: Interactive visualization and user interface

**Key Features**:
- Interactive map rendering with deck.gl
- Real-time document visualization
- User overlay with reading progress
- Semantic search interface
- Responsive design with TailwindCSS

**Technology Stack**:
- Next.js 14 with TypeScript
- deck.gl for WebGL visualization
- react-map-gl for map integration
- TailwindCSS for styling

**Key Components**:
- `KnowledgeMap.tsx`: Main visualization component
- `UserOverlay.tsx`: User reading progress overlay
- `SearchPanel.tsx`: Semantic search interface
- `DocumentDetails.tsx`: Document information display

### 2. Backend API (FastAPI)

**Purpose**: RESTful API for all system operations

**Key Features**:
- Document ingestion and processing
- Embedding generation and storage
- Dimensionality reduction and mapping
- Vector database operations
- Click handling and document retrieval
- User management and overlays

**Technology Stack**:
- FastAPI with Python 3.12
- Pydantic for data validation
- SQLAlchemy for database ORM
- Uvicorn ASGI server

**API Endpoints**:
- `/api/health`: System health check
- `/api/ingest`: Document ingestion
- `/api/embedding`: Embedding generation
- `/api/mapping`: 2D coordinate mapping
- `/api/vector-db`: Vector database operations
- `/api/click`: Map click handling
- `/api/user`: User management
- `/api/tiles`: Tile generation and retrieval

### 3. Pipeline (Processing)

**Purpose**: Document processing and data transformation

**Key Features**:
- Document ingestion with GROBID
- Boilerplate removal and text cleaning
- Embedding generation with TF-IDF weighting
- Dimensionality reduction (PCA → UMAP)
- Parametric mapper training
- Tile generation with LOD support

**Technology Stack**:
- Python 3.12
- GROBID for PDF processing
- SentenceTransformers for embeddings
- scikit-learn for TF-IDF
- UMAP for dimensionality reduction
- TensorFlow for parametric mapping
- HDBSCAN for clustering

**Key Modules**:
- `pipeline/ingest/worker.py`: Document processing
- `pipeline/embedding/generate.py`: Embedding generation
- `pipeline/dr/mapping.py`: Dimensionality reduction
- `pipeline/tilegen/generator.py`: Tile generation

### 4. Data Storage

#### PostgreSQL
- **Purpose**: Relational data storage
- **Schema**: Users, documents, mappings, clusters
- **Extensions**: PostGIS for spatial operations

#### Qdrant
- **Purpose**: Vector database for similarity search
- **Collections**: Document embeddings, user profiles
- **Indexing**: HNSW for fast approximate nearest neighbor search

#### Redis
- **Purpose**: Caching and session storage
- **Use Cases**: API response caching, user sessions

#### MinIO/S3
- **Purpose**: Object storage for documents and tiles
- **Buckets**: Raw documents, processed content, generated tiles

### 5. Monitoring

#### Prometheus
- **Purpose**: Metrics collection and storage
- **Metrics**: API performance, system resources, business metrics

#### Grafana
- **Purpose**: Metrics visualization and alerting
- **Dashboards**: System overview, API performance, document processing

#### Sentry
- **Purpose**: Error tracking and performance monitoring
- **Features**: Real-time error reporting, performance profiling

## Data Flow

### 1. Document Ingestion Flow

```
PDF Document → GROBID Processing → Text Cleaning → Metadata Extraction
     ↓
PostgreSQL Storage → Embedding Generation → Vector Database Indexing
     ↓
2D Mapping → Tile Generation → Frontend Visualization
```

### 2. User Interaction Flow

```
User Click → Spatial Query → Semantic Reranking → Document Retrieval
     ↓
User Overlay Update → Convex Hull Generation → Coverage Calculation
```

### 3. Search Flow

```
Query Text → Embedding Generation → Vector Search → Result Ranking
     ↓
Spatial Filtering → Document Details → Response Formatting
```

## Key Algorithms

### 1. Document Embedding Generation

**Algorithm**: TF-IDF Weighted Averaging
1. Split document into chunks (max 512 tokens)
2. Generate embeddings for each chunk
3. Calculate TF-IDF weights for chunks
4. Compute weighted average of chunk embeddings

**Fallback**: Simple averaging if TF-IDF is degenerate

### 2. Dimensionality Reduction

**Algorithm**: PCA → UMAP Pipeline
1. Standardize embeddings
2. Apply PCA (adaptive components: min(50, n_features/2, n_samples))
3. Apply UMAP (adaptive neighbors: min(15, n_samples-1))
4. Train parametric MLP mapper for incremental updates

### 3. Parametric Mapping

**Algorithm**: Multi-layer Perceptron
- Input: PCA-transformed embeddings
- Architecture: 128 → 64 → 32 → 2 (with dropout)
- Loss: MSE with MAE monitoring
- Training: Adam optimizer, 100 epochs

### 4. Clustering

**Algorithm**: HDBSCAN
- Adaptive parameters based on zoom level
- Convex hull generation for cluster visualization
- Noise point handling for individual documents

## Performance Characteristics

### Scalability Targets

- **Documents**: 1M+ documents
- **Users**: 10K+ concurrent users
- **API Response**: P95 < 200ms
- **Click Latency**: P95 < 200ms
- **Throughput**: 1000+ requests/second

### Resource Requirements

#### Development
- **CPU**: 4 cores
- **Memory**: 8GB RAM
- **Storage**: 100GB SSD
- **Services**: PostgreSQL, Qdrant, Redis, MinIO

#### Production
- **CPU**: 16+ cores
- **Memory**: 64GB+ RAM
- **Storage**: 1TB+ SSD
- **Services**: Managed databases, CDN, Load balancer

## Security Considerations

### Authentication & Authorization
- JWT-based authentication
- Role-based access control
- API key management for external services

### Data Protection
- Encryption at rest and in transit
- PII handling and GDPR compliance
- Secure document storage

### Network Security
- HTTPS/TLS for all communications
- Network segmentation
- Rate limiting and DDoS protection

## Deployment Architecture

### Development Environment
- Docker Compose for local development
- All services running locally
- Hot reloading for frontend and backend

### Production Environment
- Kubernetes cluster deployment
- Helm charts for configuration management
- Horizontal Pod Autoscaling
- Service mesh for traffic management

### CI/CD Pipeline
- GitHub Actions for automated testing
- Docker image building and scanning
- Automated deployment to staging/production
- Security scanning with Trivy

## Monitoring & Observability

### Metrics
- **System**: CPU, memory, disk, network
- **Application**: Request rate, response time, error rate
- **Business**: Documents processed, user engagement, search success

### Logging
- Structured logging with correlation IDs
- Centralized log aggregation
- Log retention and archival

### Tracing
- Distributed tracing for request flows
- Performance profiling
- Error tracking and debugging

## Disaster Recovery

### Backup Strategy
- Daily database backups
- Point-in-time recovery capability
- Cross-region replication
- Regular backup testing

### Recovery Procedures
- RTO: 4 hours
- RPO: 1 hour
- Automated failover for critical services
- Manual procedures for data recovery

## Future Considerations

### Scalability Improvements
- Microservices architecture
- Event-driven processing
- Caching layers
- CDN integration

### Feature Enhancements
- Real-time collaboration
- Advanced analytics
- Machine learning recommendations
- Multi-language support

### Technology Evolution
- Vector database optimization
- GPU acceleration for embeddings
- Edge computing for low latency
- AI-powered content analysis