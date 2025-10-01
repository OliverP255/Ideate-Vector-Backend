# Documentation

Comprehensive documentation for the Knowledge Map project.

## Getting Started

1. **Developer Setup**: See [DEVELOPER_SETUP.md](DEVELOPER_SETUP.md) for local development environment setup
2. **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) for system design and components
3. **API Reference**: See [API.md](API.md) for complete API documentation
4. **Operations**: See [RUNBOOK.md](RUNBOOK.md) for operational procedures

## Quick Start

```bash
# Clone repository
git clone <repo-url>
cd knowledge-map

# Setup environment
cp env.template .env
# Edit .env with your configuration

# Start services
docker-compose -f infra/docker-compose.yml up

# Verify health
curl http://localhost:8000/api/health
```

## Project Overview

The Knowledge Map is an interactive visualization system that maps knowledge resources (papers, books, articles) as points in 2D space. Users can explore the map, see their reading history as an overlay, and click to retrieve relevant documents.

### Key Features

- **One Document = One Point**: Each resource maps to exactly one point on the map
- **Incremental Updates**: New documents can be added without remapping the entire dataset
- **User Overlay**: Visual representation of user's knowledge consumption
- **Semantic Search**: Click-based retrieval of relevant documents
- **Scalable**: Designed to handle millions of documents

### Technology Stack

- **Backend**: FastAPI, PostgreSQL, Qdrant, Redis
- **Frontend**: Next.js, TypeScript, deck.gl
- **Processing**: GROBID, OpenAI embeddings, UMAP
- **Infrastructure**: Docker, Kubernetes

## Development Phases

The project is implemented in 8 phases:

1. **Phase 0**: Setup & onboarding
2. **Phase 1**: Ingestion & boilerplate removal
3. **Phase 2**: Document embeddings
4. **Phase 3**: DR mapping & parametric mapper
5. **Phase 4**: Vector DB, indexing & click API
6. **Phase 5**: Frontend & user overlay
7. **Phase 6**: Tile generation, LOD, cluster labels
8. **Phase 7**: Tests, CI, monitoring & runbook
9. **Phase 8**: Production hardening & deliverables

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make changes and add tests
4. Run tests: `pytest`
5. Submit a pull request

## Support

For questions or issues:
- Check the documentation in this directory
- Review the API documentation
- Check the runbook for operational issues
- Open an issue on GitHub
