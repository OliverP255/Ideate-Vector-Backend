# Knowledge Map Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Knowledge Map system in various environments, from development to production.

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- Memory: 8GB RAM
- Storage: 100GB SSD
- OS: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

**Recommended for Production:**
- CPU: 16+ cores
- Memory: 64GB+ RAM
- Storage: 1TB+ SSD
- OS: Linux (Ubuntu 22.04 LTS)

### Software Dependencies

- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (for production)
- Helm 3.8+ (for production)
- Python 3.12+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+

## Development Deployment

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/company/knowledge-map.git
cd knowledge-map
```

2. **Set up environment:**
```bash
cp .env.template .env
# Edit .env with your configuration
```

3. **Start services:**
```bash
docker-compose up -d
```

4. **Verify deployment:**
```bash
# Check backend health
curl http://localhost:8000/api/health

# Check frontend
curl http://localhost:3000

# View logs
docker-compose logs -f
```

### Manual Setup (Without Docker)

1. **Install Python dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Install Node.js dependencies:**
```bash
cd frontend
npm install
```

3. **Start services:**
```bash
# Terminal 1: Backend
cd backend
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

4. **Initialize data:**
```bash
# Run initial data processing
python3 pipeline/ingest/worker.py --sample-data
python3 pipeline/embedding/generate.py --sample-data
python3 pipeline/dr/mapping.py --sample-data
```

## Production Deployment

### Kubernetes Deployment

1. **Prepare cluster:**
```bash
# Create namespace
kubectl create namespace knowledge-map

# Create secrets
kubectl create secret generic knowledge-map-secrets \
  --from-literal=postgres-password=your-secure-password \
  --from-literal=redis-password=your-redis-password \
  --from-literal=api-key=your-api-key \
  --namespace=knowledge-map
```

2. **Deploy with Helm:**
```bash
# Add Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Deploy the application
helm install knowledge-map ./infra/k8s/helm/knowledge-map \
  --namespace knowledge-map \
  --values ./infra/k8s/helm/knowledge-map/values-production.yaml
```

3. **Verify deployment:**
```bash
# Check pods
kubectl get pods -n knowledge-map

# Check services
kubectl get services -n knowledge-map

# Check ingress
kubectl get ingress -n knowledge-map
```

### Production Configuration

Create `values-production.yaml`:

```yaml
# Production values
backend:
  replicaCount: 3
  resources:
    limits:
      cpu: 2000m
      memory: 4Gi
    requests:
      cpu: 1000m
      memory: 2Gi

frontend:
  replicaCount: 2
  resources:
    limits:
      cpu: 1000m
      memory: 2Gi
    requests:
      cpu: 500m
      memory: 1Gi

postgresql:
  auth:
    postgresPassword: "secure-production-password"
  primary:
    persistence:
      size: 100Gi
    resources:
      limits:
        cpu: 2000m
        memory: 4Gi

qdrant:
  replicaCount: 2
  persistence:
    size: 200Gi
  resources:
    limits:
      cpu: 4000m
      memory: 16Gi

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
    adminPassword: "secure-grafana-password"

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: api.knowledge-map.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: knowledge-map-tls
      hosts:
        - api.knowledge-map.com
```

## Environment-Specific Configurations

### Development Environment

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
    volumes:
      - ./data:/app/data
      - ./backend:/app
    depends_on:
      - postgres
      - redis
      - qdrant

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    volumes:
      - ./frontend:/app
      - /app/node_modules
    depends_on:
      - backend

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: knowledge_map
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  postgres_data:
  qdrant_data:
```

### Staging Environment

**values-staging.yaml:**
```yaml
backend:
  replicaCount: 2
  resources:
    limits:
      cpu: 1000m
      memory: 2Gi

postgresql:
  primary:
    persistence:
      size: 50Gi

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true

ingress:
  enabled: true
  hosts:
    - host: staging-api.knowledge-map.com
      paths:
        - path: /
          pathType: Prefix
```

## Database Setup

### PostgreSQL Configuration

1. **Create database:**
```sql
CREATE DATABASE knowledge_map;
CREATE USER knowledge_map_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE knowledge_map TO knowledge_map_user;
```

2. **Run migrations:**
```bash
cd backend
python3 -c "from app.db.database import init_db; import asyncio; asyncio.run(init_db())"
```

3. **Create indexes:**
```sql
-- Spatial index for coordinates
CREATE INDEX idx_document_coordinates ON documents USING GIST (coordinates);

-- Text search index
CREATE INDEX idx_document_text_search ON documents USING GIN (to_tsvector('english', title || ' ' || content));
```

### Qdrant Configuration

1. **Create collection:**
```bash
curl -X PUT http://localhost:6333/collections/documents \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 384,
      "distance": "Cosine"
    },
    "hnsw_config": {
      "m": 16,
      "ef_construct": 100,
      "full_scan_threshold": 10000
    }
  }'
```

2. **Verify collection:**
```bash
curl http://localhost:6333/collections/documents
```

## Monitoring Setup

### Prometheus Configuration

1. **Install Prometheus:**
```bash
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --set server.persistentVolume.size=20Gi
```

2. **Configure scraping:**
```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'knowledge-map-backend'
    static_configs:
      - targets: ['knowledge-map-backend:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboard

1. **Install Grafana:**
```bash
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set adminPassword=admin \
  --set persistence.enabled=true
```

2. **Import dashboard:**
```bash
# Import the provided dashboard
kubectl create configmap grafana-dashboard \
  --from-file=knowledge-map-dashboard.json=infra/monitoring/grafana-dashboard.json \
  --namespace=monitoring
```

## Security Configuration

### SSL/TLS Setup

1. **Install cert-manager:**
```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

2. **Create ClusterIssuer:**
```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@knowledge-map.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: knowledge-map-network-policy
  namespace: knowledge-map
spec:
  podSelector:
    matchLabels:
      app: knowledge-map
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: knowledge-map
    ports:
    - protocol: TCP
      port: 5432
    - protocol: TCP
      port: 6379
    - protocol: TCP
      port: 6333
```

## Backup and Restore

### Automated Backups

1. **Create backup cron job:**
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: knowledge-map-backup
  namespace: knowledge-map
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15
            command:
            - /bin/bash
            - -c
            - |
              pg_dump -h postgres -U postgres knowledge_map > /backup/postgres_$(date +%Y%m%d).sql
              # Add other backup commands
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

2. **Run backup script:**
```bash
# Manual backup
./scripts/backup.sh

# Restore from backup
./scripts/restore.sh knowledge_map_backup_20240101_120000
```

## Performance Optimization

### Database Optimization

1. **PostgreSQL tuning:**
```sql
-- Increase shared buffers
ALTER SYSTEM SET shared_buffers = '256MB';

-- Optimize work memory
ALTER SYSTEM SET work_mem = '4MB';

-- Enable query optimization
ALTER SYSTEM SET random_page_cost = 1.1;
```

2. **Qdrant optimization:**
```bash
# Optimize collection settings
curl -X PATCH http://localhost:6333/collections/documents \
  -H 'Content-Type: application/json' \
  -d '{
    "optimizers_config": {
      "default_segment_number": 2,
      "max_segment_size": 20000,
      "memmap_threshold": 50000
    }
  }'
```

### Caching Strategy

1. **Redis configuration:**
```bash
# Enable persistence
redis-cli CONFIG SET save "900 1 300 10 60 10000"

# Set memory policy
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

2. **Application caching:**
```python
# Enable response caching
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

FastAPICache.init(RedisBackend(), prefix="knowledge-map")
```

## Troubleshooting

### Common Issues

1. **Service won't start:**
```bash
# Check logs
kubectl logs -l app=knowledge-map-backend -n knowledge-map

# Check resource usage
kubectl top pods -n knowledge-map

# Check events
kubectl get events -n knowledge-map
```

2. **Database connection issues:**
```bash
# Test connection
kubectl exec -it postgres-0 -n knowledge-map -- psql -U postgres -d knowledge_map

# Check network policies
kubectl get networkpolicies -n knowledge-map
```

3. **Performance issues:**
```bash
# Check metrics
kubectl port-forward svc/prometheus-server 9090:80 -n monitoring

# Analyze slow queries
kubectl exec -it postgres-0 -n knowledge-map -- psql -U postgres -d knowledge_map -c "SELECT * FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

### Health Checks

```bash
# Backend health
curl http://localhost:8000/api/health

# Database health
kubectl exec -it postgres-0 -n knowledge-map -- pg_isready

# Vector database health
curl http://localhost:6333/health

# Redis health
kubectl exec -it redis-0 -n knowledge-map -- redis-cli ping
```

## Scaling

### Horizontal Scaling

```bash
# Scale backend
kubectl scale deployment knowledge-map-backend --replicas=5 -n knowledge-map

# Scale frontend
kubectl scale deployment knowledge-map-frontend --replicas=3 -n knowledge-map
```

### Vertical Scaling

```bash
# Update resource limits
kubectl patch deployment knowledge-map-backend -n knowledge-map -p '{"spec":{"template":{"spec":{"containers":[{"name":"backend","resources":{"limits":{"memory":"8Gi","cpu":"4000m"}}}]}}}}'
```

### Auto-scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: knowledge-map-backend-hpa
  namespace: knowledge-map
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: knowledge-map-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Maintenance

### Regular Tasks

1. **Daily:**
   - Monitor system health
   - Check error rates
   - Review resource usage

2. **Weekly:**
   - Update dependencies
   - Review performance metrics
   - Clean up old logs

3. **Monthly:**
   - Security updates
   - Backup verification
   - Capacity planning

### Updates

```bash
# Update application
helm upgrade knowledge-map ./infra/k8s/helm/knowledge-map \
  --namespace knowledge-map \
  --values values-production.yaml

# Rollback if needed
helm rollback knowledge-map 1 -n knowledge-map
```

## Support

For deployment support:
- **Email**: devops@knowledge-map.com
- **Documentation**: https://docs.knowledge-map.com/deployment
- **Status Page**: https://status.knowledge-map.com
