# Knowledge Map Operational Runbook

## Overview

This runbook provides operational procedures for the Knowledge Map system, including deployment, monitoring, troubleshooting, and maintenance tasks.

## System Architecture

- **Backend**: FastAPI service with PostgreSQL, Qdrant, Redis
- **Frontend**: Next.js application with deck.gl visualization
- **Pipeline**: Document processing, embedding generation, dimensionality reduction
- **Monitoring**: Prometheus, Grafana, Sentry

## Deployment

### Development Environment

```bash
# Start all services
docker-compose up -d

# Check service health
curl http://localhost:8000/api/health
curl http://localhost:3000

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Production Deployment

```bash
# Deploy with Helm
helm install knowledge-map ./infra/k8s/helm/knowledge-map

# Check deployment status
kubectl get pods -l app=knowledge-map
kubectl get services

# View logs
kubectl logs -l app=knowledge-map-backend
```

## Monitoring

### Key Metrics

- **API Performance**: Request rate, response time, error rate
- **Document Processing**: Processing rate, queue depth, success rate
- **Vector Database**: Index size, query performance, memory usage
- **System Resources**: CPU, memory, disk usage

### Grafana Dashboard

Access: `http://localhost:3001` (admin/admin)

Key dashboards:
- System Overview
- API Performance
- Document Processing
- Vector Database Health

### Alerts

Critical alerts:
- API error rate > 5%
- Response time P95 > 2s
- Document processing queue > 1000
- Vector database unavailable
- Memory usage > 90%

## Troubleshooting

### Common Issues

#### 1. API Service Unavailable

**Symptoms**: 503 errors, connection refused
**Diagnosis**:
```bash
# Check service status
kubectl get pods -l app=knowledge-map-backend
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>
```

**Resolution**:
- Restart service: `kubectl rollout restart deployment/knowledge-map-backend`
- Check resource limits
- Verify database connectivity

#### 2. Document Processing Failures

**Symptoms**: Documents stuck in processing queue
**Diagnosis**:
```bash
# Check processing queue
curl http://localhost:8000/api/ingest/status

# Check GROBID service
curl http://localhost:8070/api/isalive
```

**Resolution**:
- Restart GROBID service
- Check disk space for temporary files
- Verify GROBID model files

#### 3. Vector Database Issues

**Symptoms**: Search returns no results, slow queries
**Diagnosis**:
```bash
# Check Qdrant health
curl http://localhost:6333/collections

# Check collection status
curl http://localhost:6333/collections/documents
```

**Resolution**:
- Restart Qdrant service
- Rebuild index if corrupted
- Check memory usage

#### 4. Frontend Loading Issues

**Symptoms**: Map doesn't load, JavaScript errors
**Diagnosis**:
```bash
# Check frontend logs
kubectl logs -l app=knowledge-map-frontend

# Check API connectivity
curl http://localhost:8000/api/mapping/coordinates
```

**Resolution**:
- Restart frontend service
- Check API endpoint availability
- Verify Mapbox token configuration

### Performance Issues

#### Slow API Responses

1. Check database query performance
2. Monitor vector database query times
3. Review embedding generation latency
4. Check system resource usage

#### High Memory Usage

1. Monitor document processing queue
2. Check for memory leaks in embedding generation
3. Review vector database memory usage
4. Scale services if needed

## Maintenance

### Regular Tasks

#### Daily
- Monitor system health metrics
- Check error rates and alerts
- Review document processing queue

#### Weekly
- Review performance trends
- Check disk space usage
- Update monitoring dashboards

#### Monthly
- Review and update dependencies
- Analyze usage patterns
- Plan capacity scaling

### Backup Procedures

#### Database Backup

```bash
# PostgreSQL backup
pg_dump -h localhost -U postgres knowledge_map > backup_$(date +%Y%m%d).sql

# Restore
psql -h localhost -U postgres knowledge_map < backup_20240101.sql
```

#### Vector Database Backup

```bash
# Qdrant snapshot
curl -X POST http://localhost:6333/snapshots/documents

# Restore from snapshot
# (See Qdrant documentation for restore procedures)
```

#### Configuration Backup

```bash
# Backup configuration files
tar -czf config_backup_$(date +%Y%m%d).tar.gz \
  infra/ \
  .env \
  docker-compose.yml
```

### Scaling Procedures

#### Horizontal Scaling

```bash
# Scale backend service
kubectl scale deployment knowledge-map-backend --replicas=3

# Scale frontend service
kubectl scale deployment knowledge-map-frontend --replicas=2
```

#### Vertical Scaling

```bash
# Update resource limits
kubectl patch deployment knowledge-map-backend -p '{"spec":{"template":{"spec":{"containers":[{"name":"backend","resources":{"limits":{"memory":"2Gi","cpu":"1000m"}}}]}}}}'
```

## Incident Response

### Severity Levels

- **P1 (Critical)**: System down, data loss
- **P2 (High)**: Major functionality impaired
- **P3 (Medium)**: Minor functionality issues
- **P4 (Low)**: Cosmetic issues, enhancements

### Response Procedures

#### P1/P2 Incidents

1. **Immediate Response** (0-15 minutes)
   - Acknowledge incident
   - Assess impact and scope
   - Notify stakeholders

2. **Investigation** (15-60 minutes)
   - Gather logs and metrics
   - Identify root cause
   - Implement temporary fix if possible

3. **Resolution** (1-4 hours)
   - Implement permanent fix
   - Verify system stability
   - Update monitoring

4. **Post-Incident** (24-48 hours)
   - Conduct post-mortem
   - Document lessons learned
   - Update runbook

### Escalation Matrix

- **On-call Engineer**: Initial response
- **Team Lead**: P1/P2 incidents
- **Engineering Manager**: P1 incidents, major outages
- **Product Manager**: User impact assessment

## Security

### Access Control

- Use least privilege principle
- Rotate credentials regularly
- Monitor access logs
- Implement 2FA for admin access

### Vulnerability Management

- Regular security scans
- Dependency updates
- Penetration testing
- Security patch management

### Data Protection

- Encrypt data at rest and in transit
- Regular backup verification
- Access logging and monitoring
- GDPR compliance procedures

## Contact Information

- **On-call**: +1-XXX-XXX-XXXX
- **Slack**: #knowledge-map-alerts
- **Email**: knowledge-map-team@company.com
- **Documentation**: https://docs.company.com/knowledge-map

## Appendix

### Useful Commands

```bash
# Check all services
docker-compose ps

# View resource usage
docker stats

# Access service shell
docker-compose exec backend bash

# Check service health
curl http://localhost:8000/api/health
curl http://localhost:3000/api/health

# View Prometheus metrics
curl http://localhost:9090/metrics
```

### Configuration Files

- `docker-compose.yml`: Development environment
- `infra/k8s/`: Kubernetes manifests
- `infra/monitoring/`: Monitoring configuration
- `.env.template`: Environment variables template