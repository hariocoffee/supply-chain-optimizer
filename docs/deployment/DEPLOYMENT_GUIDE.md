# Deployment Guide

## Production Deployment

### Prerequisites
- Docker 24.0+
- Docker Compose 2.0+
- 16GB RAM minimum
- 4 CPU cores minimum
- 100GB storage

### Environment Setup

#### 1. Environment Variables
Create `.env` file:
```bash
# Database
POSTGRES_DB=supply_chain_optimization
POSTGRES_USER=sc_admin
POSTGRES_PASSWORD=secure_password_here
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redis_password_here

# Ollama
OLLAMA_HOST=ollama
OLLAMA_PORT=11434
OLLAMA_MODEL=qwen2.5:0.5b

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=jwt_secret_here

# Monitoring
LOG_LEVEL=INFO
METRICS_ENABLED=true
```

#### 2. Production Docker Compose
```yaml
version: '3.8'

services:
  streamlit-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - ENVIRONMENT=production
    volumes:
      - ./data:/app/data:ro
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
      - ollama
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "auth", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  postgres_data:
  redis_data:
  ollama_data:
```

### Kubernetes Deployment

#### 1. Namespace
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: supply-chain-optimization
```

#### 2. ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: supply-chain-optimization
data:
  POSTGRES_HOST: "postgres-service"
  REDIS_HOST: "redis-service"
  OLLAMA_HOST: "ollama-service"
  LOG_LEVEL: "INFO"
```

#### 3. Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: streamlit-app
  namespace: supply-chain-optimization
spec:
  replicas: 3
  selector:
    matchLabels:
      app: streamlit-app
  template:
    metadata:
      labels:
        app: streamlit-app
    spec:
      containers:
      - name: streamlit-app
        image: supply-chain-optimizer:latest
        ports:
        - containerPort: 8501
        envFrom:
        - configMapRef:
            name: app-config
        - secretRef:
            name: app-secrets
        resources:
          requests:
            memory: "4Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
```

### Load Balancer Configuration

#### NGINX Configuration
```nginx
upstream streamlit_backend {
    least_conn;
    server app1:8501 max_fails=3 fail_timeout=30s;
    server app2:8501 max_fails=3 fail_timeout=30s;
    server app3:8501 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    location / {
        proxy_pass http://streamlit_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Static files
    location /static/ {
        alias /app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### Monitoring Setup

#### Prometheus Configuration
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'streamlit-app'
    static_configs:
      - targets: ['streamlit-app:8501']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Supply Chain Optimization Metrics",
    "panels": [
      {
        "title": "Optimization Requests/min",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(optimization_requests_total[1m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, optimization_duration_seconds_bucket)"
          }
        ]
      }
    ]
  }
}
```

### Security Configuration

#### 1. SSL/TLS
- Use Let's Encrypt for automatic certificate management
- Configure HSTS headers
- Implement certificate pinning

#### 2. Network Security
```yaml
# Network Policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: app-network-policy
spec:
  podSelector:
    matchLabels:
      app: streamlit-app
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
      port: 8501
```

### Backup Strategy

#### Database Backup
```bash
#!/bin/bash
# backup-database.sh
BACKUP_DIR="/backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

docker exec postgres pg_dump -U $POSTGRES_USER $POSTGRES_DB > \
  $BACKUP_DIR/backup_$TIMESTAMP.sql

# Retain last 30 days
find $BACKUP_DIR -name "backup_*.sql" -mtime +30 -delete
```

#### Redis Backup
```bash
#!/bin/bash
# backup-redis.sh
BACKUP_DIR="/backups/redis"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

docker exec redis redis-cli --rdb - > \
  $BACKUP_DIR/dump_$TIMESTAMP.rdb
```

### Deployment Commands

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale streamlit-app=3

# Rolling update
docker-compose -f docker-compose.prod.yml up -d --no-deps streamlit-app

# Health check
curl -f http://localhost:8501/health

# View logs
docker-compose logs -f streamlit-app
```