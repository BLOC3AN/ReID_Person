# Production Deployment Guide

Complete guide for deploying the Person ReID system in production.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Load Balancer                         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐   ┌────────▼────────┐   ┌───────▼────────┐
│  Streamlit UI  │   │  Detection API  │   │  Register API  │
│   (Port 8501)  │   │   (Port 8003)   │   │  (Port 8002)   │
└────────────────┘   └─────────────────┘   └────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐   ┌────────▼────────┐   ┌───────▼────────┐
│ Triton Server  │   │    Qdrant DB    │   │   PostgreSQL   │
│ (8100-8102)    │   │   (Port 6333)   │   │  (Port 5432)   │
└────────────────┘   └─────────────────┘   └────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Redis + Kafka    │
                    │  (6379 + 9092)    │
                    └───────────────────┘
```

## Prerequisites

### Hardware Requirements

**Minimum:**
- CPU: 8 cores
- RAM: 16GB
- GPU: NVIDIA GPU with 8GB VRAM (e.g., RTX 3070)
- Storage: 50GB SSD

**Recommended (Production):**
- CPU: 16+ cores
- RAM: 32GB+
- GPU: NVIDIA GPU with 16GB+ VRAM (e.g., RTX 4090, A4000)
- Storage: 100GB+ NVMe SSD

### Software Requirements

- Ubuntu 20.04/22.04 LTS
- Docker 24.0+
- Docker Compose 2.20+
- NVIDIA Driver 525+
- nvidia-docker2
- CUDA 11.8+

## Installation Steps

### 1. Install Docker & NVIDIA Container Toolkit

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 2. Clone Repository

```bash
git clone <repository-url>
cd person_reid_system
git checkout v1.0.1  # Use stable version
```

### 3. Configure Environment

```bash
# Copy environment template
cp configs/.env.example configs/.env

# Edit configuration
nano configs/.env
```

**Key settings:**
```env
# Qdrant
QDRANT_URI=http://127.0.0.1:6333
QDRANT_COLLECTION=cross_camera_matching_id

# Triton
TRITON_URL=localhost:8101

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password

# Redis
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
```

### 4. Build Docker Images

```bash
cd deployment

# Build all services
docker-compose build

# Or build individually
docker-compose build triton
docker-compose build detection
docker-compose build register
```

### 5. Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 6. Verify Deployment

```bash
# Check Triton models
curl http://localhost:8100/v2/health/ready
curl http://localhost:8100/v2/models/bytetrack_tensorrt/ready
curl http://localhost:8100/v2/models/arcface_onnx/ready
curl http://localhost:8100/v2/models/scrfd_10g/ready

# Check Detection API
curl http://localhost:8003/health

# Check Register API
curl http://localhost:8002/health
```

## Configuration

### Triton Optimization

Edit `triton_model_repository/bytetrack_tensorrt/config.pbtxt`:

```pbtxt
# Adjust instance count based on GPU memory
instance_group [
  {
    count: 4  # 4 instances = ~2GB GPU memory
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

**Instance count guidelines:**
- 1-4 cameras: 4 instances
- 5-8 cameras: 8 instances
- 9-16 cameras: 16 instances

### Detection Parameters

Edit `configs/config.yaml`:

```yaml
detection:
  backend: triton
  conf_threshold: 0.5    # Lower = more detections
  nms_threshold: 0.45

tracking:
  track_thresh: 0.5      # Lower = more tracks
  track_buffer: 30       # Frames to keep lost tracks

reid:
  backend: triton_pipeline
  triton:
    face_conf_threshold: 0.5  # Lower = detect more faces

matching:
  similarity_threshold: 0.8   # Higher = stricter matching
```

## Monitoring

### Health Checks

```bash
# Service health
curl http://localhost:8003/health
curl http://localhost:8002/health

# Triton metrics
curl http://localhost:8102/metrics

# Redis stats
docker exec person_reid_redis redis-cli INFO stats
```

### Logs

```bash
# View logs
docker-compose logs -f detection
docker-compose logs -f triton

# Export logs
docker-compose logs detection > detection.log
```

### Resource Monitoring

```bash
# GPU usage
nvidia-smi -l 1

# Container stats
docker stats

# Disk usage
du -sh outputs/
```

## Backup & Recovery

### Backup Qdrant Database

```bash
# Create snapshot
curl -X POST http://localhost:6333/collections/cross_camera_matching_id/snapshots

# Download snapshot
curl http://localhost:6333/collections/cross_camera_matching_id/snapshots/<snapshot_name> \
  -o qdrant_backup.snapshot
```

### Backup PostgreSQL

```bash
docker exec person_reid_postgres pg_dump -U postgres hailt_imespro > postgres_backup.sql
```

### Restore

```bash
# Restore Qdrant
curl -X PUT http://localhost:6333/collections/cross_camera_matching_id/snapshots/upload \
  --data-binary @qdrant_backup.snapshot

# Restore PostgreSQL
docker exec -i person_reid_postgres psql -U postgres hailt_imespro < postgres_backup.sql
```

## Scaling

### Horizontal Scaling (Multiple Servers)

1. **Shared Storage:** Use NFS/S3 for `outputs/` directory
2. **Shared Database:** External Qdrant/PostgreSQL/Redis cluster
3. **Load Balancer:** Nginx/HAProxy for API endpoints

### Vertical Scaling (Single Server)

1. **More GPU Memory:** Increase Triton instances
2. **More CPU Cores:** Increase worker threads
3. **More RAM:** Increase batch sizes

## Security

### API Authentication

Add authentication middleware to FastAPI services:

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/detect")
async def detect(token: str = Security(security)):
    # Verify token
    pass
```

### Network Security

```bash
# Firewall rules (UFW)
sudo ufw allow 8501/tcp  # Streamlit UI
sudo ufw allow 8003/tcp  # Detection API
sudo ufw allow 8002/tcp  # Register API
sudo ufw deny 6333/tcp   # Block external Qdrant access
sudo ufw deny 5432/tcp   # Block external PostgreSQL access
```

### SSL/TLS

Use reverse proxy (Nginx) with Let's Encrypt:

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8501;
    }
}
```

## Troubleshooting

See [Troubleshooting Guide](TROUBLESHOOTING.md) for common issues.

## Maintenance

### Update System

```bash
# Pull latest code
git pull origin main

# Rebuild images
docker-compose build

# Restart services
docker-compose down
docker-compose up -d
```

### Clean Up

```bash
# Remove old outputs (keep last 7 days)
find outputs/ -type f -mtime +7 -delete

# Clean Docker
docker system prune -a

# Clean Qdrant snapshots
curl -X DELETE http://localhost:6333/collections/cross_camera_matching_id/snapshots/<old_snapshot>
```

## Support

- GitHub Issues: <repository-url>/issues
- Documentation: [docs/](.)
- API Reference: [API.md](API.md)
