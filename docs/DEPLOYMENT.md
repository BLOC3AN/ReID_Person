# Deployment Guide

## Overview

Hệ thống Person ReID hỗ trợ 2 phương thức deployment:

1. **Local Development** - Chạy trực tiếp trên máy local
2. **Docker Deployment** - Microservices với GPU support

---

## 1. Local Development Deployment

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU)
- 16GB RAM minimum
- 50GB disk space

### Installation Steps

#### 1.1. Clone Repository

```bash
cd /path/to/workspace
# Repository đã có sẵn tại person_reid_system/
```

#### 1.2. Create Virtual Environment

```bash
# Tạo virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

#### 1.3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies chính:**
- torch >= 1.10.0
- torchvision >= 0.11.0
- opencv-python >= 4.5.0
- insightface >= 0.7.3 (ArcFace)
- qdrant-client >= 1.0.0
- streamlit >= 1.28.0
- fastapi >= 0.104.0

#### 1.4. Configure Qdrant

Tạo file `configs/.env`:

```bash
cp configs/.env.example configs/.env
```

Edit `configs/.env`:

```env
QDRANT_API_KEY=your_api_key_here
QDRANT_URI=host=your-cluster.cloud.qdrant.io
QDRANT_COLLECTION=cross_camera_matching_id
QDRANT_PORT=6333
```

**Lấy Qdrant credentials:**
1. Đăng ký tại https://cloud.qdrant.io
2. Tạo cluster mới
3. Copy API key và cluster URL
4. Paste vào `.env`

#### 1.5. Verify Installation

```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check OpenCV
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Check Qdrant
python -c "from qdrant_client import QdrantClient; print('Qdrant: OK')"

# Check ArcFace (sẽ download model lần đầu ~282MB)
python -c "from core import ArcFaceExtractor; print('ArcFace: OK')"
```

#### 1.6. Check Models

```bash
ls -lh models/
```

Should show:
- `bytetrack_x_mot17.pth.tar` (757 MB)
- `yolox_x.pth` (757 MB)

### Running Services Locally

#### Option 1: Web UI (Recommended)

```bash
# Launch Streamlit UI
./run_ui.sh

# Or manually
streamlit run app.py
```

Access at: http://localhost:8501

#### Option 2: Command Line

**Extract objects:**
```bash
python scripts/extract_objects.py \
  --video data/videos/multi_person.mp4 \
  --output-dir ./output_objects \
  --model mot17
```

**Register person:**
```bash
python scripts/register_mot17.py \
  --video data/videos/person.mp4 \
  --name "John" \
  --global-id 1 \
  --sample-rate 5
```

**Detect and track:**
```bash
python scripts/detect_and_track.py \
  --video data/videos/test.mp4 \
  --model mot17 \
  --threshold 0.8
```

#### Option 3: API Services

**Terminal 1 - Extract Service:**
```bash
cd services
python extract_service.py
# Running on http://localhost:8001
```

**Terminal 2 - Register Service:**
```bash
cd services
python register_service.py
# Running on http://localhost:8002
```

**Terminal 3 - Detection Service:**
```bash
cd services
python detection_service.py
# Running on http://localhost:8003
```

**Terminal 4 - UI:**
```bash
streamlit run app.py
# Running on http://localhost:8501
```

---

## 2. Docker Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA GPU with CUDA 11.8+
- NVIDIA Container Toolkit
- 16GB RAM minimum
- 50GB disk space

### GPU Setup (First-time Only)

#### 2.1. Install NVIDIA Container Toolkit

```bash
# Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

#### 2.2. Configure Docker Daemon

```bash
# Configure NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Create daemon.json
sudo bash -c 'cat > /etc/docker/daemon.json << EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF'
```

#### 2.3. Start Docker Daemon

```bash
cd deployment
./start_docker_daemon.sh
```

**Script nội dung:**
```bash
#!/bin/bash
sudo dockerd --config-file /etc/docker/daemon.json > /tmp/dockerd.log 2>&1 &
sleep 5
sudo docker info | grep -i runtime
```

#### 2.4. Verify GPU Access

```bash
# Check Docker runtime
sudo docker info | grep -i runtime

# Test GPU in container
sudo docker run --rm --runtime=nvidia \
  nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Should show GPU information
```

### Docker Deployment Steps

#### 2.5. Configure Environment

```bash
# Ensure .env exists
ls configs/.env

# If not, create from example
cp configs/.env.example configs/.env
```

Edit `configs/.env` với Qdrant credentials.

#### 2.6. Build and Start Services

```bash
cd deployment

# Build images
sudo docker compose build

# Start all services
sudo docker compose up -d

# Check status
sudo docker compose ps
```

**Expected output:**
```
NAME                    STATUS              PORTS
person_reid_ui          Up (healthy)        0.0.0.0:8501->8501/tcp
person_reid_extract     Up (healthy)        0.0.0.0:8001->8001/tcp
person_reid_register    Up (healthy)        0.0.0.0:8002->8002/tcp
person_reid_detection   Up (healthy)        0.0.0.0:8003->8003/tcp
```

#### 2.7. Verify Services

```bash
# Check logs
sudo docker compose logs -f

# Check individual service
sudo docker compose logs extract --tail=50

# Check GPU access in container
sudo docker exec person_reid_extract nvidia-smi
sudo docker exec person_reid_extract python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### 2.8. Access Services

- **Web UI:** http://localhost:8501
- **Extract API Docs:** http://localhost:8001/docs
- **Register API Docs:** http://localhost:8002/docs
- **Detection API Docs:** http://localhost:8003/docs

### Docker Management

#### View Logs

```bash
# All services
sudo docker compose logs -f

# Specific service
sudo docker compose logs extract -f
sudo docker compose logs register -f
sudo docker compose logs detection -f
sudo docker compose logs ui -f

# Last N lines
sudo docker compose logs extract --tail=100
```

#### Restart Services

```bash
# Restart all
sudo docker compose restart

# Restart specific service
sudo docker compose restart extract
sudo docker compose restart register
sudo docker compose restart detection
sudo docker compose restart ui
```

#### Stop Services

```bash
# Stop all
sudo docker compose down

# Stop and remove volumes
sudo docker compose down -v
```

#### Rebuild Services

```bash
# Rebuild all
sudo docker compose up -d --build

# Rebuild specific service
sudo docker compose up -d --build extract
```

#### Scale Services

```bash
# Scale extract service to 2 instances
sudo docker compose up -d --scale extract=2
```

### Docker Compose Configuration

**File:** `deployment/docker-compose.yml`

**Services:**

1. **extract** (GPU)
   - Port: 8001
   - Runtime: nvidia
   - Shared memory: 4GB
   - Health check: /health endpoint

2. **register** (GPU)
   - Port: 8002
   - Runtime: nvidia
   - Shared memory: 4GB
   - Health check: /health endpoint

3. **detection** (GPU)
   - Port: 8003
   - Runtime: nvidia
   - Shared memory: 4GB
   - Health check: /health endpoint

4. **ui** (CPU)
   - Port: 8501
   - Shared memory: 2GB
   - Health check: /_stcore/health endpoint

**Shared Volumes:**
- `../outputs` → `/app/outputs`
- `../data` → `/app/data`
- `../logs` → `/app/logs`
- `../configs` → `/app/configs`
- `../models` → `/app/models`

**Network:**
- Bridge network: `person_reid_network`

---

## 3. Production Deployment

### Recommendations

#### 3.1. Resource Allocation

**Minimum:**
- CPU: 8 cores
- RAM: 16GB
- GPU: NVIDIA GPU with 8GB VRAM
- Disk: 100GB SSD

**Recommended:**
- CPU: 16 cores
- RAM: 32GB
- GPU: NVIDIA GPU with 16GB VRAM (V100, A100)
- Disk: 500GB SSD

#### 3.2. Security

**API Keys:**
```bash
# Use strong API keys
QDRANT_API_KEY=$(openssl rand -base64 32)
```

**Firewall:**
```bash
# Allow only necessary ports
sudo ufw allow 8501/tcp  # UI
sudo ufw allow 8001/tcp  # Extract API
sudo ufw allow 8002/tcp  # Register API
sudo ufw allow 8003/tcp  # Detection API
```

**HTTPS:**
```bash
# Use reverse proxy (nginx) with SSL
# Example nginx config in deployment/nginx.conf
```

#### 3.3. Monitoring

**Docker Stats:**
```bash
# Real-time stats
sudo docker stats

# Specific container
sudo docker stats person_reid_extract
```

**GPU Monitoring:**
```bash
# Watch GPU usage
watch -n 1 nvidia-smi
```

**Logs:**
```bash
# Centralized logging
sudo docker compose logs -f > /var/log/person_reid.log
```

#### 3.4. Backup

**Database:**
```bash
# Backup Qdrant data
cp data/database/reid_database.pkl backups/reid_database_$(date +%Y%m%d).pkl

# Backup Qdrant collection (via API)
# See Qdrant documentation
```

**Configuration:**
```bash
# Backup configs
tar -czf backups/configs_$(date +%Y%m%d).tar.gz configs/
```

#### 3.5. Auto-restart

**Docker Compose:**
```yaml
services:
  extract:
    restart: unless-stopped
```

**Systemd Service:**
```bash
# Create /etc/systemd/system/person-reid.service
[Unit]
Description=Person ReID System
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/path/to/person_reid_system/deployment
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable person-reid
sudo systemctl start person-reid
```

---

## 4. Troubleshooting

### GPU Issues

**Problem:** GPU not detected in container

**Solution:**
```bash
# 1. Check NVIDIA driver
nvidia-smi

# 2. Check Docker runtime
sudo docker info | grep -i runtime

# 3. Restart Docker daemon
cd deployment
./start_docker_daemon.sh

# 4. Test GPU access
sudo docker run --rm --runtime=nvidia nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Service Unhealthy

**Problem:** Service shows unhealthy status

**Solution:**
```bash
# Check logs
sudo docker compose logs <service_name>

# Check health endpoint
curl http://localhost:8001/health  # extract
curl http://localhost:8002/health  # register
curl http://localhost:8003/health  # detection

# Restart service
sudo docker compose restart <service_name>
```

### Out of Memory

**Problem:** CUDA out of memory

**Solution:**
```bash
# 1. Reduce batch size in code
# 2. Use smaller model
# 3. Increase GPU memory
# 4. Process fewer frames at once
```

### Port Already in Use

**Problem:** Port 8501 already in use

**Solution:**
```bash
# Find process using port
sudo lsof -i :8501

# Kill process
sudo kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "8502:8501"  # Use 8502 instead
```

### Qdrant Connection Failed

**Problem:** Cannot connect to Qdrant

**Solution:**
```bash
# 1. Check .env file
cat configs/.env

# 2. Test connection
python -c "
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
load_dotenv('configs/.env')
client = QdrantClient(url=os.getenv('QDRANT_URI'), api_key=os.getenv('QDRANT_API_KEY'))
print(client.get_collections())
"

# 3. Check Qdrant cloud status
# Visit https://cloud.qdrant.io
```

---

## 5. Performance Tuning

### GPU Optimization

```yaml
# docker-compose.yml
services:
  extract:
    environment:
      - NVIDIA_VISIBLE_DEVICES=0  # Use specific GPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Shared Memory

```yaml
# Increase if needed
shm_size: '8gb'  # Default: 4gb
```

### Concurrent Jobs

```python
# In service code, use ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)
```

---

## 6. Maintenance

### Regular Tasks

**Daily:**
- Check service health
- Monitor GPU usage
- Review error logs

**Weekly:**
- Backup database
- Clean old job files
- Update dependencies

**Monthly:**
- Review performance metrics
- Optimize database
- Update documentation

### Cleanup

```bash
# Remove old job files
find outputs/extracted_objects -mtime +7 -delete
find data/uploads -mtime +7 -delete

# Clean Docker
sudo docker system prune -a
```

