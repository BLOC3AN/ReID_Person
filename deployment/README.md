# Person ReID System - Docker Deployment

## 📦 Services Architecture

This deployment uses a **microservices architecture** with separate services:

- **extract** (port 8001): Video object extraction service
- **register** (port 8002): Person registration service
- **detection** (port 8003): Detection and tracking service
- **ui** (port 8501): Streamlit web interface

**Note**: Qdrant vector database runs externally at `../deployment/qdrant/`

## 🔧 GPU Support Setup

### Prerequisites

1. **NVIDIA GPU** with CUDA support (tested with Tesla V100)
2. **NVIDIA Container Toolkit** installed
3. **Docker daemon** configured with NVIDIA runtime

### First-time Setup

```bash
# 1. Install NVIDIA Container Toolkit (if not installed)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 2. Configure Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker

# 3. Create daemon.json with NVIDIA runtime
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

# 4. Start Docker daemon with GPU support
cd deployment
./start_docker_daemon.sh
```

### Verify GPU Access

```bash
# Check Docker runtime
sudo docker info | grep -i runtime

# Test GPU in container
sudo docker run --rm --runtime=nvidia nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check services GPU access
sudo docker exec person_reid_extract python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

```bash
# 1. Ensure Docker daemon is running with GPU support
cd deployment
./start_docker_daemon.sh

# 2. Start all services
sudo docker compose up -d

# 3. Check status
sudo docker compose ps

# 4. View logs
sudo docker compose logs -f

# 5. Stop services
sudo docker compose down
```

## 🌐 Access Points

- **Web UI**: http://localhost:8501
- **Register API**: http://localhost:8002/docs
- **Detection API**: http://localhost:8003/docs
- **Qdrant** (External): http://localhost:6333/dashboard

## 📁 Shared Volumes

All services share these volumes:

- `../outputs` - Generated outputs (videos, CSV, tracking data)
- `../data` - Input data and uploads
- `../logs` - Application logs
- `../configs` - Configuration files (.env)
- `../models` - Pre-trained models (YOLOX, ArcFace, etc.)

## ⚙️ Configuration

### Environment Variables

Edit `../configs/.env`:
```bash
QDRANT_HOST=host.docker.internal
QDRANT_PORT=6333
QDRANT_API_KEY=your_api_key
```

### GPU Configuration

Each backend service uses:
- `runtime: nvidia` - NVIDIA runtime
- `NVIDIA_VISIBLE_DEVICES=all` - Access to all GPUs
- `shm_size: '4gb'` - Shared memory for data loading

To use specific GPU:
```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=0  # Use only GPU 0
```

## 🔍 Monitoring

```bash
# Service status
sudo docker compose ps

# Service logs
sudo docker compose logs extract --tail=50
sudo docker compose logs register --tail=50
sudo docker compose logs detection --tail=50
sudo docker compose logs ui --tail=50

# GPU usage
nvidia-smi

# Container GPU access
sudo docker exec person_reid_extract nvidia-smi
```

## 🛠️ Troubleshooting

### GPU not detected

```bash
# 1. Check NVIDIA driver
nvidia-smi

# 2. Check Docker runtime
sudo docker info | grep -i runtime

# 3. Restart Docker daemon
./start_docker_daemon.sh

# 4. Rebuild containers
sudo docker compose down
sudo docker compose up -d --build
```

### Service unhealthy

```bash
# Check logs
sudo docker compose logs <service_name>

# Restart specific service
sudo docker compose restart <service_name>

# Check health endpoint
curl http://localhost:8001/health  # extract
curl http://localhost:8002/health  # register
curl http://localhost:8003/health  # detection
```

## 📊 System Requirements

- **GPU**: NVIDIA GPU with CUDA 11.8+ support
- **RAM**: 16GB+ recommended
- **Disk**: 50GB+ free space
- **OS**: Ubuntu 20.04+ or compatible Linux distribution

