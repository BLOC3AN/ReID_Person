# GPU Deployment Summary

## ✅ Deployment Status: SUCCESS

All services are running with GPU support enabled.

## 🎯 What Was Done

### 1. NVIDIA Container Toolkit Installation
- Installed `nvidia-container-toolkit` version 1.18.0-1
- Configured Docker daemon to use NVIDIA runtime
- Created `/etc/docker/daemon.json` with NVIDIA runtime configuration

### 2. Docker Daemon Configuration
```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

### 3. Docker Compose Updates
Changed from `deploy.resources` to `runtime: nvidia` for all backend services:

**Before:**
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
environment:
  - CUDA_VISIBLE_DEVICES=0
```

**After:**
```yaml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
```

### 4. Services with GPU Support
- ✅ **extract** (port 8001) - GPU enabled
- ✅ **register** (port 8002) - GPU enabled  
- ✅ **detection** (port 8003) - GPU enabled
- ℹ️ **ui** (port 8501) - CPU only (no GPU needed)

## 🖥️ GPU Information

### Available GPUs
```
4x Tesla V100-SXM2-16GB
- GPU 0: Tesla V100-SXM2-16GB (16GB VRAM)
- GPU 1: Tesla V100-SXM2-16GB (16GB VRAM)
- GPU 2: Tesla V100-SXM2-16GB (16GB VRAM)
- GPU 3: Tesla V100-SXM2-16GB (16GB VRAM)
```

### CUDA Version
- Container CUDA: 11.8.0
- Driver Version: 535.274.02
- Host CUDA: 12.8

## 🔍 Verification Results

### Docker Runtime
```bash
$ sudo docker info | grep -i runtime
Runtimes: runc io.containerd.runc.v2 nvidia
Default Runtime: nvidia
```

### Container GPU Access
```bash
# Extract Service
$ sudo docker exec person_reid_extract python -c "import torch; print(torch.cuda.is_available())"
CUDA available: True
Device count: 4
Current device: 0
Device name: Tesla V100-SXM2-16GB

# Register Service
$ sudo docker exec person_reid_register python -c "import torch; print(torch.cuda.is_available())"
CUDA available: True
Device: Tesla V100-SXM2-16GB

# Detection Service
$ sudo docker exec person_reid_detection python -c "import torch; print(torch.cuda.is_available())"
CUDA available: True
Device: Tesla V100-SXM2-16GB
```

### Service Health Status
```bash
$ sudo docker compose ps
NAME                    STATUS
person_reid_detection   Up (healthy)
person_reid_extract     Up (healthy)
person_reid_register    Up (healthy)
person_reid_ui          Up (healthy)
```

## 📝 Important Notes

### Docker Daemon Startup
The Docker daemon is started with custom command:
```bash
sudo dockerd \
  --data-root /home/ubuntu/data/docker \
  --config-file /etc/docker/daemon.json \
  -H unix:///var/run/docker.sock
```

**Important**: Use `./start_docker_daemon.sh` script to ensure GPU support is enabled after system restart.

### GPU Allocation
Currently all services use `NVIDIA_VISIBLE_DEVICES=all`, giving access to all 4 GPUs.

To optimize GPU usage, you can assign different GPUs to different services:
```yaml
# Extract service - GPU 0
environment:
  - NVIDIA_VISIBLE_DEVICES=0

# Register service - GPU 1  
environment:
  - NVIDIA_VISIBLE_DEVICES=1

# Detection service - GPU 2,3
environment:
  - NVIDIA_VISIBLE_DEVICES=2,3
```

### Shared Memory
All GPU services have `shm_size: '4gb'` for efficient data loading with PyTorch DataLoader.

## 🚀 Quick Commands

### Start System
```bash
cd deployment
./start_docker_daemon.sh
sudo docker compose up -d
```

### Check GPU Usage
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Real-time monitoring
```

### Check Service Logs
```bash
sudo docker compose logs extract --tail=50
sudo docker compose logs register --tail=50
sudo docker compose logs detection --tail=50
```

### Restart Services
```bash
sudo docker compose restart extract
sudo docker compose restart register
sudo docker compose restart detection
```

## 🎉 Success Metrics

- ✅ All 4 services running and healthy
- ✅ GPU detected in all backend services (extract, register, detection)
- ✅ CUDA 11.8 available in containers
- ✅ 4x Tesla V100 GPUs accessible
- ✅ NVIDIA runtime set as default
- ✅ Health checks passing
- ✅ API endpoints responding

## 📚 References

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [Docker Compose GPU](https://docs.docker.com/compose/gpu-support/)

---

**Deployment Date**: 2025-10-29  
**System**: Ubuntu with 4x Tesla V100-SXM2-16GB  
**Docker Version**: 24.0+  
**NVIDIA Driver**: 535.274.02

