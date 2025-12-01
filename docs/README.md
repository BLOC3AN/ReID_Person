# Person ReID System - Production Documentation

Production-ready documentation for deploying and operating the Person Re-Identification system.

## 📚 Documentation Structure

### Getting Started
- **[Quick Start](QUICKSTART.md)** - Get up and running in 5 minutes
- **[Deployment](DEPLOYMENT.md)** - Docker deployment guide

### Core Features
- **[Zone Monitoring](ZONE_MONITORING.md)** - Zone-based person tracking and authorization
- **[Multi-Camera](MULTI_CAMERA.md)** - Multi-stream processing setup
- **[API Reference](API.md)** - REST API endpoints and usage

### Operations
- **[Configuration](CONFIGURATION.md)** - System configuration reference
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Performance](PERFORMANCE.md)** - Optimization and tuning guide

## 🎯 Production Stack

**Current Configuration:**
- **Detection:** Triton Inference Server + TensorRT (MOT17 model)
- **ReID:** Triton Pipeline (SCRFD face detector + ArcFace)
- **Database:** Qdrant (vector database) + PostgreSQL (user management) + Redis (job tracking)
- **Messaging:** Kafka (realtime alerts)
- **Deployment:** Docker Compose

**Performance:**
- Single stream: 7-9 FPS
- Multi-stream (4 cameras): 25-30 FPS total
- GPU memory: ~2.5GB (4 Triton instances)

## 🚀 Quick Links

- [Main README](../README.md) - Project overview
- [Deployment Guide](../deployment/README.md) - Docker setup
- [Triton Setup](../deployment/TRITON_DEPLOYMENT.md) - Triton Inference Server

## 📝 Notes

- This documentation focuses on **production deployment only**
- For development/research features, see archived docs in `docs_backup/`
- System uses Triton backend exclusively (PyTorch backend deprecated for production)
