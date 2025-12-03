# Person Re-Identification System

Production-ready multi-camera person re-identification system with face recognition, zone monitoring, and real-time alerts.

[![Version](https://img.shields.io/badge/version-1.0.2-blue.svg)](https://github.com/your-repo/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](deployment/)

## 🎯 Overview

Real-time person tracking and identification system across multiple camera streams using:
- **Face Recognition:** ArcFace (512-dim embeddings, 0.85-0.95 similarity)
- **Detection:** YOLOX + ByteTrack (MOT17 optimized)
- **Inference:** Triton Inference Server + TensorRT (7-9 FPS per stream)
- **Database:** Qdrant (vector search) + PostgreSQL + Redis
- **Monitoring:** Zone-based authorization and violation detection
- **Alerts:** Real-time Kafka messaging

## ✨ Key Features

### 🎥 Multi-Camera Support
- Parallel processing of multiple streams (4-16 cameras)
- Frame synchronization across cameras
- Combined view output with per-camera tracking
- Job cancellation and progress monitoring

### 🔍 Person Re-Identification
- Face-based identification using ArcFace
- K-reciprocal reranking for improved accuracy
- First-3 voting + re-verification strategy (5.3x faster)
- Cross-camera person matching

### 🗺️ Zone Monitoring
- IoP-based zone detection (Intersection over Person)
- Authorization checking per zone
- Time tracking and violation detection
- Real-time alerts via Kafka
- UI-based zone creation (no YAML editing)

### 🚀 Performance
- **Single stream:** 7-9 FPS
- **Multi-stream (4 cameras):** 25-30 FPS total
- **GPU memory:** ~2.5GB (4 Triton instances)
- **Scalability:** Up to 16 cameras with 16 instances

### 🎨 Web UI
- Streamlit-based interface
- Person registration with video upload
- Real-time detection with progress tracking
- Zone configuration and visualization
- Download results (video, CSV, reports)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit Web UI                        │
│                     (Port 8501)                              │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼────────┐ ┌─────▼──────┐ ┌──────▼────────┐
│ Detection API  │ │ Register   │ │  Livestream   │
│  (Port 8003)   │ │   API      │ │    Service    │
│                │ │ (Port 8002)│ │               │
└───────┬────────┘ └─────┬──────┘ └──────┬────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼────────┐ ┌─────▼──────┐ ┌──────▼────────┐
│ Triton Server  │ │  Qdrant    │ │  PostgreSQL   │
│ (TensorRT)     │ │  Vector DB │ │  User DB      │
│ 8100-8102      │ │  (6333)    │ │  (5432)       │
└────────────────┘ └────────────┘ └───────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                ┌────────▼────────┐
                │ Redis + Kafka   │
                │ (6379 + 9092)   │
                └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with 8GB+ VRAM
- nvidia-docker2
- Ubuntu 20.04/22.04

### 1. Clone Repository

```bash
git clone <repository-url>
cd person_reid_system
git checkout v1.0.2
```

### 2. Configure Environment

```bash
cp configs/.env.example configs/.env
nano configs/.env  # Edit Qdrant, PostgreSQL, Redis settings
```

### 3. Start Services

```bash
cd deployment
docker-compose up -d
```

### 4. Access Web UI

```bash
cd ..
source ../hai_venv/bin/activate
streamlit run app.py
```

Open browser: http://localhost:8501

### 5. Register a Person

1. Go to "Register Person" tab
2. Upload video of person
3. Enter name and unique ID
4. Click "Register"

### 6. Run Detection

1. Go to "Detect & Track" tab
2. Upload video or enter stream URL
3. Configure parameters (optional)
4. Click "Start Detection"

**See [Quick Start Guide](docs/QUICKSTART.md) for detailed instructions.**

## 📚 Documentation

### Getting Started
- **[Quick Start](docs/QUICKSTART.md)** - Get running in 5 minutes
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment
- **[Configuration](docs/CONFIGURATION.md)** - System configuration

### Core Components
- **[Vector Database](docs/core/VECTOR_DB.md)** - Qdrant integration and k-reciprocal reranking
- **[ReID Logic](docs/core/REID_LOGIC.md)** - First-3 voting + re-verification strategy

### Deployment
- **[Docker Setup](deployment/README.md)** - Docker Compose deployment
- **[Triton Setup](deployment/TRITON_DEPLOYMENT.md)** - Triton Inference Server

## 🛠️ Technology Stack

### Detection & Tracking
- **YOLOX-X** (MOT17 model) - Person detection
- **ByteTrack** - Multi-object tracking
- **TensorRT** - GPU-optimized inference
- **Triton Inference Server** - Multi-stream batching

### Face Recognition
- **SCRFD** - Face detection
- **ArcFace** (InsightFace) - Face embedding extraction
- **K-reciprocal reranking** - Improved matching accuracy

### Database & Storage
- **Qdrant** - Vector database for face embeddings
- **PostgreSQL** - User management and metadata
- **Redis** - Job tracking and caching

### Messaging & Alerts
- **Kafka** - Real-time violation alerts
- **WebSocket** - Live progress updates

### Deployment
- **Docker** - Containerization
- **Docker Compose** - Service orchestration
- **Streamlit** - Web UI

## 📊 Performance

### Benchmarks

| Configuration | Streams | FPS (Total) | GPU Memory | Latency |
|---------------|---------|-------------|------------|---------|
| Single stream | 1 | 7-9 | ~1.5GB | ~110ms |
| Multi-stream | 4 | 25-30 | ~2.5GB | ~130ms |
| Multi-stream | 8 | 50-60 | ~4GB | ~150ms |
| Multi-stream | 16 | 100+ | ~8GB | ~160ms |

### Optimization

**For Higher FPS:**
- Increase Triton instances (4 → 8 → 16)
- Disable k-reciprocal reranking
- Increase detection threshold

**For Better Accuracy:**
- Enable k-reciprocal reranking
- Lower detection/face thresholds
- Increase re-verification frequency

See [Configuration Guide](docs/CONFIGURATION.md) for tuning details.

## 🔧 Configuration

### Key Settings

**Detection (config.yaml):**
```yaml
detection:
  backend: triton
  conf_threshold: 0.5    # Detection confidence
  triton:
    url: localhost:8101
    model_name: bytetrack_tensorrt
```

**ReID (config.yaml):**
```yaml
reid:
  backend: triton_pipeline
  use_rerank: true       # K-reciprocal reranking
  triton:
    face_conf_threshold: 0.5
```

**Matching (config.yaml):**
```yaml
matching:
  similarity_threshold: 0.8  # ReID threshold (0.7-0.9)
```

**Environment (.env):**
```env
QDRANT_URI=http://127.0.0.1:6333
TRITON_URL=localhost:8101
POSTGRES_HOST=localhost
REDIS_HOST=127.0.0.1
```

## 📁 Project Structure

```
person_reid_system/
├── app.py                      # Streamlit Web UI
├── configs/
│   ├── config.yaml             # Main configuration
│   ├── .env                    # Environment variables
│   └── zones.yaml              # Zone definitions
├── core/                       # Core components
│   ├── detector_triton.py      # Triton detector
│   ├── tracker.py              # ByteTrack wrapper
│   ├── face_recognition_triton.py  # Face recognition
│   ├── vector_db.py            # Qdrant integration
│   ├── reid_logic.py           # ReID matching logic
│   ├── zone_service.py         # Zone monitoring
│   ├── pipeline.py             # Detection pipeline
│   └── zone_processor.py       # Zone processing
├── services/                   # FastAPI services
│   ├── detection_service.py    # Detection API
│   ├── register_service.py     # Registration API
│   └── livestream_service.py   # Livestream API
├── deployment/                 # Docker deployment
│   ├── docker-compose.yml      # Service orchestration
│   ├── Dockerfile.*            # Service Dockerfiles
│   └── TRITON_DEPLOYMENT.md    # Triton setup
├── triton_model_repository/    # Triton models
│   ├── bytetrack_tensorrt/     # Detection model
│   ├── arcface_onnx/           # Face recognition
│   └── scrfd_10g/              # Face detection
├── models/                     # Model weights
├── docs/                       # Documentation
├── outputs/                    # Generated outputs
│   ├── videos/                 # Annotated videos
│   ├── csv/                    # Tracking data
│   └── logs/                   # Detailed logs
└── README.md                   # This file
```

## 🔐 Security

### Production Recommendations

1. **API Authentication:** Add JWT/OAuth to FastAPI services
2. **Network Security:** Use firewall rules (UFW) to restrict ports
3. **SSL/TLS:** Deploy behind Nginx reverse proxy with Let's Encrypt
4. **Database Security:** Use strong passwords, enable SSL connections
5. **Secrets Management:** Use Docker secrets or environment encryption

See [Deployment Guide](docs/DEPLOYMENT.md) for security setup.

## 🐛 Troubleshooting

### Common Issues

**Services won't start:**
```bash
docker-compose logs triton
docker-compose logs detection
docker-compose restart
```

**GPU not detected:**
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**Low FPS:**
- Check GPU utilization: `nvidia-smi -l 1`
- Increase Triton instances in `triton_model_repository/*/config.pbtxt`
- Disable reranking: `use_rerank: false`

**Face detection fails:**
- Lower threshold: `face_conf_threshold: 0.3`
- Check lighting conditions
- Verify face is visible in frame

See [Configuration Guide](docs/CONFIGURATION.md) for more troubleshooting.

## 📈 Monitoring

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

## 🔄 Updates

### Update System

```bash
git pull origin main
cd deployment
docker-compose build
docker-compose down
docker-compose up -d
```

### Backup

```bash
# Backup Qdrant
curl -X POST http://localhost:6333/collections/cross_camera_matching_id/snapshots

# Backup PostgreSQL
docker exec person_reid_postgres pg_dump -U postgres hailt_imespro > backup.sql
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ByteTrack** - Multi-object tracking
- **YOLOX** - Object detection
- **InsightFace** - Face recognition
- **Qdrant** - Vector database
- **NVIDIA Triton** - Inference server

## 📞 Support

- **Documentation:** [docs/](docs/)
- **Issues:** [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-repo/discussions)

## 🗺️ Roadmap

- [ ] Multi-GPU support
- [ ] Real-time dashboard
- [ ] Mobile app integration
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Advanced analytics and reporting

---

**Version:** 1.0.2  
**Last Updated:** December 2025  
**Status:** Production Ready ✅
