# Quick Start Guide

Get the Person ReID system running in 5 minutes.

## Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA support
- nvidia-docker2 installed

## 1. Start Services

```bash
cd deployment
docker-compose up -d
```

This starts:
- Triton Inference Server (port 8100-8102)
- Detection Service (port 8003)
- Register Service (port 8002)
- Qdrant (port 6333)
- PostgreSQL (port 5432)
- Redis (port 6379)
- Kafka (port 9092)

## 2. Check Service Health

```bash
# Check all services
docker-compose ps

# Check Triton models
curl http://localhost:8100/v2/models/bytetrack_tensorrt/ready
curl http://localhost:8100/v2/models/arcface_onnx/ready
curl http://localhost:8100/v2/models/scrfd_10g/ready

# Check Detection API
curl http://localhost:8003/health
```

## 3. Access Web UI

```bash
# Start Streamlit UI (outside Docker)
cd ..
source ../hai_venv/bin/activate
streamlit run app.py
```

Open browser: http://localhost:8501

## 4. Register a Person

**Via UI:**
1. Go to "Register Person" tab
2. Upload video of person
3. Enter name and global_id
4. Click "Register"

**Via API:**
```bash
curl -X POST http://localhost:8002/register \
  -F "video=@person.mp4" \
  -F "name=John" \
  -F "global_id=1"
```

## 5. Run Detection

**Via UI:**
1. Go to "Detect & Track" tab
2. Upload video or enter stream URL
3. Configure parameters
4. Click "Start Detection"

**Via API:**
```bash
curl -X POST http://localhost:8003/detect \
  -F "video=@test.mp4" \
  -F "similarity_threshold=0.8"
```

## 6. View Results

Results are saved to:
- Video: `outputs/videos/<job_id>_output.mp4`
- CSV: `outputs/csv/<job_id>_tracking.csv`
- Logs: `outputs/logs/<job_id>.log`

Download via UI or API:
```bash
curl http://localhost:8003/download/video/<job_id> -o output.mp4
```

## Next Steps

- [Zone Monitoring Setup](ZONE_MONITORING.md)
- [Multi-Camera Configuration](MULTI_CAMERA.md)
- [API Reference](API.md)
- [Configuration Guide](CONFIGURATION.md)

## Troubleshooting

**Services won't start:**
```bash
# Check logs
docker-compose logs triton
docker-compose logs detection

# Restart services
docker-compose restart
```

**GPU not detected:**
```bash
# Check nvidia-docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**Triton models not loading:**
```bash
# Check model repository
ls -la triton_model_repository/*/1/

# Verify TensorRT engines exist
ls -lh models/*.trt
```

See [Troubleshooting Guide](TROUBLESHOOTING.md) for more issues.
