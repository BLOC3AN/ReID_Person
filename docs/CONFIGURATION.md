# Configuration Reference

Complete reference for system configuration files.

## Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| `config.yaml` | Main system config | `configs/config.yaml` |
| `.env` | Environment variables | `configs/.env` |
| `docker-compose.yml` | Service orchestration | `deployment/docker-compose.yml` |
| Triton configs | Model configurations | `triton_model_repository/*/config.pbtxt` |

## config.yaml

### Detection Settings

```yaml
detection:
  backend: triton  # Fixed: triton (production)
  
  triton:
    url: localhost:8101
    model_name: bytetrack_tensorrt
    timeout: 20.0
  
  # Detection parameters
  conf_threshold: 0.5    # 0.3-0.7 (lower = more detections)
  nms_threshold: 0.45    # 0.3-0.5 (NMS suppression)
  test_size: [640, 640]  # Input size
  device: cuda
```

**Tuning:**
- `conf_threshold`: Lower for crowded scenes, higher for clean scenes
- `timeout`: Increase for slow networks

### Tracking Settings

```yaml
tracking:
  track_thresh: 0.5       # 0.4-0.6 (tracking confidence)
  track_buffer: 30        # Frames to keep lost tracks
  match_thresh: 0.8       # 0.7-0.9 (IoU matching)
  aspect_ratio_thresh: 1.6
  min_box_area: 10
```

**Tuning:**
- `track_thresh`: Lower = more tracks (may include false positives)
- `track_buffer`: Higher = better recovery from occlusion

### ReID Settings

```yaml
reid:
  backend: triton_pipeline  # Fixed: triton_pipeline (production)
  
  triton:
    url: localhost:8101
    arcface_model: arcface_onnx
    face_detector_model: scrfd_10g
    face_conf_threshold: 0.5  # 0.3-0.7 (face detection)
  
  # K-reciprocal reranking
  use_rerank: true
  rerank_k1: 20
  rerank_k2: 6
  rerank_lambda: 0.3
```

**Tuning:**
- `face_conf_threshold`: Lower for difficult angles/lighting
- `use_rerank`: Enable for better accuracy (slight performance cost)

### Database Settings

```yaml
database:
  use_qdrant: true
  qdrant_collection: cross_camera_matching_id
  max_embeddings_per_person: 100
  embedding_dim: 512  # Fixed: ArcFace dimension
```

### Matching Settings

```yaml
matching:
  similarity_threshold: 0.8  # 0.7-0.9 (ReID matching)
  metric: cosine
  top_k: 1
```

**Tuning:**
- `similarity_threshold`:
  - 0.9: Very strict (few false positives, many Unknown)
  - 0.8: Balanced (recommended)
  - 0.7: Loose (more matches, may have false positives)

### Output Settings

```yaml
output:
  save_video: true
  save_csv: true
  save_logs: true
  video_codec: mp4v
  log_level: INFO  # DEBUG | INFO | WARNING | ERROR
```

### Kafka Settings

```yaml
kafka:
  enable: true
  bootstrap_servers: localhost:9092
  topic: person_alerts
  alert_threshold: 0.0  # Seconds before alert
```

## .env File

### Qdrant Configuration

```env
QDRANT_URI=http://127.0.0.1:6333
QDRANT_COLLECTION=cross_camera_matching_id
QDRANT_USE_GRPC=true
```

### Service URLs

```env
# For Docker deployment
DETECTION_API_URL=http://detection:8003
REGISTER_API_URL=http://register:8002

# For local development
# DETECTION_API_URL=http://localhost:8003
# REGISTER_API_URL=http://localhost:8002
```

### Triton Configuration

```env
TRITON_URL=localhost:8101
TRITON_HTTP_PORT=8100
TRITON_GRPC_PORT=8101
TRITON_METRICS_PORT=8102
```

### GPU Configuration

```env
NVIDIA_VISIBLE_DEVICES=all  # or 0,1,2 for specific GPUs
CUDA_VISIBLE_DEVICES=0
```

### Database Configuration

```env
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=hailt_imespro

# Redis
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
REDIS_TTL=86400  # 1 day
```

## Triton Model Configs

### bytetrack_tensorrt/config.pbtxt

```pbtxt
name: "bytetrack_tensorrt"
platform: "tensorrt_plan"
max_batch_size: 1

input [
  {
    name: "images"
    data_type: TYPE_FP16
    dims: [ 3, 640, 640 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP16
    dims: [ 8400, 6 ]
  }
]

# Instance count: adjust based on GPU memory and camera count
instance_group [
  {
    count: 4  # 1-4 cameras: 4, 5-8 cameras: 8, 9-16 cameras: 16
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

**Instance Count Guidelines:**

| Cameras | Instances | GPU Memory | Expected FPS |
|---------|-----------|------------|--------------|
| 1-4 | 4 | ~2GB | 25-30 total |
| 5-8 | 8 | ~4GB | 50-60 total |
| 9-16 | 16 | ~8GB | 100+ total |

## Performance Tuning

### For Higher FPS

```yaml
# config.yaml
detection:
  conf_threshold: 0.6  # Higher = fewer detections
  test_size: [640, 640]  # Don't increase

tracking:
  track_thresh: 0.6  # Higher = fewer tracks

reid:
  use_rerank: false  # Disable for speed
```

### For Better Accuracy

```yaml
# config.yaml
detection:
  conf_threshold: 0.4  # Lower = more detections

tracking:
  track_thresh: 0.4  # Lower = more tracks
  track_buffer: 60  # Longer buffer

reid:
  face_conf_threshold: 0.3  # Lower = detect more faces
  use_rerank: true  # Enable reranking

matching:
  similarity_threshold: 0.85  # Higher = stricter
```

### For Crowded Scenes

```yaml
detection:
  conf_threshold: 0.4
  nms_threshold: 0.4  # Lower = keep more overlapping boxes

tracking:
  track_thresh: 0.4
  match_thresh: 0.7  # Lower = more lenient matching
```

### For Low-Light Conditions

```yaml
detection:
  conf_threshold: 0.4  # Lower threshold

reid:
  face_conf_threshold: 0.3  # Lower for difficult lighting
```

## Environment-Specific Configs

### Development

```yaml
# config.yaml
output:
  log_level: DEBUG

kafka:
  enable: false  # Disable Kafka in dev
```

### Staging

```yaml
# config.yaml
output:
  log_level: INFO

kafka:
  enable: true
```

### Production

```yaml
# config.yaml
output:
  log_level: WARNING  # Less verbose

kafka:
  enable: true
  alert_threshold: 5.0  # 5 second delay before alerts
```

## Validation

### Check Config Syntax

```bash
# Validate YAML
python3 -c "import yaml; yaml.safe_load(open('configs/config.yaml'))"

# Validate .env
source configs/.env && echo "✅ .env loaded"
```

### Test Configuration

```bash
# Test with small video
python scripts/detect_and_track.py \
  --video data/videos/test.mp4 \
  --max-frames 100
```

## Best Practices

1. **Never commit `.env`** - Use `.env.example` as template
2. **Version control `config.yaml`** - Track changes
3. **Document custom settings** - Add comments
4. **Test before production** - Validate on staging
5. **Monitor performance** - Adjust based on metrics

## See Also

- [Quick Start](QUICKSTART.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Performance Guide](PERFORMANCE.md)
- [Troubleshooting](TROUBLESHOOTING.md)
