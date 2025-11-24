# Configuration Guide

## Overview

Hệ thống Person ReID sử dụng 2 file cấu hình chính:

1. **config.yaml** - Cấu hình model, tracking, ReID parameters
2. **.env** - Qdrant credentials và environment variables

---

## 1. config.yaml

**Location:** `configs/config.yaml`

### Full Configuration

```yaml
detection:
  model_type: mot17
  model_path_mot17: models/bytetrack_x_mot17.pth.tar
  model_path_yolox: models/yolox_x.pth
  conf_threshold: 0.5
  nms_threshold: 0.45
  test_size:
  - 640
  - 640
  device: cuda
  fp16: true

tracking:
  track_thresh: 0.5
  track_buffer: 30
  match_thresh: 0.8
  aspect_ratio_thresh: 1.6
  min_box_area: 10
  mot20: false

reid:
  # ArcFace face recognition settings
  arcface_model_name: buffalo_l  # Options: buffalo_l, buffalo_s, antelopev2
  feature_dim: 512
  use_cuda: true

database:
  use_qdrant: true
  qdrant_url: ''
  qdrant_collection: cross_camera_matching_id
  max_embeddings_per_person: 100
  embedding_dim: 512

matching:
  similarity_threshold: 0.8
  metric: cosine
  top_k: 1

registration:
  sample_rate: 3
  max_frames: 50

output:
  save_video: true
  save_csv: true
  save_logs: true
  log_level: INFO
```

### Section Details

#### Detection Section

```yaml
detection:
  model_type: mot17              # Model type: 'mot17' or 'yolox'
  model_path_mot17: models/bytetrack_x_mot17.pth.tar
  model_path_yolox: models/yolox_x.pth
  conf_threshold: 0.5            # Detection confidence threshold (0-1)
  nms_threshold: 0.45            # Non-maximum suppression threshold
  test_size: [640, 640]          # Input size [height, width]
  device: cuda                   # 'cuda' or 'cpu'
  fp16: true                     # Use FP16 precision (faster on GPU)
```

**Parameters:**

- **model_type**: 
  - `mot17`: Recommended, trained on MOT17 dataset
  - `yolox`: General YOLOX model
  
- **conf_threshold**: 
  - `0.3-0.4`: Loose (more detections, more false positives)
  - `0.5`: Balanced (recommended)
  - `0.6-0.7`: Strict (fewer detections, fewer false positives)
  
- **nms_threshold**: 
  - `0.3-0.4`: Aggressive NMS (fewer overlapping boxes)
  - `0.45`: Balanced (recommended)
  - `0.5-0.6`: Loose NMS (more overlapping boxes)
  
- **test_size**: 
  - `[640, 640]`: Balanced speed/accuracy (recommended)
  - `[800, 800]`: Higher accuracy, slower
  - `[416, 416]`: Faster, lower accuracy
  
- **fp16**: 
  - `true`: Faster inference on GPU (recommended)
  - `false`: Full precision (CPU or debugging)

#### Tracking Section

```yaml
tracking:
  track_thresh: 0.5              # Tracking confidence threshold
  track_buffer: 30               # Frames to keep lost tracks
  match_thresh: 0.8              # IoU threshold for matching
  aspect_ratio_thresh: 1.6       # Max aspect ratio for valid detection
  min_box_area: 10               # Minimum bbox area (pixels)
  mot20: false                   # Use MOT20 settings
```

**Parameters:**

- **track_thresh**: 
  - `0.3-0.4`: Loose (track more objects, more false tracks)
  - `0.5`: Balanced (recommended)
  - `0.6-0.7`: Strict (track fewer objects, fewer false tracks)
  
- **track_buffer**: 
  - `15-20`: Short buffer (lose tracks quickly)
  - `30`: Balanced (recommended, 1 second at 30fps)
  - `60-90`: Long buffer (keep tracks longer during occlusion)
  
- **match_thresh**: 
  - `0.6-0.7`: Loose matching (easier to match)
  - `0.8`: Balanced (recommended)
  - `0.9`: Strict matching (harder to match)

#### ReID Section

```yaml
reid:
  arcface_model_name: buffalo_l  # ArcFace model variant
  feature_dim: 512               # Embedding dimension
  use_cuda: true                 # Use GPU for extraction
```

**Parameters:**

- **arcface_model_name**: 
  - `buffalo_l`: Large model, highest accuracy (recommended)
  - `buffalo_s`: Small model, faster but lower accuracy
  - `antelopev2`: Alternative model
  
- **feature_dim**: 
  - `512`: Standard for ArcFace (do not change)
  
- **use_cuda**: 
  - `true`: GPU acceleration (recommended)
  - `false`: CPU mode (slower)

#### Database Section

```yaml
database:
  use_qdrant: true                          # Use Qdrant backend
  qdrant_url: ''                            # Auto-load from .env
  qdrant_collection: cross_camera_matching_id
  max_embeddings_per_person: 100            # Max embeddings to store
  embedding_dim: 512                        # Must match reid.feature_dim
  use_grpc: false                           # Use gRPC protocol (faster for local connections)
```

**Parameters:**

- **use_qdrant**:
  - `true`: Use Qdrant cloud/local (recommended)
  - `false`: In-memory only (for testing)

- **use_grpc**:
  - `true`: Use gRPC protocol (faster for local Qdrant connections)
  - `false`: Use HTTP protocol (default, works everywhere)

- **max_embeddings_per_person**:
  - `50`: Fewer embeddings, faster search
  - `100`: Balanced (recommended)
  - `200`: More embeddings, better accuracy, slower search

#### Matching Section

```yaml
matching:
  similarity_threshold: 0.8      # Cosine similarity threshold
  metric: cosine                 # Distance metric
  top_k: 1                       # Return top K matches
```

**Parameters:**

- **similarity_threshold**: 
  - `0.6-0.7`: Loose (high recall, more false positives)
  - `0.8`: Balanced (recommended for ArcFace)
  - `0.9`: Strict (high precision, may miss some matches)
  
- **metric**: 
  - `cosine`: Cosine similarity (recommended for ArcFace)
  - `euclidean`: Euclidean distance (alternative)
  
- **top_k**: 
  - `1`: Return best match only (recommended)
  - `3-5`: Return top K candidates (for analysis)

#### Registration Section

```yaml
registration:
  sample_rate: 3                 # Extract 1 frame every N frames
  max_frames: 50                 # Maximum frames to process
```

**Parameters:**

- **sample_rate**: 
  - `1-2`: Extract many frames (more embeddings, slower)
  - `3-5`: Balanced (recommended)
  - `10+`: Extract fewer frames (fewer embeddings, faster)
  
- **max_frames**: 
  - `30`: Quick registration
  - `50`: Balanced (recommended)
  - `100+`: Thorough registration

#### Output Section

```yaml
output:
  save_video: true               # Save annotated video
  save_csv: true                 # Save tracking CSV
  save_logs: true                # Save detailed logs
  log_level: INFO                # Logging level
```

**Parameters:**

- **log_level**: 
  - `DEBUG`: Verbose logging (for debugging)
  - `INFO`: Standard logging (recommended)
  - `WARNING`: Only warnings and errors
  - `ERROR`: Only errors

---

## 2. .env File

**Location:** `configs/.env`

### Template

```env
# Qdrant Configuration
QDRANT_API_KEY=your_api_key_here
QDRANT_URI=host=your-cluster.cloud.qdrant.io
QDRANT_COLLECTION=cross_camera_matching_id
QDRANT_PORT=6333

# Optional: Service URLs (for Docker)
EXTRACT_API_URL=http://extract:8001
REGISTER_API_URL=http://register:8002
DETECTION_API_URL=http://detection:8003
```

### Getting Qdrant Credentials

1. **Sign up:** https://cloud.qdrant.io
2. **Create cluster:** Click "Create Cluster"
3. **Get API key:** Settings → API Keys → Create
4. **Get URL:** Cluster → Connection → Copy URL
5. **Update .env:** Paste credentials

### Example

```env
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ
QDRANT_URI=host=abc123-xyz789.us-east-1.aws.cloud.qdrant.io
QDRANT_COLLECTION=cross_camera_matching_id
QDRANT_PORT=6333
```

---

## 3. Parameter Tuning Guide

### Use Case: High Accuracy

**Goal:** Maximize accuracy, speed is secondary

```yaml
detection:
  conf_threshold: 0.6            # Strict detection
  test_size: [800, 800]          # Higher resolution

tracking:
  track_thresh: 0.6              # Strict tracking
  match_thresh: 0.9              # Strict matching

reid:
  arcface_model_name: buffalo_l  # Largest model

matching:
  similarity_threshold: 0.85     # Strict matching

registration:
  sample_rate: 2                 # More embeddings
  max_frames: 100                # More frames
```

### Use Case: High Speed

**Goal:** Maximize speed, acceptable accuracy

```yaml
detection:
  conf_threshold: 0.4            # Loose detection
  test_size: [416, 416]          # Lower resolution
  fp16: true                     # Fast inference

tracking:
  track_thresh: 0.4              # Loose tracking
  track_buffer: 15               # Short buffer

reid:
  arcface_model_name: buffalo_s  # Smaller model

matching:
  similarity_threshold: 0.7      # Loose matching

registration:
  sample_rate: 10                # Fewer embeddings
  max_frames: 30                 # Fewer frames
```

### Use Case: Balanced (Recommended)

**Goal:** Balance between accuracy and speed

```yaml
detection:
  conf_threshold: 0.5
  test_size: [640, 640]
  fp16: true

tracking:
  track_thresh: 0.5
  track_buffer: 30
  match_thresh: 0.8

reid:
  arcface_model_name: buffalo_l

matching:
  similarity_threshold: 0.8

registration:
  sample_rate: 5
  max_frames: 50
```

### Use Case: Crowded Scene

**Goal:** Handle many people in frame

```yaml
detection:
  conf_threshold: 0.6            # Reduce false positives
  nms_threshold: 0.3             # Aggressive NMS

tracking:
  track_buffer: 60               # Longer buffer for occlusion
  match_thresh: 0.7              # Easier matching

database:
  max_embeddings_per_person: 150 # More embeddings for robustness
```

### Use Case: Low-Quality Video

**Goal:** Handle poor lighting, blur, low resolution

```yaml
detection:
  conf_threshold: 0.3            # Lower threshold
  test_size: [640, 640]          # Standard size

tracking:
  track_thresh: 0.4              # Lower threshold
  track_buffer: 45               # Longer buffer

matching:
  similarity_threshold: 0.7      # Lower threshold

registration:
  sample_rate: 3                 # More samples
  max_frames: 100                # More frames
```

---

## 4. Best Practices

### Registration

1. **Video Quality:**
   - Use clear, frontal face videos
   - Good lighting
   - Minimal motion blur
   - 720p or higher resolution

2. **Sample Rate:**
   - Lower sample_rate (3-5) = more embeddings = better accuracy
   - Higher sample_rate (10+) = fewer embeddings = faster registration

3. **Global ID:**
   - Use sequential IDs: 1, 2, 3, ...
   - Never reuse IDs
   - Document person-to-ID mapping

4. **Delete Existing:**
   - Use `--delete-existing` to start fresh
   - Backup database before deleting

### Detection

1. **Threshold Tuning:**
   - Start with 0.8
   - Lower to 0.7 if missing matches
   - Raise to 0.9 if too many false positives

2. **Model Selection:**
   - Always use `mot17` for consistency
   - Same model for registration and detection

3. **Performance:**
   - First-3 + Re-verify strategy is automatic
   - ~19 FPS on GPU
   - ~3-5 FPS on CPU

### Database

1. **Qdrant:**
   - Use cloud for production
   - Use local for development
   - Backup regularly

2. **Embeddings:**
   - Limit to 100 per person
   - More embeddings = slower search
   - Quality > quantity

### Monitoring

1. **Logs:**
   - Check for `[VOTING]` events (first 3 frames)
   - Check for `[RE-VERIFY]` events (every 30 frames)
   - Monitor similarity scores

2. **Performance:**
   - Track FPS in output video
   - Monitor GPU usage with `nvidia-smi`
   - Check memory usage

---

## 5. Troubleshooting

### Low Similarity Scores

**Problem:** Similarity < 0.8 for known persons

**Solutions:**
1. Re-register with better quality video
2. Lower `similarity_threshold` to 0.7
3. Increase `sample_rate` to 3 for more embeddings
4. Ensure face is visible in both registration and detection

### No Detections

**Problem:** No persons detected in video

**Solutions:**
1. Lower `conf_threshold` to 0.3-0.4
2. Check video format and quality
3. Verify model files exist
4. Test with sample video

### Slow Processing

**Problem:** FPS < 10

**Solutions:**
1. Enable GPU: `device: cuda`
2. Enable FP16: `fp16: true`
3. Reduce resolution: `test_size: [416, 416]`
4. Use smaller model: `arcface_model_name: buffalo_s`

### Qdrant Connection Failed

**Problem:** Cannot connect to Qdrant

**Solutions:**
1. Check `.env` file exists
2. Verify API key and URL
3. Test connection manually
4. Check network/firewall

---

## 6. Advanced Configuration

### Custom ReID Strategy

Edit `scripts/detect_and_track.py`:

```python
# Change voting frames
VOTING_FRAMES = 5  # Default: 3

# Change re-verify interval
REVERIFY_INTERVAL = 60  # Default: 30 (frames)
```

### Custom Output Paths

```python
# In pipeline code
pipeline.process_video(
    video_path="input.mp4",
    output_video_path="custom/output.mp4",
    output_csv_path="custom/tracking.csv",
    output_log_path="custom/detection.log"
)
```

### Environment Variables

```bash
# Override config values
export REID_THRESHOLD=0.85
export DETECTION_DEVICE=cpu
export LOG_LEVEL=DEBUG
```

