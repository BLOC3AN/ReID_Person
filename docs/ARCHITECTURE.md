# System Architecture

## Overview

Person Re-Identification System là một hệ thống phân tán dựa trên kiến trúc microservices, sử dụng YOLOX detection, ByteTrack tracking, ArcFace face recognition và Qdrant vector database để nhận diện người qua nhiều camera.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web UI (Streamlit)                       │
│                         Port: 8501                               │
└────────────┬────────────────────────────────────┬────────────────┘
             │                                    │
             ├────────────────┬───────────────────┤
             │                │                   │
    ┌────────▼─────┐  ┌──────▼──────┐  ┌────────▼─────────┐
    │   Extract    │  │  Register   │  │   Detection      │
    │   Service    │  │  Service    │  │   Service        │
    │   Port: 8001 │  │  Port: 8002 │  │   Port: 8003     │
    └────────┬─────┘  └──────┬──────┘  └────────┬─────────┘
             │                │                   │
             │                │                   │
             └────────────────┼───────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Qdrant Vector DB  │
                    │  Port: 6333        │
                    └────────────────────┘
```

## Core Components

### 1. Detection Module (`core/detector.py`)

**Chức năng:** Phát hiện người trong video frame

**Công nghệ:**
- YOLOX-X model (MOT17 variant)
- Input: Video frame (H, W, 3) BGR
- Output: Bounding boxes [x1, y1, x2, y2, conf, cls, -1]

**Đặc điểm:**
- FP16 precision cho tốc độ cao
- Confidence threshold: 0.5 (configurable)
- NMS threshold: 0.45
- Test size: 640x640

**Performance:**
- ~144ms per frame (GPU)
- Chiếm 52.3% thời gian xử lý

### 2. Tracking Module (`core/tracker.py`)

**Chức năng:** Theo dõi người qua các frame

**Công nghệ:**
- ByteTrack algorithm
- Kalman Filter + IoU matching
- Track buffer: 30 frames

**Đặc điểm:**
- Track threshold: 0.5
- Match threshold: 0.8
- Xử lý occlusion và re-identification

**Performance:**
- ~1.45ms per frame
- Chiếm 0.5% thời gian xử lý

### 3. Feature Extraction Module (`core/feature_extractor.py`)

**Chức năng:** Trích xuất face embedding từ person bbox

**Công nghệ:**
- ArcFace (InsightFace) - buffalo_l model
- Face detection + embedding extraction
- 512-dimensional L2-normalized vectors

**Đặc điểm:**
- GPU-accelerated (CUDA)
- Detection size: 640x640
- Padding: 10px around bbox
- Returns zero vector if no face detected

**Performance:**
- ~43ms per person (GPU)
- Chiếm 47.1% thời gian xử lý (bottleneck)

### 4. Vector Database Module (`core/vector_db.py`)

**Chức năng:** Lưu trữ và tìm kiếm embeddings

**Công nghệ:**
- Qdrant vector database (cloud/local)
- In-memory fallback storage
- Cosine similarity search

**Đặc điểm:**
- Collection: cross_camera_matching_id
- Embedding dimension: 512
- Max embeddings per person: 100
- Distance metric: Cosine

**Performance:**
- ~0.06ms per search
- Chiếm 0.1% thời gian xử lý

### 5. ReID Matcher Module (`core/reid_matcher.py`)

**Chức năng:** Matching logic cho cross-camera ReID

**Strategy:** First-3 + Re-verify
- Frame 0-2: Majority voting (3 embeddings)
- Frame 30, 60, 90...: Re-verification
- Other frames: Cached labels

**Đặc điểm:**
- Similarity threshold: 0.8 (configurable)
- Voting mechanism: Highest votes + highest similarity
- Self-correction capability

**Performance:**
- 95.8% reduction in extractions
- 5.3x speedup (19 FPS vs 3.6 FPS)

## Microservices Architecture

### Service 1: Extract Service (Port 8001)

**Chức năng:** Tách video thành các object videos riêng lẻ

**Endpoints:**
- `POST /extract` - Upload video và extract objects
- `GET /status/{job_id}` - Kiểm tra trạng thái job
- `GET /results/{job_id}` - Liệt kê extracted objects
- `GET /download/{job_id}/{filename}` - Download object video
- `GET /download/zip/{job_id}` - Download tất cả as ZIP

**Processing:**
1. Upload video → temp storage
2. Detect + Track persons
3. Extract individual object videos
4. Save to `/app/outputs/extracted_objects/{job_id}/`

**GPU Usage:** NVIDIA runtime, 4GB shared memory

### Service 2: Register Service (Port 8002)

**Chức năng:** Đăng ký người vào vector database

**Endpoints:**
- `POST /register` - Register person từ video
- `GET /status/{job_id}` - Kiểm tra trạng thái
- `DELETE /jobs/{job_id}` - Xóa job

**Processing:**
1. Upload video → temp storage
2. Detect persons (MOT17)
3. Extract face embeddings (ArcFace)
4. Store to Qdrant + local database
5. Sync to `data/database/reid_database.pkl`

**Parameters:**
- `person_name`: Tên người
- `global_id`: ID duy nhất (1, 2, 3...)
- `sample_rate`: Extract 1 frame mỗi N frames
- `delete_existing`: Xóa collection cũ

**GPU Usage:** NVIDIA runtime, 4GB shared memory

### Service 3: Detection Service (Port 8003)

**Chức năng:** Detect, track và re-identify persons

**Endpoints:**
- `POST /detect` - Upload video và detect
- `GET /status/{job_id}` - Kiểm tra trạng thái
- `GET /download/video/{job_id}` - Download annotated video
- `GET /download/csv/{job_id}` - Download tracking CSV
- `GET /download/log/{job_id}` - Download detailed log

**Processing:**
1. Upload video → temp storage
2. Initialize detector + tracker + extractor
3. Load registered persons from Qdrant
4. Process video với First-3 + Re-verify strategy
5. Generate outputs:
   - Annotated video (bbox + labels + FPS)
   - CSV tracking data
   - Detailed logs

**Output Format:**
- Video: MP4 với bbox, labels, similarity scores
- CSV: frame_id, track_id, bbox, global_id, similarity, label
- Log: [VOTING], [RE-VERIFY] events

**GPU Usage:** NVIDIA runtime, 4GB shared memory

### Service 4: UI Service (Port 8501)

**Chức năng:** Web interface cho tất cả operations

**Technology:** Streamlit

**Pages:**
1. **Extract Objects** - Upload video → extract individual persons
2. **Register Person** - Upload video → register to database
3. **Detect & Track** - Upload video → detect registered persons
4. **About** - System information

**Features:**
- File upload (MP4, AVI, MKV, MOV)
- Real-time job status polling
- Download results (videos, CSV, logs, ZIP)
- Parameter configuration
- Progress tracking

**Communication:** REST API calls to backend services

## Data Flow

### Registration Flow

```
Video Upload (UI)
    ↓
Register Service (8002)
    ↓
MOT17 Detection → Extract largest bbox
    ↓
ArcFace Extraction → 512-dim embeddings
    ↓
Qdrant Storage + Local Pickle
    ↓
Success Response
```

### Detection Flow

```
Video Upload (UI)
    ↓
Detection Service (8003)
    ↓
Load Registered Persons (Qdrant)
    ↓
For each frame:
    ├─ YOLOX Detection (144ms)
    ├─ ByteTrack Tracking (1.45ms)
    └─ ReID Matching:
        ├─ Frame 0-2: Extract + Vote (43ms × 3)
        ├─ Frame 30, 60...: Re-verify (43ms)
        └─ Other frames: Cached (0ms)
    ↓
Generate Outputs:
    ├─ Annotated Video
    ├─ CSV Tracking Data
    └─ Detailed Logs
```

### Extract Flow

```
Video Upload (UI)
    ↓
Extract Service (8001)
    ↓
For each frame:
    ├─ YOLOX Detection
    └─ ByteTrack Tracking
    ↓
Group frames by track_id
    ↓
For each track:
    ├─ Extract bbox crops
    ├─ Create individual video
    └─ Save to output_dir
    ↓
Return list of object videos
```

## Storage Architecture

### Shared Volumes (Docker)

```
person_reid_system/
├── outputs/              # Shared across all services
│   ├── videos/          # Annotated videos
│   ├── csv/             # Tracking CSVs
│   ├── logs/            # Detailed logs
│   └── extracted_objects/  # Extracted object videos
├── data/
│   ├── uploads/         # Temporary uploads
│   └── database/        # reid_database.pkl
├── models/              # Pre-trained models (shared)
│   ├── bytetrack_x_mot17.pth.tar
│   └── yolox_x.pth
├── configs/             # Configuration files
│   ├── config.yaml
│   └── .env
└── logs/                # Application logs
```

### Database Storage

**Qdrant (Cloud/Local):**
- Collection: `cross_camera_matching_id`
- Vector size: 512
- Distance: Cosine
- Payload: `{global_id, name, camera_id, track_id}`

**Local Pickle:**
- Path: `data/database/reid_database.pkl`
- Format: `{db: {global_id: [embeddings]}, person_metadata, next_global_id}`
- Sync: Automatic after each registration

## Performance Optimization

### First-3 + Re-verify Strategy

**Problem:** ReID every frame = 3.6 FPS (too slow)

**Solution:**
- Extract only 3 + N/30 times (vs N times)
- 95.8% reduction in extractions
- 5.3x speedup → 19 FPS

**Breakdown (3 persons, 150 frames):**
- Detection: 144.79ms (52.3%)
- Tracking: 1.45ms (0.5%)
- Feature Extraction: 130.30ms (47.1%) ← Bottleneck
- Qdrant Search: 0.19ms (0.1%)

**Optimization Impact:**
- Naive: 150 frames × 3 persons = 450 extractions
- Optimized: (3 voting + 5 re-verify) × 3 persons = 24 extractions
- Reduction: 94.7%

### GPU Acceleration

**CUDA Configuration:**
- FP16 precision for detection
- GPU-accelerated ArcFace inference
- Batch processing where possible

**Memory Management:**
- Shared memory: 4GB per service
- Model caching
- Efficient tensor operations

## Deployment Architecture

### Docker Compose Setup

**Network:** Bridge network `person_reid_network`

**Services:**
- extract: GPU runtime, 4GB shm
- register: GPU runtime, 4GB shm
- detection: GPU runtime, 4GB shm
- ui: CPU only, 2GB shm

**Health Checks:**
- Interval: 10-30s
- Timeout: 7-10s
- Retries: 3
- Start period: 30-40s

**Restart Policy:** `unless-stopped`

### GPU Support

**Requirements:**
- NVIDIA GPU with CUDA 11.8+
- NVIDIA Container Toolkit
- Docker daemon with NVIDIA runtime

**Configuration:**
```yaml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
shm_size: '4gb'
```

## Security Considerations

**API Keys:**
- Qdrant API key in `.env` file
- Not exposed in logs or responses

**File Upload:**
- Temporary storage in `/app/data/uploads`
- Cleanup after processing
- File type validation

**Network:**
- Internal bridge network
- Only UI exposed to host
- Services communicate internally

## Scalability

**Horizontal Scaling:**
- Each service can be scaled independently
- Load balancer for multiple instances
- Shared Qdrant database

**Vertical Scaling:**
- GPU memory for batch processing
- CPU cores for parallel processing
- RAM for caching

**Limitations:**
- Qdrant connection limit
- GPU memory constraints
- Disk I/O for video processing

