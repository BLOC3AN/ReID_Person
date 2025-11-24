# Services API Documentation

## Overview

Hệ thống Person ReID sử dụng kiến trúc microservices với 4 services chính:

1. **Extract Service** (Port 8001) - Tách video thành object videos
2. **Register Service** (Port 8002) - Đăng ký người vào database
3. **Detection Service** (Port 8003) - Detect và track persons
4. **UI Service** (Port 8501) - Web interface

Tất cả services đều cung cấp OpenAPI documentation tại `/docs` endpoint.

---

## 1. Extract Service (Port 8001)

### Mục đích

Tách video có nhiều người thành các video riêng lẻ cho từng người được track.

### Base URL

```
http://localhost:8001
```

### Endpoints

#### POST /extract

Upload video và extract individual object videos.

**Request:**
```bash
curl -X POST "http://localhost:8001/extract" \
  -F "video=@input.mp4" \
  -F "model_type=mot17" \
  -F "padding=10" \
  -F "conf_thresh=0.6" \
  -F "track_thresh=0.5" \
  -F "min_frames=10"
```

**Parameters:**
- `video` (file, required): Video file (MP4, AVI, MKV, MOV)
- `model_type` (string, optional): "mot17" hoặc "yolox" (default: "mot17")
- `padding` (int, optional): Padding pixels around bbox (default: 10)
- `conf_thresh` (float, optional): Detection confidence threshold (default: 0.6)
- `track_thresh` (float, optional): Tracking threshold (default: 0.5)
- `min_frames` (int, optional): Minimum frames to save object (default: 10)

**Response:**
```json
{
  "job_id": "abc123",
  "status": "pending",
  "message": "Extraction job started"
}
```

#### GET /status/{job_id}

Kiểm tra trạng thái extraction job.

**Response:**
```json
{
  "job_id": "abc123",
  "status": "completed",
  "video_path": "/app/data/uploads/abc123_input.mp4",
  "output_dir": "/app/outputs/extracted_objects/abc123",
  "total_objects": 5,
  "error": null
}
```

**Status values:**
- `pending`: Job đang chờ xử lý
- `processing`: Đang extract
- `completed`: Hoàn thành
- `failed`: Lỗi

#### GET /results/{job_id}

Liệt kê các object videos đã extract.

**Response:**
```json
{
  "job_id": "abc123",
  "total_objects": 5,
  "files": [
    "object_1.mp4",
    "object_2.mp4",
    "object_3.mp4",
    "object_4.mp4",
    "object_5.mp4"
  ]
}
```

#### GET /download/{job_id}/{filename}

Download một object video cụ thể.

**Example:**
```bash
curl -O "http://localhost:8001/download/abc123/object_1.mp4"
```

**Response:** Video file (MP4)

#### GET /download/zip/{job_id}

Download tất cả object videos dưới dạng ZIP.

**Example:**
```bash
curl -O "http://localhost:8001/download/zip/abc123"
```

**Response:** ZIP file chứa tất cả object videos

#### DELETE /jobs/{job_id}

Xóa job và các files liên quan.

**Response:**
```json
{
  "message": "Job deleted successfully"
}
```

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### Output Structure

```
outputs/extracted_objects/{job_id}/
└── input/
    ├── object_1.mp4
    ├── object_2.mp4
    ├── object_3.mp4
    └── ...
```

---

## 2. Register Service (Port 8002)

### Mục đích

Đăng ký người vào Qdrant vector database để sử dụng cho detection.

### Base URL

```
http://localhost:8002
```

### Endpoints

#### POST /register

Register person từ video.

**Request:**
```bash
curl -X POST "http://localhost:8002/register" \
  -F "video=@person.mp4" \
  -F "person_name=John" \
  -F "global_id=1" \
  -F "sample_rate=5" \
  -F "delete_existing=false"
```

**Parameters:**
- `video` (file, required): Video chứa người cần register
- `person_name` (string, required): Tên người
- `global_id` (int, required): ID duy nhất (1, 2, 3...)
- `sample_rate` (int, optional): Extract 1 frame mỗi N frames (default: 5)
- `delete_existing` (bool, optional): Xóa collection cũ trước khi register (default: false)

**Response:**
```json
{
  "job_id": "def456",
  "status": "pending",
  "message": "Registration job started for John"
}
```

**Important Notes:**
- Mỗi person phải có `global_id` duy nhất
- `delete_existing=true` sẽ xóa toàn bộ collection và tạo mới
- Video nên có clear face visibility (ArcFace yêu cầu)
- Sample rate thấp hơn = nhiều embeddings hơn = chính xác hơn

#### GET /status/{job_id}

Kiểm tra trạng thái registration job.

**Response:**
```json
{
  "job_id": "def456",
  "status": "completed",
  "video_path": "/app/data/uploads/def456_person.mp4",
  "person_name": "John",
  "global_id": 1,
  "total_embeddings": 45,
  "error": null
}
```

#### DELETE /jobs/{job_id}

Xóa registration job và uploaded video.

**Response:**
```json
{
  "message": "Job deleted successfully"
}
```

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### Registration Process

1. Upload video → temp storage
2. Detect persons using MOT17 model
3. Extract largest bbox (assume target person)
4. Extract face embeddings using ArcFace
5. Store embeddings to Qdrant database
6. Return success with embedding count

### Database Storage

- **Qdrant:** All embeddings stored in Qdrant vector database
- **Collection:** `cross_camera_matching_id`
- **Payload:** `{global_id, name, camera_id, track_id}`

---

## 3. Detection Service (Port 8003)

### Mục đích

Detect, track và re-identify persons trong video sử dụng registered database.

### Base URL

```
http://localhost:8003
```

### Endpoints

#### POST /detect

Upload video và detect registered persons.

**Request:**
```bash
curl -X POST "http://localhost:8003/detect" \
  -F "video=@test.mp4" \
  -F "similarity_threshold=0.8" \
  -F "model_type=mot17" \
  -F "conf_thresh=0.5" \
  -F "track_thresh=0.5"
```

**Parameters:**
- `video` (file, required): Video file để detect
- `similarity_threshold` (float, optional): Cosine similarity threshold (default: 0.8)
- `model_type` (string, optional): "mot17" or "yolox" (default: from config)
- `conf_thresh` (float, optional): Detection confidence 0-1 (default: from config)
- `track_thresh` (float, optional): Tracking threshold 0-1 (default: from config)

**Response:**
```json
{
  "job_id": "ghi789",
  "status": "pending",
  "message": "Detection job started"
}
```

#### GET /status/{job_id}

Kiểm tra trạng thái detection job.

**Response:**
```json
{
  "job_id": "ghi789",
  "status": "completed",
  "video_path": "/app/data/uploads/ghi789_test.mp4",
  "output_video": "/app/outputs/videos/ghi789_output.mp4",
  "output_csv": "/app/outputs/csv/ghi789_tracking.csv",
  "output_log": "/app/outputs/logs/ghi789_detection.log",
  "total_frames": 450,
  "total_persons": 3,
  "error": null
}
```

#### GET /download/video/{job_id}

Download annotated video.

**Example:**
```bash
curl -O "http://localhost:8003/download/video/ghi789"
```

**Response:** Annotated MP4 video với:
- Bounding boxes (green=known, red=unknown)
- Person labels + similarity scores
- FPS counter (top-left)
- Frame counter

#### GET /download/csv/{job_id}

Download tracking CSV data.

**Example:**
```bash
curl -O "http://localhost:8003/download/csv/ghi789"
```

**Response:** CSV file

**CSV Format:**
```csv
frame_id,track_id,x,y,w,h,confidence,global_id,similarity,label
0,1,396,189,238,340,0.8796,1,1.0000,John
0,2,31,216,206,261,0.7674,2,0.9234,Mary
1,1,398,190,237,339,0.8823,1,1.0000,John
```

**Columns:**
- `frame_id`: Frame number (0-indexed)
- `track_id`: Local track ID
- `x, y, w, h`: Bounding box (top-left x, y, width, height)
- `confidence`: Detection confidence (0-1)
- `global_id`: Global person ID from database
- `similarity`: Cosine similarity (0-1)
- `label`: Person name or "Unknown"

#### GET /download/log/{job_id}

Download detailed processing log.

**Example:**
```bash
curl -O "http://localhost:8003/download/log/ghi789"
```

**Response:** Text log file

**Log Format:**
```
[Frame 0] Detected 2 objects, Tracked 2 persons
  Track 1 [VOTING]: 3/3 votes → John (sim=0.9606, gid=1)
  Track 2 [VOTING]: 3/3 votes → Mary (sim=0.9468, gid=2)
[Frame 30] Re-verification
  Track 1 [RE-VERIFY]: John → John (sim=0.9654, frame=30)
  Track 2 [RE-VERIFY]: Mary → Mary (sim=0.9427, frame=30)
```

**Log Events:**
- `[VOTING]`: First-3 frames majority voting
- `[RE-VERIFY]`: Re-verification at frame 30, 60, 90...

#### DELETE /jobs/{job_id}

Xóa detection job và các files.

**Response:**
```json
{
  "message": "Job deleted successfully"
}
```

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### Detection Process

1. Upload video → temp storage
2. Load registered persons from Qdrant
3. Initialize detector + tracker + extractor
4. Process video:
   - Frame 0-2: Extract + Vote (First-3 strategy)
   - Frame 30, 60, 90...: Re-verify
   - Other frames: Use cached labels
5. Generate outputs:
   - Annotated video
   - CSV tracking data
   - Detailed logs

### ReID Strategy

**First-3 + Re-verify:**
- Extract embeddings from first 3 frames → Majority vote
- Re-verify every 30 frames for self-correction
- Cache labels for other frames (fast)
- Performance: ~19 FPS (5.3x speedup)

---

## 4. UI Service (Port 8501)

### Mục đích

Web interface cho tất cả operations.

### Base URL

```
http://localhost:8501
```

### Pages

#### 1. Extract Objects

**Features:**
- Upload video (MP4, AVI, MKV, MOV)
- Configure parameters (model, padding, thresholds)
- Real-time job status
- Download individual objects or ZIP

**Workflow:**
1. Upload video
2. Configure parameters
3. Start extraction
4. Monitor progress
5. Download results

#### 2. Register Person

**Features:**
- Upload person video
- Enter name and global ID
- Configure sample rate
- Option to delete existing collection
- Real-time registration status

**Workflow:**
1. Upload video with clear face
2. Enter person details
3. Start registration
4. View embedding count
5. Success confirmation

#### 3. Detect & Track

**Features:**
- Upload test video
- Configure detection parameters (model, thresholds)
- Automatic person detection
- Real-time processing status
- Download annotated video, CSV, logs

**Parameters:**
- Model type (mot17/yolox)
- Similarity threshold (0.5-1.0)
- Detection confidence (0.1-1.0)
- Tracking threshold (0.1-1.0)

**Workflow:**
1. Upload video
2. Configure parameters (optional)
3. Start detection
4. Monitor progress
5. Download results (video/CSV/log)

#### 4. About

**Information:**
- System overview
- Pipeline description
- ReID strategy explanation
- Documentation links

### API Communication

UI service gọi backend services qua REST API:

```python
EXTRACT_API_URL = "http://extract:8001"
REGISTER_API_URL = "http://register:8002"
DETECTION_API_URL = "http://detection:8003"
```

---

## Error Handling

### Common Error Responses

**400 Bad Request:**
```json
{
  "detail": "Invalid file format. Supported: mp4, avi, mkv, mov"
}
```

**404 Not Found:**
```json
{
  "detail": "Job not found"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Failed to process video: CUDA out of memory"
}
```

### Job Status Errors

```json
{
  "job_id": "abc123",
  "status": "failed",
  "error": "No face detected in video. Please use video with clear face visibility."
}
```

---

## Best Practices

### Extract Service

1. Use `mot17` model for better accuracy
2. Set `min_frames=10` to filter short tracks
3. Adjust `conf_thresh` based on video quality
4. Use `padding=10` for better crop quality

### Register Service

1. Use video with **clear frontal face**
2. Lower `sample_rate` (3-5) for more embeddings
3. Each person needs unique `global_id`
4. Use `delete_existing=true` to start fresh

### Detection Service

1. Register persons before detection
2. Use same model (MOT17) as registration
3. Adjust `similarity_threshold` (0.7-0.9) based on accuracy needs
4. Lower `conf_thresh` (0.3-0.4) for more detections
5. Lower `track_thresh` (0.3-0.4) for easier tracking
6. Check logs for [VOTING] and [RE-VERIFY] events

### UI Service

1. Upload videos < 500MB for better performance
2. Monitor job status regularly
3. Download results before deleting jobs
4. Check logs if detection fails

