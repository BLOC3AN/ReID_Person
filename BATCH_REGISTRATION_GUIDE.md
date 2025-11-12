# Batch Registration Guide

## Overview

The Register Service now supports batch registration - allowing you to register a person with multiple videos in a single request.

## Features

✅ **Single Video Registration** - Original single video upload  
✅ **Batch Video Registration** - Upload multiple videos at once  
✅ **Parallel Processing** - All videos processed in background  
✅ **Progress Tracking** - Monitor each video's registration status  
✅ **Streamlit UI** - Easy-to-use web interface with toggle between modes  

## API Endpoints

### 1. Single Video Registration
```
POST /register
```

**Parameters:**
- `video` (File): Single video file
- `person_name` (str): Name of the person
- `global_id` (int): Unique ID for the person
- `sample_rate` (int): Extract 1 frame every N frames (default: 5)
- `delete_existing` (bool): Delete existing collection (default: false)

**Response:**
```json
{
  "job_id": "uuid",
  "status": "pending",
  "message": "Registration job started for {person_name}"
}
```

### 2. Batch Video Registration
```
POST /register-batch
```

**Parameters:**
- `videos` (List[File]): Multiple video files
- `person_name` (str): Name of the person
- `global_id` (int): Unique ID for the person
- `sample_rate` (int): Extract 1 frame every N frames (default: 5)
- `delete_existing` (bool): Delete existing collection (default: false)

**Response:**
```json
{
  "job_ids": ["uuid1", "uuid2", "uuid3"],
  "total_videos": 3,
  "status": "pending",
  "message": "Registration jobs started for {person_name} with 3 video(s)"
}
```

### 3. Get Job Status
```
GET /status/{job_id}
```

**Response:**
```json
{
  "job_id": "uuid",
  "status": "pending|processing|completed|failed",
  "message": "Status message",
  "person_name": "John Doe",
  "global_id": 1,
  "error": "Error message if failed"
}
```

### 4. Get Batch Status
```
GET /status-batch?job_ids=uuid1,uuid2,uuid3
```

**Response:**
```json
{
  "total_jobs": 3,
  "jobs": [
    {"job_id": "uuid1", "status": "completed", ...},
    {"job_id": "uuid2", "status": "processing", ...},
    {"job_id": "uuid3", "status": "pending", ...}
  ]
}
```

## Using Streamlit UI

### Single Video Mode
1. Select "Single Video" from Upload Mode
2. Upload one video file
3. Enter person name and global ID
4. Click "Register Person"
5. Monitor progress in real-time

### Multiple Videos Mode
1. Select "Multiple Videos" from Upload Mode
2. Upload multiple video files
3. Enter person name and global ID
4. Click "Register Person"
5. Monitor progress for each video in expandable sections

## Using cURL

### Single Video
```bash
curl -X POST http://localhost:8002/register \
  -F "video=@video1.mp4" \
  -F "person_name=John Doe" \
  -F "global_id=1" \
  -F "sample_rate=5"
```

### Multiple Videos
```bash
curl -X POST http://localhost:8002/register-batch \
  -F "videos=@video1.mp4" \
  -F "videos=@video2.mp4" \
  -F "videos=@video3.mp4" \
  -F "person_name=John Doe" \
  -F "global_id=1" \
  -F "sample_rate=5"
```

## Using Python

```python
import requests

# Batch registration
files = [
    ("videos", open("video1.mp4", "rb")),
    ("videos", open("video2.mp4", "rb")),
    ("videos", open("video3.mp4", "rb"))
]
data = {
    "person_name": "John Doe",
    "global_id": 1,
    "sample_rate": 5
}

response = requests.post("http://localhost:8002/register-batch", files=files, data=data)
result = response.json()
job_ids = result["job_ids"]

# Check status
for job_id in job_ids:
    status = requests.get(f"http://localhost:8002/status/{job_id}").json()
    print(f"Job {job_id}: {status['status']}")
```

## Implementation Details

### Backend Changes

**services/register_service.py:**
- Added `List` import from typing
- Added `/register-batch` endpoint for batch registration
- Added `/status-batch` endpoint for checking multiple job statuses
- Batch endpoint processes each video as a separate background task
- Only deletes existing collection on first video (if requested)

**scripts/register_mot17.py:**
- Added `import os` for environment variable access
- Uses `QDRANT_USE_GRPC` environment variable for gRPC support

**deployment/Dockerfile.register:**
- Added `COPY exps/ ./exps/` to include experiment config files
- Fixed missing exps directory that caused "Exp class not found" error

### Frontend Changes

**app.py:**
- Added radio button to toggle between "Single Video" and "Multiple Videos" modes
- Single mode: `st.file_uploader()` returns single file
- Multiple mode: `st.file_uploader(..., accept_multiple_files=True)` returns list
- Updated registration logic to use `/register` for single or `/register-batch` for multiple
- Added expandable sections to track progress for each video
- Improved progress tracking with per-video status updates

## Testing

Run the test script:
```bash
python test_batch_register.py
```

This will:
1. Create 3 dummy video files
2. Upload them to the batch registration API
3. Poll job status until completion
4. Clean up test files

## Notes

- Each video is processed as a separate background task
- Videos are processed in parallel (not sequentially)
- `delete_existing` flag only applies to the first video
- All videos for the same person use the same `global_id`
- Maximum file size depends on server configuration
- Videos should contain clear face for best results

