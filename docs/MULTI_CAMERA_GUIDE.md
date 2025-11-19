# Multi-Camera Processing & Job Cancellation Guide

## 📋 Tổng quan

Hệ thống Person ReID hỗ trợ xử lý đồng thời nhiều camera streams và cho phép dừng processing bất kỳ lúc nào.

### Tính năng chính:
- ✅ **Parallel Multi-Camera Processing**: Xử lý đồng thời nhiều camera streams
- ✅ **Frame Synchronization**: Đồng bộ frames từ các cameras
- ✅ **Job Cancellation**: Dừng processing bất kỳ lúc nào qua UI
- ✅ **Organized Output**: Mỗi camera có output riêng (video, CSV, JSON) trong ZIP file
- ✅ **Thread-Safe**: An toàn với multi-threading

---

## 🎥 Multi-Camera Processing

### 1. Kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input (UI)                          │
│  "udp://127.0.0.1:1905, udp://127.0.0.1:1906"              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              parse_stream_urls()                            │
│  Split URLs by comma or newline                             │
│  → ["udp://127.0.0.1:1905", "udp://127.0.0.1:1906"]        │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │ len(urls) > 1?          │
        └────┬────────────────┬───┘
             │ YES            │ NO
             ▼                ▼
    ┌────────────────┐  ┌──────────────┐
    │MultiStreamReader│  │StreamReader  │
    └────────┬────────┘  └──────┬───────┘
             │                  │
             ▼                  ▼
    ┌─────────────────────────────────┐
    │   Combined Frame Processing     │
    │   (Detection → Tracking → ReID) │
    └─────────────────────────────────┘
```

### 2. MultiStreamReader - Cơ chế hoạt động

#### 2.1. Threading Architecture

```
Main Thread                    Worker Thread 1              Worker Thread 2
    │                                │                           │
    │ Start workers                  │                           │
    ├───────────────────────────────►│ StreamReader(cam1)        │
    ├───────────────────────────────┼──────────────────────────►│ StreamReader(cam2)
    │                                │                           │
    │                                │ while True:               │ while True:
    │                                │   ret, frame = read()     │   ret, frame = read()
    │                                │   queue.put(frame)        │   queue.put(frame)
    │                                │   ↓                       │   ↓
    │                                │ [Queue 30 frames]         │ [Queue 30 frames]
    │                                │                           │
    │ read() called                  │                           │
    ├─ Get from queue1 ─────────────►│                           │
    ├─ Get from queue2 ──────────────┼──────────────────────────►│
    │                                │                           │
    │ Combine frames horizontally    │                           │
    │ [cam1_frame | cam2_frame]      │                           │
    │                                │                           │
    ▼                                ▼                           ▼
```

**Các thành phần:**
- **Worker Threads**: Mỗi camera có 1 thread riêng để đọc frames liên tục
- **Frame Queues**: Buffer 30 frames cho mỗi camera (configurable)
- **Main Thread**: Lấy frames từ queues và ghép lại

#### 2.2. Frame Synchronization

```python
def read(self) -> Tuple[bool, Optional[np.ndarray]]:
    frames = []
    for i, q in enumerate(self.frame_queues):
        try:
            # Wait max 0.1s for frame from each camera
            frame = q.get(timeout=self.sync_tolerance)
            frames.append(frame)
        except queue.Empty:
            # Camera lagging - handle gracefully
            return False, None
    
    # Combine frames horizontally
    combined = np.hstack(frames)
    return True, combined
```

**Sync Strategy:**
- Timeout: 0.1s (configurable via `sync_tolerance`)
- Nếu camera nào chậm → skip frame đó
- Đảm bảo frames từ các cameras gần đồng thời

#### 2.3. Frame Combination

```
Camera 1 (640x480)    Camera 2 (640x480)    →    Combined (1280x480)
┌──────────────┐      ┌──────────────┐           ┌────────────────────────┐
│              │      │              │           │            │           │
│   Person A   │  +   │   Person B   │    =      │  Person A  │ Person B  │
│              │      │              │           │            │           │
└──────────────┘      └──────────────┘           └────────────────────────┘
```

**Combination Logic:**
1. Resize frames nếu heights khác nhau (match min height)
2. Horizontal stack: `np.hstack([frame1, frame2, ...])`
3. Output: Single combined frame

### 3. Cách sử dụng

#### 3.1. Qua UI (Streamlit)

**Single Camera:**
```
Stream URL(s):
udp://127.0.0.1:1905
```

**Multiple Cameras (Comma-separated):**
```
Stream URL(s):
udp://127.0.0.1:1905, udp://127.0.0.1:1906
```

**Multiple Cameras (Newline-separated):**
```
Stream URL(s):
udp://127.0.0.1:1905
udp://127.0.0.1:1906
rtsp://192.168.1.100/stream
```

#### 3.2. Qua API

```python
import requests

# Multiple cameras
response = requests.post("http://localhost:8003/detect", json={
    "video_path": "udp://127.0.0.1:1905, udp://127.0.0.1:1906",
    "similarity_threshold": 0.8,
    "max_duration_seconds": 60
})

job_id = response.json()["job_id"]
```

#### 3.3. Qua Python Script

```python
from scripts.detect_and_track import DetectionTrackingPipeline

pipeline = DetectionTrackingPipeline(config_path="configs/config.yaml")

# Automatic multi-camera detection
pipeline.process_video(
    video_path="udp://127.0.0.1:1905, udp://127.0.0.1:1906",
    similarity_threshold=0.8,
    output_video_path="output/multi_cam.mp4"
)
```

### 4. Configuration

```python
# In utils/multi_stream_reader.py
MultiStreamReader(
    sources=["udp://127.0.0.1:1905", "udp://127.0.0.1:1906"],
    use_ffmpeg_for_udp=True,      # Use ffmpeg for UDP streams
    buffer_size=30,                # Queue size per camera (frames)
    sync_tolerance=0.1             # Max wait time for sync (seconds)
)
```

**Parameters:**
- `buffer_size`: Số frames buffer cho mỗi camera (default: 30)
  - Tăng nếu cameras có latency cao
  - Giảm để tiết kiệm memory
- `sync_tolerance`: Thời gian chờ tối đa để sync (default: 0.1s)
  - Tăng nếu cameras có jitter cao
  - Giảm để sync chặt chẽ hơn

### 5. Multi-Stream Output Structure

Khi xử lý multi-stream với zone monitoring, hệ thống tạo output riêng cho mỗi camera:

```
outputs/multi_stream_2024-01-15-14-30/
├── camera_0/
│   ├── output_20240115_143022.mp4      # Annotated video
│   ├── tracks_20240115_143022.csv      # Tracking data
│   └── zones_20240115_143022.json      # Zone monitoring report
├── camera_1/
│   ├── output_20240115_143022.mp4
│   ├── tracks_20240115_143022.csv
│   └── zones_20240115_143022.json
└── camera_2/
    ├── output_20240115_143022.mp4
    ├── tracks_20240115_143022.csv
    └── zones_20240115_143022.json
```

**Naming Convention:**
- Thư mục: `multi_stream_{YYYY-MM-DD-HH-MM}` (UTC+7 timezone)
- ZIP file: `multi_stream_{YYYY-MM-DD-HH-MM}_results.zip`
- Ví dụ: `multi_stream_2024-01-15-14-30_results.zip`

**Download từ UI:**
- Multi-stream job: Chỉ có nút **"📦 Download All Cameras (ZIP)"**
- ZIP file chứa toàn bộ cấu trúc thư mục trên
- Mỗi camera có đầy đủ video, CSV, và JSON report riêng

**Lưu ý:**
- Multi-stream **yêu cầu zone monitoring** phải được bật
- Không có "combined view" - mỗi camera được xử lý độc lập
- Parallel processing giúp tăng tốc độ xử lý

---

## 🛑 Job Cancellation

### 1. Architecture

```
UI (app.py)                Detection Service           Processing Pipeline
    │                            │                            │
    │ Click "Stop" button        │                            │
    ├───────────────────────────►│ POST /cancel/{job_id}      │
    │                            │                            │
    │                            │ Set cancellation_flag      │
    │                            │ (threading.Event)          │
    │                            ├───────────────────────────►│
    │                            │                            │
    │                            │                            │ while True:
    │                            │                            │   if flag.is_set():
    │                            │                            │     break
    │                            │                            │   process_frame()
    │                            │                            │
    │                            │ ◄──────────────────────────┤ Loop exits
    │                            │                            │
    │ ◄──────────────────────────┤ status = "cancelled"       │
    │ Show "Processing stopped"  │                            │
    │                            │                            │
```

### 2. Implementation

#### 2.1. Backend (Detection Service)

```python
# Global dictionary to track cancellation flags
cancellation_flags = {}

# When job starts
def process_detection(job_id, ...):
    # Initialize cancellation flag
    cancellation_flags[job_id] = threading.Event()
    
    try:
        # Pass flag to processing pipeline
        pipeline.process_video(
            ...,
            cancellation_flag=cancellation_flags[job_id]
        )
    finally:
        # Cleanup
        if job_id in cancellation_flags:
            del cancellation_flags[job_id]

# Cancel endpoint
@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    if job_id in cancellation_flags:
        cancellation_flags[job_id].set()  # Signal cancellation
    
    jobs[job_id]["status"] = "cancelled"
    return {"job_id": job_id, "status": "cancelled"}
```

#### 2.2. Processing Pipeline

```python
def process_video(self, ..., cancellation_flag=None):
    while True:
        # Check cancellation FIRST (every iteration)
        if cancellation_flag is not None and cancellation_flag.is_set():
            logger.info("Processing cancelled by user")
            break
        
        # Read frame
        ret, frame = stream_reader.read()
        if not ret:
            break
        
        # Process frame
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, (height, width))
        # ... ReID, visualization, etc.
```

#### 2.3. UI (Streamlit)

```python
# Create stop button container
stop_button_container = st.empty()

while True:
    # Show stop button (key changes to force re-render)
    if stop_button_container.button("🛑 Stop Processing", 
                                     key=f"stop_{job_id}_{poll_count}"):
        # Call cancel API
        response = requests.post(f"{DETECTION_API_URL}/cancel/{job_id}")
        if response.status_code == 200:
            st.warning("⚠️ Stopping processing...")
    
    # Poll job status
    status = requests.get(f"{DETECTION_API_URL}/status/{job_id}").json()
    
    if status["status"] == "cancelled":
        st.warning("⚠️ Processing stopped by user")
        stop_button_container.empty()  # Clear button
        break
    
    time.sleep(1)
```

### 3. Threading.Event Pattern

**Ưu điểm:**
- ✅ **Thread-safe**: `threading.Event` là atomic operation
- ✅ **Non-blocking**: Check nhanh (`is_set()`), không làm chậm processing
- ✅ **Clean shutdown**: Thoát vòng lặp gracefully, không force kill
- ✅ **Resource cleanup**: `finally` block đảm bảo cleanup

**So sánh với các phương pháp khác:**

| Method | Thread-safe | Non-blocking | Clean Shutdown | Complexity |
|--------|-------------|--------------|----------------|------------|
| `threading.Event` | ✅ | ✅ | ✅ | Low |
| Global flag | ❌ | ✅ | ✅ | Low |
| `multiprocessing.Event` | ✅ | ✅ | ✅ | High |
| Signal handler | ⚠️ | ❌ | ⚠️ | High |

---

## 📊 Performance Considerations

### 1. Memory Usage

**Single Camera:**
- Frame buffer: ~30 frames × 640×480×3 bytes = ~27 MB

**Multi-Camera (2 cameras):**
- Frame buffers: 2 × 27 MB = ~54 MB
- Combined frames: 1280×480×3 bytes per frame

**Optimization:**
- Giảm `buffer_size` nếu memory hạn chế
- Resize frames trước khi buffer

### 2. CPU Usage

**Threading Overhead:**
- Mỗi camera: 1 worker thread
- Main thread: Frame combination + processing
- Total: N+1 threads (N = số cameras)

**Optimization:**
- Sử dụng ffmpeg subprocess cho UDP (offload decoding)
- Limit số cameras đồng thời (recommend: ≤ 4)

### 3. Latency

**Frame Sync Latency:**
- `sync_tolerance = 0.1s` → max 100ms delay
- Tăng nếu cameras có jitter cao
- Giảm để real-time hơn

---

## 🔧 Troubleshooting

### 1. Cameras không sync

**Triệu chứng:**
- Frames từ các cameras không đồng thời
- Output video bị lag

**Giải pháp:**
```python
# Tăng sync_tolerance
MultiStreamReader(sources, sync_tolerance=0.5)  # 500ms

# Tăng buffer_size
MultiStreamReader(sources, buffer_size=60)  # 60 frames
```

### 2. Memory overflow

**Triệu chứng:**
- RAM tăng liên tục
- System crash

**Giải pháp:**
```python
# Giảm buffer_size
MultiStreamReader(sources, buffer_size=10)  # 10 frames

# Resize frames
# (Tự động resize trong MultiStreamReader nếu heights khác nhau)
```

### 3. Stop button không hoạt động

**Triệu chứng:**
- Click Stop nhưng processing vẫn chạy

**Kiểm tra:**
1. Check logs: `cancellation_flag.is_set()` có được gọi không?
2. Check API: `/cancel/{job_id}` có return 200 không?
3. Check processing loop: Có check `cancellation_flag` không?

**Debug:**
```python
# Add logging
if cancellation_flag is not None and cancellation_flag.is_set():
    logger.info("🛑 CANCELLATION DETECTED!")
    break
```

---

## 📝 Best Practices

### 1. Multi-Camera

✅ **DO:**
- Sử dụng cameras cùng resolution
- Sử dụng ffmpeg cho UDP streams
- Set reasonable `buffer_size` (30-60 frames)
- Monitor memory usage

❌ **DON'T:**
- Mix file + stream sources
- Quá nhiều cameras (>4) trên 1 machine
- Set `buffer_size` quá lớn (>100)

### 2. Job Cancellation

✅ **DO:**
- Check `cancellation_flag` ở đầu mỗi iteration
- Cleanup resources trong `finally` block
- Update job status thành "cancelled"
- Clear UI elements sau khi cancel

❌ **DON'T:**
- Force kill threads/processes
- Ignore cancellation flag
- Forget to cleanup resources
- Leave UI in inconsistent state

---

## 🎯 Examples

### Example 1: 2 UDP Cameras

```python
from scripts.detect_and_track import DetectionTrackingPipeline

pipeline = DetectionTrackingPipeline(config_path="configs/config.yaml")

pipeline.process_video(
    video_path="udp://127.0.0.1:1905, udp://127.0.0.1:1906",
    similarity_threshold=0.8,
    output_video_path="output/2_cameras.mp4",
    max_duration_seconds=60
)
```

### Example 2: 3 RTSP Cameras

```python
cameras = """
rtsp://192.168.1.100/stream
rtsp://192.168.1.101/stream
rtsp://192.168.1.102/stream
"""

pipeline.process_video(
    video_path=cameras,
    similarity_threshold=0.8,
    output_video_path="output/3_cameras.mp4"
)
```

### Example 3: With Cancellation

```python
import threading
import time

# Create cancellation flag
cancel_flag = threading.Event()

# Start processing in background thread
def process():
    pipeline.process_video(
        video_path="udp://127.0.0.1:1905, udp://127.0.0.1:1906",
        cancellation_flag=cancel_flag
    )

thread = threading.Thread(target=process)
thread.start()

# Cancel after 30 seconds
time.sleep(30)
cancel_flag.set()
print("Cancellation requested!")

thread.join()
print("Processing stopped")
```

---

## 📚 API Reference

### MultiStreamReader

```python
class MultiStreamReader:
    def __init__(
        self,
        sources: List[str],
        use_ffmpeg_for_udp: bool = True,
        buffer_size: int = 30,
        sync_tolerance: float = 0.1
    )
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]
    def release(self) -> None
    def get_properties(self) -> dict
```

### parse_stream_urls

```python
def parse_stream_urls(url_string: str) -> List[str]:
    """
    Parse multiple URLs from comma or newline separated string
    
    Args:
        url_string: String containing one or more URLs
        
    Returns:
        List of individual URLs
        
    Examples:
        >>> parse_stream_urls("udp://127.0.0.1:1905, udp://127.0.0.1:1906")
        ['udp://127.0.0.1:1905', 'udp://127.0.0.1:1906']
    """
```

### Cancel API

```http
POST /cancel/{job_id}

Response:
{
    "job_id": "abc123",
    "status": "cancelled",
    "message": "Job cancellation requested"
}
```

