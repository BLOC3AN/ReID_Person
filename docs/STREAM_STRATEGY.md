# Stream Processing Strategy Guide

## 📋 Overview

Document này mô tả chiến lược xử lý video streams trong Person ReID System, bao gồm:
- Frame reading strategies (OpenCV vs ffmpeg)
- Buffering và synchronization
- Multi-stream processing
- Error handling và recovery
- Performance optimization

---

## 🎯 Stream Types Supported

### 1. File-based Streams
- **Format**: MP4, AVI, MKV, MOV, etc.
- **Characteristics**: Fixed duration, seekable, reliable
- **Use Case**: Testing, offline processing, recorded footage

### 2. UDP Streams
- **Format**: `udp://host:port`
- **Characteristics**: Low latency, unreliable, no buffering
- **Use Case**: Real-time camera feeds, live monitoring
- **Challenges**: Missing SPS/PPS headers, packet loss

### 3. RTSP Streams
- **Format**: `rtsp://host:port/path`
- **Characteristics**: Reliable, buffered, higher latency
- **Use Case**: IP cameras, NVR systems
- **Challenges**: Network latency, authentication

### 4. HTTP/HTTPS Streams
- **Format**: `http://host:port/stream`
- **Characteristics**: Reliable, high latency
- **Use Case**: Web cameras, cloud streams

---

## 🏗️ Stream Reader Architecture

### Single Stream Architecture

```
┌──────────────┐
│ Video Source │ (UDP/RTSP/File)
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  StreamReader    │
│  ┌────────────┐  │
│  │ OpenCV Cap │  │ ← Try first (fast)
│  └────────────┘  │
│        │         │
│        ▼ (fail)  │
│  ┌────────────┐  │
│  │   ffmpeg   │  │ ← Fallback (reliable)
│  │ subprocess │  │
│  └────────────┘  │
└──────┬───────────┘
       │
       ▼
   Frame (BGR)
```

### Multi-Stream Architecture

```
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Camera 1│  │ Camera 2│  │ Camera N│
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Reader 1│  │ Reader 2│  │ Reader N│
│(Thread) │  │(Thread) │  │(Thread) │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Queue 1 │  │ Queue 2 │  │ Queue N │
│(30 frms)│  │(30 frms)│  │(30 frms)│
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     └────────────┼────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │ Synchronization │
        │  (timestamp)    │
        └────────┬────────┘
                 │
                 ▼
          Combined Frame
```

---

## 🔧 Frame Reading Strategies

### Strategy 1: OpenCV VideoCapture (Default)

**Pros:**
- ✅ Fast (native C++ implementation)
- ✅ Low overhead
- ✅ Direct memory access
- ✅ Works for most file formats

**Cons:**
- ❌ Fails on UDP streams with missing SPS/PPS headers
- ❌ Limited codec support
- ❌ Poor error recovery

**Code:**
```python
cap = cv2.VideoCapture(source)
ret, frame = cap.read()
```

**Use Cases:**
- File-based videos
- RTSP streams with proper headers
- HTTP streams

---

### Strategy 2: ffmpeg Subprocess (Fallback)

**Pros:**
- ✅ Handles problematic UDP streams
- ✅ Excellent codec support
- ✅ Robust error handling
- ✅ Timeout support

**Cons:**
- ❌ Higher CPU overhead
- ❌ Subprocess management complexity
- ❌ Slightly higher latency (~10-20ms)

**Implementation:**
```python
ffmpeg_cmd = [
    'ffmpeg',
    '-timeout', '5000000',  # 5 second timeout
    '-i', source,
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-an',  # No audio
    'pipe:1'
]

process = subprocess.Popen(
    ffmpeg_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    bufsize=frame_size * 10
)

# Read frame
raw_frame = process.stdout.read(frame_size)
frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
```

**Use Cases:**
- UDP streams (especially with missing headers)
- Problematic RTSP streams
- When OpenCV fails

---

### Strategy 3: Automatic Fallback (Recommended)

**Logic:**
```python
class StreamReader:
    def __init__(self, source, use_ffmpeg_for_udp=True):
        self.is_udp = source.startswith('udp://')
        
        if self.is_udp and use_ffmpeg_for_udp:
            # Use ffmpeg directly for UDP
            self._initialize_ffmpeg_udp()
        else:
            try:
                # Try OpenCV first
                self._initialize_opencv()
            except Exception:
                # Fallback to ffmpeg
                self._initialize_ffmpeg()
```

**Benefits:**
- ✅ Best of both worlds
- ✅ Automatic error recovery
- ✅ Optimal performance for each source type

---

## 📦 Frame Buffering Strategy

### Single Stream Buffering

**No buffering** - Read frame on demand:
```python
ret, frame = stream_reader.read()
if ret:
    process_frame(frame)
```

**Pros:**
- Simple implementation
- Low memory usage
- No synchronization needed

**Cons:**
- Blocking I/O
- Can't handle frame drops
- No lookahead

---

### Multi-Stream Buffering

**Queue-based buffering** with worker threads:

```python
class MultiStreamReader:
    def __init__(self, sources, buffer_size=30):
        self.buffer_size = buffer_size  # 30 frames ≈ 1 second @ 30 FPS
        self.frame_queues = [Queue(maxsize=buffer_size) for _ in sources]
        
    def _read_stream_worker(self, stream_id, reader, frame_queue):
        """Worker thread continuously reads frames"""
        while self.running:
            ret, frame = reader.read()
            if ret:
                timestamp = time.time()
                try:
                    # Non-blocking put
                    frame_queue.put((timestamp, frame), block=False)
                except queue.Full:
                    # Drop oldest frame
                    frame_queue.get_nowait()
                    frame_queue.put((timestamp, frame), block=False)
```

**Buffer Size Guidelines:**

| FPS | Buffer Size | Memory per Stream | Latency |
|-----|-------------|-------------------|---------|
| 15 | 15 frames | ~30MB | 1 second |
| 25 | 25 frames | ~50MB | 1 second |
| 30 | 30 frames | ~60MB | 1 second |

**Formula**: `buffer_size = fps * desired_latency_seconds`

---

## 🔄 Frame Synchronization

### Problem: Multi-camera frame alignment

Cameras may have:
- Different FPS
- Network delays
- Processing time variations

### Solution: Timestamp-based synchronization

```python
def read(self) -> Tuple[bool, np.ndarray]:
    """Read synchronized frames from all streams"""
    frames = []
    timestamps = []
    
    # Get one frame from each stream
    for frame_queue in self.frame_queues:
        timestamp, frame = frame_queue.get(timeout=1.0)
        frames.append(frame)
        timestamps.append(timestamp)
    
    # Check synchronization
    time_diff = max(timestamps) - min(timestamps)
    if time_diff > self.sync_tolerance:  # Default: 0.1 seconds
        logger.warning(f"Frames out of sync: {time_diff:.3f}s")
    
    # Combine frames (horizontal stack)
    combined = np.hstack(frames)
    return True, combined
```

**Sync Tolerance Guidelines:**

| Use Case | Tolerance | Trade-off |
|----------|-----------|-----------|
| Strict sync | 0.033s (1 frame @ 30 FPS) | May drop frames |
| Balanced | 0.1s (3 frames @ 30 FPS) | Good for most cases |
| Loose sync | 0.5s (15 frames @ 30 FPS) | Rarely drops frames |

---

## 🛡️ Error Handling & Recovery

### Frame Read Failures

**Strategy: Retry with exponential backoff**

```python
consecutive_failures = 0
max_failures = 10

while running:
    ret, frame = reader.read()
    
    if ret and frame is not None:
        consecutive_failures = 0
        process_frame(frame)
    else:
        consecutive_failures += 1
        if consecutive_failures >= max_failures:
            logger.error("Stream failed, stopping")
            break
        time.sleep(0.01 * consecutive_failures)  # Exponential backoff
```

### Stream Disconnection

**Strategy: Automatic reconnection**

```python
def _reconnect_stream(self):
    """Attempt to reconnect to stream"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Reconnection attempt {attempt + 1}/{max_retries}")
            self._initialize()
            logger.info("✓ Reconnected successfully")
            return True
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
    return False
```

### UDP Packet Loss

**Strategy: Skip incomplete frames**

```python
frame_size = width * height * 3
raw_frame = process.stdout.read(frame_size)

if len(raw_frame) != frame_size:
    logger.warning(f"Incomplete frame: {len(raw_frame)}/{frame_size} bytes")
    return False, None  # Skip this frame

frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
return True, frame
```

---

## ⚡ Performance Optimization

### 1. Frame Preprocessing Pipeline

**Optimize preprocessing order:**

```python
# ❌ Bad: Preprocess every frame
for frame in stream:
    preprocessed = preprocess(frame)  # Resize, normalize, etc.
    detections = detector.detect(preprocessed)

# ✅ Good: Batch preprocessing
frames = []
for _ in range(batch_size):
    ret, frame = stream.read()
    if ret:
        frames.append(frame)

# Preprocess batch (vectorized)
preprocessed_batch = preprocess_batch(frames)
detections_batch = detector.detect_batch(preprocessed_batch)
```

### 2. Frame Skipping Strategy

**Skip frames when processing is slow:**

```python
class AdaptiveFrameSkip:
    def __init__(self, target_fps=25):
        self.target_fps = target_fps
        self.target_interval = 1.0 / target_fps
        self.last_process_time = time.time()
    
    def should_process_frame(self) -> bool:
        """Decide if we should process this frame"""
        current_time = time.time()
        elapsed = current_time - self.last_process_time
        
        if elapsed >= self.target_interval:
            self.last_process_time = current_time
            return True
        return False

# Usage
skipper = AdaptiveFrameSkip(target_fps=15)
while True:
    ret, frame = stream.read()
    if ret and skipper.should_process_frame():
        process_frame(frame)
```

### 3. Multi-threading Strategy

**Separate I/O and processing threads:**

```
Thread 1 (I/O):          Thread 2 (Processing):
  Read frame    ──────►    Queue (buffer)
  Read frame    ──────►    Queue (buffer)
  Read frame    ──────►    Queue (buffer)
                              │
                              ▼
                         Process frame
                         Process frame
                         Process frame
```

**Benefits:**
- I/O doesn't block processing
- Processing doesn't block I/O
- Better CPU utilization

---

## 📊 Performance Metrics

### Single Stream Performance

| Source Type | Read Time | Preprocess Time | Total Overhead |
|-------------|-----------|-----------------|----------------|
| File (OpenCV) | 1-2ms | 3-5ms | 4-7ms |
| UDP (ffmpeg) | 5-10ms | 3-5ms | 8-15ms |
| RTSP (OpenCV) | 10-20ms | 3-5ms | 13-25ms |

### Multi-Stream Performance (4 cameras)

| Strategy | Total Read Time | Memory Usage | CPU Usage |
|----------|----------------|--------------|-----------|
| Sequential | 40-80ms | ~100MB | 20-30% |
| **Parallel (threads)** | **10-20ms** | **~200MB** | **40-60%** |

**Speedup**: 2-4x faster with parallel reading

---

## 🎯 Best Practices

### 1. Choose Right Buffer Size
```python
# For real-time (low latency)
buffer_size = fps * 0.5  # 0.5 second buffer

# For reliability (handle jitter)
buffer_size = fps * 1.0  # 1 second buffer

# For offline processing
buffer_size = fps * 2.0  # 2 second buffer
```

### 2. Handle Stream Timeouts
```python
# Always set timeout for network streams
stream_reader = StreamReader(
    "udp://127.0.0.1:1905",
    use_ffmpeg_for_udp=True
)

# ffmpeg automatically uses 5-second timeout
```

### 3. Monitor Frame Drops
```python
total_frames = 0
dropped_frames = 0

ret, frame = stream.read()
if not ret:
    dropped_frames += 1
total_frames += 1

drop_rate = dropped_frames / total_frames
if drop_rate > 0.05:  # > 5% drop rate
    logger.warning(f"High frame drop rate: {drop_rate:.1%}")
```

### 4. Graceful Shutdown
```python
class StreamReader:
    def close(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait(timeout=5)
```

---

## 🔗 Related Documents

- [Backend Strategy](BACKEND_STRATEGY.md)
- [Stream Troubleshooting](STREAM_TROUBLESHOOTING.md)
- [Multi-Camera Guide](MULTI_CAMERA_GUIDE.md)
- [Configuration Guide](CONFIGURATION.md)

---

**Last Updated**: 2025-11-11

