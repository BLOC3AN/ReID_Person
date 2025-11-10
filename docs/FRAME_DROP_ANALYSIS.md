# Frame Drop Analysis - Real-time Stream Processing

## 📋 Executive Summary

Trong quá trình xử lý video stream real-time, hệ thống **có chủ đích drop một số frames** để đảm bảo:
- ✅ **Real-time processing** - Không bị delay tích lũy
- ✅ **System stability** - Tránh memory overflow
- ✅ **Graceful degradation** - Xử lý được network jitter và packet loss

**Trade-off:** Ưu tiên **real-time responsiveness** hơn **frame completeness**.

---

## 🎯 Tại sao cần Drop Frame?

### Vấn đề cơ bản

```
Camera FPS:     30 FPS (33.3 ms/frame)
Processing:     25 FPS (40 ms/frame)

→ Processing chậm hơn 6.7 ms/frame
→ Sau 1 giây: Delay tích lũy = 200 ms
→ Sau 10 giây: Delay tích lũy = 2 giây
→ Sau 1 phút: Delay tích lũy = 12 giây ❌
```

**Nếu KHÔNG drop frame:**
- Video output sẽ bị delay ngày càng lớn
- Memory buffer sẽ tràn (overflow)
- System crash hoặc freeze

**Khi CÓ drop frame:**
- Giữ được real-time processing
- Video có thể bị giật nhưng không bị delay
- System ổn định

---

## 🔍 3 Điểm Drop Frame trong Hệ thống

### **1. StreamReader - Network/Decode Level**

**Vị trí:** `utils/stream_reader.py` (Line 266-285)

**Cơ chế:**
```python
# Timeout 3 giây chờ data từ ffmpeg
ready, _, _ = select.select([ffmpeg_stdout], [], [], 3.0)
if not ready:
    logger.warning("No frame available (3s timeout) - skipping")
    return False, None  # ❌ DROP FRAME

# Kiểm tra frame integrity
if len(raw_frame) != expected_size:
    logger.warning(f"Incomplete frame - skipping")
    return False, None  # ❌ DROP FRAME
```

**Nguyên nhân drop:**
- **Network packet loss** (UDP stream)
- **ffmpeg decode timeout** (stream lag)
- **Incomplete frame data** (corrupted packets)

**Tần suất:**
- UDP stream: 1-5% frames (tùy network quality)
- RTSP stream: 0.1-1% frames
- Local file: ~0%

**Ví dụ:**
```
Input stream:  F1 → F2 → [LOST] → F4 → F5 → F6
StreamReader:  ✓    ✓     ❌      ✓    ✓    ✓
Output:        F1   F2           F4   F5   F6
```

---

### **2. Processing Loop - Consecutive Failures**

**Vị trí:** `scripts/detect_and_track.py` (Line 322-337)

**Cơ chế:**
```python
consecutive_failures = 0
max_consecutive_failures = 30  # For streams

while True:
    ret, frame = stream_reader.read()
    
    if not ret:
        consecutive_failures += 1
        if consecutive_failures <= max_consecutive_failures:
            logger.debug(f"Skipped frame ({consecutive_failures}/30)")
            continue  # ❌ SKIP FRAME (không tăng frame_id)
        else:
            logger.error("Stream lost after 30 failures")
            break
    
    consecutive_failures = 0  # Reset on success
    # Process frame...
```

**Nguyên nhân drop:**
- **Cascading failures** từ StreamReader
- **Temporary stream interruption**
- **Network congestion**

**Tần suất:**
- Bình thường: 0-2% frames
- Network unstable: 5-10% frames
- Stream reconnecting: 10-20% frames

**Ví dụ:**
```
Read attempts:
  Frame 100: ✓ (success) → process
  Frame 101: ❌ (fail) → continue → skip
  Frame 102: ❌ (fail) → continue → skip
  Frame 103: ❌ (fail) → continue → skip
  Frame 104: ✓ (success) → process

Output video:
  Frame 100 → Frame 104 (thiếu 101, 102, 103)
```

---

### **3. MultiStreamReader - Queue Overflow**

**Vị trí:** `utils/multi_stream_reader.py` (Line 126-135)

**Cơ chế:**
```python
# Worker thread đọc frames liên tục
while running:
    ret, frame = reader.read()
    if ret:
        try:
            # Non-blocking put
            frame_queue.put((timestamp, frame), block=False)
        except queue.Full:
            # Queue đầy (30 frames) → drop oldest
            frame_queue.get_nowait()  # ❌ DROP oldest frame
            frame_queue.put((timestamp, frame), block=False)
```

**Nguyên nhân drop:**
- **Processing slower than camera FPS**
- **Heavy computation** (detection, tracking, ReID)
- **Multi-camera** (nhiều streams cùng lúc)

**Tần suất:**
- Processing = Camera FPS: 0% drop
- Processing 90% of FPS: ~10% drop
- Processing 80% of FPS: ~20% drop

**Ví dụ:**
```
Camera: 30 FPS (33ms/frame)
Processing: 25 FPS (40ms/frame)

Queue state over time:
  t=0s:   [F1, F2, F3, ..., F30] ← FULL (30 frames)
  t=1s:   [F6, F7, F8, ..., F35] ← Dropped F1-F5
  t=2s:   [F11, F12, ..., F40]   ← Dropped F6-F10
  
→ Drop rate: 5 frames/second = 16.7%
```

---

## 📊 Frame Drop Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                Camera Stream (30 FPS)                       │
│  F1 → F2 → F3 → [LOST] → F5 → F6 → F7 → F8 → F9 → F10    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         ① StreamReader (Network/Decode Level)               │
│  ✓    ✓    ✓    ❌    ✓    ✓    ✓    ✓    ✓    ✓         │
│  F1   F2   F3  SKIP   F5   F6   F7   F8   F9   F10        │
│                 ↑                                           │
│         Packet loss / Timeout                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         ② Processing Loop (Consecutive Failures)            │
│  ✓    ✓    ✓    ✓    ❌   ❌   ✓    ✓    ✓    ✓          │
│  F1   F2   F3   F5   F6  F7   F8   F9   F10               │
│                      ↑    ↑                                 │
│              Read failures (continue)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         ③ MultiStreamReader Queue (Buffer Overflow)         │
│                                                             │
│  Queue (30 frames max):                                     │
│  [F1, F2, F3, F5, F8, F9, F10, ...]                        │
│                                                             │
│  If processing slow (25 FPS):                               │
│  [F3, F5, F8, F9, F10, ...]  ← F1, F2 dropped              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Output Video                               │
│  F3 → F5 → F8 → F9 → F10 → ...                            │
│  (Thiếu F1, F2, F4, F6, F7)                                │
│  Drop rate: ~40% trong ví dụ này                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 Performance Analysis

### Scenario 1: Ideal Conditions
```
Camera FPS:      30 FPS
Processing FPS:  30 FPS
Network:         Stable (0% packet loss)

Result:
  - Frame drop: 0-1%
  - Video quality: Smooth
  - Latency: <100ms
```

### Scenario 2: Heavy Processing
```
Camera FPS:      30 FPS
Processing FPS:  25 FPS (Detection + Tracking + ReID)
Network:         Stable

Result:
  - Frame drop: ~16.7% (queue overflow)
  - Video quality: Slightly jerky
  - Latency: <200ms
```

### Scenario 3: Unstable Network
```
Camera FPS:      30 FPS
Processing FPS:  30 FPS
Network:         Unstable (5% packet loss)

Result:
  - Frame drop: 5-10% (network + cascading failures)
  - Video quality: Jerky
  - Latency: <100ms
```

### Scenario 4: Multi-Camera + Heavy Load
```
Cameras:         3 cameras × 30 FPS = 90 FPS total
Processing FPS:  60 FPS (combined)
Network:         Moderate (2% packet loss)

Result:
  - Frame drop: 30-35% (queue overflow + network)
  - Video quality: Very jerky
  - Latency: <300ms
```

---

## 🎯 Drop Frame Metrics

### Measurement Points

**1. Network Level (StreamReader):**
```python
total_read_attempts = 1000
failed_reads = 50
network_drop_rate = 50/1000 = 5%
```

**2. Processing Level:**
```python
total_frames_received = 950  # After network drops
frames_processed = 900
processing_drop_rate = 50/950 = 5.3%
```

**3. Overall:**
```python
total_input_frames = 1000
total_output_frames = 900
overall_drop_rate = 100/1000 = 10%
```

### Acceptable Thresholds

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Network drop | <1% | 1-5% | >5% |
| Processing drop | <5% | 5-15% | >15% |
| Overall drop | <5% | 5-20% | >20% |
| Output FPS | >25 | 20-25 | <20 |

---

## ⚙️ Optimization Strategies

### 1. Reduce Processing Time

**A. Lower Resolution:**
```python
# Before detection, resize frame
frame = cv2.resize(frame, (640, 480))  # From 1280x720
# → 2.25x faster processing
```

**B. Skip Frames:**
```python
# Process every 2nd frame
if frame_id % 2 == 0:
    continue
# → 2x faster, but 50% drop rate
```

**C. Optimize Detection:**
```python
# Use smaller YOLOX model
detector = YOLOXDetector(model_size='s')  # Instead of 'x'
# → 3x faster, slightly lower accuracy
```

### 2. Increase Buffer Size

```python
# In MultiStreamReader
MultiStreamReader(
    sources=urls,
    buffer_size=60  # Increase from 30 to 60
)
# → Tolerate 2x longer processing spikes
# → But uses 2x memory
```

### 3. Adjust Timeouts

```python
# In StreamReader._read_ffmpeg()
ready, _, _ = select.select([...], [], [], 5.0)  # From 3.0 to 5.0
# → More tolerant to network jitter
# → But higher latency on failures
```

### 4. Reduce Camera FPS

```python
# Configure camera to send 15 FPS instead of 30 FPS
# → 50% less data to process
# → Smoother output with same processing power
```

### 5. Hardware Upgrade

```
CPU: i5 → i7/i9 (2x faster)
GPU: GTX 1660 → RTX 3080 (4x faster)
RAM: 16GB → 32GB (larger buffers)
Network: 100Mbps → 1Gbps (less packet loss)
```

---

## 📊 Monitoring & Logging

### Key Metrics to Track

**1. Frame Drop Rate:**
```python
logger.info(f"Frame drop rate: {dropped_frames}/{total_frames} = {drop_rate:.1f}%")
```

**2. Processing FPS:**
```python
logger.info(f"Processing FPS: {avg_fps:.1f} (target: {camera_fps})")
```

**3. Queue Depth:**
```python
logger.info(f"Queue depth: {queue.qsize()}/{buffer_size}")
```

**4. Network Stats:**
```python
logger.info(f"Network drops: {network_drops} ({network_drop_rate:.1f}%)")
```

### Log Examples

**Normal Operation:**
```
[INFO] Frame 1000/3000 (33.3%) - FPS: 29.8 - Drop: 1.2%
[INFO] Queue depth: 5/30 (16.7%)
[INFO] Network drops: 12/1000 (1.2%)
```

**Heavy Load:**
```
[WARNING] Frame 1000/3000 (33.3%) - FPS: 24.5 - Drop: 18.3%
[WARNING] Queue depth: 30/30 (100%) - FULL
[WARNING] Network drops: 35/1000 (3.5%)
[WARNING] Processing slower than camera FPS
```

**Network Issues:**
```
[ERROR] Frame 1000/3000 (33.3%) - FPS: 28.1 - Drop: 25.7%
[ERROR] Consecutive failures: 15/30
[ERROR] Network drops: 257/1000 (25.7%)
[ERROR] Stream unstable - consider reconnecting
```

---

## 🎬 Real-world Examples

### Example 1: Office Monitoring (Good)
```
Setup:
  - 2 cameras, 1280x720, 25 FPS
  - RTX 3060 GPU
  - Stable LAN network
  
Results:
  - Processing FPS: 24.8
  - Drop rate: 2.1%
  - Video quality: Smooth
  - Verdict: ✅ Excellent
```

### Example 2: Warehouse (Acceptable)
```
Setup:
  - 4 cameras, 1920x1080, 30 FPS
  - GTX 1660 GPU
  - WiFi network (occasional drops)
  
Results:
  - Processing FPS: 22.3
  - Drop rate: 12.8%
  - Video quality: Slightly jerky
  - Verdict: ⚠️ Acceptable, consider optimization
```

### Example 3: Outdoor (Poor)
```
Setup:
  - 3 cameras, 1920x1080, 30 FPS
  - CPU only (no GPU)
  - 4G network (unstable)
  
Results:
  - Processing FPS: 8.5
  - Drop rate: 71.7%
  - Video quality: Very jerky, unusable
  - Verdict: ❌ Needs hardware upgrade
```

---

## 🔧 Troubleshooting Guide

### Problem: High drop rate (>20%)

**Diagnosis:**
```bash
# Check processing FPS
grep "FPS:" output.log | tail -20

# Check queue status
grep "Queue" output.log | tail -20

# Check network drops
grep "Skipped frame" output.log | wc -l
```

**Solutions:**
1. Lower camera resolution
2. Reduce number of cameras
3. Upgrade GPU
4. Increase buffer size
5. Optimize network

### Problem: Video very jerky

**Diagnosis:**
- Check if drop rate > 15%
- Check if drops are clustered (bursts)

**Solutions:**
1. Increase `max_consecutive_failures` to smooth out bursts
2. Increase buffer size to absorb spikes
3. Improve network stability

### Problem: Increasing latency

**Diagnosis:**
- Check if queue is always full
- Check if processing FPS < camera FPS

**Solutions:**
1. **Must** reduce processing time or camera FPS
2. Current settings are unsustainable
3. System will eventually crash

---

## 📝 Recommendations for Production

### Minimum Requirements
- Processing FPS ≥ 90% of camera FPS
- Drop rate < 10%
- Network packet loss < 2%
- Queue rarely full (<50% depth)

### Best Practices
1. **Monitor continuously** - Log drop rates every 100 frames
2. **Set alerts** - Alert if drop rate > 15% for 1 minute
3. **Test under load** - Simulate peak conditions
4. **Have fallback** - Reduce quality if drop rate too high
5. **Document baselines** - Know normal drop rates for your setup

### Configuration Template
```yaml
# For stable production
camera_fps: 25  # Lower than 30 for headroom
buffer_size: 60  # 2x default for spike tolerance
max_consecutive_failures: 50  # Tolerant to brief outages
timeout: 5.0  # Generous timeout for network jitter
```

---

## 📚 Summary

### Key Takeaways

1. **Frame drop is intentional and necessary** for real-time processing
2. **3 drop points**: Network, Processing Loop, Queue Overflow
3. **Trade-off**: Real-time > Completeness
4. **Acceptable drop rate**: <10% for production
5. **Main causes**: Slow processing, network issues, queue overflow

### Decision Matrix

| Scenario | Action |
|----------|--------|
| Drop < 5% | ✅ No action needed |
| Drop 5-10% | ⚠️ Monitor closely |
| Drop 10-20% | ⚠️ Optimize or reduce load |
| Drop > 20% | ❌ Must fix - unsustainable |

### Contact & Support

For questions about frame drop behavior:
- Check logs: `output/*.log`
- Monitor metrics: FPS, drop rate, queue depth
- Adjust configuration based on your hardware
- Consider hardware upgrade if consistently high drop rate

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-10  
**Author:** Person ReID System Team

