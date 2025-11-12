# 📊 PERSON REID SYSTEM - PERFORMANCE BENCHMARK REPORT

**Date**: 2025-11-12  
**System**: 4x Tesla V100-SXM2-16GB (64GB total VRAM), 80-core Xeon Gold 6148, 251GB RAM

---

## 🎯 EXECUTIVE SUMMARY

**Current Performance**: **KHÔNG ĐẠT REAL-TIME** với 2 camera streams  
**Bottleneck chính**: **Detection (ByteTrack) - chiếm 88.3% thời gian xử lý**  
**Frame drop rate**: **80-92%** - Rất cao, không chấp nhận được cho production

---

## 📈 TEST 1: SINGLE VIDEO FILE (cam1.mkv)

### Video Info
- **Resolution**: 1280x720
- **FPS**: 25.0
- **Format**: H.264 (MKV container)
- **Size**: 7.7MB

### Performance Results
```
⏱️  PROCESSING PERFORMANCE:
  Total frames:        250
  Elapsed time:        23.51s
  Average FPS:         10.64
  Video FPS:           25.00
  Real-time factor:    0.43x  ⚠️ (Cần 2.35x speedup)
  Dropped frames:      202 (80.8%)
```

### Component Breakdown
```
🔍 COMPONENT BREAKDOWN (ms):
  Detection:    avg= 57.96  max=120.92  min= 40.15
  Tracking:     avg=  2.44  max=  5.30  min=  1.32
  ReID:         avg=  2.51  max= 72.07  min=  0.01
  Total/frame:  avg= 65.65  max=139.69  min= 42.91

📈 TIME DISTRIBUTION:
  Detection:     88.3%  ⚠️ BOTTLENECK
  Tracking:       3.7%
  ReID:           3.8%
  Other:          4.2%
```

### System Resources
```
💻 SYSTEM RESOURCES:
  GPU Util:     avg= 11.3%  max= 29.0%  ⚠️ Underutilized
  GPU Memory:   avg= 47.6%  max= 47.7%
  CPU Util:     avg= 17.8%  max= 26.8%
  RAM Util:     avg= 36.0%  max= 36.1%
```

---

## 📈 TEST 2: DUAL CAMERA STREAMS (UDP)

### Stream Info
- **Streams**: 2x UDP streams (ports 1905, 1906)
- **Source**: Same cam1.mkv video (looped)
- **Protocol**: UDP/MPEGTS
- **Duration**: 30 seconds

### Performance Results
```
⏱️  OVERALL PERFORMANCE:
  Duration:            32.14s
  Total frames:        299
  Combined FPS:        9.30  ⚠️ Rất thấp
  Dropped frames:      276 (92.3%)  ⚠️ Cực kỳ cao
```

### Per-Stream Breakdown
```
📹 STREAM 0:
  Frames:          81
  FPS:             2.44  ⚠️ Chỉ 10% của video FPS
  Dropped:         60 (74.1%)
  Detection:       avg= 53.96ms  max= 95.26ms
  Tracking:        avg=  2.13ms  max=  6.82ms
  ReID:            avg= 49.64ms  max=3788.90ms  ⚠️ Spike lớn

📹 STREAM 1:
  Frames:          218
  FPS:             6.58  ⚠️ Chỉ 26% của video FPS
  Dropped:         216 (99.1%)  ⚠️ Gần như drop toàn bộ
  Detection:       avg= 72.24ms  max=185.04ms  ⚠️ Chậm hơn stream 0
  Tracking:        avg=  2.22ms  max=  5.16ms
  ReID:            avg=  2.81ms  max= 79.26ms
```

### System Resources
```
💻 SYSTEM RESOURCES:
  GPU Util:        avg=  9.6%  max= 48.0%  ⚠️ Rất thấp
  GPU Memory:      avg= 47.7%  max= 48.0%
  CPU Util:        avg= 18.0%  max= 30.6%
  RAM Util:        avg= 35.9%  max= 36.0%
```

---

## 🔍 ROOT CAUSE ANALYSIS

### 1. **Detection Bottleneck (88.3% thời gian)**
- **Triton ByteTrack TensorRT**: 54-72ms/frame (avg ~60ms)
- **Expected**: ~10-15ms/frame cho TensorRT FP16
- **Vấn đề**: 
  - ❌ Triton chạy **SEQUENTIAL** (không batch)
  - ❌ Dynamic batching **DISABLED** trong config
  - ❌ Mỗi stream gọi Triton riêng lẻ → không tận dụng batching
  - ❌ GPU utilization chỉ 9-11% → GPU idle phần lớn thời gian

### 2. **Multi-Stream Contention**
- 2 streams cùng gọi Triton → **serialize** requests
- Stream 1 chậm hơn Stream 0 (72ms vs 54ms) → chờ đợi lẫn nhau
- Không có queue management → frame drops

### 3. **ReID Spikes**
- Stream 0 có spike lên **3788ms** (3.8 giây!) cho 1 frame
- Nguyên nhân: InsightFace face detection thất bại nhiều lần
- Không có timeout mechanism

### 4. **GPU Underutilization**
- GPU 0: 9-11% utilization (should be 60-80%)
- GPU 1-3: **IDLE** (0% usage)
- Không tận dụng multi-GPU

---

## 💡 RECOMMENDED SOLUTIONS

### ✅ SOLUTION 1: Enable Dynamic Batching (Quick Win)
**Impact**: 2-3x speedup  
**Effort**: 5 minutes

**Action**:
```protobuf
# triton_model_repository/bytetrack_tensorrt/config.pbtxt
dynamic_batching {
  preferred_batch_size: [ 2, 4 ]
  max_queue_delay_microseconds: 5000
}
```

**Expected Result**:
- 2 streams → batch size 2 → ~30ms/batch → 15ms/frame
- FPS: 10.64 → **~20-25 FPS** (real-time cho 1 stream)

---

### ✅ SOLUTION 2: Add ArcFace to Triton on GPU 1 (Recommended)
**Impact**: 10-16x speedup cho ReID, giảm spikes  
**Effort**: 1 hour

**Benefits**:
- Loại bỏ ReID spikes (3788ms → <10ms)
- Batch processing cho faces
- Tách workload: GPU 0 (Detection) + GPU 1 (ReID)
- Support 8-16+ cameras

**Steps**:
1. Convert ArcFace ONNX → TensorRT
2. Setup Triton model repository
3. Create `core/feature_extractor_triton.py`
4. Update config

---

### ✅ SOLUTION 3: Optimize Triton Config (Medium Win)
**Impact**: 1.5-2x speedup  
**Effort**: 15 minutes

**Actions**:
```protobuf
# Increase instances
instance_group [
  {
    count: 8  # Reduce from 16 to 8 (enough for 2-4 streams)
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

# Enable CUDA graphs
optimization {
  cuda {
    graphs: true
  }
}
```

---

### ✅ SOLUTION 4: Implement Frame Skipping Strategy
**Impact**: Maintain real-time at cost of some frames  
**Effort**: 30 minutes

**Logic**:
- If processing time > frame interval → skip next frame
- Adaptive skip rate based on queue depth
- Prioritize tracking continuity over all frames

---

## 📊 PROJECTED PERFORMANCE (After Optimizations)

### Scenario 1: Dynamic Batching Only
```
Single stream:  10.64 → 20-25 FPS  ✅ Real-time
Dual streams:   9.30  → 15-18 FPS  ⚠️ Marginal
```

### Scenario 2: Dynamic Batching + ArcFace Triton
```
Single stream:  10.64 → 25-30 FPS  ✅ Real-time+
Dual streams:   9.30  → 20-25 FPS  ✅ Real-time
4 streams:      N/A   → 12-15 FPS  ⚠️ Marginal
8 streams:      N/A   → 8-10 FPS   ❌ Below real-time
```

### Scenario 3: Full Optimization (All solutions)
```
Single stream:  10.64 → 30-35 FPS  ✅ Real-time++
Dual streams:   9.30  → 25-30 FPS  ✅ Real-time+
4 streams:      N/A   → 15-20 FPS  ✅ Real-time
8 streams:      N/A   → 10-12 FPS  ⚠️ Marginal
16 streams:     N/A   → 6-8 FPS    ❌ Below real-time
```

---

## 🎯 NEXT STEPS

### Immediate (Today)
1. ✅ **Enable dynamic batching** trong Triton config
2. ✅ **Test lại** với 2 streams
3. ✅ **Measure improvement**

### Short-term (This Week)
1. ✅ **Convert ArcFace to TensorRT**
2. ✅ **Add ArcFace to Triton** on GPU 1
3. ✅ **Integrate** vào pipeline
4. ✅ **Benchmark** với 4-8 streams

### Medium-term (Next Week)
1. ⏸️ Implement frame skipping strategy
2. ⏸️ Add monitoring dashboard
3. ⏸️ Optimize preprocessing pipeline
4. ⏸️ Consider model quantization (INT8)

---

## 📝 CONCLUSION

**Current State**: System **KHÔNG ĐẠT** real-time performance cho 2 cameras  
**Root Cause**: Detection bottleneck (88.3%) do không dùng batching  
**Quick Fix**: Enable dynamic batching → 2-3x speedup  
**Long-term**: Add ArcFace to Triton → support 8-16 cameras  

**Recommendation**: Triển khai Solution 1 (dynamic batching) ngay lập tức, sau đó Solution 2 (ArcFace Triton) trong tuần này.

