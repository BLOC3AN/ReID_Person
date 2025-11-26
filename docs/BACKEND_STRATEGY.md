# Detection Backend Strategy Guide

## 📋 Overview

Person ReID System hỗ trợ 3 detection backends với các đặc điểm và use cases khác nhau:

1. **PyTorch** - Standard inference với PyTorch
2. **TensorRT** - Optimized GPU inference với NVIDIA TensorRT
3. **Triton Inference Server** - Multi-stream optimization với dynamic batching

Document này giúp bạn lựa chọn backend phù hợp cho use case của mình.

---

## 🎯 Backend Comparison

### Performance Comparison

| Backend | Single Stream | Multi-Stream (4 cams) | GPU Memory | Setup Complexity |
|---------|---------------|----------------------|------------|------------------|
| **PyTorch** | 21-22 FPS | 21-22 FPS total | ~2GB | ⭐ Easy |
| **TensorRT** | 28-30 FPS | 28-30 FPS total | ~2GB | ⭐⭐ Medium |
| **Triton** | 26-28 FPS | **66+ FPS total** | ~3-4GB | ⭐⭐⭐ Advanced |

### Latency Comparison

| Backend | Avg Inference Time | P95 Latency | P99 Latency |
|---------|-------------------|-------------|-------------|
| **PyTorch FP16** | 45.63ms | ~50ms | ~55ms |
| **TensorRT FP16** | 35.58ms | ~40ms | ~45ms |
| **Triton + TensorRT** | **22.89ms** | **~25ms** | **~30ms** |

**Key Insight**: Triton giảm latency **50%** so với PyTorch và **36%** so với TensorRT standalone.

---

## 🔍 Detailed Backend Analysis

### 1. PyTorch Backend

**Architecture:**
```
Frame → Preprocess → PyTorch Model (GPU) → Postprocess → Detections
```

**Pros:**
- ✅ Dễ setup, không cần convert model
- ✅ Debugging dễ dàng với Python
- ✅ Flexible, dễ modify model
- ✅ Hỗ trợ cả CPU và GPU

**Cons:**
- ❌ Chậm nhất trong 3 backends
- ❌ Không tối ưu cho production
- ❌ Không hỗ trợ dynamic batching
- ❌ Python GIL overhead

**Use Cases:**
- Development và debugging
- Single camera với FPS thấp (< 15 FPS)
- Prototype và testing
- Không có GPU hoặc GPU yếu

**Configuration:**
```yaml
# configs/config.yaml
detection:
  backend: pytorch
  model_path: models/bytetrack_x_mot17.pth.tar
  conf_threshold: 0.01
  nms_threshold: 0.7
  test_size: [640, 640]
  fp16: true  # Enable FP16 for faster inference
```

**Code Example:**
```python
from core import YOLOXDetector

detector = YOLOXDetector(
    model_path="models/bytetrack_x_mot17.pth.tar",
    model_type="mot17",
    conf_thresh=0.01,
    nms_thresh=0.7,
    test_size=(640, 640),
    fp16=True
)

detections = detector.detect(frame)  # Returns (N, 7) array
```

---

### 2. TensorRT Backend

**Architecture:**
```
Frame → Preprocess → TensorRT Engine (GPU) → Postprocess → Detections
```

**Pros:**
- ✅ **1.3-1.5x faster** than PyTorch
- ✅ Optimized CUDA kernels
- ✅ FP16 precision với minimal accuracy loss
- ✅ Low latency (~35ms)
- ✅ Không cần Docker

**Cons:**
- ❌ Cần convert ONNX → TensorRT
- ❌ Engine specific to GPU architecture
- ❌ Không hỗ trợ dynamic batching
- ❌ Cần rebuild engine khi đổi GPU

**Use Cases:**
- Single camera với high FPS (25-30 FPS)
- 2-3 cameras sequential processing
- Production deployment trên single GPU
- Khi cần low latency nhưng không cần Triton complexity

**Setup Steps:**

1. **Export ONNX:**
```bash
python tools/export_onnx.py \
    --model models/bytetrack_x_mot17.pth.tar \
    --output models/bytetrack_x_mot17_fp16.onnx \
    --opset 11
```

2. **Convert to TensorRT:**
```bash
python tools/convert_tensorrt.py \
    --onnx models/bytetrack_x_mot17_fp16.onnx \
    --output models/bytetrack_x_mot17_fp16.trt \
    --fp16 \
    --workspace 4096
```

3. **Configure:**
```yaml
# configs/config.yaml
detection:
  backend: tensorrt
  tensorrt:
    engine_path: models/bytetrack_x_mot17_fp16.trt
    fp16: true
```

4. **Code Example:**
```python
from core import TensorRTDetector

detector = TensorRTDetector(
    engine_path="models/bytetrack_x_mot17_fp16.trt",
    conf_thresh=0.01,
    nms_thresh=0.7,
    test_size=(640, 640)
)

detections = detector.detect(frame)  # Returns (N, 7) array
```

---

### 3. Triton Inference Server Backend

**Architecture:**
```
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Camera 1│  │ Camera 2│  │ Camera N│
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     └────────────┼────────────┘
                  │ gRPC (concurrent)
          ┌───────▼────────┐
          │ Triton Server  │
          │ Dynamic Batch  │ ← Queues requests
          │ (max 500μs)    │
          └───────┬────────┘
                  │
          ┌───────▼────────┐
          │ TensorRT Engine│
          │ CUDA Graphs    │ ← Optimized execution
          └────────────────┘
```

**Pros:**
- ✅ **2-3x faster** than PyTorch for multi-stream
- ✅ **Dynamic batching** - tự động gộp requests
- ✅ **Concurrent execution** - xử lý đồng thời nhiều streams
- ✅ **CUDA Graphs** - giảm kernel launch overhead
- ✅ **Model versioning** - hot-reload models
- ✅ **Metrics & monitoring** - Prometheus metrics
- ✅ **Production-ready** - battle-tested by NVIDIA

**Cons:**
- ❌ Complex setup (Docker, model repository)
- ❌ Cần hiểu Triton configuration
- ❌ Overhead cho single stream
- ❌ Cần thêm ~1-2GB GPU memory

**Use Cases:**
- **4+ cameras** concurrent processing
- High throughput requirements (> 60 FPS total)
- Production deployment với multiple streams
- Khi cần monitoring và metrics
- Microservices architecture

**Setup Steps:**

See [TRITON_DEPLOYMENT.md](../deployment/TRITON_DEPLOYMENT.md) for detailed setup.

**Quick Start:**
```bash
# 1. Setup model repository
cd deployment
bash setup_triton.sh

# 2. Start Triton server
sudo docker compose up -d triton

# 3. Verify
curl http://localhost:8100/v2/health/ready

# 4. Configure
# Edit configs/config.yaml:
detection:
  backend: triton
  triton:
    url: localhost:8101  # gRPC endpoint
    model_name: bytetrack_tensorrt
```

**Code Example:**
```python
from core import TritonDetector

detector = TritonDetector(
    triton_url="localhost:8101",
    model_name="bytetrack_tensorrt",
    conf_thresh=0.01,
    nms_thresh=0.7,
    test_size=(640, 640),
    timeout=10.0
)

detections = detector.detect(frame)  # Returns (N, 7) array
```

---

## 🎯 Decision Tree

```
Start
  │
  ├─ Single camera?
  │   ├─ Yes → FPS < 20?
  │   │   ├─ Yes → PyTorch ✅
  │   │   └─ No → TensorRT ✅
  │   │
  │   └─ No → Multiple cameras?
  │       ├─ 2-3 cameras → TensorRT ✅
  │       └─ 4+ cameras → Triton ✅
  │
  ├─ Need low latency (< 30ms)?
  │   └─ Yes → Triton ✅
  │
  ├─ Need monitoring/metrics?
  │   └─ Yes → Triton ✅
  │
  └─ Development/debugging?
      └─ Yes → PyTorch ✅
```

---

## 📊 Benchmark Results

### Test Setup
- **GPU**: Tesla V100-SXM2-16GB
- **Input**: 640x640 RGB images
- **Model**: ByteTrack YOLOX-X (FP16)
- **Batch Size**: 1 (single frame)
- **Iterations**: 100 warmup + 1000 test

### Single Stream Results

| Backend | Avg Time | Min Time | Max Time | Std Dev | FPS |
|---------|----------|----------|----------|---------|-----|
| PyTorch FP16 | 45.63ms | 42.1ms | 52.3ms | 2.1ms | 21.91 |
| TensorRT FP16 | 35.58ms | 33.2ms | 39.4ms | 1.5ms | 28.11 |
| Triton + TRT | **22.89ms** | **21.1ms** | **26.7ms** | **1.2ms** | **43.69** |

### Multi-Stream Results (4 cameras concurrent)

| Backend | Total Throughput | Avg Latency per Request |
|---------|------------------|------------------------|
| PyTorch (sequential) | 21.91 FPS | 182ms |
| TensorRT (sequential) | 28.11 FPS | 142ms |
| **Triton (batched)** | **66+ FPS** | **~60ms** |

**Speedup**: Triton is **3x faster** than PyTorch for multi-camera scenarios.

---

## 🔧 Configuration Best Practices

### PyTorch Configuration
```yaml
detection:
  backend: pytorch
  model_path: models/bytetrack_x_mot17.pth.tar
  model_type: mot17
  conf_threshold: 0.01
  nms_threshold: 0.7
  test_size: [640, 640]
  fp16: true  # Always enable for 2x speedup
  fuse: true  # Fuse Conv+BN layers
```

### TensorRT Configuration
```yaml
detection:
  backend: tensorrt
  tensorrt:
    engine_path: models/bytetrack_x_mot17_fp16.trt
    fp16: true
  conf_threshold: 0.01
  nms_threshold: 0.7
  test_size: [640, 640]
```

### Triton Configuration
```yaml
detection:
  backend: triton
  triton:
    url: localhost:8101  # gRPC endpoint
    model_name: bytetrack_tensorrt
    model_version: ''  # Empty = latest
    timeout: 10.0  # Request timeout (seconds)
    verbose: false
    
    # Dynamic batching (configured in model config.pbtxt)
    max_batch_size: 8
    max_queue_delay_ms: 0.5  # 500μs for low latency
    preferred_batch_sizes: [1, 2, 4, 8]
```

---

## 🚀 Migration Guide

### From PyTorch → TensorRT

1. Export ONNX model
2. Convert to TensorRT engine
3. Update config.yaml
4. Test with single video

**Estimated time**: 15-30 minutes

### From TensorRT → Triton

1. Setup Triton model repository
2. Copy TensorRT engine to repository
3. Create config.pbtxt
4. Start Triton Docker container
5. Update config.yaml
6. Test with multiple streams

**Estimated time**: 1-2 hours

### From PyTorch → Triton

Combine both migrations above.

**Estimated time**: 2-3 hours

---

## 📝 Summary

| Scenario | Recommended Backend | Reason |
|----------|-------------------|--------|
| Development | PyTorch | Easy debugging |
| Single camera (< 20 FPS) | PyTorch | Simple setup |
| Single camera (20-30 FPS) | TensorRT | Best single-stream performance |
| 2-3 cameras | TensorRT | Good balance |
| **4+ cameras** | **Triton** | **Dynamic batching advantage** |
| Low latency required | Triton | CUDA Graphs optimization |
| Production deployment | Triton | Monitoring, versioning, scalability |

**General Rule**: 
- Use **PyTorch** for development
- Use **TensorRT** for single-stream production
- Use **Triton** for multi-stream production

---

## 🔗 Related Documents

- [Triton Deployment Guide](../deployment/TRITON_DEPLOYMENT.md)
- [Stream Processing Strategy](STREAM_STRATEGY.md)
- [Configuration Guide](CONFIGURATION.md)
- [Troubleshooting](TROUBLESHOOTING.md)

---

**Last Updated**: 2025-11-11

