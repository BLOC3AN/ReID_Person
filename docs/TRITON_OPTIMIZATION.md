# Triton Inference Server Optimization Guide

## 📊 Overview

This guide explains how to optimize Triton Inference Server for maximum throughput and resource utilization when processing multiple camera streams.

---

## 🎯 Current Configuration (Optimized for 12-16+ Streams)

### GPU Resource Usage
- **Current**: ~1.9GB / 16GB (12% utilization)
- **Optimized**: ~8-10GB / 16GB (50-60% utilization)
- **Headroom**: ~6GB for other services (ArcFace, tracking, etc.)

### Performance Targets
| Configuration | Instances | GPU Memory | Streams | Total FPS | Latency |
|---------------|-----------|------------|---------|-----------|---------|
| **Low Latency** | 2 | ~1.5GB | 2-4 | 60-80 | 25-35ms |
| **Balanced** | 4 | ~2.5GB | 4-8 | 120-160 | 35-50ms |
| **High Throughput** | 8 | ~5GB | 8-12 | 200-280 | 50-80ms |
| **Maximum** | 16 | ~8-10GB | 12-16+ | 320-400+ | 70-120ms |

---

## 🔧 Configuration Files

### 1. Triton Model Config (`triton_model_repository/bytetrack_tensorrt/config.pbtxt`)

#### Key Parameters

**A. Instance Count** (Line 53)
```protobuf
instance_group [
  {
    count: 16  # Number of parallel instances
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

**Tuning Guide:**
- **2 instances**: 2-4 cameras, low latency priority
- **4 instances**: 4-8 cameras, balanced (recommended starting point)
- **8 instances**: 8-12 cameras, high throughput
- **16 instances**: 12-16+ cameras, maximum throughput

**Memory per instance**: ~500-600MB

---

**B. Queue Configuration** (Lines 37-42)
```protobuf
default_queue_policy {
  timeout_action: REJECT
  default_timeout_microseconds: 20000  # 20ms
  allow_timeout_override: true
  max_queue_size: 256  # Maximum pending requests
}
```

**Tuning Guide:**
- `max_queue_size`: Number of requests that can wait in queue
  - 32: Low latency (2-4 cameras)
  - 128: Balanced (4-8 cameras)
  - 256: High throughput (8-16 cameras)
  - 512: Maximum (16+ cameras)

- `default_timeout_microseconds`: How long requests wait before rejection
  - 10000 (10ms): Low latency
  - 20000 (20ms): Balanced (current)
  - 30000 (30ms): High throughput
  - 50000 (50ms): Maximum

---

**C. Dynamic Batching** (Lines 28-32)
```protobuf
dynamic_batching {
  preferred_batch_size: [ 1 ]
  max_queue_delay_microseconds: 5000  # 5ms
  preserve_ordering: false
}
```

**Tuning Guide:**
- `max_queue_delay_microseconds`: How long to wait for batching
  - 500-1000 (0.5-1ms): Low latency
  - 5000 (5ms): Balanced (current)
  - 10000 (10ms): High throughput
  - 20000 (20ms): Maximum

- `preserve_ordering`: 
  - `true`: Maintain request order (adds overhead)
  - `false`: Better throughput (current)

---

### 2. Docker Compose (`deployment/docker-compose.yml`)

**Shared Memory** (Line 29)
```yaml
triton:
  shm_size: '8gb'  # Shared memory for IPC
```

**Tuning Guide:**
- 4GB: 1-4 instances
- 8GB: 4-16 instances (current)
- 12GB: 16+ instances (if needed)

---

### 3. Application Config (`configs/config.yaml`)

**Timeout** (Line 20)
```yaml
triton:
  timeout: 20.0  # Request timeout in seconds
```

**Tuning Guide:**
- 10s: Low latency, small queue
- 20s: Balanced (current)
- 30s: High throughput, large queue

---

## 📈 Performance Optimization Strategies

### Strategy 1: Maximize Throughput (Current Configuration)

**Goal**: Process 12-16+ camera streams simultaneously

**Configuration:**
```protobuf
# config.pbtxt
instance_group [{ count: 16 }]
max_queue_size: 256
max_queue_delay_microseconds: 5000
preserve_ordering: false
```

**Expected Results:**
- Total FPS: 320-400+
- Per-camera FPS: 20-30
- Latency: 70-120ms
- GPU Memory: ~8-10GB

**Best for:**
- Large deployments (12+ cameras)
- Throughput > latency priority
- Batch processing scenarios

---

### Strategy 2: Balanced Performance

**Goal**: Balance latency and throughput for 4-8 cameras

**Configuration:**
```protobuf
# config.pbtxt
instance_group [{ count: 4 }]
max_queue_size: 128
max_queue_delay_microseconds: 5000
preserve_ordering: false
```

**Expected Results:**
- Total FPS: 120-160
- Per-camera FPS: 25-35
- Latency: 35-50ms
- GPU Memory: ~2.5GB

**Best for:**
- Medium deployments (4-8 cameras)
- Balanced latency/throughput
- General purpose

---

### Strategy 3: Low Latency

**Goal**: Minimize latency for 2-4 cameras

**Configuration:**
```protobuf
# config.pbtxt
instance_group [{ count: 2 }]
max_queue_size: 32
max_queue_delay_microseconds: 1000
preserve_ordering: true
```

**Expected Results:**
- Total FPS: 60-80
- Per-camera FPS: 25-35
- Latency: 25-35ms
- GPU Memory: ~1.5GB

**Best for:**
- Small deployments (2-4 cameras)
- Real-time applications
- Latency-critical scenarios

---

## 🔍 Monitoring & Tuning

### Check GPU Usage
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Expected output:
# GPU Memory: 8-10GB / 16GB (50-60%)
# GPU Utilization: 70-90%
```

### Check Triton Metrics
```bash
# Queue statistics
curl -s http://localhost:8102/metrics | grep "nv_inference_queue"

# Instance utilization
curl -s http://localhost:8102/metrics | grep "nv_inference_exec_count"

# Request statistics
curl -s http://localhost:8102/metrics | grep "nv_inference_request"
```

### Key Metrics to Monitor

**1. Queue Duration**
```bash
curl -s http://localhost:8102/metrics | grep "nv_inference_queue_duration_us"
```
- **Good**: < 10ms average
- **Acceptable**: 10-30ms
- **Bad**: > 30ms (increase instances or queue size)

**2. Request Success Rate**
```bash
curl -s http://localhost:8102/metrics | grep "nv_inference_request_success"
```
- **Good**: > 99%
- **Acceptable**: 95-99%
- **Bad**: < 95% (increase timeout or queue size)

**3. GPU Memory Usage**
```bash
curl -s http://localhost:8102/metrics | grep "nv_gpu_memory_used_bytes"
```
- **Good**: 50-70% of total
- **Acceptable**: 30-50% or 70-85%
- **Bad**: > 90% (reduce instances) or < 20% (increase instances)

---

## 🚨 Troubleshooting

### Issue 1: High Latency (> 100ms)

**Symptoms:**
- Slow frame processing
- Queue duration > 30ms

**Solutions:**
1. Reduce `max_queue_delay_microseconds` to 1000-2000
2. Increase instance count
3. Reduce number of concurrent streams

---

### Issue 2: Requests Timing Out

**Symptoms:**
- Error: "Request timeout"
- Low success rate (< 95%)

**Solutions:**
1. Increase `default_timeout_microseconds` to 30000-50000
2. Increase `max_queue_size` to 512
3. Increase instance count

---

### Issue 3: GPU Out of Memory

**Symptoms:**
- Error: "CUDA out of memory"
- Triton crashes

**Solutions:**
1. Reduce instance count (16 → 8 → 4)
2. Reduce `max_queue_size`
3. Check other GPU processes: `nvidia-smi`

---

### Issue 4: Low GPU Utilization (< 30%)

**Symptoms:**
- GPU memory < 30%
- Underutilized resources

**Solutions:**
1. Increase instance count (4 → 8 → 16)
2. Increase number of concurrent streams
3. Reduce `max_queue_delay_microseconds` for faster processing

---

## 📋 Quick Reference

### Restart Triton After Config Changes
```bash
# Docker Compose
cd deployment
docker-compose restart triton

# Standalone Docker
docker restart triton_inference_server

# Wait for ready
curl http://localhost:8100/v2/health/ready
```

### Test Configuration
```bash
# Single stream test
python test_triton_detection.py

# Multi-stream test (if available)
python test_multi_stream.py --num_cameras 8

# Benchmark
python benchmark_triton.py --streams 16 --duration 60
```

---

## 🎯 Recommended Configurations by Use Case

### Small Deployment (2-4 Cameras)
```protobuf
instance_group [{ count: 2 }]
max_queue_size: 64
max_queue_delay_microseconds: 1000
default_timeout_microseconds: 10000
```
- GPU Memory: ~1.5GB
- Total FPS: 60-80
- Latency: 25-35ms

### Medium Deployment (4-8 Cameras)
```protobuf
instance_group [{ count: 4 }]
max_queue_size: 128
max_queue_delay_microseconds: 5000
default_timeout_microseconds: 20000
```
- GPU Memory: ~2.5GB
- Total FPS: 120-160
- Latency: 35-50ms

### Large Deployment (8-12 Cameras)
```protobuf
instance_group [{ count: 8 }]
max_queue_size: 256
max_queue_delay_microseconds: 10000
default_timeout_microseconds: 30000
```
- GPU Memory: ~5GB
- Total FPS: 200-280
- Latency: 50-80ms

### Maximum Deployment (12-16+ Cameras) - **CURRENT**
```protobuf
instance_group [{ count: 16 }]
max_queue_size: 256
max_queue_delay_microseconds: 5000
default_timeout_microseconds: 20000
```
- GPU Memory: ~8-10GB
- Total FPS: 320-400+
- Latency: 70-120ms

---

## 📚 Additional Resources

- [Triton Model Configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
- [Dynamic Batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#dynamic-batcher)
- [Instance Groups](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#instance-groups)
- [Performance Tuning](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html)

---

**Last Updated**: 2025-11-11
**Configuration**: 16 instances, 256 queue size, 5ms delay
**Target**: 12-16+ camera streams, 320-400+ FPS total

