# TensorRT Integration - Dependencies & Requirements

## 📋 OVERVIEW

Tài liệu này ghi lại toàn bộ dependencies và requirements để chạy hệ thống Person ReID với TensorRT optimization.

---

## 🖥️ SYSTEM REQUIREMENTS

### Hardware
- **GPU**: NVIDIA Tesla V100 (SM 70 - Volta architecture)
- **CUDA Compute Capability**: 7.0
- **GPU Memory**: 16GB+ recommended
- **System RAM**: 32GB+ recommended

### Software
- **OS**: Ubuntu 20.04/22.04
- **CUDA**: 12.8 (system-wide installation)
- **Python**: 3.10.x (REQUIRED - TensorRT 8.6.1 không hỗ trợ Python 3.12)

---

## 🐍 PYTHON ENVIRONMENT

### Virtual Environment Setup
```bash
# Tạo virtual environment với Python 3.10
python3.10 -m venv ~/data/hai_venv_py310

# Activate
source ~/data/hai_venv_py310/bin/activate

# Verify Python version
python --version  # Phải là Python 3.10.x
```

---

## 📦 CORE DEPENDENCIES

### 1. PyTorch Stack (CRITICAL VERSION REQUIREMENTS)

**⚠️ QUAN TRỌNG: Phải dùng PyTorch 2.0.1 để tương thích với CUDNN 8.x**

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

**Lý do:**
- PyTorch 2.5.x yêu cầu CUDNN 9.x
- TensorRT 8.6.1 yêu cầu CUDNN 8.x
- PyTorch 2.0.1 tương thích với CUDNN 8.x

**Dependencies của PyTorch 2.0.1:**
- `triton==2.0.0` (auto-installed)
- `sympy==1.13.1`
- `networkx`
- `jinja2`
- `filelock`
- `typing-extensions`

### 2. TensorRT (CRITICAL VERSION)

**⚠️ QUAN TRỌNG: Phải dùng TensorRT 8.6.1 cho GPU SM 70 (V100)**

```bash
pip install tensorrt==8.6.1.post1 --extra-index-url https://pypi.nvidia.com
```

**Lý do:**
- TensorRT 10.x chỉ hỗ trợ SM 75+ (Turing và mới hơn)
- TensorRT 8.6.1 hỗ trợ SM 70 (Volta/V100)
- TensorRT 8.6.1 chỉ có bindings cho Python 3.10, không có cho Python 3.12

**Dependencies của TensorRT 8.6.1:**
- `tensorrt-libs==8.6.1.post1` (auto-installed)
- `tensorrt-bindings==8.6.1` (auto-installed)

### 3. CUDNN (CRITICAL VERSION)

**⚠️ QUAN TRỌNG: Phải dùng CUDNN 8.9.6.50**

```bash
pip install nvidia-cudnn-cu12==8.9.6.50 --no-deps
```

**Lý do:**
- TensorRT 8.6.1 yêu cầu CUDNN 8.x
- PyTorch 2.0.1 tương thích với CUDNN 8.x
- CUDNN 9.x không tương thích với TensorRT 8.6.1

### 4. PyCUDA

```bash
pip install pycuda
```

**Dependencies:**
- Cần CUDA toolkit đã cài đặt trên system
- Sử dụng cho GPU memory management trong TensorRT

### 5. NumPy (CRITICAL VERSION)

**⚠️ QUAN TRỌNG: Phải dùng NumPy < 2.0**

```bash
pip install "numpy<2"
```

**Lý do:**
- PyTorch 2.0.1 không tương thích với NumPy 2.x
- NumPy 2.x có breaking changes với compiled modules
- Recommended: `numpy==1.26.4`

### 6. ONNX Stack

```bash
pip install onnx onnxruntime-gpu
```

**Versions:**
- `onnx>=1.14.0` (latest compatible)
- `onnxruntime-gpu>=1.15.0` (for ONNX verification)

---

## 📦 PROJECT DEPENDENCIES

### Computer Vision & ML
```bash
pip install opencv-python scipy pillow
```

### YOLOX Dependencies
```bash
pip install thop tabulate tqdm pycocotools
```

### Tracking & ReID
```bash
pip install lap cython_bbox
```

### Utilities
```bash
pip install loguru pyyaml python-dotenv
```

### Vector Database (Optional - for ReID)
```bash
pip install qdrant-client
```

---

## 🔧 COMPLETE INSTALLATION SCRIPT

```bash
#!/bin/bash

# 1. Create Python 3.10 virtual environment
python3.10 -m venv ~/data/hai_venv_py310
source ~/data/hai_venv_py310/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch 2.0.1 (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 4. Install NumPy < 2.0
pip install "numpy<2"

# 5. Install TensorRT 8.6.1
pip install tensorrt==8.6.1.post1 --extra-index-url https://pypi.nvidia.com

# 6. Install CUDNN 8.9.6.50
pip install nvidia-cudnn-cu12==8.9.6.50 --no-deps

# 7. Install PyCUDA
pip install pycuda

# 8. Install ONNX stack
pip install onnx onnxruntime-gpu

# 9. Install CV & ML libraries
pip install opencv-python scipy pillow

# 10. Install YOLOX dependencies
pip install thop tabulate tqdm pycocotools

# 11. Install tracking dependencies
pip install lap cython_bbox

# 12. Install utilities
pip install loguru pyyaml python-dotenv

# 13. Install vector database (optional)
pip install qdrant-client

echo "✅ Installation completed!"
```

---

## ⚠️ CRITICAL VERSION CONSTRAINTS

### Version Matrix

| Package | Version | Reason |
|---------|---------|--------|
| Python | **3.10.x** | TensorRT 8.6.1 bindings chỉ có cho Python 3.10 |
| PyTorch | **2.0.1+cu118** | Tương thích CUDNN 8.x |
| TensorRT | **8.6.1.post1** | Hỗ trợ GPU SM 70 (V100) |
| CUDNN | **8.9.6.50** | Yêu cầu của TensorRT 8.6.1 |
| NumPy | **< 2.0** | PyTorch 2.0.1 không tương thích NumPy 2.x |
| CUDA | **12.8** | System-wide (compatible với cu118 wheels) |

### Dependency Conflicts to Avoid

❌ **KHÔNG CÀI:**
- PyTorch 2.5.x (yêu cầu CUDNN 9.x)
- TensorRT 10.x (không hỗ trợ SM 70)
- NumPy 2.x (không tương thích PyTorch 2.0.1)
- Python 3.12 (không có TensorRT 8.6.1 bindings)
- CUDNN 9.x (không tương thích TensorRT 8.6.1)

---

## 🧪 VERIFICATION

### 1. Verify Python Version
```bash
python --version
# Expected: Python 3.10.x
```

### 2. Verify PyTorch
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
# Expected: 2.0.1+cu118, True, 11.8
```

### 3. Verify TensorRT
```python
import tensorrt as trt
print(f"TensorRT: {trt.__version__}")
# Expected: 8.6.1
```

### 4. Verify CUDNN
```bash
python -c "import torch; print(torch.backends.cudnn.version())"
# Expected: 8906 (CUDNN 8.9.6)
```

### 5. Verify PyCUDA
```python
import pycuda.autoinit
import pycuda.driver as cuda
print(f"PyCUDA initialized: {cuda.Device(0).name()}")
# Expected: Tesla V100-SXM2-16GB
```

### 6. Verify NumPy
```python
import numpy as np
print(f"NumPy: {np.__version__}")
# Expected: 1.26.4 (< 2.0)
```

---

## 🚀 USAGE

### 1. Activate Environment
```bash
source ~/data/hai_venv_py310/bin/activate
cd /home/ubuntu/data/person_reid_system
```

### 2. Run TensorRT Inference
```python
from core.detector_trt import TensorRTDetector

detector = TensorRTDetector(
    engine_path='models/bytetrack_x_mot17_fp16.trt',
    conf_threshold=0.5,
    nms_threshold=0.45,
    test_size=(640, 640)
)

detections = detector.detect(frame)
```

### 3. Run Benchmark
```bash
python tools/benchmark.py \
    --pytorch-model models/bytetrack_x_mot17.pth.tar \
    --tensorrt-engine models/bytetrack_x_mot17_fp16.trt \
    --warmup 10 \
    --iterations 100
```

---

## 🐛 TROUBLESHOOTING

### Issue 1: `libcudnn.so.8: cannot open shared object file`
**Solution:**
```bash
pip uninstall -y nvidia-cudnn-cu12
pip install nvidia-cudnn-cu12==8.9.6.50 --no-deps
```

### Issue 2: `Target GPU SM 70 is not supported`
**Solution:** Downgrade TensorRT từ 10.x xuống 8.6.1
```bash
pip uninstall -y tensorrt tensorrt_cu12 tensorrt_cu12_libs tensorrt_cu12_bindings
pip install tensorrt==8.6.1.post1 --extra-index-url https://pypi.nvidia.com
```

### Issue 3: `Numpy is not available`
**Solution:** Downgrade NumPy
```bash
pip install "numpy<2"
```

### Issue 4: `Could not find a version that satisfies tensorrt_bindings`
**Solution:** Sử dụng Python 3.10 thay vì Python 3.12
```bash
python3.10 -m venv ~/data/hai_venv_py310
source ~/data/hai_venv_py310/bin/activate
```

---

## 📊 PERFORMANCE EXPECTATIONS

### Benchmark Results (Tesla V100)

| Metric | PyTorch FP16 | TensorRT FP16 | Improvement |
|--------|--------------|---------------|-------------|
| Latency | 45.63 ms | 35.58 ms | **-22%** |
| FPS | 21.91 | 28.11 | **+28%** |
| Speedup | 1.00x | **1.28x** | - |
| Accuracy | 100% | 100% | Same |

### Expected Speedup on Different GPUs

| GPU | Architecture | SM | Expected Speedup |
|-----|--------------|-----|------------------|
| V100 | Volta | 70 | 1.2-1.5x |
| T4 | Turing | 75 | 2.0-2.5x |
| RTX 3090 | Ampere | 86 | 3.0-4.0x |
| A100 | Ampere | 80 | 3.5-5.0x |

---

## 📝 NOTES

1. **Speedup thấp hơn mong đợi (1.28x vs 3-5x)** vì:
   - GPU V100 (SM 70) cũ, không tối ưu cho TensorRT 8.6
   - PyTorch đã dùng FP16 và fused model
   - Batch size = 1 (không tận dụng parallel processing)

2. **Để tăng speedup:**
   - Sử dụng GPU mới hơn (A100, RTX 3090)
   - Tăng batch size (nếu use case cho phép)
   - Thử INT8 quantization (cần calibration dataset)

3. **Environment isolation:**
   - Luôn sử dụng virtual environment riêng
   - Không cài TensorRT vào system Python
   - Tránh conflict với các project khác

---

## 🔄 MAINTENANCE

### Update Dependencies
```bash
# Backup current environment
pip freeze > requirements_backup.txt

# Update specific package (cẩn thận với version constraints)
pip install --upgrade <package>

# Verify after update
python tools/benchmark.py --warmup 5 --iterations 10
```

### Recreate Environment
```bash
# Deactivate current environment
deactivate

# Remove old environment
rm -rf ~/data/hai_venv_py310

# Recreate from scratch
python3.10 -m venv ~/data/hai_venv_py310
source ~/data/hai_venv_py310/bin/activate

# Run installation script (see above)
```

---

## 📞 SUPPORT

Nếu gặp vấn đề, kiểm tra theo thứ tự:

1. ✅ Python version = 3.10.x
2. ✅ PyTorch version = 2.0.1+cu118
3. ✅ TensorRT version = 8.6.1
4. ✅ CUDNN version = 8.9.6.50
5. ✅ NumPy version < 2.0
6. ✅ CUDA available in PyTorch
7. ✅ GPU SM = 70 (V100)

---

**Last Updated:** 2025-11-11  
**Environment:** Python 3.10 + PyTorch 2.0.1 + TensorRT 8.6.1 + CUDNN 8.9.6.50

