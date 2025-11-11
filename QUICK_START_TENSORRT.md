# TensorRT Quick Start Guide

## 🚀 QUICK INSTALLATION (5 phút)

### Prerequisites
- Ubuntu 20.04/22.04
- NVIDIA GPU (Tesla V100 hoặc tương đương)
- CUDA 12.x đã cài đặt
- Python 3.10 available

### Step 1: Create Virtual Environment
```bash
python3.10 -m venv ~/data/hai_venv_py310
source ~/data/hai_venv_py310/bin/activate
```

### Step 2: Install Dependencies (One-liner)
```bash
pip install --upgrade pip && \
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && \
pip install "numpy<2" && \
pip install tensorrt==8.6.1.post1 --extra-index-url https://pypi.nvidia.com && \
pip install nvidia-cudnn-cu12==8.9.6.50 --no-deps && \
pip install pycuda onnx onnxruntime-gpu opencv-python scipy pillow thop tabulate tqdm pycocotools lap cython_bbox loguru pyyaml python-dotenv qdrant-client
```

### Step 3: Verify Installation
```bash
python -c "import torch; import tensorrt as trt; print(f'PyTorch: {torch.__version__}, TensorRT: {trt.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.0.1+cu118, TensorRT: 8.6.1, CUDA: True
```

---

## 🎯 USAGE

### 1. Run TensorRT Inference
```python
from core.detector_trt import TensorRTDetector
import cv2

# Initialize detector
detector = TensorRTDetector(
    engine_path='models/bytetrack_x_mot17_fp16.trt',
    conf_threshold=0.5,
    nms_threshold=0.45,
    test_size=(640, 640)
)

# Run inference
frame = cv2.imread('test.jpg')
detections = detector.detect(frame)
print(f"Detected {len(detections)} objects")
```

### 2. Run Benchmark
```bash
source ~/data/hai_venv_py310/bin/activate
cd /home/ubuntu/data/person_reid_system

python tools/benchmark.py \
    --pytorch-model models/bytetrack_x_mot17.pth.tar \
    --tensorrt-engine models/bytetrack_x_mot17_fp16.trt \
    --warmup 10 \
    --iterations 100
```

---

## 📊 EXPECTED RESULTS

### Performance (Tesla V100)
- **PyTorch FP16**: 45.63 ms (21.91 FPS)
- **TensorRT FP16**: 35.58 ms (28.11 FPS)
- **Speedup**: 1.28x (28% faster)
- **Accuracy**: 100% (identical to PyTorch)

---

## ⚠️ CRITICAL REQUIREMENTS

| Component | Version | Why |
|-----------|---------|-----|
| Python | **3.10.x** | TensorRT 8.6.1 bindings chỉ có cho Python 3.10 |
| PyTorch | **2.0.1+cu118** | Tương thích CUDNN 8.x |
| TensorRT | **8.6.1** | Hỗ trợ GPU SM 70 (V100) |
| CUDNN | **8.9.6.50** | Yêu cầu của TensorRT 8.6.1 |
| NumPy | **< 2.0** | PyTorch 2.0.1 không tương thích NumPy 2.x |

**❌ KHÔNG CÀI:**
- PyTorch 2.5.x (yêu cầu CUDNN 9.x)
- TensorRT 10.x (không hỗ trợ SM 70)
- NumPy 2.x
- Python 3.12

---

## 🐛 COMMON ISSUES

### Issue: `libcudnn.so.8: cannot open shared object file`
```bash
pip install nvidia-cudnn-cu12==8.9.6.50 --no-deps
```

### Issue: `Target GPU SM 70 is not supported`
```bash
pip install tensorrt==8.6.1.post1 --extra-index-url https://pypi.nvidia.com
```

### Issue: `Numpy is not available`
```bash
pip install "numpy<2"
```

---

## 📁 FILES

```
models/
├── bytetrack_x_mot17_fp16.trt (192 MB) ← TensorRT engine
├── bytetrack_x_mot17.pth.tar (757 MB)  ← PyTorch weights
└── bytetrack_x_mot17.onnx (381 MB)     ← ONNX export

tools/
├── export_onnx_simple.py      ← Export PyTorch → ONNX
├── convert_tensorrt.py        ← Convert ONNX → TensorRT
├── benchmark.py               ← Benchmark PyTorch vs TensorRT
└── verify_onnx.py             ← Verify ONNX model

core/
├── detector.py                ← PyTorch detector
└── detector_trt.py            ← TensorRT detector
```

---

## 📖 FULL DOCUMENTATION

Xem `TENSORRT_SETUP.md` để biết chi tiết về:
- Dependency conflicts và cách giải quyết
- Troubleshooting guide
- Performance tuning
- Advanced usage

---

**Last Updated:** 2025-11-11  
**Environment:** Python 3.10 + PyTorch 2.0.1 + TensorRT 8.6.1

