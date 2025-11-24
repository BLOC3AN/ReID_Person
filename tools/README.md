# TensorRT Optimization Tools

Công cụ để export và tối ưu YOLOX model với TensorRT cho tốc độ inference nhanh hơn 3-5x.

## 📋 Yêu cầu

### Dependencies cơ bản
```bash
pip install onnx onnxruntime-gpu onnx-simplifier
```

### TensorRT (cho conversion và inference)
```bash
# Option 1: Cài từ NVIDIA
# Download từ: https://developer.nvidia.com/tensorrt
# Làm theo hướng dẫn cài đặt

# Option 2: Dùng pip (nếu có sẵn)
pip install tensorrt pycuda
```

## 🚀 Workflow

### Bước 1: Export ONNX từ PyTorch

Export model PyTorch sang ONNX với FP32 precision:

```bash
python tools/export_onnx.py \
    --model models/bytetrack_x_mot17.pth.tar \
    --output models/bytetrack_x_mot17_fp32.onnx \
    --size 640 640 \
    --opset 11
```

**Tham số:**
- `--model`: Path đến PyTorch weights (.pth.tar)
- `--output`: Path output ONNX model
- `--size`: Input size (height width), mặc định: 640 640
- `--opset`: ONNX opset version (11, 12, hoặc 13), khuyến nghị: 11
- `--dynamic-batch`: Enable dynamic batch size (không khuyến nghị cho TensorRT)
- `--no-simplify`: Skip ONNX simplification

**Best Practices:**
- ✅ Export FP32 (TensorRT sẽ tự optimize)
- ✅ Fixed batch size = 1 (nhanh nhất)
- ✅ Opset 11 hoặc 12 (tương thích tốt với TensorRT)
- ❌ KHÔNG dùng `fuse_model` trước khi export
- ❌ KHÔNG dùng `.half()` trước khi export

### Bước 2: Verify ONNX Model

Kiểm tra ONNX model structure và inference:

```bash
python tools/verify_onnx.py \
    --model models/bytetrack_x_mot17_fp32.onnx \
    --test-image data/test_image.jpg  # Optional
```

**Output:**
- Model information (IR version, opset, input/output shapes)
- ONNX validity check
- Inference test với onnxruntime
- Benchmark (100 iterations)
- Accuracy comparison với PyTorch (nếu có test image)

### Bước 3: Convert ONNX sang TensorRT

Convert ONNX sang TensorRT engine với FP16 precision:

```bash
python tools/convert_tensorrt.py \
    --onnx models/bytetrack_x_mot17_fp32.onnx \
    --output models/bytetrack_x_mot17_fp16.trt \
    --fp16 \
    --workspace 2048
```

**Tham số:**
- `--onnx`: Path đến ONNX model
- `--output`: Path output TensorRT engine (auto-generated nếu không chỉ định)
- `--fp16`: Enable FP16 precision (khuyến nghị, ~3-4x speedup)
- `--int8`: Enable INT8 precision (cần calibration, ~4-5x speedup)
- `--workspace`: Max workspace size in MB (mặc định: 2048)
- `--verbose`: Verbose logging

**Precision Options:**
- **FP32**: Baseline, không tối ưu
- **FP16**: ~3-4x nhanh hơn, 99.5% accuracy ✅ Khuyến nghị
- **INT8**: ~4-5x nhanh hơn, 98-99% accuracy (cần calibration)

### Bước 4: Benchmark PyTorch vs TensorRT

So sánh tốc độ và accuracy:

```bash
python tools/benchmark.py \
    --pytorch-model models/bytetrack_x_mot17.pth.tar \
    --tensorrt-engine models/bytetrack_x_mot17_fp16.trt \
    --video data/test_video.mp4 \
    --num-frames 100 \
    --iterations 100
```

**Tham số:**
- `--pytorch-model`: PyTorch model path
- `--tensorrt-engine`: TensorRT engine path
- `--video`: Test video path
- `--num-frames`: Số frames để test (mặc định: 100)
- `--warmup`: Warmup iterations (mặc định: 10)
- `--iterations`: Benchmark iterations (mặc định: 100)
- `--skip-pytorch`: Skip PyTorch benchmark
- `--skip-tensorrt`: Skip TensorRT benchmark

**Output:**
- Timing statistics (avg, std, min, max, P50, P95, P99)
- FPS comparison
- Speedup ratio
- Accuracy metrics (precision, recall, F1)

## 🎯 Sử dụng trong Production

### Cấu hình Backend

Chỉnh sửa `configs/config.yaml`:

```yaml
detection:
  # Chọn backend: pytorch hoặc tensorrt
  backend: tensorrt  # Đổi từ pytorch sang tensorrt
  
  # PyTorch model paths
  model_path_mot17: models/bytetrack_x_mot17.pth.tar
  
  # TensorRT engine paths
  tensorrt_engine_mot17: models/bytetrack_x_mot17_fp16.trt
  
  # Detection parameters
  conf_threshold: 0.5
  nms_threshold: 0.45
  test_size: [640, 640]
```

### Chạy Pipeline với TensorRT

```python
from core.preloaded_manager import PreloadedPipelineManager

# Initialize với TensorRT backend
manager = PreloadedPipelineManager()
manager.initialize()  # Sẽ load TensorRT detector theo config

# Detector sẽ tự động dùng TensorRT
detector = manager.detector
detections = detector.detect(frame)
```

## 📊 Expected Performance

### RTX 3090 (ví dụ)

| Backend | Precision | Avg Time | FPS | Speedup | Accuracy |
|---------|-----------|----------|-----|---------|----------|
| PyTorch | FP16 | ~12ms | ~83 | 1.0x | 100% |
| TensorRT | FP16 | ~3ms | ~333 | 4.0x | 99.5% |
| TensorRT | INT8 | ~2ms | ~500 | 6.0x | 98-99% |

*Lưu ý: Kết quả thực tế phụ thuộc vào GPU, CUDA version, TensorRT version*

## 🔧 Troubleshooting

### Lỗi: ONNX export failed

**Nguyên nhân:** Model có operations không support bởi ONNX

**Giải pháp:**
- Thử opset version khác (11, 12, 13)
- Kiểm tra PyTorch version compatibility

### Lỗi: TensorRT build failed

**Nguyên nhân:** ONNX model có operations không support bởi TensorRT

**Giải pháp:**
- Verify ONNX model trước: `python tools/verify_onnx.py`
- Thử workspace size lớn hơn: `--workspace 4096`
- Kiểm tra TensorRT version compatibility

### Lỗi: CUDNN_STATUS_BAD_PARAM

**Nguyên nhân:** CUDNN version không tương thích

**Giải pháp:**
- Cài đúng CUDNN version cho TensorRT
- Hoặc dùng TensorRT để inference thay vì onnxruntime

### Lỗi: Engine file not found

**Nguyên nhân:** Chưa convert ONNX sang TensorRT

**Giải pháp:**
```bash
python tools/convert_tensorrt.py \
    --onnx models/bytetrack_x_mot17_fp32.onnx
```

## 📝 Notes

1. **ONNX Export:**
   - Luôn export FP32, để TensorRT tự optimize
   - Không dùng `fuse_model()` hoặc `.half()` trước khi export
   - Fixed batch size = 1 cho tốc độ tốt nhất

2. **TensorRT Conversion:**
   - FP16 là lựa chọn tốt nhất (speedup cao, accuracy tốt)
   - INT8 cần calibration dataset để đạt accuracy tốt
   - Engine file phụ thuộc vào GPU, không portable

3. **Production:**
   - Build engine trên GPU production
   - Kiểm tra accuracy trước khi deploy
   - Monitor performance và accuracy trong production

## 🔗 References

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ONNX Documentation](https://onnx.ai/)
- [YOLOX GitHub](https://github.com/Megvii-BaseDetection/YOLOX)

