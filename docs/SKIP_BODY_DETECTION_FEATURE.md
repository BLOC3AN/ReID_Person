# Skip Body Detection Feature

## 📋 Tổng quan

Feature này cho phép bỏ qua bước body detection (YOLOX MOT17) và sử dụng trực tiếp toàn bộ ảnh làm bbox để extract face embedding. Điều này hữu ích khi bạn đã có ảnh augmented face (chỉ khuôn mặt, không có body).

## 🎯 Vấn đề giải quyết

**Trước đây:**
- Ảnh augmented face (chỉ có khuôn mặt) → YOLOX không detect được body → Lỗi
- Logic: `Image → YOLOX detect body → Crop bbox → ArcFace extract face → Save to VectorDB`

**Bây giờ:**
- Ảnh augmented face → Skip YOLOX → Dùng full image → ArcFace extract face → Save to VectorDB
- Logic: `Image → [SKIP YOLOX] → Use full image as bbox → ArcFace extract face → Save to VectorDB`

## 🔧 Các thay đổi

### 1. Core Functions (`scripts/register_mot17.py`)

#### `register_person_mot17()`
- **Thêm param:** `skip_body_detection: bool = False`
- **Logic:**
  ```python
  if skip_body_detection:
      # Use full image as bbox
      h, w = frame.shape[:2]
      bbox = [0, 0, w, h]
  else:
      # Detect body using YOLOX
      detections = detector.detect(frame)
      # ... get largest bbox
  ```

#### `register_person_from_images()`
- **Thêm param:** `skip_body_detection: bool = False`
- **Logic tương tự như trên**

### 2. Service Layer (`services/register_service.py`)

#### `process_registration()`
- **Thêm param:** `skip_body_detection: bool = False`
- **Truyền param xuống:** `register_person_mot17(..., skip_body_detection=skip_body_detection)`

#### `process_image_registration()`
- **Thêm param:** `skip_body_detection: bool = False`
- **Truyền param xuống:** `register_person_from_images(..., skip_body_detection=skip_body_detection)`

### 3. API Endpoints (`services/register_service.py`)

#### `/register` (POST)
- **Thêm form field:** `skip_body_detection: bool = Form(False)`
- **Truyền vào background task**

#### `/register-batch` (POST)
- **Thêm form field:** `skip_body_detection: bool = Form(False)`
- **Truyền vào background task**

#### `/register-images` (POST)
- **Thêm form field:** `skip_body_detection: bool = Form(False)`
- **Truyền vào background task**

### 4. Streamlit UI (`app.py`)

#### Registration Form
- **Thêm checkbox:**
  ```python
  skip_body_detection = st.checkbox(
      "Skip Body Detection",
      value=False,
      help="⚠️ Use full image as bbox (for augmented face images without body)"
  )
  ```
- **Thêm vào data payload khi gọi API**

### 5. CLI (`scripts/register_mot17.py`)

#### Command Line Arguments
- **Thêm flag:**
  ```bash
  --skip-body-detection
  ```

## 📖 Cách sử dụng

### 1. CLI

```bash
# Register với augmented face images
python scripts/register_mot17.py \
  --video data/augmented_faces/person1.mp4 \
  --name "John Doe" \
  --global-id 1 \
  --skip-body-detection

# Register từ folder ảnh augmented
python scripts/register_mot17.py \
  --video data/augmented_faces/person1/ \
  --name "John Doe" \
  --global-id 1 \
  --skip-body-detection
```

### 2. API

```bash
# Register images với skip body detection
curl -X POST http://localhost:8001/register-images \
  -F 'images=@face1.jpg' \
  -F 'images=@face2.jpg' \
  -F 'person_name=John Doe' \
  -F 'global_id=1' \
  -F 'skip_body_detection=true'
```

### 3. Streamlit UI

1. Mở UI: `http://localhost:8501`
2. Chọn tab "📝 Register Person"
3. Upload ảnh augmented face
4. **✅ Check "Skip Body Detection"**
5. Nhập thông tin và click "Register Person"

## ✅ Lợi ích

1. **Giải quyết lỗi:** Không còn lỗi "No person detected" với ảnh augmented face
2. **Linh hoạt:** Có thể dùng cho cả ảnh thường và ảnh augmented
3. **Backward compatible:** Mặc định `skip_body_detection=False` giữ nguyên logic cũ
4. **Đơn giản:** Chỉ thêm 1 flag boolean, không phá vỡ logic hiện tại

## 🧪 Testing

Chạy test script:
```bash
python test_skip_body_detection.py
```

## 📝 Notes

- Khi `skip_body_detection=True`, detector sẽ không được khởi tạo (tiết kiệm memory)
- Face embedding vẫn được extract bởi ArcFace (InsightFace)
- Logic lưu vào VectorDB không thay đổi
- Phù hợp cho ảnh đã được crop/augment chỉ chứa khuôn mặt

