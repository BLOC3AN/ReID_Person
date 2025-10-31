# Zone Monitoring - Quick Guide

## 🎯 Tổng quan

Zone Monitoring cho phép theo dõi người làm việc trong các khu vực được định nghĩa trước, kết hợp với Person ReID để xác định danh tính.

### Tính năng chính

- ✅ **IoP-based zone detection**: Tính % diện tích person nằm trong zone (Intersection over Person)
- ✅ **R-tree spatial indexing**: O(log n) performance cho zone lookup
- ✅ **Authorization checking**: Mỗi zone có danh sách authorized IDs
- ✅ **Time tracking**: Tính thời gian presence trong mỗi zone
- ✅ **Violation detection**: Phát hiện unauthorized entries

### 📐 IoP vs IOU

**IoP (Intersection over Person)** - Phương pháp hiện tại ⭐:
```
IoP = Intersection / Area_Person
```
- ✅ Đo % diện tích person nằm trong zone
- ✅ Hoạt động chính xác khi zone lớn hơn person nhiều
- ✅ Dễ hiểu: "60% cơ thể trong zone" = trong zone
- ✅ Không phụ thuộc kích thước zone

**Ví dụ:**
```
Zone: 1000x1000 pixels (1,000,000 px²)
Person: 50x100 pixels (5,000 px²) - hoàn toàn trong zone

IOU (cũ) = 5,000 / 1,000,000 = 0.5% ❌ (không detect!)
IoP (mới) = 5,000 / 5,000 = 100% ✅ (detect chính xác!)
```

## 📊 Pipeline hoàn chỉnh

```
Video Input
    ↓
[YOLOX Detection] → Phát hiện bbox người
    ↓
[ByteTrack] → Gán track_id
    ↓
[ArcFace + Qdrant] → ReID matching → Global ID
    ↓
[Zone Monitor] → IoP-based zone detection (% person in zone)
    ↓
Output: Video (zones drawn) + CSV + JSON report
```

## 🚀 Sử dụng qua UI

### Phương án 1: Tạo Zones trực tiếp trong UI (Recommended)

1. Mở UI: `http://localhost:8501`
2. Chọn **"Detect & Track"**
3. Upload video
4. Mở **"🗺️ Zone Monitoring (Optional)"**
5. Chọn **"Create Zones in UI"**
6. Nhập **Number of Zones** (ví dụ: 2)
7. Cấu hình từng zone:
   - **Zone Name**: "Assembly Area"
   - **Authorized IDs**: `1,2` (comma-separated)
   - **Polygon Points**: `100,100; 400,100; 400,300; 100,300`
8. Preview YAML và download nếu muốn lưu
9. Điều chỉnh **IoP Threshold** (mặc định: 0.6 = 60% cơ thể trong zone)
10. Click **"🚀 Start Detection"**

**Ví dụ cấu hình:**

**Zone 1 - Assembly Area:**
- Name: `Assembly Area`
- Authorized IDs: `1`
- Polygon: `100,100; 400,100; 400,300; 100,300`

**Zone 2 - Packaging Area:**
- Name: `Packaging Area`
- Authorized IDs: `2`
- Polygon: `450,100; 750,100; 750,300; 450,300`

### Phương án 2: Upload YAML File

1. Tạo file `zones.yaml`:

```yaml
zones:
  zone1:
    name: "Assembly Area"
    polygon:
      - [100, 100]  # Top-left
      - [400, 100]  # Top-right
      - [400, 300]  # Bottom-right
      - [100, 300]  # Bottom-left
    authorized_ids: [1]  # Duong (Global ID từ Qdrant)

  zone2:
    name: "Packaging Area"
    polygon:
      - [450, 100]
      - [750, 100]
      - [750, 300]
      - [450, 300]
    authorized_ids: [2]  # Khiem
```

2. Chạy qua UI:
   - Mở UI: `http://localhost:8501`
   - Chọn **"Detect & Track"**
   - Upload video
   - Mở **"🗺️ Zone Monitoring (Optional)"**
   - Chọn **"Upload YAML File"**
   - Upload file `zones.yaml`
   - Điều chỉnh **IoP Threshold** (0.6 = 60% cơ thể trong zone)
   - Click **"🚀 Start Detection"**

### Kết quả

**Video Output:**
- **Ruler (Thước đo)**: Hiển thị ở 4 cạnh video với tọa độ pixel
  - Major ticks: Mỗi 100 pixels (có số)
  - Minor ticks: Mỗi 50 pixels
  - Corner coordinates: (0,0) và (width, height)
- **Zones**: Vẽ polygon màu vàng
- **Bbox người**: Xanh=authorized, Đỏ=unauthorized, Xám=unknown
- **Zone name**: Hiển thị trên mỗi người
- **Real-time progress**: Progress bar và track info trong UI

**CSV Output:**
```csv
frame_id,track_id,global_id,person_name,similarity,x,y,w,h,zone_id,zone_name,authorized,duration
0,1,1,Duong,0.92,150,120,80,200,zone1,Assembly Area,True,0.0
```

**JSON Report:**
```json
{
  "summary": {
    "zone1": {
      "name": "Assembly Area",
      "current_persons": [
        {"id": 1, "name": "Duong", "duration": 45.3, "authorized": true}
      ]
    }
  },
  "violations": [
    {
      "global_id": 2,
      "name": "Khiem",
      "zone": "zone1",
      "time": 12.5,
      "type": "unauthorized_entry"
    }
  ]
}
```

## ⚙️ Tham số

### IoP Threshold (Intersection over Person)

**Ý nghĩa:** % diện tích cơ thể người nằm trong zone

- **0.5 (50%)**: Loose - Nửa cơ thể trong zone là được
- **0.6 (60%)**: Recommended ⭐ - 60% cơ thể trong zone
- **0.7 (70%)**: Strict - 70% cơ thể trong zone (chính xác cao)

**Lưu ý:**
- IoP = 1.0 (100%) = Người hoàn toàn nằm trong zone
- IoP = 0.6 (60%) = 60% cơ thể trong zone, 40% ngoài zone
- Không phụ thuộc kích thước zone (khác với IOU cũ)

### Similarity Threshold

- **0.7**: Loose matching
- **0.8**: Recommended
- **0.9**: Strict matching

## 🔧 Chạy qua Script (không dùng UI)

```bash
python scripts/zone_monitor.py \
    --video data/videos/factory.mp4 \
    --zones configs/zones.yaml \
    --similarity 0.8 \
    --iou 0.6 \
    --output outputs/
```

## 📏 Ruler (Thước đo tọa độ)

Video output tự động hiển thị ruler ở 4 cạnh để dễ xác định tọa độ:

### Cách đọc Ruler:

**Horizontal Ruler (Top & Bottom):**
- Major ticks (dài): Mỗi 100 pixels, có số hiển thị
- Minor ticks (ngắn): Mỗi 50 pixels
- Ví dụ: 0, 100, 200, 300, ... đến width

**Vertical Ruler (Left & Right):**
- Major ticks (dài): Mỗi 100 pixels, có số hiển thị
- Minor ticks (ngắn): Mỗi 50 pixels
- Ví dụ: 0, 100, 200, 300, ... đến height

**Corner Coordinates:**
- Top-left: `0,0` (màu xanh lá)
- Bottom-right: `width,height` (màu xanh lá)

### Cách sử dụng Ruler để tạo Zone:

1. **Chạy video một lần** để xem ruler
2. **Pause video** tại frame cần định nghĩa zone
3. **Đọc tọa độ** từ ruler:
   - Nhìn vào horizontal ruler (top/bottom) để lấy tọa độ X
   - Nhìn vào vertical ruler (left/right) để lấy tọa độ Y
4. **Ghi lại 4 góc** của zone (hoặc nhiều hơn nếu polygon phức tạp)
5. **Nhập vào UI** hoặc YAML config

**Ví dụ:**
```
Video resolution: 1920x1080

Muốn tạo zone ở góc trên bên trái (1/4 màn hình):
- Top-left: Đọc ruler → (0, 0)
- Top-right: Đọc ruler → (960, 0)
- Bottom-right: Đọc ruler → (960, 540)
- Bottom-left: Đọc ruler → (0, 540)

→ Polygon: 0,0; 960,0; 960,540; 0,540
```

## 📝 Lưu ý

1. **Zone polygon**: Tọa độ theo pixel của video
2. **Authorized IDs**: Phải là Global ID đã register trong Qdrant
3. **IOU calculation**: Tính overlap giữa person bbox và zone bbox
4. **R-tree indexing**: Tối ưu performance cho nhiều zones

## 🎓 Use Cases

### Factory Safety
- Zone nguy hiểm: Chỉ nhân viên được training
- Phát hiện vi phạm real-time
- Báo cáo thời gian làm việc

### Office Access Control
- Phòng server: Chỉ IT staff
- Meeting rooms: Theo lịch đặt
- Track thời gian làm việc

### Retail Analytics
- Khu vực sản phẩm cao cấp: Track thời gian khách
- Khu vực nhân viên: Không cho khách vào
- Phân tích hành vi

## 🔍 Troubleshooting

### Zone không detect được người

**Giải pháp:**
- Giảm IOU threshold xuống 0.5
- Kiểm tra polygon coordinates
- Verify video resolution

### Performance chậm

**Giải pháp:**
- R-tree đã tối ưu O(log n)
- Dùng rectangle thay vì polygon phức tạp
- Giảm số zones nếu có thể

### Violations không chính xác

**Giải pháp:**
- Tăng similarity threshold lên 0.85
- Kiểm tra database registration
- Verify authorized_ids trong zones.yaml

