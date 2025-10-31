# UI Zone Configuration Guide

## 🎯 Tổng quan

Bạn có thể cấu hình zones trực tiếp trong UI mà không cần tạo file YAML thủ công.

## 📝 Hướng dẫn từng bước

### Bước 1: Mở Zone Configuration

1. Truy cập UI: `http://localhost:8501`
2. Chọn tab **"Detect & Track"**
3. Upload video cần phân tích
4. Click vào **"🗺️ Zone Monitoring (Optional)"** để mở rộng

### Bước 2: Chọn phương thức cấu hình

Bạn có 2 lựa chọn:

#### Option A: Create Zones in UI ⭐ (Recommended)
- Tạo và chỉnh sửa zones trực tiếp trong giao diện
- Không cần tạo file YAML thủ công
- Preview và download config nếu muốn

#### Option B: Upload YAML File
- Upload file `zones.yaml` đã chuẩn bị sẵn
- Phù hợp khi đã có config từ trước

### Bước 3: Tạo Zones trong UI

#### 3.1. Nhập số lượng zones

```
Number of Zones: 2
```

Hệ thống sẽ tự động tạo 2 zones với config mặc định.

#### 3.2. Cấu hình Zone 1

**Zone Name:**
```
Assembly Area
```

**Authorized IDs (comma-separated):**
```
1
```
- Nhập Global ID của người được phép vào zone này
- Có thể nhập nhiều IDs: `1,2,3`
- Để trống nếu không ai được phép (restricted area)

**Polygon Points:**
```
100,100; 400,100; 400,300; 100,300
```

Format: `x1,y1; x2,y2; x3,y3; x4,y4`

- Mỗi điểm là tọa độ pixel (x, y)
- Phân cách các điểm bằng dấu `;`
- Tối thiểu 3 điểm (tam giác)
- Thường dùng 4 điểm (hình chữ nhật)

**Ví dụ hình chữ nhật:**
```
Top-left: (100, 100)
Top-right: (400, 100)
Bottom-right: (400, 300)
Bottom-left: (100, 300)

→ Polygon: 100,100; 400,100; 400,300; 100,300
```

#### 3.3. Cấu hình Zone 2

**Zone Name:**
```
Packaging Area
```

**Authorized IDs:**
```
2
```

**Polygon Points:**
```
450,100; 750,100; 750,300; 450,300
```

### Bước 4: Preview và Download (Optional)

Click vào **"📄 Preview YAML Config"** để xem config dạng YAML:

```yaml
zones:
  zone1:
    name: Assembly Area
    polygon:
    - [100, 100]
    - [400, 100]
    - [400, 300]
    - [100, 300]
    authorized_ids:
    - 1
  zone2:
    name: Packaging Area
    polygon:
    - [450, 100]
    - [750, 100]
    - [750, 300]
    - [450, 300]
    authorized_ids:
    - 2
```

Click **"💾 Download Zone Config"** để lưu file `zones.yaml` cho lần sau.

### Bước 5: Điều chỉnh IoP Threshold

```
Zone IoP Threshold: 0.6 (60% of person in zone)
```

**IoP (Intersection over Person)** = % diện tích cơ thể người nằm trong zone

- **0.5 (50%)**: Loose - Nửa cơ thể trong zone là được
- **0.6 (60%)**: Recommended - 60% cơ thể trong zone ✅
- **0.7 (70%)**: Strict - 70% cơ thể trong zone (chính xác cao)

**Ưu điểm IoP:**
- ✅ Hoạt động chính xác khi zone lớn hơn person
- ✅ Dễ hiểu: "60% cơ thể trong zone" = trong zone
- ✅ Không phụ thuộc kích thước zone

### Bước 6: Start Detection

Click **"🚀 Start Detection"** để bắt đầu xử lý.

## 📊 Kết quả

Sau khi xử lý xong, bạn sẽ thấy:

### 1. Zone Monitoring Report

```
🗺️ Zone Monitoring Report

Zone Summary
📍 Assembly Area (zone1)
  Authorized IDs: [1]
  Current Persons: 1
    ✅ Duong (ID:1) - 45.3s

📍 Packaging Area (zone2)
  Authorized IDs: [2]
  Current Persons: 1
    ✅ Khiem (ID:2) - 120.5s

⚠️ Violations Detected
  🚫 Khiem (ID:2) entered unauthorized zone Assembly Area at 12.5s
```

### 2. Download Files

- **📹 Video**: Zones vẽ trên video, bbox người có màu
- **📊 CSV**: Tracking data với zone info
- **🗺️ Zone Report**: JSON report chi tiết
- **📄 Log**: Detection log

## 💡 Tips & Tricks

### Cách xác định tọa độ polygon

**Phương pháp 1: Sử dụng Ruler trên video (Recommended) ⭐**

Video output tự động có ruler (thước đo) ở 4 cạnh:

1. **Chạy detection một lần** với video (không cần zone config)
2. **Mở video output** và pause tại frame cần tạo zone
3. **Đọc tọa độ từ ruler:**
   - Horizontal ruler (top/bottom): Tọa độ X
   - Vertical ruler (left/right): Tọa độ Y
   - Major ticks: Mỗi 100 pixels (có số)
   - Minor ticks: Mỗi 50 pixels
4. **Ghi lại tọa độ 4 góc** của zone
5. **Nhập vào UI** hoặc YAML config

**Ví dụ đọc ruler:**
```
Video 1920x1080, muốn tạo zone ở giữa màn hình:

Nhìn vào ruler:
- Top-left: X=500 (horizontal ruler), Y=300 (vertical ruler) → (500, 300)
- Top-right: X=1400, Y=300 → (1400, 300)
- Bottom-right: X=1400, Y=700 → (1400, 700)
- Bottom-left: X=500, Y=700 → (500, 700)

→ Polygon: 500,300; 1400,300; 1400,700; 500,700
```

**Phương pháp 2: Ước lượng từ video resolution**

Nếu video 1920x1080:
- Top-left corner: `0,0`
- Top-right corner: `1920,0`
- Bottom-right corner: `1920,1080`
- Bottom-left corner: `0,1080`

Zone ở góc trên bên trái (1/4 màn hình):
```
0,0; 960,0; 960,540; 0,540
```

**Phương pháp 3: Dùng video player**

1. Mở video trong VLC/MPV
2. Pause tại frame cần xác định zone
3. Di chuột để xem tọa độ pixel
4. Ghi lại các điểm góc

**Phương pháp 4: Dùng OpenCV (advanced)**

```python
import cv2

cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read()

# Click on frame to get coordinates
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Point: {x},{y}")

cv2.imshow("Frame", frame)
cv2.setMouseCallback("Frame", mouse_callback)
cv2.waitKey(0)
```

### Authorized IDs

**Cách lấy Global ID:**

1. Vào tab **"Register Person"** trong UI
2. Xem danh sách người đã đăng ký
3. Mỗi người có một Global ID (số nguyên)

**Ví dụ:**
- Duong → Global ID: 1
- Khiem → Global ID: 2
- Huy → Global ID: 3

**Cấu hình:**
- Zone chỉ cho Duong: `1`
- Zone cho Duong và Khiem: `1,2`
- Zone cho tất cả: `1,2,3`
- Zone không cho ai (restricted): để trống

### Polygon shapes

**Rectangle (4 points):**
```
100,100; 400,100; 400,300; 100,300
```

**Triangle (3 points):**
```
100,100; 300,100; 200,300
```

**Pentagon (5 points):**
```
100,100; 200,80; 300,100; 280,200; 120,200
```

**Irregular shape (6+ points):**
```
100,100; 200,100; 250,150; 200,200; 100,200; 80,150
```

## 🔧 Troubleshooting

### "Invalid polygon format"

**Nguyên nhân:** Sai format tọa độ

**Giải pháp:**
- Đảm bảo format: `x1,y1; x2,y2; x3,y3`
- Dùng dấu `;` để phân cách các điểm
- Dùng dấu `,` để phân cách x và y
- Không có khoảng trắng thừa

### "Need at least 3 points"

**Nguyên nhân:** Polygon cần tối thiểu 3 điểm

**Giải pháp:**
- Nhập ít nhất 3 cặp tọa độ
- Ví dụ: `100,100; 200,100; 150,200`

### "Invalid ID format"

**Nguyên nhân:** Authorized IDs không đúng format

**Giải pháp:**
- Chỉ nhập số nguyên
- Phân cách bằng dấu `,`
- Ví dụ đúng: `1,2,3`
- Ví dụ sai: `1, 2, 3` (có khoảng trắng)

### Zone không detect được người

**Nguyên nhân:** IOU threshold quá cao hoặc polygon sai

**Giải pháp:**
- Giảm IOU threshold xuống 0.5
- Kiểm tra lại tọa độ polygon bằng ruler trên video
- Verify video resolution
- Chạy lại video và xem ruler để đảm bảo tọa độ chính xác

### Progress bar không hiển thị

**Nguyên nhân:** Backend không gửi progress updates

**Giải pháp:**
- Kiểm tra logs của detection service
- Refresh browser page
- Đảm bảo API endpoint `/progress/{job_id}` hoạt động
- Progress updates mỗi 5 frames để giảm overhead

## 📚 Tham khảo

- [Zone Monitoring Guide](./ZONE_MONITORING_GUIDE.md)
- [Example Config](../configs/zones_example.yaml)

