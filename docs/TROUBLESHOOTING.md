# Troubleshooting Guide

## Common Issues and Solutions

### 1. Low Similarity Scores (< 0.85 with ArcFace)

**Symptom:**
```
Track 1: similarity=0.57, global_id=1 → Unknown (should be Khiem)
```

**Cause:**
- No face detected in bbox (ArcFace requires visible face)
- Poor video quality or lighting
- Face not visible in registration video

**Solution:**
```bash
# Re-register using video with clear frontal face
python scripts/register_mot17.py --video data/videos/person.mp4 --name Khiem --sample-rate 3

# Then run detection with MOT17
python scripts/detect_and_track.py --video data/videos/test.mp4 --model mot17 --threshold 0.6
```

**Why:** ArcFace extracts face embeddings from person bbox. If face is not visible or unclear, similarity will be low.

**Note:** With ArcFace, expected similarity is 0.85-0.95 for good matches

---

### 2. Qdrant Connection Failed

**Symptom:**
```
WARNING | core.vector_db:_init_qdrant:87 - ⚠️ Failed to init Qdrant: [Errno 111] Connection refused. Using in-memory storage.
```

**Cause:** Missing or incorrect Qdrant credentials.

**Solution:**

1. Check `configs/.env` exists:
```bash
ls -la configs/.env
```

2. Verify credentials format:
```env
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
QDRANT_URI=host=cluster-id.region.aws.cloud.qdrant.io
QDRANT_COLLECTION=cross_camera_matching_id
QDRANT_PORT=6333
```

3. Test connection:
```python
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv("configs/.env")
client = QdrantClient(
    url=os.getenv("QDRANT_URI"),
    api_key=os.getenv("QDRANT_API_KEY")
)
print(client.get_collections())
```

---

### 3. CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solution:**

**Option 1:** Use CPU mode
```python
# In configs/config.yaml
feature_extraction:
  device: "cpu"
```

**Option 2:** Reduce batch size
```python
# Process frames one by one instead of batching
```

**Option 3:** Use smaller model
```bash
# Use YOLOX-S instead of YOLOX-X (not recommended due to accuracy loss)
```

---

### 4. No Detections in Video

**Symptom:**
```
[Frame 0] Detected 0 objects, Tracked 0 persons
[Frame 1] Detected 0 objects, Tracked 0 persons
```

**Cause:** Detection threshold too high or wrong model.

**Solution:**

1. Lower detection threshold in `configs/config.yaml`:
```yaml
detection:
  conf_thresh: 0.3  # Lower from 0.5
```

2. Check video format:
```bash
ffprobe data/videos/test.mp4
```

3. Test detection manually:
```python
from core.detector import YOLOXDetector
import cv2

detector = YOLOXDetector("models/bytetrack_x_mot17.pth.tar", conf_thresh=0.3)
frame = cv2.imread("test_frame.jpg")
detections = detector.detect(frame)
print(f"Detected {len(detections)} objects")
```

---

### 5. Wrong Person Detected

**Symptom:**
```
Track 1: similarity=0.85, global_id=2 → John (should be Khiem)
```

**Cause:** Multiple registered persons with similar appearance.

**Solution:**

1. **Increase threshold:**
```bash
python scripts/detect_and_track.py --threshold 0.9  # More strict
```

2. **Re-register with more samples:**
```bash
python scripts/register_mot17.py --video data/videos/khiem.mp4 --name Khiem --sample-rate 3
# Lower sample-rate = more embeddings
```

3. **Clear Qdrant database and re-register:**
```bash
# Use --delete-existing flag to clear collection
python scripts/register_mot17.py --video data/videos/khiem.mp4 --name Khiem --delete-existing
```

---

### 6. Video Output Not Generated

**Symptom:**
```
No file in outputs/videos/
```

**Cause:** Video writer failed or wrong codec.

**Solution:**

1. Check logs:
```bash
cat outputs/logs/cam1_*.log | grep -i error
```

2. Install video codecs:
```bash
sudo apt-get install ffmpeg libavcodec-extra
```

3. Try different codec in code:
```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Instead of 'avc1'
```

---

### 7. Import Errors

**Symptom:**
```
ModuleNotFoundError: No module named 'loguru'
```

**Solution:**

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or install specific package
pip install loguru
```

---

### 8. Qdrant Connection Issues

**Symptom:**
```
Error connecting to Qdrant: Connection refused
```

**Solution:**

1. **Check Qdrant service status:**
```bash
# For Docker deployment
docker ps | grep qdrant

# Check Qdrant logs
docker logs qdrant
```

2. **Verify Qdrant configuration in configs/.env:**
```bash
QDRANT_URI=http://qdrant:6333
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION=cross_camera_matching_id
```

3. **Recreate Qdrant collection:**
```bash
python scripts/register_mot17.py --video data/videos/person.mp4 --name PersonName --delete-existing
```

---

### 9. Slow Processing Speed

**Symptom:**
```
Processing 1 FPS (should be 10-30 FPS)
```

**Solution:**

1. **Use GPU:**
```yaml
# configs/config.yaml
feature_extraction:
  device: "cuda"
```

2. **Process fewer frames:**
```bash
python scripts/detect_and_track.py --max-frames 100
```

3. **Skip frames:**
```python
# Process every 2nd frame
if frame_id % 2 == 0:
    continue
```

4. **Reduce resolution:**
```python
frame = cv2.resize(frame, (960, 540))  # Half of 1080p
```

---

### 10. Qdrant Collection Not Found

**Symptom:**
```
qdrant_client.http.exceptions.UnexpectedResponse: Collection not found
```

**Solution:**

1. **Create collection:**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url=..., api_key=...)
client.create_collection(
    collection_name="cross_camera_matching_id",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
)
```

2. **Or let register script create it:**
```bash
python scripts/register_mot17.py --video data/videos/person.mp4 --name PersonName
# This will auto-create collection
```

---

## Debug Mode

Enable debug logging:

```python
# Add to top of script
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use loguru:

```python
from loguru import logger
logger.add("debug.log", level="DEBUG")
```

---

## Getting Help

If issue persists:

1. Check logs: `outputs/logs/*.log`
2. Check configuration: `configs/config.yaml`
3. Verify models: `ls -lh models/`
4. Test components individually (see API.md)
5. Check system resources: `nvidia-smi`, `free -h`, `df -h`

---

## Known Limitations

1. **Single person per track:** System assumes one person per bounding box
2. **Occlusion:** Heavy occlusion may cause tracking loss
3. **Lighting:** Extreme lighting changes affect similarity
4. **Angle:** Large viewpoint changes reduce similarity
5. **Resolution:** Low resolution crops reduce accuracy
6. **Face visibility (ArcFace):** Requires visible face in person bbox for accurate matching
7. **Face angle (ArcFace):** Works best with frontal or near-frontal faces

---

## ArcFace-Specific Issues

### No Face Detected in BBox

**Symptom:**
```
DEBUG | core.feature_extractor:extract:231 - No face detected in bbox [473, 184, 254, 398]
```

**Cause:** ArcFace cannot find face within person bounding box.

**Solution:**
1. Ensure video shows clear face (not back view)
2. Use videos with frontal or near-frontal face angles
3. Check if person bbox includes head region
4. Re-register with better quality video showing clear faces

### ArcFace Requirements

**System uses ArcFace for face recognition:**
- Videos must show clear face visibility
- Camera angles should show frontal or near-frontal faces
- Higher accuracy (0.85-0.95 similarity)
- Face-based identification only

**Configuration:**
```yaml
# configs/config.yaml
reid:
  arcface_model_name: buffalo_l  # Options: buffalo_l, buffalo_s, antelopev2
  feature_dim: 512
  use_cuda: true
```

