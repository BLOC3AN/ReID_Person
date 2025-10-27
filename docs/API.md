# API Documentation

## Core Modules

### 1. Detector (`core/detector.py`)

**Class:** `YOLOXDetector`

```python
from core.detector import YOLOXDetector

detector = YOLOXDetector(
    model_path="models/bytetrack_x_mot17.pth.tar",
    conf_thresh=0.5,
    nms_thresh=0.45
)

# Detect objects in frame
detections = detector.detect(frame)
# Returns: List of [x1, y1, x2, y2, conf, cls, -1]
```

**Methods:**
- `detect(frame)` - Detect objects in frame
- `preprocess(frame)` - Preprocess frame for detection
- `postprocess(outputs)` - Postprocess detection outputs

---

### 2. Tracker (`core/tracker.py`)

**Class:** `ByteTracker`

```python
from core.tracker import ByteTracker

tracker = ByteTracker(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8
)

# Update tracker with detections
tracks = tracker.update(detections, frame_shape)
# Returns: List of Track objects with .tlbr, .track_id, .score
```

**Methods:**
- `update(detections, frame_shape)` - Update tracker with new detections
- `reset()` - Reset tracker state

---

### 3. Feature Extractor (`core/feature_extractor.py`)

**Class:** `OSNetExtractor`

```python
from core.feature_extractor import OSNetExtractor

extractor = OSNetExtractor(
    model_name="osnet_x0_5",
    device="cuda"
)

# Extract feature from person crop
feature = extractor.extract(person_crop)
# Returns: numpy array (512,) L2-normalized
```

**Methods:**
- `extract(image)` - Extract feature from single image
- `extract_batch(images)` - Extract features from batch of images
- `preprocess(image)` - Preprocess image for extraction

---

### 4. Vector Database (`core/vector_db.py`)

**Class:** `VectorDatabase`

```python
from core.vector_db import VectorDatabase

db = VectorDatabase(
    db_path="data/database/reid_database.pkl",
    use_qdrant=True
)

# Add embedding for person
db.add_embedding(person_id=1, embedding=feature, metadata={})

# Find best match
matches = db.find_best_match(query_embedding=feature, top_k=3)
# Returns: [(person_id, similarity), ...]

# Create new person
person_id = db.create_new_person(name="John", embeddings=[feature])
```

**Methods:**
- `add_embedding(person_id, embedding, metadata)` - Add embedding to database
- `find_best_match(query_embedding, top_k)` - Find best matching person
- `create_new_person(name, embeddings)` - Create new person entry
- `get_person_info(person_id)` - Get person information
- `save_to_file()` - Save database to file
- `load_from_file()` - Load database from file

---

### 5. ReID Matcher (`core/reid_matcher.py`)

**Class:** `ReIDMatcher`

```python
from core.reid_matcher import ReIDMatcher

matcher = ReIDMatcher(
    vector_db=db,
    feature_extractor=extractor,
    threshold=0.8
)

# Match person in frame
result = matcher.match_person(
    frame=frame,
    bbox=[x, y, w, h],
    known_person_id=1
)
# Returns: {
#     'matched': True/False,
#     'similarity': 0.95,
#     'person_id': 1,
#     'person_name': 'John'
# }
```

**Methods:**
- `match_person(frame, bbox, known_person_id)` - Match person against known ID
- `identify_person(frame, bbox)` - Identify person from all registered persons
- `register_person(name, frames, bboxes)` - Register new person

---

## Scripts

### 1. Register Person (`scripts/register_mot17.py`)

```bash
python scripts/register_mot17.py \
  --video data/videos/person.mp4 \
  --name "PersonName" \
  --sample-rate 5
```

**Arguments:**
- `--video` - Path to video containing person
- `--name` - Person name to register
- `--sample-rate` - Extract 1 frame every N frames (default: 5)

**Output:**
- Updates `data/database/reid_database.pkl`
- Syncs to Qdrant cloud
- Prints person ID and number of embeddings

---

### 2. Detect and Track (`scripts/detect_and_track.py`)

```bash
python scripts/detect_and_track.py \
  --video data/videos/test.mp4 \
  --model mot17 \
  --known-person "PersonName" \
  --threshold 0.8 \
  --max-frames 100
```

**Arguments:**
- `--video` - Input video path
- `--model` - Model type: `mot17` or `yolox` (default: mot17)
- `--known-person` - Name of registered person to match
- `--threshold` - Similarity threshold (default: 0.8)
- `--max-frames` - Maximum frames to process (optional)

**Output:**
- `outputs/videos/` - Annotated video
- `outputs/csv/` - Tracking data CSV
- `outputs/logs/` - Detailed logs

---

## Configuration

### config.yaml

```yaml
detection:
  model_path: "models/bytetrack_x_mot17.pth.tar"
  conf_thresh: 0.5
  nms_thresh: 0.45

tracking:
  track_thresh: 0.5
  track_buffer: 30
  match_thresh: 0.8

feature_extraction:
  model_name: "osnet_x0_5"
  device: "cuda"

database:
  db_path: "data/database/reid_database.pkl"
  use_qdrant: true
  qdrant_collection: "cross_camera_matching_id"

reid:
  similarity_threshold: 0.8
  max_embeddings_per_person: 50
```

---

## Data Formats

### CSV Output Format

```csv
frame_id,track_id,x,y,w,h,confidence,global_id,similarity,label
0,1,396,189,238,340,0.8796,1,1.0000,Khiem
0,2,31,216,206,261,0.7674,1,0.7668,Unknown
```

**Columns:**
- `frame_id` - Frame number (0-indexed)
- `track_id` - Local track ID in current video
- `x, y, w, h` - Bounding box (top-left x, y, width, height)
- `confidence` - Detection confidence (0-1)
- `global_id` - Global person ID from database
- `similarity` - Cosine similarity to known person (0-1)
- `label` - Person name or "Unknown"

### Log Format

```
[Frame 0] Detected 2 objects, Tracked 2 persons
  Track 1: bbox=[396,189,238,340], similarity=1.0000, global_id=1 → Khiem
  Track 2: bbox=[31,216,206,261], similarity=0.7668, global_id=1 → Unknown
```

---

## Examples

### Example 1: Register and Detect

```python
from core.detector import YOLOXDetector
from core.tracker import ByteTracker
from core.feature_extractor import OSNetExtractor
from core.vector_db import VectorDatabase
from core.reid_matcher import ReIDMatcher

# Initialize components
detector = YOLOXDetector("models/bytetrack_x_mot17.pth.tar")
tracker = ByteTracker()
extractor = OSNetExtractor()
db = VectorDatabase("data/database/reid_database.pkl")
matcher = ReIDMatcher(db, extractor, threshold=0.8)

# Process frame
detections = detector.detect(frame)
tracks = tracker.update(detections, frame.shape[:2])

for track in tracks:
    bbox = track.tlbr  # [x1, y1, x2, y2]
    result = matcher.identify_person(frame, bbox)
    print(f"Track {track.track_id}: {result['person_name']} (sim={result['similarity']:.4f})")
```

### Example 2: Custom Threshold

```python
# Use different thresholds for different scenarios
matcher_strict = ReIDMatcher(db, extractor, threshold=0.9)  # High precision
matcher_balanced = ReIDMatcher(db, extractor, threshold=0.8)  # Balanced
matcher_loose = ReIDMatcher(db, extractor, threshold=0.7)  # High recall
```

---

## Performance Tips

1. **Batch Processing:** Use `extract_batch()` for multiple crops
2. **GPU Acceleration:** Set `device="cuda"` for faster inference
3. **Frame Sampling:** Process every N frames for faster processing
4. **Threshold Tuning:** Adjust threshold based on your use case
5. **Database Size:** Limit embeddings per person to 50-100 for speed

