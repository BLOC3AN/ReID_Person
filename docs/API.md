# API Documentation

## ReID Matching Strategy

The system uses an optimized **"First-3 + Re-verify"** strategy for person re-identification:

### Strategy Overview

```
New Track Detected
    ↓
Frame 0-2: Extract 3 embeddings → Majority Voting → Assign Label
    ↓
Frame 3-29: Use Cached Label (no extraction)
    ↓
Frame 30: Re-extract → Re-match → Update if needed
    ↓
Frame 31-59: Use Cached Label
    ↓
Frame 60: Re-verify again...
```

### Implementation Details

**1. First-3 Voting (Robust Initialization)**
```python
# Collect embeddings from first 3 frames
for frame in [0, 1, 2]:
    embedding = extract(frame, bbox)
    track_embeddings[track_id].append(embedding)

# After 3rd frame: Majority voting
votes = {}
for emb in track_embeddings[track_id]:
    global_id, sim, name = find_best_match(emb)
    votes[(global_id, name)] += 1

# Select winner: highest votes + highest similarity
best_label = max(votes, key=lambda x: (votes[x], similarity[x]))
```

**2. Re-verification (Self-Correction)**
```python
# Every 30 frames
if frame_count % 30 == 0:
    embedding = extract(frame, bbox)
    global_id, sim, name = find_best_match(embedding)

    # Update if changed or high confidence
    if new_label != old_label or sim >= threshold:
        update_label(track_id, new_label)
```

**3. Performance Metrics**
- **Embedding extractions:** 3 (voting) + N/30 (re-verify) vs N (every frame)
- **Speedup:** ~5.3x (19 FPS vs 3.6 FPS)
- **Reduction:** 95.8% fewer extractions (19 vs 450 for 150 frames)

---

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
# Returns: [(global_id, similarity, name), ...]

# Create new person
person_id = db.create_new_person(name="John", embeddings=[feature])
```

**Methods:**
- `add_embedding(person_id, embedding, metadata)` - Add embedding to database
- `find_best_match(query_embedding, top_k)` - Find best matching person, returns `[(global_id, similarity, name), ...]`
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
# Register first person
python scripts/register_mot17.py \
  --video data/videos/person.mp4 \
  --name "PersonName" \
  --global-id 1 \
  --sample-rate 5

# Register additional person
python scripts/register_mot17.py \
  --video data/videos/person2.mp4 \
  --name "Person2" \
  --global-id 2

# Delete existing collection and start fresh
python scripts/register_mot17.py \
  --video data/videos/person.mp4 \
  --name "PersonName" \
  --global-id 1 \
  --delete-existing
```

**Arguments:**
- `--video` - Path to video containing person
- `--name` - Person name to register
- `--global-id` - Unique ID for person (required, e.g., 1, 2, 3)
- `--sample-rate` - Extract 1 frame every N frames (default: 5)
- `--delete-existing` - Delete existing collection before registering (optional)

**Output:**
- Updates `data/database/reid_database.pkl`
- Syncs to Qdrant cloud
- Prints person ID and number of embeddings

**Important:**
- Each person must have a unique `--global-id`
- Use `--delete-existing` to recreate collection from scratch
- Without `--delete-existing`, new person is added to existing collection

---

### 2. Detect and Track (`scripts/detect_and_track.py`)

```bash
python scripts/detect_and_track.py \
  --video data/videos/test.mp4 \
  --model mot17 \
  --threshold 0.8 \
  --max-frames 100
```

**Arguments:**
- `--video` - Input video path
- `--model` - Model type: `mot17` or `yolox` (default: mot17)
- `--threshold` - Similarity threshold (default: 0.8)
- `--max-frames` - Maximum frames to process (optional)

**Note:** Person names are automatically retrieved from Qdrant database. All registered persons will be detected and labeled with their names.

**ReID Strategy:**
- **First-3 Voting:** Extract embeddings from first 3 frames → Majority vote
- **Re-verification:** Re-match every 30 frames for self-correction
- **Cached Labels:** Use cached results for other frames (fast)
- **Performance:** ~19 FPS (5.3x speedup vs ReID every frame)

**Output:**
- `outputs/videos/` - Annotated video with FPS counter + frame counter
- `outputs/csv/` - Tracking data CSV (frame_id, track_id, bbox, global_id, similarity, label)
- `outputs/logs/` - Detailed logs with [VOTING] and [RE-VERIFY] events

**Video Features:**
- Real-time FPS counter (top-left)
- Frame counter
- Person labels with similarity scores
- Color-coded bounding boxes (green=known, red=unknown)

**Log Events:**
```
Track 1 [VOTING]: 3/3 votes → Duong (sim=0.9606, gid=2)
Track 1 [RE-VERIFY]: Duong → Duong (sim=0.9654, frame=30)
Track 2 [RE-VERIFY]: Khiem → Khiem (sim=0.9427, frame=30)
```

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

