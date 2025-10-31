# Person Re-Identification System

Multi-camera Person Re-Identification system using YOLOX detection, ByteTrack tracking, **ArcFace (InsightFace)** face recognition, Qdrant vector database, and **Zone Monitoring** with IOU-based detection.

## 🎯 Key Features

### ArcFace Face Recognition
- ✅ High accuracy (similarity 0.85-0.95)
- ✅ Robust to pose/lighting changes
- ✅ Face-focused detection (512-dim embeddings)
- ✅ GPU-accelerated inference

### Zone Monitoring
- ✅ **IOU-based zone detection** (60% overlap threshold)
- ✅ **R-tree spatial indexing** for O(log n) performance
- ✅ **Authorization checking** per zone
- ✅ **Time tracking** for presence in zones
- ✅ **Violation detection** for unauthorized entries
- ✅ **Ruler overlay** on video for easy coordinate reference

### UI Features
- ✅ **Create zones in UI** - No YAML editing needed
- ✅ **Real-time progress bar** with track info
- ✅ **Zone report visualization** with summary and violations
- ✅ **Download zone config** for reuse

📖 **Documentation**: See [docs/](docs/) for detailed guides
🔧 **Configuration**: Edit `configs/config.yaml` for model settings
🗺️ **Zone Monitoring**: See [docs/ZONE_MONITORING_GUIDE.md](docs/ZONE_MONITORING_GUIDE.md)

## Pipeline

```
Video → YOLOX Detection → ByteTrack Tracking → ArcFace ReID → Qdrant Search → Zone Monitoring → Output
                                                                      ↓
                                                            IOU-based Zone Detection
                                                                      ↓
                                                            Authorization Check
                                                                      ↓
                                                            Time Tracking + Violations
```



## ReID Matching Strategy

The system uses an optimized **"First-3 + Re-verify"** strategy for robust and efficient person identification:

### 1. First-3 Voting (Frame 0-2 of each track)
- Extract embeddings from **first 3 frames** of each new track
- Perform **majority voting** from 3 matching results
- Select label with highest votes + highest similarity
- **Purpose:** Robust initialization, reduce false positives from single bad frame

### 2. Re-verification (Every 30 frames)
- Re-extract embedding at frame 30, 60, 90, 120...
- Re-match against database
- Update label if changed or confidence is high
- **Purpose:** Self-correction, handle occlusion/pose changes

### 3. Cached Labels (Other frames)
- Use cached label from voting/re-verification
- No embedding extraction → **Very fast**
- **Purpose:** High performance (19+ FPS vs 3.6 FPS if ReID every frame)

**Performance:** ~5.3x speedup with 95.8% reduction in embedding extractions while maintaining accuracy.

## 🚀 Quick Start

### Option 1: Web UI (Recommended)

```bash
# Activate virtual environment
source ../hai_venv/bin/activate

# Install Streamlit (if not installed)
pip install streamlit

# Launch UI
./run_ui.sh

# Or manually
streamlit run app.py
```

Then open browser at `http://localhost:8501`

📖 **See [UI_GUIDE.md](UI_GUIDE.md) for detailed UI usage**

### Option 2: Command Line

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Qdrant

Edit `configs/.env`:

```env
QDRANT_API_KEY=your_api_key
QDRANT_URI=host=your_qdrant_host
QDRANT_COLLECTION=cross_camera_matching_id
```

### 3. Extract Objects from Video (Optional)

If you have a video with multiple people and want to extract individual person videos:

```bash
python scripts/extract_objects.py \
  --video data/videos/multi_person.mp4 \
  --output-dir ./output_objects \
  --model mot17 \
  --min-frames 10
```

This will create separate video files for each tracked person in `output_objects/<video_name>/object_X.mp4`.

### 4. Register Person (IMPORTANT: Use MOT17)

```bash
# Register first person
python scripts/register_mot17.py \
  --video data/videos/person.mp4 \
  --name "PersonName" \
  --global-id 1 \
  --sample-rate 5

# Register additional person (add to existing collection)
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

**⚠️ IMPORTANT:**
- Each person must have a unique `--global-id` (1, 2, 3, ...)
- Use `--delete-existing` to recreate collection from scratch
- System uses ArcFace for face recognition (requires clear face visibility)

### 5. Run Detection

```bash
python scripts/detect_and_track.py \
  --video data/videos/test.mp4 \
  --model mot17 \
  --threshold 0.8
```

**Note:** Person names are automatically retrieved from Qdrant database. All registered persons will be detected and labeled with their names.

## Parameters

**extract_objects.py:**
- `--video`: Input video with multiple people
- `--output-dir`: Output directory (default: ./output_objects)
- `--model`: `mot17` (recommended) or `yolox`
- `--padding`: Padding pixels around bbox (default: 10)
- `--min-frames`: Minimum frames to save object (default: 10)

**register_mot17.py:**
- `--video`: Video containing person to register
- `--name`: Person name
- `--global-id`: Unique ID for person (required, e.g., 1, 2, 3)
- `--sample-rate`: Extract 1 frame every N frames (default: 5)
- `--delete-existing`: Delete existing collection before registering

**detect_and_track.py:**
- `--video`: Input video
- `--model`: `mot17` (recommended) or `yolox`
- `--threshold`: Similarity threshold (0.8 = strict, 0.7 = loose)
- `--max-frames`: Limit frames for testing (optional)

## Output

```
outputs/
├── videos/     # Annotated video with bbox + labels + FPS counter
├── csv/        # Tracking data (frame_id, track_id, bbox, global_id, similarity, label)
└── logs/       # Detailed per-frame logs (voting, re-verification events)
```

**Video Features:**
- Real-time FPS counter (top-left)
- Frame counter
- Person labels with similarity scores
- Color-coded bounding boxes (green=known, red=unknown)

**CSV Columns:**
- `frame_id`, `track_id`, `x`, `y`, `w`, `h`, `confidence`
- `global_id`, `similarity`, `label`

**Log Events:**
- `[VOTING]`: First-3 frames majority voting results
- `[RE-VERIFY]`: Re-verification at frame 30, 60, 90...

## Project Structure

```
person_reid_system/
├── configs/
│   ├── config.yaml          # Main config
│   └── .env                 # Qdrant credentials
├── core/
│   ├── detector.py          # YOLOX detector
│   ├── tracker.py           # ByteTrack tracker
│   ├── feature_extractor.py # ArcFace face recognition
│   └── vector_db.py         # Qdrant database
├── yolox/                   # ByteTrack YOLOX modules (integrated)
├── exps/                    # YOLOX experiment configs
├── scripts/
│   ├── extract_objects.py   # Extract individual objects from video
│   ├── register_mot17.py    # Register person (USE THIS)
│   └── detect_and_track.py  # Detection pipeline
├── data/
│   ├── videos/              # Input videos
│   └── database/            # reid_database.pkl
├── models/
│   ├── bytetrack_x_mot17.pth.tar  # MOT17 model (756 MB)
│   └── yolox_x.pth                # YOLOX model (756 MB)
└── outputs/                 # Generated outputs
```

## Important Notes

1. **Always use `register_mot17.py`** - Ensures model consistency
2. **Threshold tuning:** 0.8 = strict, 0.7 = balanced, 0.6 = loose
3. **Qdrant sync** - Database synced to local file + Qdrant cloud
4. **ReID Strategy:** First-3 voting + Re-verify every 30 frames for optimal speed/accuracy
5. **Performance:** ~19 FPS (5.3x faster than ReID every frame)

## Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Detailed installation steps
- **[API Documentation](docs/API.md)** - API reference and examples
- **[ReID Strategy](docs/REID_STRATEGY.md)** - Detailed explanation of First-3 + Re-verify strategy
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Usage Examples](docs/USAGE.txt)** - Quick usage examples
- **[Package Manifest](docs/MANIFEST.txt)** - Package structure
- **[Package Info](docs/PACKAGE_INFO.txt)** - Package information

## License

MIT
