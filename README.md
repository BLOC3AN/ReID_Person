# Person Re-Identification System

Multi-camera Person Re-Identification system using YOLOX detection, ByteTrack tracking, OSNet feature extraction, and Qdrant vector database.

## Pipeline

```
Video → YOLOX MOT17 Detection → ByteTrack Tracking → OSNet Extraction → Qdrant Search → ReID Decision → Output
```

## Quick Start

### 1. Install Dependencies

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

**⚠️ CRITICAL:**
- Always use `register_mot17.py` (NOT `register_person.py`) to ensure model consistency
- Each person must have a unique `--global-id` (1, 2, 3, ...)
- Use `--delete-existing` to recreate collection from scratch

### 5. Run Detection

```bash
python scripts/detect_and_track.py \
  --video data/videos/test.mp4 \
  --model mot17 \
  --known-person "PersonName" \
  --threshold 0.8
```

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
- `--known-person`: Registered person name
- `--threshold`: Similarity threshold (0.8 = strict, 0.7 = loose)
- `--max-frames`: Limit frames for testing (optional)

## Output

```
outputs/
├── videos/     # Annotated video with bbox + labels
├── csv/        # Tracking data
└── logs/       # Detailed per-frame logs
```

## Project Structure

```
person_reid_system/
├── configs/
│   ├── config.yaml          # Main config
│   └── .env                 # Qdrant credentials
├── core/
│   ├── detector.py          # YOLOX detector
│   ├── tracker.py           # ByteTrack tracker
│   ├── feature_extractor.py # OSNet extractor
│   └── vector_db.py         # Qdrant database
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

## Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Detailed installation steps
- **[API Documentation](docs/API.md)** - API reference and examples
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Usage Examples](docs/USAGE.txt)** - Quick usage examples
- **[Package Manifest](docs/MANIFEST.txt)** - Package structure
- **[Package Info](docs/PACKAGE_INFO.txt)** - Package information

## License

MIT
