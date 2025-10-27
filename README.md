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

### 3. Register Person (IMPORTANT: Use MOT17)

```bash
python scripts/register_mot17.py \
  --video data/videos/person.mp4 \
  --name "PersonName" \
  --sample-rate 5
```

**⚠️ CRITICAL:** Always use `register_mot17.py` (NOT `register_person.py`) to ensure model consistency.

### 4. Run Detection

```bash
python scripts/detect_and_track.py \
  --video data/videos/test.mp4 \
  --model mot17 \
  --known-person "PersonName" \
  --threshold 0.8
```

## Parameters

**register_mot17.py:**
- `--video`: Video containing person to register
- `--name`: Person name
- `--sample-rate`: Extract 1 frame every N frames (default: 5)

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
