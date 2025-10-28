# Installation Guide

## System Requirements

- **OS:** Linux (Ubuntu 18.04+)
- **Python:** 3.8+
- **GPU:** CUDA-capable GPU (recommended)
- **RAM:** 2 GB minimum
- **Disk:** 2 GB for models

## Installation Steps

### 1. Extract Package

```bash
tar -xzf person_reid_system_*.tar.gz
cd person_reid_system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- torch >= 1.7.0
- torchvision >= 0.8.0
- opencv-python >= 4.5.0
- numpy >= 1.19.0
- pyyaml >= 5.4.0
- loguru >= 0.5.0
- qdrant-client >= 1.0.0
- python-dotenv >= 0.19.0
- insightface >= 0.7.3 (ArcFace face recognition)
- onnxruntime-gpu >= 1.12.0 (for GPU acceleration)

### 3. Configure Qdrant

Edit `configs/.env`:

```env
QDRANT_API_KEY=your_api_key_here
QDRANT_URI=host=your_cluster.cloud.qdrant.io
QDRANT_COLLECTION=cross_camera_matching_id
QDRANT_PORT=6333
```

**Get Qdrant credentials:**
1. Sign up at https://cloud.qdrant.io
2. Create a cluster
3. Get API key and cluster URL
4. Update `configs/.env`

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from qdrant_client import QdrantClient; print('Qdrant: OK')"
python -c "from core import ArcFaceExtractor; print('ArcFace: OK')"
```

**Note:** First run will download ArcFace model (~282MB) to `~/.insightface/models/buffalo_l/`

### 5. Check Models

```bash
ls -lh models/
```

Should show:
- `bytetrack_x_mot17.pth.tar` (757 MB)
- `yolox_x.pth` (757 MB)

## Troubleshooting

### CUDA not available

If you don't have GPU:
```bash
# Install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Note:** CPU mode will be slower but still works.

### Qdrant connection failed

Check `configs/.env`:
- API key is correct
- URI format: `host=cluster-id.region.aws.cloud.qdrant.io`
- Collection name matches

### Import errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Next Steps

After installation:
1. See `docs/USAGE.txt` for usage examples
2. See `README.md` for quick start
3. Register your first person with `scripts/register_mot17.py`

