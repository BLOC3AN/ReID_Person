# Lazy import ArcFaceExtractor to avoid torch dependency
# from .feature_extractor import ArcFaceExtractor

from .face_recognition_triton import FaceRecognitionTriton
from .reid_logic import process_reid_logic

# Lazy load ArcFaceExtractor
def __getattr__(name):
    if name == 'ArcFaceExtractor':
        from .feature_extractor import ArcFaceExtractor
        return ArcFaceExtractor
    raise AttributeError(f"module 'core.reid' has no attribute '{name}'")

__all__ = ['ArcFaceExtractor', 'FaceRecognitionTriton', 'process_reid_logic']
