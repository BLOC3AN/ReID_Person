from .detector_triton import TritonDetector

# Keep YOLOXDetector for future use (body detection in registration)
# Import only when needed to avoid loading PyTorch unnecessarily
def get_yolox_detector(*args, **kwargs):
    from .detector import YOLOXDetector
    return YOLOXDetector(*args, **kwargs)

__all__ = ['TritonDetector', 'get_yolox_detector']
