"""
Person ReID System - Core Modules
"""

from .detection import TritonDetector
from .reid import FaceRecognitionTriton, process_reid_logic
from .database import QdrantVectorDB, RedisTrackManager
from .pipeline import PersonReIDPipeline, register_person_mot17, register_person_from_images
from .preloaded_manager import preloaded_manager

# Lazy import ArcFaceExtractor (uses torch - InsightFace fallback)
def _get_arcface():
    from .reid import ArcFaceExtractor
    return ArcFaceExtractor

# Lazy import tracking to avoid yolox dependency
def _get_tracking():
    from .tracking import ByteTrackWrapper
    return ByteTrackWrapper

# Lazy import zone modules to avoid heavy dependencies (rtree, tabulate)
def _get_zone_modules():
    from .zone import ZoneMonitoringService, ZoneTask, ZoneResult, ZoneMonitor, process_video_with_zones, process_multi_stream_with_zones
    return ZoneMonitoringService, ZoneTask, ZoneResult, ZoneMonitor, process_video_with_zones, process_multi_stream_with_zones

# Export modules via __getattr__ for lazy loading
def __getattr__(name):
    if name == 'ArcFaceExtractor':
        return _get_arcface()
    
    if name == 'ByteTrackWrapper':
        return _get_tracking()
    
    if name in ['ZoneMonitoringService', 'ZoneTask', 'ZoneResult', 'ZoneMonitor', 'process_video_with_zones', 'process_multi_stream_with_zones']:
        zone_modules = _get_zone_modules()
        module_map = {
            'ZoneMonitoringService': zone_modules[0],
            'ZoneTask': zone_modules[1],
            'ZoneResult': zone_modules[2],
            'ZoneMonitor': zone_modules[3],
            'process_video_with_zones': zone_modules[4],
            'process_multi_stream_with_zones': zone_modules[5],
        }
        return module_map[name]
    
    raise AttributeError(f"module 'core' has no attribute '{name}'")

__all__ = [
    # Detection
    'TritonDetector',
    # Tracking (lazy loaded)
    'ByteTrackWrapper',
    # ReID
    'ArcFaceExtractor',  # Lazy loaded (torch dependency)
    'FaceRecognitionTriton',
    'process_reid_logic',
    # Database
    'QdrantVectorDB',
    'RedisTrackManager',
    # Zone (lazy loaded)
    'ZoneMonitoringService',
    'ZoneTask',
    'ZoneResult',
    'ZoneMonitor',
    'process_video_with_zones',
    'process_multi_stream_with_zones',
    # Pipeline
    'PersonReIDPipeline',
    'register_person_mot17',
    'register_person_from_images',
    # Manager
    'preloaded_manager',
]
