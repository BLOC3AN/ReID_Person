"""
Person ReID System - Core Modules
"""

from .detection import TritonDetector
from .tracking import ByteTrackWrapper
from .reid import ArcFaceExtractor, FaceRecognitionTriton, process_reid_logic
from .database import QdrantVectorDB, RedisTrackManager
from .zone import ZoneMonitoringService, ZoneTask, ZoneResult, ZoneMonitor, process_video_with_zones, process_multi_stream_with_zones
from .pipeline import PersonReIDPipeline, register_person_mot17, register_person_from_images
from .preloaded_manager import preloaded_manager

__all__ = [
    # Detection
    'TritonDetector',
    # Tracking
    'ByteTrackWrapper',
    # ReID
    'ArcFaceExtractor',
    'FaceRecognitionTriton',
    'process_reid_logic',
    # Database
    'QdrantVectorDB',
    'RedisTrackManager',
    # Zone
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
