#!/usr/bin/env python3
"""
Preloaded Pipeline Manager
Singleton manager for pre-loaded components to avoid lazy loading delays
"""

import threading
import time
import yaml
from pathlib import Path
from loguru import logger
from typing import Optional

from .detector import YOLOXDetector
from .detector_triton import TritonDetector
from .tracker import ByteTrackWrapper
from .feature_extractor import ArcFaceExtractor
from .vector_db import QdrantVectorDB


class PreloadedPipelineManager:
    """
    Singleton manager for pre-loaded pipeline components
    Ensures components are loaded once at service startup
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        # Components
        self.detector = None
        self.tracker = None
        self.extractor = None
        self.database = None
        self.config = None
        
        # State
        self._initialized = False
        self._loading = False
    
    def initialize(self, config_path: Optional[str] = None) -> None:
        """
        Initialize all components (thread-safe)
        
        Args:
            config_path: Path to config file (optional)
        """
        if self._initialized:
            logger.debug("Components already initialized")
            return
            
        with self._lock:
            if self._initialized:  # Double-check locking
                return
                
            if self._loading:
                logger.warning("Components are currently being loaded by another thread")
                return
                
            self._loading = True
            
            try:
                logger.info("=" * 80)
                logger.info("🚀 PRE-LOADING PIPELINE COMPONENTS")
                logger.info("=" * 80)
                overall_start = time.time()

                # Load config
                config_start = time.time()
                self._load_config(config_path)
                logger.info(f"✓ Config loaded in {time.time() - config_start:.2f}s")

                # Initialize components in order with timing
                detector_start = time.time()
                self._init_detector()
                logger.info(f"✓ Detector loaded in {time.time() - detector_start:.2f}s")

                tracker_start = time.time()
                self._init_tracker()
                logger.info(f"✓ Tracker loaded in {time.time() - tracker_start:.2f}s")

                extractor_start = time.time()
                self._init_extractor()
                logger.info(f"✓ Extractor loaded in {time.time() - extractor_start:.2f}s")

                database_start = time.time()
                self._init_database()
                logger.info(f"✓ Database loaded in {time.time() - database_start:.2f}s")

                load_time = time.time() - overall_start
                logger.info("=" * 80)
                logger.info(f"✅ ALL COMPONENTS LOADED IN {load_time:.2f}s")
                logger.info("🎯 Pipeline ready for instant inference!")
                logger.info("=" * 80)

                self._initialized = True

            except Exception as e:
                logger.error(f"❌ Failed to initialize components: {e}")
                self._cleanup_partial_init()
                raise
            finally:
                self._loading = False
    
    def _load_config(self, config_path: Optional[str]) -> None:
        """Load configuration file"""
        if config_path is None:
            # Priority: configs/config.yaml (for services), then .streamlit/configs/config.yaml (for UI)
            primary_path = Path(__file__).parent.parent / "configs" / "config.yaml"
            fallback_path = Path(__file__).parent.parent / ".streamlit" / "configs" / "config.yaml"

            if primary_path.exists():
                config_path = primary_path
            elif fallback_path.exists():
                config_path = fallback_path
            else:
                raise FileNotFoundError(f"Config not found at {primary_path} or {fallback_path}")

        logger.info(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def _init_detector(self) -> None:
        """Initialize detector (PyTorch, TensorRT, or Triton)"""
        cfg = self.config['detection']
        backend = cfg.get('backend', 'pytorch').lower()

        logger.info(f"Loading detector with backend: {backend}")

        if backend == 'triton':
            # Triton Inference Server backend
            triton_cfg = cfg['triton']

            self.detector = TritonDetector(
                triton_url=triton_cfg['url'],
                model_name=triton_cfg['model_name'],
                model_version=triton_cfg.get('model_version', ''),
                conf_thresh=cfg['conf_threshold'],
                nms_thresh=cfg['nms_threshold'],
                test_size=tuple(cfg['test_size']),
                timeout=triton_cfg.get('timeout', 10.0),
                verbose=triton_cfg.get('verbose', False)
            )
            logger.info("✓ Triton Detector loaded")
            logger.info(f"  Server: {triton_cfg['url']}")
            logger.info(f"  Model: {triton_cfg['model_name']}")

        else:
            # PyTorch backend (default)
            model_type = cfg.get('model_type', 'mot17')
            if model_type == 'mot17':
                model_path = Path(__file__).parent.parent / cfg['model_path_mot17']
            else:
                model_path = Path(__file__).parent.parent / cfg['model_path_yolox']

            self.detector = YOLOXDetector(
                model_path=str(model_path),
                model_type=model_type,
                device=cfg['device'],
                fp16=cfg['fp16'],
                conf_thresh=cfg['conf_threshold'],
                nms_thresh=cfg['nms_threshold'],
                test_size=tuple(cfg['test_size'])
            )
            logger.info("✓ PyTorch Detector loaded")
    
    def _init_tracker(self) -> None:
        """Initialize ByteTrack tracker"""
        logger.info("Loading ByteTrack tracker...")
        cfg = self.config['tracking']

        self.tracker = ByteTrackWrapper(
            track_thresh=cfg.get('track_thresh', 0.5),
            track_buffer=cfg.get('track_buffer', 30),
            match_thresh=cfg.get('match_thresh', 0.8),
            frame_rate=30,
            mot20=cfg.get('mot20', False)
        )
        logger.info("✓ Tracker loaded")
    
    def _init_extractor(self) -> None:
        """Initialize ArcFace feature extractor (Triton Pipeline, Triton, or InsightFace)"""
        logger.info("Loading ArcFace extractor...")
        cfg = self.config['reid']
        reid_backend = cfg.get('backend', 'insightface')
        logger.info(f"  Backend: {reid_backend}")

        if reid_backend == 'triton_pipeline':
            # Triton pipeline: SCRFD face detector + ArcFace
            from .face_recognition_triton import FaceRecognitionTriton

            triton_cfg = cfg.get('triton', {})
            triton_url = triton_cfg.get('url', 'localhost:8101')
            arcface_model = triton_cfg.get('arcface_model', 'arcface_tensorrt')
            face_detector_model = triton_cfg.get('face_detector_model', 'scrfd_10g')
            feature_dim = triton_cfg.get('feature_dim', 512)
            face_conf_threshold = triton_cfg.get('face_conf_threshold', 0.5)

            self.extractor = FaceRecognitionTriton(
                triton_url=triton_url,
                face_detector_model=face_detector_model,
                arcface_model=arcface_model,
                feature_dim=feature_dim,
                face_conf_threshold=face_conf_threshold
            )
            logger.info("✓ Triton Face Recognition Pipeline loaded")
            logger.info(f"  Server: {triton_url}")
            logger.info(f"  Face Detector: {face_detector_model}")
            logger.info(f"  ArcFace: {arcface_model}")

        else:
            # InsightFace backend (default)
            triton_cfg = cfg.get('triton', {})
            self.extractor = ArcFaceExtractor(
                model_name=cfg.get('arcface_model_name', 'buffalo_l'),
                use_cuda=cfg.get('use_cuda', True),
                feature_dim=cfg.get('feature_dim', 512),
                face_conf_thresh=triton_cfg.get('face_conf_threshold', 0.5)
            )
            logger.info("✓ InsightFace ArcFace Extractor loaded")
    
    def _init_database(self) -> None:
        """Initialize Qdrant vector database"""
        logger.info("Loading vector database...")
        cfg = self.config['database']

        self.database = QdrantVectorDB(
            use_qdrant=cfg['use_qdrant'],
            collection_name=cfg['qdrant_collection'],
            max_embeddings_per_person=cfg['max_embeddings_per_person'],
            embedding_dim=cfg['embedding_dim'],
            use_grpc=cfg.get('use_grpc', False)
        )

        # Sync metadata from Qdrant (if available)
        if self.database.client:
            person_count = self.database.sync_metadata_from_qdrant()
            logger.info(f"✅ Synced metadata from Qdrant: {person_count} persons")

        logger.info("✓ Database loaded")
    
    def _cleanup_partial_init(self) -> None:
        """Cleanup partially initialized components"""
        logger.warning("Cleaning up partially initialized components")
        self.detector = None
        self.tracker = None
        self.extractor = None
        self.database = None
        self.config = None
    
    def is_initialized(self) -> bool:
        """Check if components are initialized"""
        return self._initialized
    
    def get_components(self) -> tuple:
        """
        Get all initialized components
        
        Returns:
            Tuple of (detector, tracker, extractor, database, config)
            
        Raises:
            RuntimeError: If components are not initialized
        """
        if not self._initialized:
            raise RuntimeError("Components not initialized. Call initialize() first.")
        
        return self.detector, self.tracker, self.extractor, self.database, self.config


# Global singleton instance
preloaded_manager = PreloadedPipelineManager()
