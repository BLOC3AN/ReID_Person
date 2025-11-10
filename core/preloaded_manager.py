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
                logger.info("🚀 Pre-loading pipeline components...")
                start_time = time.time()
                
                # Load config
                self._load_config(config_path)
                
                # Initialize components in order
                self._init_detector()
                self._init_tracker()
                self._init_extractor()
                self._init_database()
                
                load_time = time.time() - start_time
                logger.info(f"✅ All components loaded in {load_time:.2f}s")
                logger.info("🎯 Pipeline ready for instant inference")
                
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
            config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def _init_detector(self) -> None:
        """Initialize YOLOX detector"""
        logger.info("Loading YOLOX detector...")
        cfg = self.config['detection']
        
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
        logger.info("✓ Detector loaded")
    
    def _init_tracker(self) -> None:
        """Initialize ByteTrack tracker"""
        logger.info("Loading ByteTrack tracker...")
        cfg = self.config['tracking']
        
        self.tracker = ByteTrackWrapper(
            track_thresh=cfg['track_thresh'],
            track_buffer=cfg['track_buffer'],
            match_thresh=cfg['match_thresh'],
            frame_rate=30,
            mot20=cfg['mot20']
        )
        logger.info("✓ Tracker loaded")
    
    def _init_extractor(self) -> None:
        """Initialize ArcFace feature extractor"""
        logger.info("Loading ArcFace extractor...")
        cfg = self.config['reid']
        
        self.extractor = ArcFaceExtractor(
            model_name=cfg.get('arcface_model_name', 'buffalo_l'),
            use_cuda=cfg['use_cuda'],
            feature_dim=cfg['feature_dim']
        )
        logger.info("✓ Extractor loaded")
    
    def _init_database(self) -> None:
        """Initialize Qdrant vector database"""
        logger.info("Loading vector database...")
        cfg = self.config['database']
        
        self.database = QdrantVectorDB(
            use_qdrant=cfg['use_qdrant'],
            collection_name=cfg['qdrant_collection'],
            max_embeddings_per_person=cfg['max_embeddings_per_person'],
            embedding_dim=cfg['embedding_dim']
        )
        
        # Load existing database
        db_file = Path(__file__).parent.parent / "data" / "database" / "reid_database.pkl"
        if db_file.exists():
            logger.info(f"Loading database from {db_file}")
            self.database.load_from_file(str(db_file))
            logger.info(f"Database loaded: {self.database.get_person_count()} persons")
        
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
