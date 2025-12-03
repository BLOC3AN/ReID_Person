"""
Centralized ReID Processor
Handles all ReID logic in one place to avoid code duplication
"""

from loguru import logger
from typing import Dict, Any, Optional
from .reid_logic import process_reid_logic


class ReIDProcessor:
    """
    Centralized processor for ReID operations
    Eliminates code duplication between pipeline.py and zone_processor.py
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full config dict containing 'reid' section
        """
        self.config = config
        self.enabled = config.get('reid', {}).get('enable', True)
        self.extract_interval = 60  # Extract every 60 frames
        
        if not self.enabled:
            logger.warning("⚠️ ReIDProcessor initialized with ReID disabled")
    
    def should_extract_embedding(self, current_frame_count: int) -> bool:
        """
        Determine if we should extract embedding for this frame
        
        Args:
            current_frame_count: Frame count for this track
            
        Returns:
            True if should extract (first frame or every 60 frames)
        """
        return (current_frame_count == 1) or (current_frame_count % self.extract_interval == 0)
    
    def process_track(
        self,
        track_id: int,
        frame_id: int,
        current_frame_count: int,
        frame,
        bbox: list,
        extractor,
        database,
        similarity_threshold: float,
        redis_manager,
        track_labels: Dict[int, Dict[str, Any]],
        log_file=None,
        camera_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Process ReID for a single track
        
        Args:
            track_id: Track ID
            frame_id: Current frame ID
            current_frame_count: Frame count for this track
            frame: Current frame image
            bbox: Bounding box [x, y, w, h]
            extractor: Face extractor instance
            database: Vector database instance
            similarity_threshold: Similarity threshold for matching
            redis_manager: Redis manager instance
            track_labels: In-memory track labels dict
            log_file: Optional file handle for logging
            camera_idx: Camera index
            
        Returns:
            Track info dict with keys: global_id, similarity, label, person_name
        """
        should_extract = self.should_extract_embedding(current_frame_count)
        
        if should_extract and self.enabled and extractor is not None and database is not None:
            # Extract embedding and run ReID logic
            embedding = extractor.extract(frame, bbox)
            process_reid_logic(
                track_id=track_id,
                frame_id=frame_id,
                current_frame_count=current_frame_count,
                embedding=embedding,
                database=database,
                similarity_threshold=similarity_threshold,
                redis_manager=redis_manager,
                track_labels=track_labels,
                log_file=log_file,
                camera_idx=camera_idx,
                use_rerank=self.config.get('reid', {}).get('use_rerank', False),
                rerank_k1=self.config.get('reid', {}).get('rerank_k1', 20),
                rerank_k2=self.config.get('reid', {}).get('rerank_k2', 6),
                rerank_lambda=self.config.get('reid', {}).get('rerank_lambda', 0.3)
            )
        elif should_extract and (not self.enabled or extractor is None or database is None):
            logger.debug(f"⏭️ Skipping ReID for track_id={track_id} (ReID disabled or components not initialized)")
        
        # Return track info (will be Unknown if ReID disabled)
        return track_labels.get(track_id, {
            'global_id': -1,
            'similarity': 0.0,
            'label': 'Unknown',
            'person_name': 'Unknown'
        })
