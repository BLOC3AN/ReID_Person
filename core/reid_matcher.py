"""
Unified ReID Matching Logic
Cross-camera person re-identification using cosine distance
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple
from loguru import logger

try:
    from .vector_db import QdrantVectorDB
except ImportError:
    from vector_db import QdrantVectorDB


class ReIDMatcher:
    """Match persons across cameras using ReID embeddings"""
    
    def __init__(self, distance_threshold=0.8, metric='cosine',
                 use_qdrant=False, qdrant_url="http://localhost:6333", use_grpc=False):
        """
        Args:
            distance_threshold: Cosine similarity threshold (0-1)
                - 0.6-0.7: Loose (high recall)
                - 0.7-0.8: Balanced (recommended)
                - 0.8-0.9: Strict (high precision)
            metric: Distance metric ('cosine', 'euclidean')
            use_qdrant: Use Qdrant backend
            qdrant_url: Qdrant server URL
            use_grpc: Use gRPC protocol instead of HTTP
        """
        self.distance_threshold = distance_threshold
        self.metric = metric
        self.db = QdrantVectorDB(
            use_qdrant=use_qdrant,
            qdrant_url=qdrant_url,
            embedding_dim=512,
            use_grpc=use_grpc
        )
        self.camera_track_to_global = {}  # {(camera_id, track_id): global_id}
    
    def match_single_embedding(self, embedding: np.ndarray,
                              camera_id: str, track_id: int) -> int:
        """
        Match single embedding to global ID
        Args:
            embedding: Feature vector (512,)
            camera_id: Camera identifier
            track_id: Local track ID
        Returns:
            Global ID (new or existing)
        """
        track_key = (camera_id, track_id)
        
        # Check if already matched
        if track_key in self.camera_track_to_global:
            return self.camera_track_to_global[track_key]
        
        # Try to find match
        matches = self.db.find_best_match(embedding, self.distance_threshold, top_k=1)

        if matches:
            global_id, distance, name = matches[0]
            logger.debug(f"Match found: {camera_id}_{track_id} → Global_{global_id} ({name}, dist={distance:.4f})")
        else:
            # Create new person
            global_id = self.db.create_new_person(
                embedding,
                metadata={'camera_id': camera_id, 'track_id': track_id}
            )
            logger.debug(f"New person: {camera_id}_{track_id} → Global_{global_id}")
        
        # Store mapping
        self.camera_track_to_global[track_key] = global_id
        self.db.add_embedding(global_id, embedding)
        
        return global_id
    
    def match_batch_embeddings(self, embeddings: Dict[Tuple[str, int], np.ndarray]) -> Dict[Tuple[str, int], int]:
        """
        Match batch of embeddings
        Args:
            embeddings: {(camera_id, track_id): embedding_vector}
        Returns:
            {(camera_id, track_id): global_id}
        """
        mapping = {}
        for (camera_id, track_id), embedding in embeddings.items():
            global_id = self.match_single_embedding(embedding, camera_id, track_id)
            mapping[(camera_id, track_id)] = global_id
        return mapping
    
    def match_cameras(self, features_list: List[Dict[int, List[np.ndarray]]],
                     camera_ids: List[str]) -> Dict[str, int]:
        """
        Match persons across multiple cameras (batch mode)
        Args:
            features_list: List of {track_id: [feat1, feat2, ...]} for each camera
            camera_ids: List of camera identifiers
        Returns:
            {f"{camera_id}_{track_id}": global_id}
        """
        if len(features_list) == 0:
            return {}
        
        mapping = {}
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🔗 CROSS-CAMERA MATCHING (threshold={self.distance_threshold})")
        logger.info(f"{'='*70}")
        
        for cam_id, avg_feats in zip(camera_ids, features_list):
            logger.info(f"  {cam_id}: {len(avg_feats)} tracks")
        logger.info(f"{'='*70}\n")
        
        # Process each camera
        for camera_id, features_dict in zip(camera_ids, features_list):
            logger.info(f"\n📷 Processing {camera_id}:")
            
            for track_id, feats in features_dict.items():
                # Calculate average feature
                if isinstance(feats, list):
                    avg_feat = np.mean(feats, axis=0)
                else:
                    avg_feat = feats
                
                # Match
                global_id = self.match_single_embedding(avg_feat, camera_id, track_id)
                mapping[f"{camera_id}_{track_id}"] = global_id
                
                logger.info(f"  Track {track_id} → Global_ID {global_id}")
        
        return mapping
    
    def match_pair_cameras(self, features_cam1: Dict[int, List[np.ndarray]],
                          features_cam2: Dict[int, List[np.ndarray]],
                          camera_id1: str = "cam1",
                          camera_id2: str = "cam2") -> Dict[str, int]:
        """
        Match persons between 2 cameras
        Args:
            features_cam1: {track_id: [feat1, feat2, ...]}
            features_cam2: {track_id: [feat1, feat2, ...]}
            camera_id1: ID for camera 1
            camera_id2: ID for camera 2
        Returns:
            {f"{camera_id}_{track_id}": global_id}
        """
        return self.match_cameras(
            [features_cam1, features_cam2],
            [camera_id1, camera_id2]
        )
    
    def get_global_tracks(self, mapping: Dict[str, int]) -> Dict[int, List[str]]:
        """
        Get all camera tracks for each global ID
        Args:
            mapping: {f"{camera_id}_{track_id}": global_id}
        Returns:
            {global_id: [f"{camera_id}_{track_id}", ...]}
        """
        global_tracks = {}
        for track_key, global_id in mapping.items():
            if global_id not in global_tracks:
                global_tracks[global_id] = []
            global_tracks[global_id].append(track_key)
        return global_tracks
    
    def get_stats(self) -> Dict:
        """Get matcher statistics"""
        return self.db.get_stats()
    
    def reset(self):
        """Reset matcher state"""
        self.camera_track_to_global.clear()
        self.db.clear()

