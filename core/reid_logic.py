#!/usr/bin/env python3
"""
ReID Logic Processing Module
Centralized ReID decision logic to avoid code duplication
"""

import time
from typing import Optional, Dict, Any
from loguru import logger


def process_reid_logic(
    track_id: int,
    frame_id: int,
    current_frame_count: int,
    embedding,
    database,
    similarity_threshold: float,
    redis_manager,
    track_labels: Dict[int, Dict[str, Any]],
    log_file=None,
    camera_idx: int = 0
) -> Optional[Dict[str, Any]]:
    """
    Process ReID logic with Redis storage
    
    Decision Logic:
    - Frame 1: Extract → Assign (first time)
    - Frame 60+: Extract → Compare with old
    - Update only if new_sim > old_sim AND passes threshold
    - Reset when track lost (TTL expires)
    
    Args:
        track_id: Track ID
        frame_id: Current frame ID
        current_frame_count: Frame count for this track
        embedding: Feature embedding vector
        database: Vector database instance
        similarity_threshold: Similarity threshold for matching
        redis_manager: Redis manager instance (optional)
        track_labels: In-memory track labels dict
        log_file: Optional file handle for logging
        camera_idx: Camera index (default: 0)
    
    Returns:
        Updated track_data dict or None if no match
    """
    matches = database.find_best_match(embedding, threshold=0.0, top_k=1)
    
    if not matches:
        return None
    
    new_global_id, new_similarity, new_person_name = matches[0]
    
    # Get old data from Redis or in-memory
    old_data = None
    if redis_manager:
        old_data = redis_manager.get_track(track_id)
    if old_data is None:
        old_data = track_labels.get(track_id)
    
    # Decision logic
    if new_similarity >= similarity_threshold:
        if old_data is None:
            # New track (first time or recovered after TTL)
            new_data = {
                'global_id': new_global_id,
                'similarity': new_similarity,
                'best_similarity': new_similarity,
                'person_name': new_person_name,
                'label': new_person_name,
                'first_assignment_frame': frame_id,
                'last_update_frame': frame_id,
                'timestamp': time.time(),
                'camera_idx': camera_idx,
                'status': 'active'
            }
            log_msg = f"✅ Track {track_id}: ASSIGN {new_person_name} (ID:{new_global_id}, sim={new_similarity:.4f}, frame={frame_id})"
            if log_file:
                log_file.write(log_msg + "\n")
            logger.info(log_msg)
            
        else:
            # Existing track
            old_global_id = old_data.get('global_id', -1)
            old_similarity = old_data.get('similarity', 0.0)
            old_person_name = old_data.get('person_name', 'Unknown')
            
            if new_similarity > old_similarity:
                # UPDATE: Better match found
                new_data = old_data.copy()
                new_data['global_id'] = new_global_id
                new_data['similarity'] = new_similarity
                new_data['best_similarity'] = max(new_similarity, old_data.get('best_similarity', 0.0))
                new_data['person_name'] = new_person_name
                new_data['label'] = new_person_name
                new_data['last_update_frame'] = frame_id
                new_data['timestamp'] = time.time()
                
                log_msg = f"✅ Track {track_id}: UPDATE {old_person_name} → {new_person_name} (ID:{old_global_id} → {new_global_id}, sim={old_similarity:.4f} → {new_similarity:.4f}, frame={frame_id})"
                if log_file:
                    log_file.write(log_msg + "\n")
                logger.info(f"✅ Track {track_id}: UPDATE {old_person_name} → {new_person_name} (sim {old_similarity:.4f} → {new_similarity:.4f})")
                
            else:
                # REJECT: Same or worse match
                new_data = old_data.copy()
                new_data['last_update_frame'] = frame_id
                new_data['timestamp'] = time.time()
                
                log_msg = f"❌ Track {track_id}: REJECT {new_person_name} (ID:{new_global_id}, sim={new_similarity:.4f} < {old_similarity:.4f}, frame={frame_id})"
                if log_file:
                    log_file.write(log_msg + "\n")
                logger.debug(f"❌ Track {track_id}: REJECT {new_person_name} (sim {new_similarity:.4f} < {old_similarity:.4f})")
    
    else:
        # FAIL_THRESHOLD
        if old_data is None:
            new_data = {
                'global_id': -1,
                'similarity': 0.0,
                'best_similarity': 0.0,
                'person_name': 'Unknown',
                'label': 'Unknown',
                'first_assignment_frame': frame_id,
                'last_update_frame': frame_id,
                'timestamp': time.time(),
                'camera_idx': camera_idx,
                'status': 'unknown'
            }
            log_msg = f"❌ Track {track_id}: FAIL_THRESHOLD {new_person_name} (ID:{new_global_id}, sim={new_similarity:.4f} < {similarity_threshold:.4f}, frame={frame_id})"
            if log_file:
                log_file.write(log_msg + "\n")
            logger.debug(f"❌ Track {track_id}: FAIL_THRESHOLD {new_person_name} (sim {new_similarity:.4f} < {similarity_threshold:.4f})")
        else:
            new_data = old_data.copy()
            new_data['last_update_frame'] = frame_id
            new_data['timestamp'] = time.time()
            
            log_msg = f"❌ Track {track_id}: FAIL_THRESHOLD {new_person_name} (ID:{new_global_id}, sim={new_similarity:.4f} < {similarity_threshold:.4f}, frame={frame_id})"
            if log_file:
                log_file.write(log_msg + "\n")
            logger.debug(f"❌ Track {track_id}: FAIL_THRESHOLD {new_person_name} (sim {new_similarity:.4f} < {similarity_threshold:.4f})")
    
    # Save to Redis and in-memory
    if redis_manager:
        redis_manager.set_track(track_id, new_data)
    track_labels[track_id] = new_data
    
    return new_data

