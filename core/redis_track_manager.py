"""
Redis Track Manager for persistent track label storage
Handles track data persistence with TTL-based auto-cleanup
"""

import redis
import json
import time
from typing import Dict, Optional, List
from loguru import logger


class RedisTrackManager:
    """
    Manages track labels in Redis with TTL support
    
    Data Structure:
        Key: track:{track_id}
        Type: Hash
        Fields:
            - global_id: Person ID from database
            - similarity: Current similarity score
            - best_similarity: Best similarity ever seen
            - person_name: Person name
            - label: Display label
            - first_assignment_frame: Frame when first assigned
            - last_update_frame: Frame when last updated
            - timestamp: Unix timestamp of last update
            - camera_idx: Camera index
            - status: Track status (active/unknown)
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 ttl: int = 300, db: int = 0):
        """
        Initialize Redis Track Manager
        
        Args:
            host: Redis server host
            port: Redis server port
            ttl: Time-to-live for track keys in seconds (default: 300s = 5 min)
            db: Redis database number
        """
        self.host = host
        self.port = port
        self.ttl = ttl
        self.db = db
        
        try:
            self.redis = redis.Redis(
                host=host, 
                port=port, 
                db=db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            # Test connection
            self.redis.ping()
            logger.info(f"✅ Redis connected: {host}:{port} (TTL={ttl}s)")
        except redis.ConnectionError as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    def set_track(self, track_id: int, track_data: Dict) -> bool:
        """
        Store track data with TTL
        
        Args:
            track_id: Track ID
            track_data: Track data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"track:{track_id}"
            # Convert numeric values to strings for Redis storage
            data_to_store = {
                k: str(v) if isinstance(v, (int, float)) else v 
                for k, v in track_data.items()
            }
            self.redis.hset(key, mapping=data_to_store)
            self.redis.expire(key, self.ttl)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to set track {track_id}: {e}")
            return False
    
    def get_track(self, track_id: int) -> Optional[Dict]:
        """
        Get track data from Redis
        
        Args:
            track_id: Track ID
            
        Returns:
            Track data dictionary or None if not found
        """
        try:
            key = f"track:{track_id}"
            data = self.redis.hgetall(key)
            if not data:
                return None
            
            # Convert string values back to appropriate types
            result = {}
            for k, v in data.items():
                if k in ['global_id', 'first_assignment_frame', 'last_update_frame', 'camera_idx']:
                    result[k] = int(v)
                elif k in ['similarity', 'best_similarity', 'timestamp']:
                    result[k] = float(v)
                else:
                    result[k] = v
            return result
        except Exception as e:
            logger.error(f"❌ Failed to get track {track_id}: {e}")
            return None
    
    def delete_track(self, track_id: int) -> bool:
        """
        Delete track data from Redis
        
        Args:
            track_id: Track ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"track:{track_id}"
            self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete track {track_id}: {e}")
            return False
    
    def get_all_track_ids(self) -> List[int]:
        """
        Get all active track IDs
        
        Returns:
            List of track IDs
        """
        try:
            keys = self.redis.keys("track:*")
            return [int(key.split(':')[1]) for key in keys]
        except Exception as e:
            logger.error(f"❌ Failed to get all track IDs: {e}")
            return []
    
    def clear_all_tracks(self) -> bool:
        """
        Clear all track data (use with caution)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            keys = self.redis.keys("track:*")
            if keys:
                self.redis.delete(*keys)
            logger.info(f"✅ Cleared {len(keys)} tracks from Redis")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear all tracks: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """
        Get Redis statistics
        
        Returns:
            Dictionary with stats
        """
        try:
            info = self.redis.info()
            keys = self.redis.keys("track:*")
            return {
                'total_tracks': len(keys),
                'redis_memory_used': info.get('used_memory_human', 'N/A'),
                'redis_connected_clients': info.get('connected_clients', 0),
                'ttl': self.ttl
            }
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {}

