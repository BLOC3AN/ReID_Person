"""
Redis Track Manager for persistent track label storage
Handles track data persistence with TTL-based auto-cleanup
Each job has isolated namespace to prevent conflicts
"""

import redis
import json
import time
from typing import Dict, Optional, List
from loguru import logger
import os
from datetime import datetime


class RedisTrackManager:
    """
    Manages track labels in Redis with job-based isolation and TTL support
    
    Data Structure:
        Key: track:{job_id}:{track_id}
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
    
    def __init__(self, job_id: str = None, host: str = None, port: int = None, 
                 ttl: int = None, db: int = None):
        """
        Initialize Redis Track Manager with job isolation
        
        Args:
            job_id: Unique job identifier (auto-generated if not provided)
            host: Redis server host (default: from REDIS_HOST env or 'localhost')
            port: Redis server port (default: from REDIS_PORT env or 6379)
            ttl: Time-to-live for track keys in seconds (default: from REDIS_TTL env or 300s)
            db: Redis database number (default: from REDIS_DB env or 0)
        """
        # Generate job_id if not provided
        if not job_id:
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.job_id = job_id
        self.host = host 
        self.port = port
        self.ttl = ttl 
        self.db = db or os.getenv('REDIS_DB', '0')
        
        try:
            self.redis = redis.Redis(
                host=self.host, 
                port=self.port, 
                db=self.db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            # Test connection
            self.redis.ping()
            logger.info(f"✅ Redis connected: {self.host}:{self.port} (job_id={self.job_id}, TTL={self.ttl}s)")
        except redis.ConnectionError as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    def _get_key(self, track_id: int) -> str:
        """
        Generate Redis key with job_id namespace
        
        Args:
            track_id: Track ID
            
        Returns:
            Redis key string
        """
        return f"track:{self.job_id}:{track_id}"
    
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
            key = self._get_key(track_id)
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
            key = self._get_key(track_id)
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
            key = self._get_key(track_id)
            self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete track {track_id}: {e}")
            return False
    
    def get_all_track_ids(self) -> List[int]:
        """
        Get all active track IDs for current job
        
        Returns:
            List of track IDs
        """
        try:
            keys = self.redis.keys(f"track:{self.job_id}:*")
            return [int(key.split(':')[2]) for key in keys]
        except Exception as e:
            logger.error(f"❌ Failed to get all track IDs: {e}")
            return []
    
    def clear_job_tracks(self) -> bool:
        """
        Clear all tracks for current job
        
        Returns:
            True if successful, False otherwise
        """
        try:
            keys = self.redis.keys(f"track:{self.job_id}:*")
            if keys:
                self.redis.delete(*keys)
            logger.info(f"✅ Cleared {len(keys)} tracks for job {self.job_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear job tracks: {e}")
            return False
    
    def clear_all_tracks(self) -> bool:
        """
        Clear all track data across all jobs (use with caution)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            keys = self.redis.keys("track:*")
            if keys:
                self.redis.delete(*keys)
            logger.info(f"✅ Cleared {len(keys)} tracks from Redis (all jobs)")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear all tracks: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """
        Get Redis statistics for current job

        Returns:
            Dictionary with stats
        """
        try:
            info = self.redis.info()
            job_keys = self.redis.keys(f"track:{self.job_id}:*")
            all_keys = self.redis.keys("track:*")
            return {
                'job_id': self.job_id,
                'job_tracks': len(job_keys),
                'total_tracks_all_jobs': len(all_keys),
                'redis_memory_used': info.get('used_memory_human', 'N/A'),
                'redis_connected_clients': info.get('connected_clients', 0),
                'ttl': self.ttl
            }
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {}

    def set_users_dict(self, users_dict: Dict[int, str], ttl: int = 3600) -> bool:
        """
        Cache users dictionary (global_id -> name mapping) in Redis

        Args:
            users_dict: Dictionary mapping global_id to person name
            ttl: Time-to-live in seconds (default: 3600s = 1 hour)

        Returns:
            True if successful, False otherwise
        """
        try:
            key = "users:dict"
            # Convert int keys to strings for Redis
            data_to_store = {str(k): v for k, v in users_dict.items()}
            self.redis.hset(key, mapping=data_to_store)
            self.redis.expire(key, ttl)
            logger.info(f"✅ Cached {len(users_dict)} users in Redis (TTL={ttl}s)")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to cache users dict: {e}")
            return False

    def get_users_dict(self) -> Dict[int, str]:
        """
        Get cached users dictionary from Redis

        Returns:
            Dictionary mapping global_id to person name, or empty dict if not found
        """
        try:
            key = "users:dict"
            data = self.redis.hgetall(key)
            if not data:
                return {}
            # Convert string keys back to int
            return {int(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"❌ Failed to get users dict: {e}")
            return {}

