"""
Qdrant Vector Database Integration for ReID
Persistent storage and similarity search for person embeddings using Qdrant only
"""

import os
import uuid
import numpy as np
from loguru import logger
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv


class QdrantVectorDB:
    """
    Vector database for storing and searching person embeddings using Qdrant
    """
    
    def __init__(self, qdrant_url=None, collection_name=None,
                 embedding_dim=512, api_key=None, use_grpc=False):
        """
        Args:
            qdrant_url: Qdrant server URL (auto-load from .env if None)
            collection_name: Collection name (auto-load from .env if None)
            embedding_dim: Embedding dimension (512 for ArcFace)
            api_key: Qdrant API key (auto-load from .env if None)
            use_grpc: Use gRPC protocol instead of HTTP (default: False)
        
        Raises:
            RuntimeError: If Qdrant client initialization fails
        """
        # Load from .env if not provided
        env_path = Path(__file__).parent.parent / "configs" / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        self.use_grpc = use_grpc
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "cross_camera_matching_id")
        self.embedding_dim = embedding_dim
        self.next_global_id = 1
        self.person_metadata = {}

        # Qdrant client (required)
        qdrant_url = qdrant_url or os.getenv("QDRANT_URI", "http://localhost:6333")
        api_key = api_key or os.getenv("QDRANT_API_KEY")
        use_grpc = use_grpc or os.getenv("QDRANT_USE_GRPC", "false").lower() == "true"
        self._init_qdrant(qdrant_url, api_key, use_grpc)
    
    def _init_qdrant(self, qdrant_url, api_key=None, use_grpc=False):
        """Initialize Qdrant client

        Args:
            qdrant_url: Qdrant server URL (http://host:port or host:port for gRPC)
            api_key: Optional API key for authentication
            use_grpc: Use gRPC protocol instead of HTTP
        
        Raises:
            RuntimeError: If Qdrant client initialization fails
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            # Parse URL if it contains "host=" prefix
            if qdrant_url.startswith("host="):
                qdrant_url = qdrant_url.replace("host=", "https://")

            # Initialize client with gRPC or HTTP
            if use_grpc:
                if qdrant_url.startswith("http://"):
                    qdrant_url = qdrant_url.replace("http://", "")
                elif qdrant_url.startswith("https://"):
                    qdrant_url = qdrant_url.replace("https://", "")

                # Initialize with gRPC
                if api_key:
                    self.client = QdrantClient(host=qdrant_url.split(":")[0],
                                              port=int(qdrant_url.split(":")[1]) if ":" in qdrant_url else 6334,
                                              api_key=api_key, grpc_port=6334)
                else:
                    self.client = QdrantClient(host=qdrant_url.split(":")[0],
                                              port=int(qdrant_url.split(":")[1]) if ":" in qdrant_url else 6334,
                                              grpc_port=6334)
                logger.info(f"✅ Initialized Qdrant client with gRPC protocol")
            else:
                # Initialize with HTTP (default)
                if api_key:
                    self.client = QdrantClient(url=qdrant_url, api_key=api_key)
                else:
                    self.client = QdrantClient(url=qdrant_url)
                logger.info(f"✅ Initialized Qdrant client with HTTP protocol")

            # Test connection and create collection if needed
            try:
                self.client.get_collection(self.collection_name)
                logger.info(f"✅ Connected to Qdrant collection: {self.collection_name}")
            except Exception:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qdrant client: {e}")
    
    def add_embedding(self, global_id: int, embedding: np.ndarray,
                     metadata: Optional[Dict] = None):
        """
        Add embedding for person
        Args:
            global_id: Global person ID
            embedding: Embedding vector (512,)
            metadata: Optional metadata (camera_id, track_id, etc.)
        """
        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Store metadata
        if metadata:
            self.person_metadata[global_id] = metadata

        # Store in Qdrant
        point_id = str(uuid.uuid4())
        payload = metadata or {}
        payload['global_id'] = global_id

        self.client.upsert(
            collection_name=self.collection_name,
            points=[{
                "id": point_id,
                "vector": embedding.tolist(),
                "payload": payload
            }]
        )
    
    def find_best_match(self, embedding: np.ndarray, threshold: float = 0.8,
                       top_k: int = 1) -> List[Tuple[int, float, str]]:
        """
        Find best matching persons using Qdrant vector search
        
        Priority 2 (first): Query with UI threshold, if results exist → return
        Priority 1 (fallback): If no Priority 2 results, query with 0.5, if all same GID → match
        
        Args:
            embedding: Query embedding (512,)
            threshold: Cosine similarity threshold (0-1, default 0.8)
            top_k: Return top K matches
        Returns:
            List of (global_id, similarity, name) tuples
        """
        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        logger.info("="*50)
        logger.info(f"🔍 Starting ReID matching (UI threshold={threshold})")
        logger.info("="*50)

        # Priority 2: Query with UI threshold (try strict matching first)
        logger.debug(f"[Priority 2] Querying Qdrant with UI threshold={threshold}...")
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding.tolist(),
            limit=top_k * 10,
            score_threshold=threshold
        )

        logger.debug(f"[Priority 2] Qdrant returned {len(results.points)} results")

        if results.points:
            # Group by global_id and get best score + name for each person
            best_per_person = {}
            for r in results.points:
                global_id = r.payload.get('global_id', r.id)
                name = r.payload.get('name', f'Person_{global_id}')
                logger.debug(f"[Priority 2]   Result: GID={global_id}, name={name}, score={r.score:.4f}")
                if global_id not in best_per_person or r.score > best_per_person[global_id][1]:
                    best_per_person[global_id] = (global_id, r.score, name)

            # Return top K persons
            matches = sorted(best_per_person.values(), key=lambda x: x[1], reverse=True)[:top_k]
            logger.info("="*50)
            logger.info(f"✅ [Priority 2] MATCH: {[(f'GID={gid} ({name})', f'score={score:.4f}') for gid, score, name in matches]}")
            logger.info("="*50)
            return matches

        # Priority 1: Fallback with lower threshold (0.5) if Priority 2 found nothing
        logger.debug(f"[Priority 2] No results found, falling back to Priority 1")
        
        priority1_threshold = 0.5
        logger.debug(f"[Priority 1] Querying Qdrant with threshold={priority1_threshold}...")
        results_p1 = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding.tolist(),
            limit=top_k * 10,
            score_threshold=priority1_threshold
        )

        logger.debug(f"[Priority 1] Qdrant returned {len(results_p1.points)} results")

        if results_p1.points:
            all_global_ids = [r.payload.get('global_id', r.id) for r in results_p1.points]
            unique_gids = set(all_global_ids)
            
            logger.debug(f"[Priority 1] Unique GIDs: {unique_gids}")
            
            if len(unique_gids) == 1:  # All same global_id
                avg_score = np.mean([r.score for r in results_p1.points])
                gid = all_global_ids[0]
                name = results_p1.points[0].payload.get('name', f'Person_{gid}')
                
                logger.debug(f"[Priority 1] All {len(results_p1.points)} results have same GID={gid} ({name})")
                logger.debug(f"[Priority 1] Scores: {[f'{r.score:.4f}' for r in results_p1.points[:5]]}{'...' if len(results_p1.points) > 5 else ''}")
                logger.debug(f"[Priority 1] avg_score={avg_score:.4f}, checking if >= {priority1_threshold}")
                
                if avg_score >= priority1_threshold:
                    logger.info("="*50)
                    logger.info(f"✅ [Priority 1] MATCH: GID={gid} ({name}), avg_score={avg_score:.4f}")
                    logger.info(f"   Bypassing UI threshold={threshold} (all {len(results_p1.points)} results agree)")
                    logger.info("="*50)
                    return [(gid, avg_score, name)]
                else:
                    logger.debug(f"[Priority 1] avg_score={avg_score:.4f} < {priority1_threshold}, skipping")
            else:
                logger.debug(f"[Priority 1] Multiple GIDs found: {unique_gids}, no match")

        # No match found
        logger.info("="*50)
        logger.info(f"❌ NO MATCH: No results found in both Priority 2 and Priority 1")
        logger.info("="*50)
        return []
    
    def create_new_person(self, embedding: np.ndarray,
                         metadata: Optional[Dict] = None) -> int:
        """Create new person and return global ID"""
        global_id = self.next_global_id
        self.next_global_id += 1
        self.add_embedding(global_id, embedding, metadata)
        return global_id
    
    def get_person_count(self) -> int:
        """Get number of persons in database"""
        # Get unique global_ids from Qdrant
        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,  # Get all points
            with_payload=True,
            with_vectors=False
        )[0]

        unique_global_ids = set()
        for point in results:
            gid = point.payload.get('global_id', point.id)
            unique_global_ids.add(gid)

        return len(unique_global_ids)

    def sync_metadata_from_qdrant(self) -> int:
        """
        Sync person metadata from Qdrant to in-memory storage
        This is used to populate self.person_metadata

        Returns:
            Number of persons synced
        """
        # Get all points from Qdrant
        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,  # Get all points
            with_payload=True,
            with_vectors=False
        )[0]

        # Group by global_id and extract metadata
        persons = {}
        for point in results:
            gid = point.payload.get('global_id')
            if gid is None:
                continue

            name = point.payload.get('name', f'Person_{gid}')

            if gid not in persons:
                persons[gid] = {
                    'name': name,
                    'global_id': gid,
                    'source': point.payload.get('source', 'unknown'),
                    'num_embeddings': 0
                }

            persons[gid]['num_embeddings'] += 1

        # Update person_metadata
        self.person_metadata.clear()
        for gid, metadata in persons.items():
            self.person_metadata[gid] = metadata

        # Update next_global_id
        if persons:
            max_id = max(persons.keys())
            self.next_global_id = max_id + 1

        logger.info(f"✅ Synced {len(persons)} persons from Qdrant")
        for gid, meta in sorted(persons.items()):
            logger.info(f"   - GID {gid} ({meta['name']}): {meta['num_embeddings']} embeddings")

        return len(persons)

    def get_stats(self) -> Dict:
        """Get database statistics"""
        # Get all points from Qdrant
        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )[0]

        unique_global_ids = set()
        for point in results:
            gid = point.payload.get('global_id', point.id)
            unique_global_ids.add(gid)

        num_persons = len(unique_global_ids)
        total_embeddings = len(results)

        return {
            'num_persons': num_persons,
            'total_embeddings': total_embeddings,
            'avg_embeddings_per_person': total_embeddings / num_persons if num_persons > 0 else 0,
            'next_global_id': self.next_global_id,
        }

    def clear(self):
        """Clear database"""
        self.person_metadata.clear()
        self.next_global_id = 1

        from qdrant_client.models import Distance, VectorParams
        self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE
            )
        )

