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
        
        Priority 1: If all top K results have same global_id and avg_score >= 0.5 → match
        Priority 2: Group by global_id and return best score per person
        
        Args:
            embedding: Query embedding (512,)
            threshold: Cosine similarity threshold (0-1, default 0.8)
            top_k: Return top K matches
        Returns:
            List of (global_id, similarity, name) tuples
        """
        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Query Qdrant
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding.tolist(),
            limit=top_k * 10,  # Get more results to group by global_id
            score_threshold=threshold  # Qdrant uses similarity directly
        )

        logger.debug(f"Qdrant query_points returned {len(results.points)} results (threshold={threshold})")

        if not results.points:
            return []

        # Priority 1: Check if all top K results have same global_id
        top_results = results.points[:top_k]
        global_ids = [r.payload.get('global_id', r.id) for r in top_results]
        
        if len(set(global_ids)) == 1:  # All same global_id
            avg_score = np.mean([r.score for r in top_results])
            if avg_score >= 0.5:
                gid = global_ids[0]
                name = top_results[0].payload.get('name', f'Person_{gid}')
                logger.debug(f"[Priority 1] All top {len(top_results)} results have same GID={gid}, avg_score={avg_score:.4f}")
                return [(gid, avg_score, name)]

        # Priority 2: Group by global_id and get best score + name for each person
        best_per_person = {}
        for r in results.points:
            global_id = r.payload.get('global_id', r.id)
            name = r.payload.get('name', f'Person_{global_id}')
            logger.debug(f"  Result: GID={global_id}, name={name}, score={r.score:.4f}")
            if global_id not in best_per_person or r.score > best_per_person[global_id][1]:
                best_per_person[global_id] = (global_id, r.score, name)

        # Return top K persons
        matches = sorted(best_per_person.values(), key=lambda x: x[1], reverse=True)[:top_k]
        logger.info("="*50)
        logger.debug(f"[Priority 2] Best matches: {[(gid, f'{score:.4f}', name) for gid, score, name in matches]}")
        logger.info("="*50)
        return matches
    
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

