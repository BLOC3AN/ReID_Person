"""
Qdrant Vector Database Integration for ReID
Persistent storage and similarity search for person embeddings
"""

import os
import uuid
import numpy as np
from collections import deque
from scipy.spatial.distance import cdist
from loguru import logger
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv


class QdrantVectorDB:
    """
    Vector database for storing and searching person embeddings
    Supports both in-memory and Qdrant backend
    """
    
    def __init__(self, use_qdrant=False, qdrant_url=None,
                 collection_name=None, max_embeddings_per_person=100,
                 embedding_dim=512, api_key=None, use_grpc=False):
        """
        Args:
            use_qdrant: Use Qdrant backend (default: False, use in-memory)
            qdrant_url: Qdrant server URL (auto-load from .env if None)
            collection_name: Collection name (auto-load from .env if None)
            max_embeddings_per_person: Max embeddings to store per person
            embedding_dim: Embedding dimension (512 for ArcFace)
            api_key: Qdrant API key (auto-load from .env if None)
            use_grpc: Use gRPC protocol instead of HTTP (default: False)
        """
        # Load from .env if not provided
        env_path = Path(__file__).parent.parent / "configs" / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        self.use_qdrant = use_qdrant
        self.use_grpc = use_grpc
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "cross_camera_matching_id")
        self.max_embeddings = max_embeddings_per_person
        self.embedding_dim = embedding_dim
        self.next_global_id = 1

        # In-memory storage (always available as fallback)
        self.db = {}  # {global_id: deque of embeddings}
        self.person_metadata = {}  # {global_id: {camera_id, track_id, ...}}

        # Qdrant client (optional)
        self.client = None
        if use_qdrant:
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
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            # Parse URL if it contains "host=" prefix
            if qdrant_url.startswith("host="):
                qdrant_url = qdrant_url.replace("host=", "https://")

            # Initialize client with gRPC or HTTP
            if use_grpc:
                # For gRPC, remove http:// or https:// prefix if present
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
            except Exception as collection_error:
                logger.debug(f"Collection check failed: {collection_error}")
                try:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.embedding_dim,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"✅ Created Qdrant collection: {self.collection_name}")
                except Exception as create_error:
                    logger.warning(f"⚠️ Failed to create collection: {create_error}. Using in-memory storage.")
                    self.client = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to init Qdrant: {e}. Using in-memory storage.")
            self.client = None
    
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

        # Store in memory
        if global_id not in self.db:
            self.db[global_id] = deque(maxlen=self.max_embeddings)
        self.db[global_id].append(embedding)

        # Store metadata
        if metadata:
            self.person_metadata[global_id] = metadata

        # Store in Qdrant if available
        if self.client:
            try:
                # Use UUID for each embedding to avoid conflicts, store global_id in payload
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
            except Exception as e:
                logger.warning(f"Failed to store in Qdrant: {e}")
    
    def get_avg_embedding(self, global_id: int) -> Optional[np.ndarray]:
        """Get average embedding for person"""
        if global_id not in self.db or len(self.db[global_id]) == 0:
            return None
        
        embeddings = np.array(list(self.db[global_id]))
        avg = np.mean(embeddings, axis=0)
        return avg / (np.linalg.norm(avg) + 1e-8)
    
    def find_best_match(self, embedding: np.ndarray, threshold: float = 0.8,
                       top_k: int = 1) -> List[Tuple[int, float, str]]:
        """
        Find best matching persons using cosine similarity
        Args:
            embedding: Query embedding (512,)
            threshold: Cosine similarity threshold (0-1, default 0.8)
            top_k: Return top K matches
        Returns:
            List of (global_id, similarity, name) tuples
        """
        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Use Qdrant if available (preferred)
        if self.client:
            try:
                # Use query_points (new API) instead of search (old API)
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=embedding.tolist(),
                    limit=top_k * 10,  # Get more results to group by global_id
                    score_threshold=threshold  # Qdrant uses similarity directly
                )

                logger.debug(f"Qdrant query_points returned {len(results.points)} results (threshold={threshold})")

                # Group by global_id and get best score + name for each person
                best_per_person = {}
                for r in results.points:
                    global_id = r.payload.get('global_id', r.id)
                    name = r.payload.get('name', f'Person_{global_id}')
                    logger.debug(f"  Result: GID={global_id}, name={name}, score={r.score:.4f}")
                    if global_id not in best_per_person or r.score > best_per_person[global_id][1]:
                        best_per_person[global_id] = (global_id, r.score, name)

                # Return top K persons
                matches = sorted(best_per_person.values(), key=lambda x: x[1], reverse=True)[:top_k]
                logger.debug(f"Best matches: {[(gid, f'{score:.4f}', name) for gid, score, name in matches]}")
                return matches
            except Exception as e:
                logger.warning(f"Qdrant query_points failed: {e}. Falling back to in-memory.")

        # Fallback to in-memory search using cosine similarity
        if len(self.db) == 0:
            return []

        global_ids = list(self.db.keys())
        avg_embeddings = np.array([self.get_avg_embedding(gid) for gid in global_ids])

        # Calculate cosine similarity (1 - cosine_distance)
        distances = cdist([embedding], avg_embeddings, metric='cosine')[0]
        similarities = 1.0 - distances

        # Get top K by similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]
        matches = []
        for idx in top_indices:
            sim = similarities[idx]
            if sim >= threshold:
                gid = global_ids[idx]
                name = self.person_metadata.get(gid, {}).get('name', f'Person_{gid}')
                matches.append((gid, sim, name))

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
        # If using Qdrant, get count from Qdrant
        if self.client:
            try:
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
            except Exception as e:
                logger.warning(f"Failed to get person count from Qdrant: {e}")

        # Fallback to in-memory count
        return len(self.db)

    def sync_metadata_from_qdrant(self) -> int:
        """
        Sync person metadata from Qdrant to in-memory storage
        This is used to populate self.person_metadata without loading embeddings

        Returns:
            Number of persons synced
        """
        if not self.client:
            logger.warning("Qdrant client not available. Cannot sync metadata.")
            return 0

        try:
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

        except Exception as e:
            logger.error(f"Failed to sync metadata from Qdrant: {e}")
            return 0

    def get_stats(self) -> Dict:
        """Get database statistics"""
        total_embeddings = sum(len(embs) for embs in self.db.values())
        return {
            'num_persons': len(self.db),
            'total_embeddings': total_embeddings,
            'avg_embeddings_per_person': total_embeddings / len(self.db) if len(self.db) > 0 else 0,
            'next_global_id': self.next_global_id,
            'using_qdrant': self.client is not None,
        }

    def clear(self):
        """Clear database"""
        self.db.clear()
        self.person_metadata.clear()
        self.next_global_id = 1

        if self.client:
            try:
                self.client.delete_collection(self.collection_name)
                self._init_qdrant(self.client.url)
            except Exception as e:
                logger.warning(f"Failed to clear Qdrant: {e}")

