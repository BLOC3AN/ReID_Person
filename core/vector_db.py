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
    
    def find_best_match(self, embedding: np.ndarray, threshold: float = 0.0,
                       top_k: int = 1, use_rerank=False, k1=20, k2=6, lambda_value=0.3):
        """
        Find best matching persons using Qdrant vector search with optional k-reciprocal reranking
        
        Args:
            embedding: Query embedding (512,)
            threshold: Cosine similarity threshold (0-1, default 0.8)
            top_k: Return top K matches
            use_rerank: Enable k-reciprocal reranking (default: False)
            k1: K-reciprocal set size (default: 20)
            k2: K-nearest neighbors for expansion (default: 6)
            lambda_value: Weight for original distance (default: 0.3)
        Returns:
            Dict with matches, all_gids, query_threshold
        """
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding.tolist(),
            limit=50 if use_rerank else top_k * 10,
            score_threshold=0.3 if use_rerank else threshold,
            with_vectors=use_rerank
        )

        logger.debug(f"Qdrant returned {len(results.points)} results (threshold={threshold}, rerank={use_rerank})")

        if not results.points:
            return {'matches': [], 'all_gids': [], 'query_threshold': threshold}

        # Apply reranking if enabled
        if use_rerank:
            reranked_points = self._rerank_embeddings(embedding, results.points, k1, k2, lambda_value)
            results.points = reranked_points

        # Group by global_id and get best score
        best_per_person = {}
        for r in results.points:
            gid = r.payload.get('global_id', r.id)
            name = r.payload.get('name', f'Person_{gid}')
            if r.score >= threshold and (gid not in best_per_person or r.score > best_per_person[gid][1]):
                best_per_person[gid] = (gid, r.score, name)
        
        matches = sorted(best_per_person.values(), key=lambda x: x[1], reverse=True)[:top_k]
        all_gids = [m[0] for m in matches]
        return {'matches': matches, 'all_gids': all_gids, 'query_threshold': threshold}

    def _rerank_embeddings(self, query_emb, points, k1, k2, lambda_value):
        """K-reciprocal reranking - returns points with updated scores"""
        gallery_embs = np.array([p.vector for p in points])
        
        # Compute distances
        query_sim = np.dot(query_emb.reshape(1, -1), gallery_embs.T).flatten()
        query_dist = 1 - query_sim
        gallery_dist = 1 - np.dot(gallery_embs, gallery_embs.T)
        
        # K-reciprocal neighbors
        k1 = min(k1, len(points))
        initial_rank = np.argsort(query_dist)[:k1]
        k_reciprocal_idx = [i for i in initial_rank if i in initial_rank]
        
        # Expand with k2
        k_reciprocal_exp = k_reciprocal_idx.copy()
        for i in k_reciprocal_idx:
            candidate_k = np.argsort(gallery_dist[i])[:int(k2)]
            candidate_reciprocal = [j for j in candidate_k if j in np.argsort(gallery_dist[j])[:int(k2/2)]]
            if len(np.intersect1d(candidate_reciprocal, k_reciprocal_idx)) > 2/3 * len(candidate_reciprocal):
                k_reciprocal_exp.extend(candidate_reciprocal)
        k_reciprocal_exp = list(set(k_reciprocal_exp))
        
        # Jaccard distance
        jaccard_dist = np.zeros(len(points))
        for i in range(len(points)):
            temp_reciprocal = [j for j in np.argsort(gallery_dist[i])[:k1] if i in np.argsort(gallery_dist[j])[:k1]]
            if len(temp_reciprocal) == 0:
                jaccard_dist[i] = 1.0
            else:
                intersect = np.intersect1d(k_reciprocal_exp, temp_reciprocal)
                union = np.union1d(k_reciprocal_exp, temp_reciprocal)
                jaccard_dist[i] = 1 - len(intersect) / len(union)
        
        # Final distance and update scores
        final_dist = jaccard_dist * (1 - lambda_value) + query_dist * lambda_value
        for i, point in enumerate(points):
            point.score = 1 - final_dist[i]
        
        return points
    
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

