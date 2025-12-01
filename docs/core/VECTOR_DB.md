# Vector Database (vector_db.py)

Documentation for the Qdrant vector database integration used for person re-identification.

## Overview

`vector_db.py` implements the vector database layer using Qdrant for storing and searching person face embeddings. It provides similarity search capabilities for matching detected faces against registered persons.

**File:** `core/vector_db.py`  
**Class:** `QdrantVectorDB`  
**Purpose:** Persistent storage and similarity search for 512-dim ArcFace embeddings

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   QdrantVectorDB                        │
├─────────────────────────────────────────────────────────┤
│  Initialization                                         │
│  ├─ Load .env configuration                            │
│  ├─ Connect to Qdrant (HTTP or gRPC)                   │
│  ├─ Create/verify collection                           │
│  └─ Sync metadata (person_id → name mapping)           │
├─────────────────────────────────────────────────────────┤
│  Core Operations                                        │
│  ├─ add_person()        - Register new person          │
│  ├─ add_embeddings()    - Add face embeddings          │
│  ├─ find_best_match()   - Search similar faces         │
│  ├─ _rerank_embeddings()- K-reciprocal reranking       │
│  └─ get_all_persons()   - List registered persons      │
├─────────────────────────────────────────────────────────┤
│  Storage                                                │
│  ├─ Qdrant Collection   - Vector embeddings            │
│  └─ Metadata Dict       - {global_id: name}            │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Initialization

**Purpose:** Connect to Qdrant and setup collection

**Process:**
```python
__init__(qdrant_url, collection_name, embedding_dim=512, use_grpc=False)
  ↓
Load .env file (configs/.env)
  ↓
Initialize Qdrant client (HTTP or gRPC)
  ↓
Create collection if not exists
  - Vector size: 512 (ArcFace)
  - Distance metric: Cosine
  - HNSW index for fast search
  ↓
Sync metadata from Qdrant
  - Build person_metadata dict
  - Track next_global_id
```

**Configuration Sources:**
- `.env` file: `QDRANT_URI`, `QDRANT_COLLECTION`, `QDRANT_USE_GRPC`
- Parameters: Override .env if provided

### 2. Person Registration

**Method:** `add_person(global_id, name)`

**Purpose:** Register a new person in the system

**Logic:**
```
Input: global_id (int), name (str)
  ↓
Validate global_id is unique
  ↓
Store in metadata dict: person_metadata[global_id] = name
  ↓
Update next_global_id if needed
  ↓
Return: global_id
```

**Important:**
- `global_id` must be unique across all persons
- Name can be duplicated (different persons with same name)
- No embeddings stored yet (only metadata)

### 3. Embedding Storage

**Method:** `add_embeddings(global_id, embeddings, metadata=None)`

**Purpose:** Store face embeddings for a registered person

**Logic:**
```
Input: 
  - global_id: Person identifier
  - embeddings: List of 512-dim numpy arrays
  - metadata: Optional dict (e.g., video_name, frame_id)
  ↓
Validate person exists (global_id in person_metadata)
  ↓
For each embedding:
  ├─ Generate unique UUID
  ├─ Create payload:
  │   {
  │     "global_id": int,
  │     "name": str,
  │     "metadata": dict (optional)
  │   }
  └─ Upsert to Qdrant collection
  ↓
Return: Number of embeddings added
```

**Storage Format:**
```json
{
  "id": "uuid-string",
  "vector": [512 floats],
  "payload": {
    "global_id": 1,
    "name": "John",
    "metadata": {
      "video": "person1.mp4",
      "frame": 100
    }
  }
}
```

### 4. Similarity Search

**Method:** `find_best_match(query_embedding, threshold=0.8, top_k=1, use_rerank=False, ...)`

**Purpose:** Find the most similar person for a query face embedding

**Logic Flow:**

#### Standard Search (use_rerank=False):
```
Input: query_embedding (512-dim array)
  ↓
Search Qdrant with cosine similarity
  - Limit: top_k results
  - Score filter: >= threshold
  ↓
Get top result
  ↓
If score >= threshold:
  └─ Return (global_id, similarity, name)
Else:
  └─ Return (None, 0.0, "Unknown")
```

#### K-Reciprocal Reranking (use_rerank=True):
```
Input: query_embedding + rerank parameters
  ↓
Initial search: Get top rerank_k1 candidates
  ↓
Call _rerank_embeddings():
  ├─ For each candidate:
  │   ├─ Find its k-nearest neighbors (k2)
  │   ├─ Check if query is in candidate's neighbors
  │   └─ Build reciprocal set
  ├─ Compute Jaccard distance between reciprocal sets
  ├─ Combine with original distance:
  │   final_distance = λ * original + (1-λ) * jaccard
  └─ Re-sort by final distance
  ↓
Return best match after reranking
```

**Reranking Benefits:**
- Reduces false positives from similar-looking persons
- Considers gallery-gallery relationships (not just query-gallery)
- More robust to noise and outliers

**Parameters:**
- `rerank_k1`: Size of initial candidate set (default: 20)
- `rerank_k2`: K-nearest neighbors for reciprocal check (default: 6)
- `rerank_lambda`: Weight for original distance (default: 0.3)

### 5. K-Reciprocal Reranking Algorithm

**Method:** `_rerank_embeddings(query_embedding, initial_results, k1, k2, lambda_value)`

**Purpose:** Improve search accuracy by considering reciprocal nearest neighbors

**Algorithm:**
```
Step 1: Build reciprocal sets
  For each candidate in top-k1:
    ├─ Find k2 nearest neighbors of candidate
    ├─ Check if query is in those neighbors
    └─ If yes: Add to reciprocal set R(q,g)

Step 2: Compute Jaccard distance
  For each pair of candidates (i, j):
    ├─ Compute intersection: |R(i) ∩ R(j)|
    ├─ Compute union: |R(i) ∪ R(j)|
    └─ Jaccard distance: 1 - (intersection / union)

Step 3: Combine distances
  final_distance = λ * cosine_distance + (1-λ) * jaccard_distance

Step 4: Re-sort by final distance
  Return top-1 match
```

**Example:**
```
Query: Person A's face
Initial top-3:
  1. Person B (0.85 similarity)
  2. Person C (0.83 similarity)  
  3. Person D (0.82 similarity)

After reranking:
  1. Person C (0.88 final score) ← Promoted due to reciprocal neighbors
  2. Person B (0.84 final score)
  3. Person D (0.80 final score)
```

## Data Flow

### Registration Flow
```
register_mot17.py
  ↓
Extract face embeddings (ArcFace)
  ↓
QdrantVectorDB.add_person(global_id, name)
  ↓
QdrantVectorDB.add_embeddings(global_id, embeddings)
  ↓
Qdrant Collection
```

### Detection Flow
```
detect_and_track.py
  ↓
Detect face (SCRFD)
  ↓
Extract embedding (ArcFace)
  ↓
QdrantVectorDB.find_best_match(embedding, use_rerank=True)
  ↓
Return (global_id, similarity, name)
  ↓
Annotate video with person name
```

## Performance Characteristics

### Search Performance
- **HNSW Index:** O(log n) search time
- **Typical latency:** 1-5ms for 1000 persons
- **Scalability:** Handles 10,000+ persons efficiently

### Memory Usage
- **Per embedding:** ~2KB (512 floats + metadata)
- **1000 embeddings:** ~2MB
- **10,000 embeddings:** ~20MB

### Reranking Overhead
- **Standard search:** 1-2ms
- **With reranking (k1=20, k2=6):** 5-10ms
- **Trade-off:** 3-5x slower but more accurate

## Configuration

### Environment Variables (.env)
```env
QDRANT_URI=http://127.0.0.1:6333
QDRANT_COLLECTION=cross_camera_matching_id
QDRANT_USE_GRPC=true
```

### Collection Settings
```python
VectorParams(
    size=512,              # ArcFace embedding dimension
    distance=Distance.COSINE,  # Cosine similarity
    hnsw_config=HnswConfigDiff(
        m=16,              # HNSW parameter (connections per node)
        ef_construct=100   # Construction time accuracy
    )
)
```

## Error Handling

### Common Errors

**1. Connection Failed**
```
RuntimeError: Failed to connect to Qdrant at <url>
```
**Solution:** Check Qdrant service is running, verify URL in .env

**2. Collection Not Found**
```
Collection 'cross_camera_matching_id' not found
```
**Solution:** Collection auto-created on first use, check permissions

**3. Duplicate Global ID**
```
ValueError: Person with global_id X already exists
```
**Solution:** Use unique global_id for each person

**4. Person Not Found**
```
ValueError: Person with global_id X not found
```
**Solution:** Call add_person() before add_embeddings()

## Best Practices

### 1. Global ID Management
- Use sequential IDs: 1, 2, 3, ...
- Never reuse IDs (even after deletion)
- Track next_global_id in database

### 2. Embedding Quality
- Store 10-50 embeddings per person (diverse poses/angles)
- Filter low-quality faces (confidence < 0.5)
- Avoid duplicate embeddings from same frame

### 3. Search Optimization
- Use reranking for critical applications (security, access control)
- Disable reranking for real-time applications (live streaming)
- Tune threshold based on false positive/negative trade-off

### 4. Maintenance
- Periodically backup Qdrant snapshots
- Monitor collection size and search latency
- Clean up unused embeddings

## Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Collection Stats
```python
db = QdrantVectorDB()
info = db.client.get_collection(db.collection_name)
print(f"Vectors: {info.vectors_count}")
print(f"Points: {info.points_count}")
```

### Test Search
```python
# Search with dummy embedding
dummy = np.random.randn(512).astype(np.float32)
result = db.find_best_match(dummy, threshold=0.0)
print(f"Result: {result}")
```

## Related Files

- `core/reid_logic.py` - Uses vector_db for ReID matching
- `scripts/register_mot17.py` - Registers persons via vector_db
- `scripts/detect_and_track.py` - Searches persons via vector_db
- `configs/.env` - Qdrant connection configuration

## See Also

- [ReID Logic Documentation](REID_LOGIC.md)
- [Configuration Guide](../CONFIGURATION.md)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
