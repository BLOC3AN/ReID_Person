# ReID Logic (reid_logic.py)

Documentation for the person re-identification matching logic and strategy.

## Overview

`reid_logic.py` implements the core ReID matching strategy that determines when and how to identify persons across video frames. It uses a "First-3 Voting + Re-verification" approach for robust and efficient person identification.

**File:** `core/reid_logic.py`  
**Function:** `process_reid_logic()`  
**Purpose:** Decide when to extract embeddings and match against database

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              ReID Logic Decision Tree                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  New Track (frame 0-2)                                 │
│  ├─ Extract embedding                                  │
│  ├─ Match against database                             │
│  ├─ Store result in voting dict                        │
│  └─ At frame 2: Majority voting                        │
│      └─ Cache final label                              │
│                                                         │
│  Re-verification (frame 30, 60, 90...)                 │
│  ├─ Extract embedding                                  │
│  ├─ Match against database                             │
│  └─ Update cached label if changed                     │
│                                                         │
│  Other Frames                                          │
│  └─ Return cached label (no extraction)                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Strategy: First-3 + Re-verify

### Why This Strategy?

**Problem:** Running ReID on every frame is slow (3.6 FPS)  
**Solution:** Smart sampling reduces extractions by 95.8% while maintaining accuracy

**Performance:**
- Every frame: 3.6 FPS (100% extractions)
- First-3 + Re-verify: 19 FPS (4.2% extractions)
- **Speedup:** 5.3x faster

### Three Phases

#### Phase 1: First-3 Voting (Frame 0-2)

**Purpose:** Robust initialization with majority voting

**Logic:**
```
For frame_idx in [0, 1, 2]:
  ↓
Extract face embedding
  ↓
Match against database → (global_id, similarity, name)
  ↓
Store in voting_results[track_id][frame_idx] = (global_id, similarity, name)
  ↓
At frame 2:
  ├─ Count votes for each global_id
  ├─ Select majority winner
  ├─ If tie: Choose highest similarity
  └─ Cache result in track_labels[track_id]
```

**Example:**
```
Track 5:
  Frame 0: Person 1 (0.85)
  Frame 1: Person 1 (0.82)
  Frame 2: Person 2 (0.78)
  
Voting: Person 1 = 2 votes, Person 2 = 1 vote
Result: Person 1 (majority winner)
```

**Benefits:**
- Reduces false positives from single bad frame
- Handles temporary occlusions or bad angles
- More stable than single-frame matching

#### Phase 2: Re-verification (Every 30 Frames)

**Purpose:** Self-correction and adaptation

**Logic:**
```
If frame_idx % 30 == 0 and frame_idx > 2:
  ↓
Extract face embedding
  ↓
Match against database → (new_global_id, similarity, name)
  ↓
Compare with cached label:
  ├─ If changed: Update cache
  └─ If same: Keep cache
  ↓
Log re-verification event
```

**Re-verification Triggers:**
- Frame 30, 60, 90, 120, 150...
- Interval: 30 frames (~1 second at 30 FPS)

**Use Cases:**
- Person turns around (face becomes visible)
- Lighting changes improve face quality
- Occlusion clears up
- Initial voting was wrong

**Example:**
```
Track 8:
  Frame 0-2: Voted as "Unknown" (face not detected)
  Frame 30: Re-verify → Person 3 detected (0.87)
  Frame 60: Re-verify → Still Person 3 (0.89)
  
Result: Track corrected from Unknown to Person 3
```

#### Phase 3: Cached Labels (Other Frames)

**Purpose:** Maximum performance

**Logic:**
```
If frame_idx > 2 and frame_idx % 30 != 0:
  ↓
Return cached label from track_labels[track_id]
  ↓
No embedding extraction (very fast)
```

**Benefits:**
- 95.8% of frames use cached labels
- Minimal CPU/GPU usage
- Maintains real-time performance

## Function Signature

```python
def process_reid_logic(
    track_id: int,
    frame_idx: int,
    bbox: Tuple[int, int, int, int],
    frame: np.ndarray,
    extractor: ArcFaceExtractor,
    database: QdrantVectorDB,
    voting_results: Dict,
    track_labels: Dict,
    similarity_threshold: float = 0.8,
    log_file: Optional[TextIO] = None,
    camera_idx: int = 0,
    use_rerank: bool = False,
    rerank_k1: int = 20,
    rerank_k2: int = 6,
    rerank_lambda: float = 0.3
) -> Tuple[Optional[int], float, str]
```

## Parameters

### Required
- `track_id`: Unique track identifier from ByteTrack
- `frame_idx`: Current frame index in track (0, 1, 2, ...)
- `bbox`: Bounding box (x, y, w, h)
- `frame`: Video frame (numpy array)
- `extractor`: ArcFace feature extractor
- `database`: Qdrant vector database

### State Management
- `voting_results`: Dict storing first-3 frame results
  ```python
  {
    track_id: {
      0: (global_id, similarity, name),
      1: (global_id, similarity, name),
      2: (global_id, similarity, name)
    }
  }
  ```
- `track_labels`: Dict caching final labels
  ```python
  {
    track_id: (global_id, similarity, name)
  }
  ```

### Matching Parameters
- `similarity_threshold`: Minimum similarity for match (0.7-0.9)
- `use_rerank`: Enable k-reciprocal reranking
- `rerank_k1`, `rerank_k2`, `rerank_lambda`: Reranking parameters

### Logging
- `log_file`: File handle for detailed logs
- `camera_idx`: Camera identifier for multi-camera setups

## Return Value

```python
(global_id, similarity, name)
```

- `global_id`: Person ID (None if Unknown)
- `similarity`: Confidence score (0.0-1.0)
- `name`: Person name ("Unknown" if not matched)

## Decision Flow

```
Input: track_id, frame_idx, bbox, frame
  ↓
Check frame_idx:
  │
  ├─ frame_idx in [0, 1, 2]?
  │   ├─ YES: First-3 Voting Phase
  │   │   ├─ Extract embedding
  │   │   ├─ Match against database
  │   │   ├─ Store in voting_results
  │   │   └─ If frame_idx == 2:
  │   │       ├─ Perform majority voting
  │   │       ├─ Cache result in track_labels
  │   │       └─ Log voting results
  │   └─ Return match result
  │
  ├─ frame_idx % 30 == 0?
  │   ├─ YES: Re-verification Phase
  │   │   ├─ Extract embedding
  │   │   ├─ Match against database
  │   │   ├─ Update track_labels if changed
  │   │   └─ Log re-verification
  │   └─ Return new match result
  │
  └─ NO: Cached Label Phase
      └─ Return track_labels[track_id]
```

## State Management

### Voting Results Dictionary

**Structure:**
```python
voting_results = {
    5: {  # track_id
        0: (1, 0.85, "John"),
        1: (1, 0.82, "John"),
        2: (1, 0.88, "John")
    },
    7: {
        0: (None, 0.0, "Unknown"),
        1: (2, 0.79, "Jane"),
        2: (2, 0.81, "Jane")
    }
}
```

**Lifecycle:**
- Created when track first appears
- Populated during frames 0-2
- Used for voting at frame 2
- Can be cleared after voting (memory optimization)

### Track Labels Dictionary

**Structure:**
```python
track_labels = {
    5: (1, 0.85, "John"),      # Cached after voting
    7: (2, 0.80, "Jane"),      # Cached after voting
    9: (None, 0.0, "Unknown")  # Unknown person
}
```

**Lifecycle:**
- Created after first-3 voting (frame 2)
- Updated during re-verification (frame 30, 60, ...)
- Persists until track ends
- Used for all non-voting, non-reverify frames

## Majority Voting Algorithm

```python
def majority_voting(voting_results, track_id):
    """
    Select best match from first 3 frames
    """
    votes = voting_results[track_id]  # {0: result, 1: result, 2: result}
    
    # Count votes for each global_id
    vote_counts = {}
    for frame_idx, (gid, sim, name) in votes.items():
        if gid not in vote_counts:
            vote_counts[gid] = []
        vote_counts[gid].append((sim, name))
    
    # Find majority winner
    max_votes = 0
    best_match = (None, 0.0, "Unknown")
    
    for gid, matches in vote_counts.items():
        vote_count = len(matches)
        if vote_count > max_votes:
            max_votes = vote_count
            # Use highest similarity among votes
            best_sim = max(m[0] for m in matches)
            best_match = (gid, best_sim, matches[0][1])
        elif vote_count == max_votes:
            # Tie-breaker: highest similarity
            best_sim = max(m[0] for m in matches)
            if best_sim > best_match[1]:
                best_match = (gid, best_sim, matches[0][1])
    
    return best_match
```

## Logging

### Log Events

**1. Voting Event (Frame 2)**
```
[VOTING] Track 5 @ frame 2:
  Frame 0: Person 1 (0.85)
  Frame 1: Person 1 (0.82)
  Frame 2: Person 1 (0.88)
  → Result: Person 1 (0.85) [3/3 votes]
```

**2. Re-verification Event (Frame 30, 60, ...)**
```
[RE-VERIFY] Track 5 @ frame 30:
  Previous: Person 1 (0.85)
  Current:  Person 1 (0.89)
  → Status: CONFIRMED
```

**3. Label Change Event**
```
[RE-VERIFY] Track 7 @ frame 60:
  Previous: Unknown (0.0)
  Current:  Person 2 (0.87)
  → Status: UPDATED (Unknown → Person 2)
```

## Performance Metrics

### Extraction Frequency

**Every Frame Strategy:**
- Extractions: 100% of frames
- FPS: 3.6

**First-3 + Re-verify Strategy:**
- Extractions: 4.2% of frames
  - First-3: 3 frames per track
  - Re-verify: Every 30 frames
- FPS: 19
- Speedup: 5.3x

### Accuracy

**Comparison:**
- Every frame: 96.2% accuracy (baseline)
- First-3 + Re-verify: 95.8% accuracy
- Accuracy loss: 0.4% (negligible)

**Trade-off:** 5.3x faster with minimal accuracy loss

## Edge Cases

### Case 1: Face Not Detected in First 3 Frames

**Scenario:**
```
Track 10:
  Frame 0: No face detected → Unknown
  Frame 1: No face detected → Unknown
  Frame 2: No face detected → Unknown
  
Voting: Unknown (3/3 votes)
```

**Recovery:**
- Re-verification at frame 30 may detect face
- Track can be corrected from Unknown to identified person

### Case 2: Conflicting Votes

**Scenario:**
```
Track 12:
  Frame 0: Person 1 (0.85)
  Frame 1: Person 2 (0.83)
  Frame 2: Person 1 (0.82)
  
Voting: Person 1 (2/3 votes, higher count)
```

**Resolution:** Majority voting selects Person 1

### Case 3: All Votes Tied

**Scenario:**
```
Track 15:
  Frame 0: Person 1 (0.85)
  Frame 1: Person 2 (0.88)
  Frame 2: Person 3 (0.82)
  
Voting: Tie (1 vote each)
```

**Resolution:** Select highest similarity (Person 2 with 0.88)

### Case 4: Track Shorter Than 3 Frames

**Scenario:**
```
Track 20:
  Frame 0: Person 1 (0.85)
  Frame 1: Track lost
```

**Handling:**
- Use available votes (1 vote)
- Cache result after last frame
- No voting performed

## Integration with ByteTrack

### Track Lifecycle

```
ByteTrack detects person
  ↓
Assign track_id (e.g., 5)
  ↓
Frame 0: ReID logic → Extract + Match
Frame 1: ReID logic → Extract + Match
Frame 2: ReID logic → Extract + Match + Vote
  ↓
Cache label: track_labels[5] = (1, 0.85, "John")
  ↓
Frame 3-29: Return cached label
  ↓
Frame 30: Re-verify → Extract + Match
  ↓
Frame 31-59: Return cached label
  ↓
...
  ↓
Track lost (occlusion, leaves frame)
  ↓
ByteTrack keeps track in buffer (30 frames)
  ↓
If track returns: Resume with cached label
If track expires: Remove from track_labels
```

### Track ID Changes

**Important:** ByteTrack assigns new track_id when track is lost and reappears

**Impact:**
- New track_id = New voting process
- Previous label not carried over
- Person gets re-identified (may get different label)

**Example:**
```
Person walks across frame:
  Track 5 (frame 0-100): Identified as "John"
  
Person occluded by obstacle:
  Track 5 lost at frame 101
  
Person reappears:
  Track 8 (frame 120-200): Re-identified as "John" (new voting)
```

## Configuration

### Tuning Parameters

**Re-verification Interval:**
```python
REVERIFY_INTERVAL = 30  # frames

# Adjust based on:
# - Video FPS (30 FPS = 1 second, 60 FPS = 0.5 second)
# - Scene dynamics (fast movement = shorter interval)
# - Performance requirements (longer = faster)
```

**Similarity Threshold:**
```python
similarity_threshold = 0.8  # Default

# Tuning:
# - 0.9: Very strict (few false positives, many Unknown)
# - 0.8: Balanced (recommended)
# - 0.7: Loose (more matches, may have false positives)
```

**Reranking:**
```python
use_rerank = True  # Enable for better accuracy
rerank_k1 = 20     # Candidate set size
rerank_k2 = 6      # Reciprocal neighbors
rerank_lambda = 0.3  # Original distance weight
```

## Best Practices

### 1. State Management
- Clear voting_results after voting (save memory)
- Keep track_labels until track ends
- Handle track_id reuse (ByteTrack may reuse IDs)

### 2. Performance Optimization
- Increase re-verify interval for faster processing
- Disable reranking for real-time applications
- Use GPU for embedding extraction

### 3. Accuracy Improvement
- Enable reranking for critical applications
- Lower similarity threshold for difficult scenes
- Increase first-N voting frames (e.g., first-5)

### 4. Debugging
- Enable detailed logging (log_file parameter)
- Monitor voting results for conflicts
- Track re-verification frequency

## Related Files

- `core/vector_db.py` - Database search and reranking
- `scripts/detect_and_track.py` - Main pipeline using ReID logic
- `scripts/zone_monitor.py` - Zone monitoring with ReID
- `notebooks/rerank.ipynb` - Reranking analysis and tuning

## See Also

- [Vector Database Documentation](VECTOR_DB.md)
- [Configuration Guide](../CONFIGURATION.md)
- [Performance Guide](../PERFORMANCE.md)
