# Person Re-Identification Strategy

## Overview

This document describes the optimized ReID matching strategy used in the Person Re-Identification System. The strategy balances **accuracy** and **performance** through a "First-3 + Re-verify" approach.

---

## Problem Statement

### Naive Approach: ReID Every Frame
```python
for each frame:
    for each track:
        embedding = extract(frame, bbox)  # ~43ms
        match = search_database(embedding)  # ~0.06ms
        assign_label(track, match)
```

**Issues:**
- ❌ **Very slow:** ~3.6 FPS (276ms/frame for 3 persons)
- ❌ **Redundant:** Same person extracted 100s of times
- ❌ **Label flickering:** Label may change between frames
- ❌ **Bottleneck:** Feature extraction takes 47% of processing time

### Our Solution: First-3 + Re-verify
```python
for each track:
    if frame_count <= 3:
        # Robust initialization
        collect_embeddings()
        if frame_count == 3:
            majority_vote()
    elif frame_count % 30 == 0:
        # Self-correction
        re_verify()
    else:
        # Fast path
        use_cached_label()
```

**Benefits:**
- ✅ **Fast:** ~19 FPS (5.3x speedup)
- ✅ **Robust:** Voting reduces false positives
- ✅ **Self-correcting:** Re-verify handles occlusion/pose changes
- ✅ **Stable:** No label flickering

---

## Strategy Details

### 1. First-3 Voting (Frame 0-2)

**Purpose:** Robust initialization to avoid false positives from single bad frame.

**Algorithm:**
```python
def first_3_voting(track_id):
    embeddings = []
    
    # Collect embeddings from first 3 frames
    for frame in [0, 1, 2]:
        bbox = get_bbox(track_id, frame)
        emb = extractor.extract(frame, bbox)
        embeddings.append(emb)
    
    # Match each embedding
    votes = {}  # {(global_id, name): count}
    similarities = {}  # {(global_id, name): max_similarity}
    
    for emb in embeddings:
        matches = database.find_best_match(emb, threshold=0.0, top_k=1)
        if matches:
            global_id, sim, name = matches[0]
            key = (global_id, name)
            votes[key] = votes.get(key, 0) + 1
            similarities[key] = max(similarities.get(key, 0), sim)
    
    # Select winner: highest votes, then highest similarity
    best_key = max(votes.items(), key=lambda x: (x[1], similarities[x[0]]))[0]
    global_id, person_name = best_key
    similarity = similarities[best_key]
    
    # Assign label based on threshold
    if similarity >= threshold:
        label = person_name
    else:
        label = "Unknown"
    
    return {
        'global_id': global_id,
        'similarity': similarity,
        'label': label,
        'person_name': person_name,
        'votes': votes[best_key]
    }
```

**Example Output:**
```
Track 1 [VOTING]: 3/3 votes → Duong (sim=0.9606, gid=2)
Track 2 [VOTING]: 3/3 votes → Khiem (sim=0.9468, gid=1)
```

**Edge Cases:**
- **Tie votes (1-1-1):** Select by highest similarity
- **2-1 vote:** Winner is the one with 2 votes
- **All Unknown:** If all 3 have sim < threshold → "Unknown"

---

### 2. Re-verification (Every 30 frames)

**Purpose:** Self-correction to handle occlusion, pose changes, or initial misidentification.

**Algorithm:**
```python
def re_verify(track_id, frame_count):
    if frame_count % 30 != 0:
        return  # Not a re-verify frame
    
    # Extract current embedding
    bbox = get_bbox(track_id, frame)
    embedding = extractor.extract(frame, bbox)
    
    # Re-match
    matches = database.find_best_match(embedding, threshold=0.0, top_k=1)
    if not matches:
        return
    
    global_id, similarity, person_name = matches[0]
    
    # Get old label
    old_label = track_labels[track_id]['label']
    
    # Determine new label
    if similarity >= threshold:
        new_label = person_name
    else:
        new_label = "Unknown"
    
    # Update if changed or high confidence
    if new_label != old_label or similarity >= threshold:
        track_labels[track_id] = {
            'global_id': global_id,
            'similarity': similarity,
            'label': new_label,
            'person_name': person_name
        }
        
        log(f"Track {track_id} [RE-VERIFY]: {old_label} → {new_label} "
            f"(sim={similarity:.4f}, frame={frame_count})")
```

**Example Output:**
```
Track 1 [RE-VERIFY]: Duong → Duong (sim=0.9654, frame=30)
Track 2 [RE-VERIFY]: Khiem → Khiem (sim=0.9427, frame=30)
Track 1 [RE-VERIFY]: Unknown → Duong (sim=0.9664, frame=60)  # Correction!
```

**Re-verify Frequency:**
- **30 frames = 1 second** at 30 FPS
- **Adjustable:** Change `frame_count % 30` to `% 60` for 2-second interval

---

### 3. Cached Labels (Other frames)

**Purpose:** High performance by avoiding redundant feature extraction.

**Algorithm:**
```python
def get_label(track_id, frame_count):
    # Check if need to extract
    if track_id not in track_labels:
        # New track: start voting process
        return None  # Will be handled by voting
    
    if frame_count <= 3:
        # Voting in progress
        return None
    
    if frame_count % 30 == 0:
        # Re-verify frame
        return None  # Will be handled by re-verify
    
    # Use cached label
    return track_labels[track_id]
```

**Performance:**
- **No extraction:** 0ms for feature extraction
- **No search:** 0ms for database search
- **Only tracking:** ~146ms for detection + tracking
- **Result:** ~6.85 FPS for cached frames

---

## Performance Analysis

### Extraction Count (150 frames, 2 tracks)

| Strategy | Extractions | Frames | Total |
|----------|-------------|--------|-------|
| **Every frame** | 2 tracks × 150 frames | 150 | **300** |
| **First-3 + Re-verify** | 2×3 (voting) + 2×5 (re-verify) | 150 | **16** |
| **Reduction** | - | - | **94.7%** |

### FPS Comparison

| Strategy | FPS | Speedup |
|----------|-----|---------|
| ReID every frame | 3.6 | 1.0x |
| **First-3 + Re-verify** | **19.25** | **5.3x** |
| Only detect + track | 25.0 | 6.9x (theoretical max) |

### Time Breakdown (per frame, 3 persons)

| Component | Time | % |
|-----------|------|---|
| Detection | 144.79 ms | 52.3% |
| Tracking | 1.45 ms | 0.5% |
| **Feature Extraction (×3)** | **130.30 ms** | **47.1%** |
| Qdrant Search (×3) | 0.19 ms | 0.1% |
| **Total** | **276.73 ms** | **100%** |

**Key Insight:** Feature extraction is the bottleneck (47%), not database search (0.1%).

---

## Configuration

### Adjustable Parameters

```python
# In scripts/detect_and_track.py

# Number of frames for voting
VOTING_FRAMES = 3  # Default: 3

# Re-verification interval (frames)
REVERIFY_INTERVAL = 30  # Default: 30 (1 second at 30fps)

# Similarity threshold
SIMILARITY_THRESHOLD = 0.8  # Default: 0.8
```

### Tuning Guidelines

**Voting Frames:**
- `VOTING_FRAMES = 1`: Fastest, but less robust
- `VOTING_FRAMES = 3`: Balanced (recommended)
- `VOTING_FRAMES = 5`: Most robust, but slower initialization

**Re-verify Interval:**
- `REVERIFY_INTERVAL = 15`: More frequent correction, slower
- `REVERIFY_INTERVAL = 30`: Balanced (recommended)
- `REVERIFY_INTERVAL = 60`: Less frequent, faster

**Similarity Threshold:**
- `threshold = 0.6`: Loose (high recall, more false positives)
- `threshold = 0.7`: Balanced
- `threshold = 0.8`: Strict (high precision, recommended)
- `threshold = 0.9`: Very strict (may miss some matches)

---

## Example Timeline

### Track 1 (Duong) - 150 frames

| Frame | Action | Result | Similarity |
|-------|--------|--------|------------|
| 0 | Extract #1 | Pending | - |
| 1 | Extract #2 | Pending | - |
| 2 | Extract #3 + Vote | **Duong** (3/3 votes) | 0.9606 |
| 3-29 | Cached | Duong | 0.9606 |
| 30 | Re-verify | Duong → Duong ✓ | 0.9654 |
| 31-59 | Cached | Duong | 0.9654 |
| 60 | Re-verify | Duong → Duong ✓ | 0.9664 |
| 61-89 | Cached | Duong | 0.9664 |
| 90 | Re-verify | Duong → Duong ✓ | 0.9668 |
| 91-119 | Cached | Duong | 0.9668 |
| 120 | Re-verify | Duong → Duong ✓ | 0.9716 |
| 121-149 | Cached | Duong | 0.9716 |
| 150 | Re-verify | Duong → Duong ✓ | 0.9659 |

**Total extractions:** 3 (voting) + 5 (re-verify) = **8 extractions** (vs 150 if every frame)

---

## Advantages

### 1. Robust Initialization
- **Problem:** Single bad frame (blur, occlusion) → wrong label forever
- **Solution:** Vote from 3 frames → majority wins
- **Example:** Frame 0 (blur, 0.6), Frame 1 (good, 0.95), Frame 2 (good, 0.94) → 2/3 vote for correct person

### 2. Self-Correction
- **Problem:** Person occluded → misidentified → stuck with wrong label
- **Solution:** Re-verify every 30 frames → can correct mistakes
- **Example:** Frame 0-29 (Unknown), Frame 30 (clear view) → corrected to "Duong"

### 3. High Performance
- **Problem:** Feature extraction is slow (43ms/person)
- **Solution:** Extract only when needed (3 + N/30 times)
- **Result:** 94.7% reduction in extractions → 5.3x speedup

### 4. Stable Labels
- **Problem:** Label flickers between frames (Duong → Unknown → Duong)
- **Solution:** Cache labels, only update on re-verify
- **Result:** Smooth, stable labels in output video

---

## Limitations

### 1. Initial Delay
- **Issue:** Label appears after 3rd frame (not immediately)
- **Impact:** First 2 frames show "Unknown"
- **Mitigation:** Acceptable for most use cases (60-120ms delay)

### 2. Re-verify Latency
- **Issue:** Correction happens at frame 30, 60, 90... (not immediately)
- **Impact:** Wrong label persists for up to 30 frames
- **Mitigation:** Reduce `REVERIFY_INTERVAL` if needed (trade-off with speed)

### 3. Track ID Dependency
- **Issue:** If ByteTrack loses track → new track_id → re-voting
- **Impact:** Same person may get new label temporarily
- **Mitigation:** ByteTrack is robust, rarely loses tracks

---

## Future Improvements

### 1. Adaptive Re-verification
```python
# Re-verify more frequently if similarity is low
if similarity < 0.85:
    reverify_interval = 15  # More frequent
else:
    reverify_interval = 60  # Less frequent
```

### 2. Confidence-based Voting
```python
# Weight votes by similarity
weighted_votes = {}
for emb in embeddings:
    gid, sim, name = find_best_match(emb)
    weighted_votes[(gid, name)] += sim  # Weight by similarity
```

### 3. Temporal Smoothing
```python
# Average embeddings over time
avg_embedding = np.mean(embeddings[-5:], axis=0)
match = find_best_match(avg_embedding)
```

---

## References

- **ByteTrack Paper:** [https://arxiv.org/abs/2110.06864](https://arxiv.org/abs/2110.06864)
- **ArcFace Paper:** [https://arxiv.org/abs/1801.07698](https://arxiv.org/abs/1801.07698)
- **Qdrant Documentation:** [https://qdrant.tech/documentation/](https://qdrant.tech/documentation/)

