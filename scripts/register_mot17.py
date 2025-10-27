#!/usr/bin/env python3
"""
Register person using MOT17 model (same as detect_and_track.py)
This ensures embeddings are consistent between registration and detection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from loguru import logger
from core import YOLOXDetector, OSNetExtractor, QdrantVectorDB
from qdrant_client.models import Distance, VectorParams


def register_person_mot17(video_path: str, person_name: str, sample_rate: int = 5):
    """
    Register a person using MOT17 model
    
    Args:
        video_path: Path to video file containing the person
        person_name: Name of the person to register
        sample_rate: Extract 1 frame every N frames (default: 5)
    """
    
    logger.info("=" * 80)
    logger.info(f"REGISTERING PERSON: {person_name}")
    logger.info(f"Video: {video_path}")
    logger.info(f"Sample rate: {sample_rate}")
    logger.info("=" * 80)
    
    # Initialize detector (MOT17 - same as detect_and_track.py)
    logger.info("\nInitializing MOT17 detector...")
    model_path = Path(__file__).parent.parent / "models" / "bytetrack_x_mot17.pth.tar"
    detector = YOLOXDetector(
        model_path=str(model_path),
        model_type="mot17",
        conf_thresh=0.6,
        nms_thresh=0.45
    )
    
    # Initialize OSNet extractor
    logger.info("Initializing OSNet extractor...")
    extractor = OSNetExtractor(use_cuda=True)
    
    # Initialize database
    logger.info("Initializing database...")
    db = QdrantVectorDB(use_qdrant=True, embedding_dim=512)
    
    # Extract frames
    logger.info(f"\nExtracting frames (sample rate={sample_rate})...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_rate == 0:
            frames.append(frame)
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {frame_count} total frames")
    
    if len(frames) == 0:
        logger.error("No frames extracted!")
        return
    
    # Extract embeddings
    logger.info(f"\nExtracting embeddings from {len(frames)} frames...")
    embeddings = []
    
    for i, frame in enumerate(frames):
        # Detect persons in frame
        detections = detector.detect(frame)
        if len(detections) == 0:
            continue
        
        # Get largest bounding box (assume it's the target person)
        areas = [(det[2] - det[0]) * (det[3] - det[1]) for det in detections]
        largest_idx = np.argmax(areas)
        det = detections[largest_idx]
        
        # Convert to [x, y, w, h] format
        x1, y1, x2, y2 = det[:4]
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        
        # Extract embedding
        embedding = extractor.extract(frame, bbox)
        if np.linalg.norm(embedding) > 0:
            embeddings.append(embedding)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{len(frames)} frames")
    
    logger.info(f"✅ Extracted {len(embeddings)} valid embeddings")
    
    if len(embeddings) == 0:
        logger.error("No valid embeddings extracted!")
        return
    
    # Clear Qdrant collection
    logger.info("\nClearing Qdrant collection...")
    try:
        db.client.delete_collection(db.collection_name)
        logger.info(f"✅ Deleted old collection: {db.collection_name}")
    except Exception as e:
        logger.info(f"No old collection to delete (this is OK for first registration)")
    
    # Recreate collection
    logger.info("Creating new Qdrant collection...")
    db.client.create_collection(
        collection_name=db.collection_name,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
    )
    logger.info(f"✅ Created collection: {db.collection_name}")
    
    # Register person
    logger.info(f"\nRegistering {person_name} to database...")
    metadata = {
        'name': person_name,
        'video_path': video_path,
        'num_frames': len(frames),
        'num_embeddings': len(embeddings)
    }
    
    global_id = db.create_new_person(embeddings[0], metadata=metadata)
    
    # Add embeddings (limit to 50 to avoid bloat)
    max_embeddings = min(len(embeddings), 50)
    logger.info(f"Adding {max_embeddings} embeddings to Qdrant...")
    
    for emb in embeddings[:max_embeddings]:
        db.add_embedding(global_id, emb, metadata=metadata)
    
    # Save to local file
    db_file = Path(__file__).parent.parent / "data" / "database" / "reid_database.pkl"
    db.save_to_file(str(db_file))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ REGISTRATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Person: {person_name}")
    logger.info(f"Global ID: {global_id}")
    logger.info(f"Embeddings stored: {max_embeddings}")
    logger.info(f"Local database: {db_file}")
    logger.info(f"Qdrant collection: {db.collection_name} ({max_embeddings} points)")
    logger.info("=" * 80)
    logger.info("\nNext step:")
    logger.info(f"  python scripts/detect_and_track.py --video <VIDEO> --model mot17 --known-person {person_name}")
    logger.info("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Register a person using MOT17 model for Person Re-Identification"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to video file containing the person to register"
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the person to register"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=5,
        help="Extract 1 frame every N frames (default: 5)"
    )
    
    args = parser.parse_args()
    
    register_person_mot17(args.video, args.name, args.sample_rate)

