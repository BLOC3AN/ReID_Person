#!/usr/bin/env python3
"""
Register person using MOT17 model (same as detect_and_track.py)
This ensures embeddings are consistent between registration and detection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import cv2
import numpy as np
from loguru import logger
from core import YOLOXDetector, ArcFaceExtractor, ArcFaceTritonClient, QdrantVectorDB
from qdrant_client.models import Distance, VectorParams
from typing import List, Union


def register_person_mot17(video_path: str, person_name: str, global_id: int, sample_rate: int = 5, delete_existing: bool = False, detector=None, extractor=None):
    """
    Register a person using MOT17 model

    Args:
        video_path: Path to video file containing the person
        person_name: Name of the person to register
        global_id: Global ID for the person (unique identifier)
        sample_rate: Extract 1 frame every N frames (default: 5)
        delete_existing: Delete existing collection before registering
        detector: Pre-loaded YOLOX detector (optional, will create new if None)
        extractor: Pre-loaded ArcFace extractor (optional, will create new if None)
    """

    logger.info("=" * 80)
    logger.info(f"REGISTERING PERSON: {person_name}")
    logger.info(f"Global ID: {global_id}")
    logger.info(f"Video: {video_path}")
    logger.info(f"Sample rate: {sample_rate}")
    logger.info("=" * 80)

    # Use preloaded detector or initialize new one
    if detector is None:
        logger.info("\nInitializing MOT17 detector...")
        model_path = Path(__file__).parent.parent / "models" / "bytetrack_x_mot17.pth.tar"
        detector = YOLOXDetector(
            model_path=str(model_path),
            model_type="mot17",
            conf_thresh=0.6,
            nms_thresh=0.45
        )
    else:
        logger.info("\n✅ Using preloaded MOT17 detector")

    # Use preloaded extractor or initialize new one
    if extractor is None:
        logger.info("Initializing ArcFace extractor (fallback to InsightFace)...")
        extractor = ArcFaceExtractor(model_name='buffalo_l', use_cuda=True)
    else:
        logger.info("✅ Using preloaded ArcFace extractor (Triton or InsightFace)")
    
    # Initialize database
    logger.info("Initializing database...")
    use_grpc = os.getenv("QDRANT_USE_GRPC", "false").lower() == "true"
    db = QdrantVectorDB(use_qdrant=True, embedding_dim=512, use_grpc=use_grpc)
    
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
    
    # Check if collection exists
    logger.info("\nChecking Qdrant collection...")
    collection_exists = False
    try:
        db.client.get_collection(db.collection_name)
        collection_exists = True
        logger.info(f"Collection '{db.collection_name}' already exists")
    except Exception:
        logger.info(f"Collection '{db.collection_name}' does not exist")

    # Delete collection if requested or if it exists
    if collection_exists:
        if delete_existing:
            logger.info(f"Deleting existing collection: {db.collection_name}")
            db.client.delete_collection(db.collection_name)
            logger.info(f"✅ Deleted old collection: {db.collection_name}")
            collection_exists = False
        else:
            logger.info(f"⚠️  Collection already exists. Use --delete-existing to recreate it.")
            logger.info(f"Adding embeddings to existing collection...")

    # Create collection if it doesn't exist
    if not collection_exists:
        logger.info("Creating new Qdrant collection...")
        db.client.create_collection(
            collection_name=db.collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
        logger.info(f"✅ Created collection: {db.collection_name}")
    
    # Register person
    logger.info(f"\nRegistering {person_name} (ID: {global_id}) to database...")
    metadata = {
        'name': person_name,
        'global_id': global_id,
        'video_path': video_path,
        'num_frames': len(frames),
        'num_embeddings': len(embeddings)
    }

    # Add all embeddings
    logger.info(f"Adding {len(embeddings)} embeddings to Qdrant...")

    for emb in embeddings:
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
    logger.info(f"Embeddings stored: {len(embeddings)}")
    logger.info(f"Local database: {db_file}")
    logger.info(f"Qdrant collection: {db.collection_name} ({len(embeddings)} points)")
    logger.info("=" * 80)
    logger.info("\nNext step:")
    logger.info(f"  python scripts/detect_and_track.py --video <VIDEO> --model mot17 --known-person {person_name}")
    logger.info("=" * 80)


def register_person_from_images(image_paths: Union[str, List[str]], person_name: str, global_id: int,
                                 delete_existing: bool = False, detector=None, extractor=None):
    """
    Register a person using images instead of video

    Args:
        image_paths: Path to image file or folder, or list of image paths
        person_name: Name of the person to register
        global_id: Global ID for the person (unique identifier)
        delete_existing: Delete existing collection before registering
        detector: Pre-loaded YOLOX detector (optional, will create new if None)
        extractor: Pre-loaded ArcFace extractor (optional, will create new if None)
    """

    logger.info("=" * 80)
    logger.info(f"REGISTERING PERSON FROM IMAGES: {person_name}")
    logger.info(f"Global ID: {global_id}")
    logger.info("=" * 80)

    # Collect image paths
    images_to_process = []

    if isinstance(image_paths, str):
        path = Path(image_paths)
        if path.is_dir():
            # Load all images from folder
            logger.info(f"Loading images from folder: {image_paths}")
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                images_to_process.extend(list(path.glob(ext)))
                images_to_process.extend(list(path.glob(ext.upper())))
        elif path.is_file():
            images_to_process = [path]
        else:
            logger.error(f"Invalid path: {image_paths}")
            return
    else:
        # List of paths
        images_to_process = [Path(p) for p in image_paths]

    logger.info(f"Found {len(images_to_process)} images to process")

    if len(images_to_process) == 0:
        logger.error("No images found!")
        return

    # Use preloaded detector or initialize new one
    if detector is None:
        logger.info("\nInitializing MOT17 detector...")
        model_path = Path(__file__).parent.parent / "models" / "bytetrack_x_mot17.pth.tar"
        detector = YOLOXDetector(
            model_path=str(model_path),
            model_type="mot17",
            conf_thresh=0.6,
            nms_thresh=0.45
        )
    else:
        logger.info("\n✅ Using preloaded MOT17 detector")

    # Use preloaded extractor or initialize new one
    if extractor is None:
        logger.info("Initializing ArcFace extractor (fallback to InsightFace)...")
        extractor = ArcFaceExtractor(model_name='buffalo_l', use_cuda=True)
    else:
        logger.info("✅ Using preloaded ArcFace extractor (Triton or InsightFace)")

    # Initialize database
    logger.info("Initializing database...")
    use_grpc = os.getenv("QDRANT_USE_GRPC", "false").lower() == "true"
    db = QdrantVectorDB(use_qdrant=True, embedding_dim=512, use_grpc=use_grpc)

    # Extract embeddings from images
    logger.info(f"\nExtracting embeddings from {len(images_to_process)} images...")
    embeddings = []
    processed_count = 0

    for i, img_path in enumerate(images_to_process):
        # Read image
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning(f"  ⚠️  Cannot read image: {img_path}")
            continue

        # Detect persons in image
        detections = detector.detect(frame)
        if len(detections) == 0:
            logger.warning(f"  ⚠️  No person detected in: {img_path.name}")
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
            processed_count += 1
            logger.info(f"  ✅ [{processed_count}/{len(images_to_process)}] {img_path.name}")
        else:
            logger.warning(f"  ⚠️  Invalid embedding from: {img_path.name}")

    logger.info(f"✅ Extracted {len(embeddings)} valid embeddings from {processed_count} images")

    if len(embeddings) == 0:
        logger.error("No valid embeddings extracted!")
        return

    # Check if collection exists
    logger.info("\nChecking Qdrant collection...")
    collection_exists = False
    try:
        db.client.get_collection(db.collection_name)
        collection_exists = True
        logger.info(f"Collection '{db.collection_name}' already exists")
    except Exception:
        logger.info(f"Collection '{db.collection_name}' does not exist")

    # Delete collection if requested
    if collection_exists:
        if delete_existing:
            logger.info(f"Deleting existing collection: {db.collection_name}")
            db.client.delete_collection(db.collection_name)
            logger.info(f"✅ Deleted old collection: {db.collection_name}")
            collection_exists = False
        else:
            logger.info(f"⚠️  Collection already exists. Use delete_existing=True to recreate it.")
            logger.info(f"Adding embeddings to existing collection...")

    # Create collection if it doesn't exist
    if not collection_exists:
        logger.info("Creating new Qdrant collection...")
        db.client.create_collection(
            collection_name=db.collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
        logger.info(f"✅ Created collection: {db.collection_name}")

    # Register person
    logger.info(f"\nRegistering {person_name} (ID: {global_id}) to database...")
    metadata = {
        'name': person_name,
        'global_id': global_id,
        'source': 'images',
        'num_images': len(images_to_process),
        'num_embeddings': len(embeddings)
    }

    # Add all embeddings
    logger.info(f"Adding {len(embeddings)} embeddings to Qdrant...")

    for emb in embeddings:
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
    logger.info(f"Images processed: {processed_count}/{len(images_to_process)}")
    logger.info(f"Embeddings stored: {len(embeddings)}")
    logger.info(f"Local database: {db_file}")
    logger.info(f"Qdrant collection: {db.collection_name} ({len(embeddings)} points)")
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
        "--global-id",
        type=int,
        required=True,
        help="Global ID for the person (unique identifier)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=5,
        help="Extract 1 frame every N frames (default: 5)"
    )
    
    parser.add_argument(
        "--delete-existing",
        action='store_true',
        help="Delete existing collection before registering (default: False)"
    )
    
    args = parser.parse_args()

    register_person_mot17(args.video, args.name, args.global_id, args.sample_rate, args.delete_existing)

