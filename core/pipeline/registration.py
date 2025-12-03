"""Person registration logic"""

import os
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from typing import List, Union

from core.database import QdrantVectorDB


def register_person_mot17(video_path: str, person_name: str, global_id: int, 
                          sample_rate: int = 5, delete_existing: bool = False, 
                          face_conf_thresh: float = 0.5, skip_body_detection: bool = False, 
                          detector=None, extractor=None):
    """
    Register a person from video
    
    Args:
        video_path: Path to video file
        person_name: Person name
        global_id: Unique person ID
        sample_rate: Extract 1 frame every N frames
        delete_existing: Delete existing person data
        face_conf_thresh: Face detection confidence threshold
        skip_body_detection: Skip body detection, use full image (default: False)
        detector: Pre-loaded detector (from preloaded_manager)
        extractor: Pre-loaded face extractor (from preloaded_manager)
    """
    
    logger.info(f"Registering person: {person_name} (ID: {global_id})")
    
    # Detector and extractor should be passed from preloaded_manager
    if not skip_body_detection and detector is None:
        raise ValueError("Detector required when skip_body_detection=False. Pass detector from preloaded_manager.")
    
    if extractor is None:
        raise ValueError("Extractor required. Pass extractor from preloaded_manager.")
    
    # Initialize database
    use_grpc = os.getenv("QDRANT_USE_GRPC", "false").lower() == "true"
    db = QdrantVectorDB(embedding_dim=512, use_grpc=use_grpc)
    
    # Extract frames
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
    logger.info(f"Extracted {len(frames)} frames")
    
    if len(frames) == 0:
        logger.error("No frames extracted!")
        return
    
    # Extract embeddings
    embeddings = []
    for i, frame in enumerate(frames):
        if skip_body_detection:
            # Use full image
            h, w = frame.shape[:2]
            bbox = [0, 0, w, h]
        else:
            # Detect person body first
            detections = detector.detect(frame)
            if len(detections) == 0:
                continue
            # Use first detection
            x1, y1, x2, y2, conf = detections[0]
            bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
        
        # Extract face embedding
        embedding = extractor.extract(frame, bbox)
        if embedding is not None:
            embeddings.append(embedding)
    
    logger.info(f"Extracted {len(embeddings)} embeddings")
    
    if len(embeddings) == 0:
        logger.error("No embeddings extracted!")
        return
    
    # Register in database
    db.register_person(global_id, person_name, embeddings)
    logger.info(f"✅ Registered {person_name} with {len(embeddings)} embeddings")


def register_person_from_images(image_paths: List[str], person_name: str, global_id: int,
                                face_conf_thresh: float = 0.5, delete_existing: bool = False,
                                skip_body_detection: bool = True, detector=None, extractor=None):
    """
    Register a person from images
    
    Args:
        image_paths: List of image paths
        person_name: Person name
        global_id: Unique person ID
        face_conf_thresh: Face detection confidence threshold
        delete_existing: Delete existing person data (not implemented)
        skip_body_detection: Skip body detection, use full image (default: True)
        detector: Pre-loaded detector (not used for images)
        extractor: Pre-loaded face extractor (from preloaded_manager)
    """
    
    logger.info(f"Registering person from images: {person_name} (ID: {global_id})")
    
    if extractor is None:
        raise ValueError("Extractor required. Pass extractor from preloaded_manager.")
    
    # Initialize database
    use_grpc = os.getenv("QDRANT_USE_GRPC", "false").lower() == "true"
    db = QdrantVectorDB(embedding_dim=512, use_grpc=use_grpc)
    
    # Extract embeddings from images
    embeddings = []
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"Cannot read image: {img_path}")
            continue
        
        # Use full image as bbox
        h, w = frame.shape[:2]
        bbox = [0, 0, w, h]
        
        # Extract embedding
        embedding = extractor.extract(frame, bbox)
        if embedding is not None:
            embeddings.append(embedding)
    
    logger.info(f"Extracted {len(embeddings)} embeddings from {len(image_paths)} images")
    
    if len(embeddings) == 0:
        logger.error("No embeddings extracted!")
        return
    
    # Register in database
    db.register_person(global_id, person_name, embeddings)
    logger.info(f"✅ Registered {person_name} with {len(embeddings)} embeddings")
