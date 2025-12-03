"""Person registration logic"""

import os
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from typing import List, Union

from core import YOLOXDetector, ArcFaceExtractor, QdrantVectorDB


def register_person_mot17(video_path: str, person_name: str, global_id: int, 
                          sample_rate: int = 5, delete_existing: bool = False, 
                          face_conf_thresh: float = 0.5, skip_body_detection: bool = False, 
                          detector=None, extractor=None):
    """Register a person using MOT17 model"""
    
    logger.info(f"Registering person: {person_name} (ID: {global_id})")
    
    # Initialize detector if needed
    if skip_body_detection:
        detector = None
    elif detector is None:
        model_path = Path(__file__).parent.parent / "models" / "bytetrack_x_mot17.pth.tar"
        detector = YOLOXDetector(
            model_path=str(model_path),
            model_type="mot17",
            conf_thresh=0.6,
            nms_thresh=0.45
        )
    
    # Initialize extractor if needed
    if extractor is None:
        extractor = ArcFaceExtractor(model_name='buffalo_l', use_cuda=True, face_conf_thresh=face_conf_thresh)
    
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
            # Detect person
            detections = detector.detect(frame)
            if len(detections) == 0:
                continue
            # Use first detection
            x1, y1, x2, y2, conf = detections[0]
            bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
        
        # Extract embedding
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
                                face_conf_thresh: float = 0.5, extractor=None):
    """Register a person from a list of images"""
    
    logger.info(f"Registering person from images: {person_name} (ID: {global_id})")
    
    # Initialize extractor if needed
    if extractor is None:
        extractor = ArcFaceExtractor(model_name='buffalo_l', use_cuda=True, face_conf_thresh=face_conf_thresh)
    
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
