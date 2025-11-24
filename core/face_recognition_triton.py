"""
Face Recognition Pipeline with Triton
Combines SCRFD face detector + ArcFace embedding extractor
"""

import numpy as np
import cv2
from loguru import logger

try:
    from .scrfd_triton_client import SCRFDTritonClient
    from .arcface_triton_client import ArcFaceTritonClient
except ImportError:
    from scrfd_triton_client import SCRFDTritonClient
    from arcface_triton_client import ArcFaceTritonClient


class FaceRecognitionTriton:
    """
    Complete face recognition pipeline using Triton
    Pipeline: Person bbox → SCRFD face detection → ArcFace embedding
    """
    
    def __init__(self, triton_url='localhost:8101', 
                 face_detector_model='scrfd_10g',
                 arcface_model='arcface_tensorrt',
                 feature_dim=512,
                 face_conf_threshold=0.5):
        """
        Args:
            triton_url: Triton server gRPC URL
            face_detector_model: SCRFD model name in Triton
            arcface_model: ArcFace model name in Triton
            feature_dim: Embedding dimension (512)
            face_conf_threshold: Confidence threshold for face detection
        """
        self.triton_url = triton_url
        self.feature_dim = feature_dim
        self.face_conf_threshold = face_conf_threshold
        
        # Initialize face detector
        logger.info(f"Initializing SCRFD face detector: {face_detector_model}")
        self.face_detector = SCRFDTritonClient(
            triton_url=triton_url,
            model_name=face_detector_model,
            conf_threshold=face_conf_threshold
        )
        logger.info("✅ SCRFD face detector initialized")
        
        # Initialize ArcFace extractor
        logger.info(f"Initializing ArcFace extractor: {arcface_model}")
        self.arcface = ArcFaceTritonClient(
            triton_url=triton_url,
            model_name=arcface_model,
            feature_dim=feature_dim
        )
        logger.info("✅ ArcFace extractor initialized")
        
        logger.info("✅ Face Recognition Triton pipeline ready")
    
    def extract(self, frame, person_bbox):
        """
        Extract face embedding from person bbox
        Pipeline: Person bbox → Crop → Face detection → Face embedding
        
        Args:
            frame: Input frame (H, W, 3) BGR
            person_bbox: Person bounding box [x, y, w, h]
        Returns:
            Face embedding (512,) normalized, or zero vector if no face detected
        """
        x, y, w, h = [int(v) for v in person_bbox]
        
        # Validate person bbox
        if w <= 0 or h <= 0:
            logger.debug(f"Invalid person bbox: {person_bbox}")
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # Crop person region with padding
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            logger.debug(f"Empty person crop for bbox: {person_bbox}")
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # Detect faces in person crop
        face_bboxes = self.face_detector.detect_faces_in_crop(person_crop)
        
        if len(face_bboxes) == 0:
            logger.debug(f"No face detected in person bbox: {person_bbox}")
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # Select largest face (by area)
        if len(face_bboxes) > 1:
            areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in face_bboxes]
            largest_idx = np.argmax(areas)
            face_bbox = face_bboxes[largest_idx]
        else:
            face_bbox = face_bboxes[0]
        
        # Convert face bbox from [x1, y1, x2, y2, conf] to [x, y, w, h]
        fx1, fy1, fx2, fy2, conf = face_bbox
        face_bbox_xywh = [fx1, fy1, fx2 - fx1, fy2 - fy1]
        
        logger.debug(f"Face detected: bbox={face_bbox_xywh}, conf={conf:.3f}")
        
        # Extract face embedding from person crop
        embedding = self.arcface.extract(person_crop, face_bbox_xywh)
        
        return embedding
    
    def extract_batch(self, frame, person_bboxes):
        """
        Extract face embeddings for multiple person bboxes
        
        Args:
            frame: Input frame (H, W, 3) BGR
            person_bboxes: List of person bounding boxes [[x, y, w, h], ...]
        Returns:
            Face embeddings (N, 512) normalized
        """
        if len(person_bboxes) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        
        embeddings = []
        for person_bbox in person_bboxes:
            embedding = self.extract(frame, person_bbox)
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)


def build_face_recognition_triton(triton_url='localhost:8101',
                                   face_detector_model='scrfd_10g',
                                   arcface_model='arcface_tensorrt',
                                   feature_dim=512,
                                   face_conf_threshold=0.5):
    """Factory function to build face recognition pipeline"""
    return FaceRecognitionTriton(
        triton_url=triton_url,
        face_detector_model=face_detector_model,
        arcface_model=arcface_model,
        feature_dim=feature_dim,
        face_conf_threshold=face_conf_threshold
    )

