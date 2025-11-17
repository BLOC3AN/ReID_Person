"""
ArcFace-based Face Recognition Feature Extractor
Uses InsightFace for high-accuracy face recognition
Embedding dimension: 512
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from loguru import logger


class ArcFaceExtractor:
    """Extract face embeddings using ArcFace (InsightFace) model"""

    def __init__(self, model_name='buffalo_l', use_cuda=True, feature_dim=512, face_conf_thresh=0.5):
        """
        Args:
            model_name: InsightFace model name (default: 'buffalo_l' for high accuracy)
                       Options: 'buffalo_l', 'buffalo_s', 'antelopev2'
            use_cuda: Use GPU if available
            feature_dim: Feature embedding dimension (default: 512)
            face_conf_thresh: Face detection confidence threshold 0-1 (default: 0.5)
        """
        self.feature_dim = feature_dim
        self.use_cuda = use_cuda
        self.face_conf_thresh = face_conf_thresh

        # Initialize InsightFace
        try:
            import insightface
            from insightface.app import FaceAnalysis

            # Determine device
            if use_cuda and torch.cuda.is_available():
                ctx_id = 0  # GPU 0
                self.device = 'cuda'
            else:
                ctx_id = -1  # CPU
                self.device = 'cpu'

            # Initialize face analysis app
            logger.info(f"Initializing ArcFace model: {model_name} on {self.device}")
            self.app = FaceAnalysis(
                name=model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if ctx_id >= 0 else ['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

            logger.info(f"✅ Loaded ArcFace model: {model_name}")

        except Exception as e:
            logger.error(f"❌ Failed to load ArcFace model: {e}")
            raise

    def extract(self, frame, bbox):
        """
        Extract face embedding from bbox
        Args:
            frame: Input frame (H, W, 3) BGR
            bbox: [x, y, w, h]
        Returns:
            Feature vector (512,) normalized
        """
        x, y, w, h = [int(v) for v in bbox]

        # Validate bbox
        if w <= 0 or h <= 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Crop face region with padding
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Detect and extract face embedding
        try:
            faces = self.app.get(crop)

            if len(faces) == 0:
                # No face detected, return zero vector
                logger.debug(f"No face detected in bbox [{x}, {y}, {w}, {h}]")
                return np.zeros(self.feature_dim, dtype=np.float32)

            # Get the largest face (by bbox area)
            if len(faces) > 1:
                areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
                largest_idx = np.argmax(areas)
                face = faces[largest_idx]
            else:
                face = faces[0]

            # Get embedding (already normalized by InsightFace)
            embedding = face.embedding.astype(np.float32)

            # Ensure it's 512-dim (some models may output different dims)
            if embedding.shape[0] != self.feature_dim:
                logger.warning(f"Embedding dim mismatch: {embedding.shape[0]} != {self.feature_dim}")
                # Pad or truncate to match expected dimension
                if embedding.shape[0] < self.feature_dim:
                    embedding = np.pad(embedding, (0, self.feature_dim - embedding.shape[0]))
                else:
                    embedding = embedding[:self.feature_dim]

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.warning(f"Failed to extract face embedding: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)

    def extract_batch(self, frame, bboxes):
        """
        Extract face embeddings for multiple bboxes
        Args:
            frame: Input frame (H, W, 3) BGR
            bboxes: List of [x, y, w, h]
        Returns:
            Features (N, 512) normalized
        """
        if len(bboxes) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        embeddings = []
        for bbox in bboxes:
            embedding = self.extract(frame, bbox)
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)


def build_arcface_extractor(model_name='buffalo_l', use_cuda=True):
    """Factory function to build ArcFace extractor"""
    return ArcFaceExtractor(model_name=model_name, use_cuda=use_cuda)

