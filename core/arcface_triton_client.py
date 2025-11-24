"""
ArcFace Triton Client for Face Embedding Extraction
Uses Triton Inference Server for GPU-accelerated batch inference
"""

import numpy as np
import cv2
import tritonclient.grpc as grpcclient
from loguru import logger


class ArcFaceTritonClient:
    """ArcFace client using Triton Inference Server"""
    
    def __init__(self, triton_url='localhost:8101', model_name='arcface_tensorrt', feature_dim=512):
        """
        Args:
            triton_url: Triton server gRPC URL
            model_name: Model name in Triton repository
            feature_dim: Feature embedding dimension (512)
        """
        self.triton_url = triton_url
        self.model_name = model_name
        self.feature_dim = feature_dim
        
        # Initialize Triton client
        try:
            self.client = grpcclient.InferenceServerClient(url=triton_url)
            
            # Check server health
            if not self.client.is_server_live():
                raise RuntimeError(f"Triton server not live at {triton_url}")
            
            # Check model ready
            if not self.client.is_model_ready(model_name):
                raise RuntimeError(f"Model {model_name} not ready")
            
            logger.info(f"✅ ArcFace Triton client connected: {triton_url}/{model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Triton: {e}")
            raise
    
    def preprocess_face(self, frame, bbox):
        """
        Preprocess face crop for ArcFace
        Args:
            frame: Input frame (H, W, 3) BGR
            bbox: [x, y, w, h]
        Returns:
            Preprocessed face (3, 112, 112) FP32 or None if invalid
        """
        x, y, w, h = [int(v) for v in bbox]
        
        # Validate bbox
        if w <= 0 or h <= 0:
            return None
        
        # Crop face region with padding
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
        
        # Resize to 112x112
        face = cv2.resize(crop, (112, 112))
        
        # Convert BGR to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        face = face.astype(np.float32) / 255.0
        
        # Transpose to (3, 112, 112)
        face = np.transpose(face, (2, 0, 1))
        
        return face
    
    def extract_batch(self, frame, bboxes, timeout=0.05):
        """
        Extract face embeddings for multiple bboxes using Triton batch inference
        Args:
            frame: Input frame (H, W, 3) BGR
            bboxes: List of [x, y, w, h]
            timeout: Inference timeout in seconds (default: 50ms)
        Returns:
            Features (N, 512) normalized FP32
        """
        if len(bboxes) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        
        # Preprocess all faces
        faces = []
        valid_indices = []
        
        for i, bbox in enumerate(bboxes):
            face = self.preprocess_face(frame, bbox)
            if face is not None:
                faces.append(face)
                valid_indices.append(i)
        
        # If no valid faces, return zeros
        if len(faces) == 0:
            return np.zeros((len(bboxes), self.feature_dim), dtype=np.float32)
        
        # Stack to batch (N, 3, 112, 112)
        batch = np.stack(faces, axis=0).astype(np.float32)
        
        # Create Triton input
        inputs = [
            grpcclient.InferInput('input.1', batch.shape, 'FP32')
        ]
        inputs[0].set_data_from_numpy(batch)
        
        # Create Triton output
        outputs = [
            grpcclient.InferRequestedOutput('683')
        ]
        
        # Inference
        try:
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
                timeout=int(timeout * 1e6)  # Convert to microseconds
            )
            
            # Get embeddings (N, 512)
            embeddings = response.as_numpy('683').astype(np.float32)
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)  # Avoid division by zero
            embeddings = embeddings / norms
            
            # Map back to original indices (fill zeros for invalid faces)
            result = np.zeros((len(bboxes), self.feature_dim), dtype=np.float32)
            for i, valid_idx in enumerate(valid_indices):
                result[valid_idx] = embeddings[i]
            
            return result
            
        except Exception as e:
            logger.warning(f"Triton inference failed: {e}")
            return np.zeros((len(bboxes), self.feature_dim), dtype=np.float32)

    def extract(self, frame, bbox, timeout=0.05):
        """
        Extract face embedding for a single bbox (wrapper for extract_batch)
        Args:
            frame: Input frame (H, W, 3) BGR
            bbox: [x, y, w, h]
            timeout: Inference timeout in seconds (default: 50ms)
        Returns:
            Feature vector (512,) normalized FP32
        """
        # Call extract_batch with single bbox
        embeddings = self.extract_batch(frame, [bbox], timeout=timeout)
        return embeddings[0]

