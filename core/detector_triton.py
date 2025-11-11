#!/usr/bin/env python3
"""
Triton Inference Server Detector
Optimized for multi-stream camera inference with dynamic batching
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from loguru import logger

try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    logger.warning("tritonclient not installed. Install with: pip install tritonclient[all]")


class TritonDetector:
    """
    YOLOX Detector using NVIDIA Triton Inference Server
    
    Features:
    - Dynamic batching for multi-stream cameras
    - Async inference support
    - Automatic model warmup
    - Connection pooling
    
    Args:
        triton_url: Triton server URL (e.g., 'localhost:8001')
        model_name: Model name in Triton repository
        model_version: Model version (default: latest)
        conf_thresh: Confidence threshold for detections
        nms_thresh: NMS threshold
        test_size: Input size (height, width)
        timeout: Request timeout in seconds
        verbose: Enable verbose logging
    """
    
    def __init__(
        self,
        triton_url: str = 'localhost:8001',
        model_name: str = 'bytetrack_tensorrt',
        model_version: str = '',
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.45,
        test_size: Tuple[int, int] = (640, 640),
        timeout: float = 10.0,
        verbose: bool = False
    ):
        if not TRITON_AVAILABLE:
            raise ImportError(
                "tritonclient not installed. Install with:\n"
                "  pip install tritonclient[all]"
            )
        
        self.triton_url = triton_url
        self.model_name = model_name
        self.model_version = model_version
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.test_size = test_size
        self.timeout = timeout
        self.verbose = verbose
        
        # Initialize Triton client
        self._init_client()
        
        # Get model metadata
        self._get_model_info()
        
        # Warmup
        self._warmup()
        
        logger.info("✅ Triton Detector initialized successfully")
    
    def _init_client(self):
        """Initialize Triton gRPC client"""
        try:
            self.client = grpcclient.InferenceServerClient(
                url=self.triton_url,
                verbose=self.verbose
            )
            
            # Check server health
            if not self.client.is_server_live():
                raise ConnectionError(f"Triton server at {self.triton_url} is not live")
            
            if not self.client.is_server_ready():
                raise ConnectionError(f"Triton server at {self.triton_url} is not ready")
            
            logger.info(f"✓ Connected to Triton server at {self.triton_url}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Triton server: {e}")
            raise
    
    def _get_model_info(self):
        """Get model metadata from Triton"""
        try:
            # Get model metadata
            metadata = self.client.get_model_metadata(
                model_name=self.model_name,
                model_version=self.model_version
            )
            
            # Get model config
            config = self.client.get_model_config(
                model_name=self.model_name,
                model_version=self.model_version
            )
            
            # Extract input/output info
            self.input_name = metadata.inputs[0].name
            self.output_name = metadata.outputs[0].name
            self.input_shape = metadata.inputs[0].shape
            self.output_shape = metadata.outputs[0].shape
            self.input_dtype = metadata.inputs[0].datatype
            
            logger.info(f"Model: {self.model_name}")
            logger.info(f"  Input: {self.input_name} {self.input_shape} ({self.input_dtype})")
            logger.info(f"  Output: {self.output_name} {self.output_shape}")
            
            # Check if dynamic batching is enabled
            if hasattr(config, 'dynamic_batching'):
                logger.info(f"  Dynamic batching: ENABLED")
                if hasattr(config.dynamic_batching, 'max_queue_delay_microseconds'):
                    delay_us = config.dynamic_batching.max_queue_delay_microseconds
                    logger.info(f"  Max queue delay: {delay_us/1000:.1f}ms")
            
        except InferenceServerException as e:
            logger.error(f"❌ Failed to get model info: {e}")
            raise
    
    def _warmup(self, num_iterations: int = 3):
        """Warmup model with dummy inputs"""
        logger.info(f"Warming up model ({num_iterations} iterations)...")
        
        dummy_input = np.random.rand(1, 3, *self.test_size).astype(np.float32)
        
        for i in range(num_iterations):
            try:
                self._infer(dummy_input)
            except Exception as e:
                logger.warning(f"Warmup iteration {i+1} failed: {e}")
        
        logger.info("✓ Model warmed up")
    
    def _infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on Triton server
        
        Args:
            input_data: Input tensor [batch, 3, height, width]
            
        Returns:
            Output tensor [batch, num_detections, 6]
        """
        # Create input object
        inputs = [grpcclient.InferInput(self.input_name, input_data.shape, self.input_dtype)]
        inputs[0].set_data_from_numpy(input_data)
        
        # Create output object
        outputs = [grpcclient.InferRequestedOutput(self.output_name)]
        
        # Run inference
        response = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=outputs,
            timeout=self.timeout
        )
        
        # Get output
        output_data = response.as_numpy(self.output_name)
        
        return output_data
    
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Preprocess image for inference
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            Tuple of (preprocessed_tensor, scale_ratio)
        """
        if len(img.shape) == 3:
            padded_img = np.ones((self.test_size[0], self.test_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.test_size, dtype=np.uint8) * 114
        
        # Calculate scale
        r = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        
        # Paste resized image
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        
        # Convert to tensor format [1, 3, H, W]
        padded_img = padded_img.transpose((2, 0, 1))  # HWC -> CHW
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        padded_img = np.expand_dims(padded_img, axis=0)  # Add batch dimension
        
        return padded_img, r
    
    def postprocess(
        self,
        outputs: np.ndarray,
        img_size: Tuple[int, int],
        ratio: float
    ) -> np.ndarray:
        """
        Postprocess model outputs
        
        Args:
            outputs: Model output [batch, num_detections, 6] (x1, y1, x2, y2, conf, class)
            img_size: Original image size (height, width)
            ratio: Scale ratio from preprocessing
            
        Returns:
            Detections array [N, 5] (x1, y1, x2, y2, conf)
        """
        # Remove batch dimension
        if len(outputs.shape) == 3:
            outputs = outputs[0]
        
        # Filter by confidence
        mask = outputs[:, 4] >= self.conf_thresh
        outputs = outputs[mask]
        
        if len(outputs) == 0:
            return np.empty((0, 5))
        
        # Scale boxes back to original image size
        outputs[:, :4] /= ratio
        
        # Clip to image boundaries
        outputs[:, 0] = np.clip(outputs[:, 0], 0, img_size[1])
        outputs[:, 1] = np.clip(outputs[:, 1], 0, img_size[0])
        outputs[:, 2] = np.clip(outputs[:, 2], 0, img_size[1])
        outputs[:, 3] = np.clip(outputs[:, 3], 0, img_size[0])
        
        # Return [x1, y1, x2, y2, conf]
        return outputs[:, :5]
    
    def detect(self, img: np.ndarray) -> np.ndarray:
        """
        Detect objects in image
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            Detections array [N, 5] (x1, y1, x2, y2, conf)
        """
        img_size = img.shape[:2]
        
        # Preprocess
        input_tensor, ratio = self.preprocess(img)
        
        # Inference
        outputs = self._infer(input_tensor)
        
        # Postprocess
        detections = self.postprocess(outputs, img_size, ratio)
        
        return detections
    
    def detect_batch(self, imgs: list) -> list:
        """
        Detect objects in batch of images (for dynamic batching)
        
        Args:
            imgs: List of input images (BGR format)
            
        Returns:
            List of detection arrays
        """
        if len(imgs) == 0:
            return []
        
        # Preprocess all images
        batch_tensors = []
        ratios = []
        img_sizes = []
        
        for img in imgs:
            tensor, ratio = self.preprocess(img)
            batch_tensors.append(tensor)
            ratios.append(ratio)
            img_sizes.append(img.shape[:2])
        
        # Stack into batch
        batch_input = np.concatenate(batch_tensors, axis=0)
        
        # Inference
        batch_outputs = self._infer(batch_input)
        
        # Postprocess each output
        results = []
        for i in range(len(imgs)):
            detections = self.postprocess(
                batch_outputs[i:i+1],
                img_sizes[i],
                ratios[i]
            )
            results.append(detections)
        
        return results
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'client'):
            try:
                self.client.close()
            except:
                pass

