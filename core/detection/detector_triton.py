#!/usr/bin/env python3
"""
Triton Inference Server detector for YOLOX/ByteTrack
Optimized for multi-stream inference with dynamic batching
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from loguru import logger

try:
    import tritonclient.grpc as grpcclient
except ImportError:
    logger.error("tritonclient not installed. Install with: pip install tritonclient[all]")
    raise


class TritonDetector:
    """
    YOLOX detector using Triton Inference Server
    Supports gRPC protocol for low-latency inference
    """
    
    def __init__(
        self,
        triton_url: str,
        model_name: str,
        model_version: str = '',
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.45,
        test_size: Tuple[int, int] = (640, 640),
        timeout: float = 10.0,
        verbose: bool = False
    ):
        """
        Initialize Triton detector
        
        Args:
            triton_url: Triton server URL (host:port for gRPC)
            model_name: Model name in Triton repository
            model_version: Model version (empty string for latest)
            conf_thresh: Confidence threshold
            nms_thresh: NMS threshold
            test_size: Input size (height, width)
            timeout: Request timeout in seconds
            verbose: Enable verbose logging
        """
        self.triton_url = triton_url
        self.model_name = model_name
        self.model_version = model_version
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.test_size = test_size
        self.verbose = verbose
        
        # Convert timeout to milliseconds (integer)
        self.timeout = int(timeout * 1000)
        
        # Preprocessing params (same as YOLOX)
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
        logger.info(f"Initializing Triton Detector...")
        logger.info(f"  Server: {triton_url}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Timeout: {timeout}s")
        
        # Initialize gRPC client
        self._init_client()
        
        # Get model metadata
        self._get_model_info()
        
        # Warmup
        self._warmup(num_iterations=3)
    
    def _init_client(self) -> None:
        """Initialize Triton gRPC client"""
        try:
            self.client = grpcclient.InferenceServerClient(
                url=self.triton_url,
                verbose=self.verbose
            )
            
            # Check server health
            if not self.client.is_server_live():
                raise RuntimeError(f"Triton server at {self.triton_url} is not live")
            
            if not self.client.is_server_ready():
                raise RuntimeError(f"Triton server at {self.triton_url} is not ready")
            
            logger.info(f"✅ Connected to Triton server at {self.triton_url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Triton server: {e}")
            raise
    
    def _get_model_info(self) -> None:
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
            else:
                logger.info(f"  Dynamic batching: DISABLED")
            
            logger.info("✅ Model metadata loaded")
            
        except Exception as e:
            logger.error(f"Failed to get model metadata: {e}")
            raise
    
    def _warmup(self, num_iterations: int = 3) -> None:
        """Warmup model with dummy inputs"""
        logger.info(f"Warming up model ({num_iterations} iterations)...")

        # Create dummy input with correct shape and dtype
        h, w = self.test_size
        if self.input_dtype == 'FP16':
            dummy_input = np.random.rand(1, 3, h, w).astype(np.float16)
        else:
            dummy_input = np.random.rand(1, 3, h, w).astype(np.float32)

        for i in range(num_iterations):
            try:
                self._infer(dummy_input)
            except Exception as e:
                logger.warning(f"Warmup iteration {i+1} failed: {e}")

        logger.info("✅ Model warmed up")
    
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
            client_timeout=self.timeout
        )
        
        # Get output
        output_data = response.as_numpy(self.output_name)
        
        return output_data
    
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Preprocess image using YOLOX's preproc function

        Args:
            img: Input image (BGR format)

        Returns:
            Tuple of (preprocessed_tensor, scale_ratio)
        """
        # Use YOLOX's preproc function for consistency
        from yolox.data.data_augment import preproc

        # preproc returns (img_tensor, ratio)
        # img_tensor shape: [3, H, W], dtype: float32, normalized
        img_tensor, ratio = preproc(img, self.test_size, self.rgb_means, self.std)

        # Add batch dimension: [1, 3, H, W]
        img_tensor = np.expand_dims(img_tensor, axis=0)

        # Convert to FP16 if model expects FP16
        if self.input_dtype == 'FP16':
            img_tensor = img_tensor.astype(np.float16)

        return img_tensor, ratio
    
    def postprocess(
        self,
        outputs: np.ndarray,
        img_size: Tuple[int, int],
        ratio: float
    ) -> np.ndarray:
        """
        Postprocess model outputs

        Args:
            outputs: Model output [batch, num_detections, 6] - RAW format (cx, cy, w, h, obj_conf, cls_conf)
            img_size: Original image size (height, width)
            ratio: Scale ratio from preprocessing

        Returns:
            detections: (N, 5) - [x1, y1, x2, y2, conf]
        """
        from yolox.utils.demo_utils import multiclass_nms

        # Step 1: Decode outputs from grid format to absolute coordinates
        # outputs shape: [batch, 8400, 6] where 8400 = 80x80 + 40x40 + 20x20
        # Format: [cx_offset, cy_offset, w_log, h_log, objectness_logit, class_conf_logit]

        # IMPORTANT: Make a writable copy
        outputs = np.array(outputs, copy=True)

        # Remove batch dimension first
        if len(outputs.shape) == 3:
            outputs = outputs[0]  # [num_detections, 6]

        # Check if outputs are ALREADY DECODED (w/h > 100 means absolute pixels, not log values)
        # The TensorRT model has decode_in_inference=True by default, so outputs are already decoded
        max_w_h = max(np.max(outputs[:, 2]), np.max(outputs[:, 3]))

        if max_w_h > 100:
            # Model outputs are ALREADY DECODED: [cx, cy, w, h, obj_conf, cls_conf]
            # Confidence scores are already probabilities (0-1), no need for sigmoid
            obj_conf = outputs[:, 4]
            cls_conf = outputs[:, 5]
        else:
            # Model outputs are RAW: [cx_offset, cy_offset, w_log, h_log, obj_logit, cls_logit]
            # Need to decode grid offsets and apply sigmoid

            strides = [8, 16, 32]
            hsizes = [self.test_size[0] // stride for stride in strides]
            wsizes = [self.test_size[1] // stride for stride in strides]

            grids = []
            expanded_strides = []

            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                expanded_strides.append(np.full((*shape, 1), stride))

            grids = np.concatenate(grids, 1)
            expanded_strides = np.concatenate(expanded_strides, 1)

            # Decode center coordinates
            outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides

            # Decode width/height
            outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

            # Apply sigmoid to confidence scores
            obj_conf = 1.0 / (1.0 + np.exp(-outputs[:, 4]))
            cls_conf = 1.0 / (1.0 + np.exp(-outputs[:, 5]))

        # Compute final confidence (objectness * class_conf)
        scores = obj_conf * cls_conf

        # Step 4: Filter by confidence threshold
        mask = scores >= self.conf_thresh

        if not np.any(mask):
            return np.empty((0, 5))

        # Apply mask to all data
        filtered_outputs = outputs[mask]
        filtered_scores = scores[mask]

        if len(filtered_outputs) == 0:
            return np.empty((0, 5))

        boxes = np.zeros((len(filtered_outputs), 4), dtype=np.float32)
        boxes[:, 0] = filtered_outputs[:, 0] - filtered_outputs[:, 2] / 2  # x1 = cx - w/2
        boxes[:, 1] = filtered_outputs[:, 1] - filtered_outputs[:, 3] / 2  # y1 = cy - h/2
        boxes[:, 2] = filtered_outputs[:, 0] + filtered_outputs[:, 2] / 2  # x2 = cx + w/2
        boxes[:, 3] = filtered_outputs[:, 1] + filtered_outputs[:, 3] / 2  # y2 = cy + h/2

        # Step 6: Scale boxes back to original image size
        boxes /= ratio

        # Step 7: Apply NMS (Non-Maximum Suppression)
        # multiclass_nms expects: (boxes, scores, nms_thr, score_thr)
        # scores shape should be [N, num_classes]
        scores_expanded = filtered_scores[:, np.newaxis]  # [N, 1] for single class

        nms_output = multiclass_nms(
            boxes,
            scores_expanded,
            nms_thr=self.nms_thresh,
            score_thr=self.conf_thresh
        )

        if nms_output is None or len(nms_output) == 0:
            return np.empty((0, 5))

        # nms_output format: [x1, y1, x2, y2, score, class_id]
        # We need: [x1, y1, x2, y2, score]
        detections = nms_output[:, :5]

        # Clip to image boundaries
        detections[:, 0] = np.clip(detections[:, 0], 0, img_size[1])
        detections[:, 1] = np.clip(detections[:, 1], 0, img_size[0])
        detections[:, 2] = np.clip(detections[:, 2], 0, img_size[1])
        detections[:, 3] = np.clip(detections[:, 3], 0, img_size[0])

        # Return [x1, y1, x2, y2, conf]
        return detections
    
    def detect(self, img: np.ndarray) -> np.ndarray:
        """
        Detect objects in image
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            detections: (N, 5) - [x1, y1, x2, y2, conf]
        """
        # Preprocess
        img_tensor, ratio = self.preprocess(img)
        
        # Inference
        outputs = self._infer(img_tensor)
        
        # Postprocess
        detections = self.postprocess(outputs, img.shape[:2], ratio)
        
        return detections
    
    def detect_batch(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Detect objects in batch of images
        
        Args:
            imgs: List of input images (BGR format)
            
        Returns:
            List of detections for each image
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
            detections = self.postprocess(batch_outputs[i:i+1], img_sizes[i], ratios[i])
            results.append(detections)
        
        return results
    
    def get_info(self) -> dict:
        """Get detector information"""
        return {
            'triton_url': self.triton_url,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'conf_thresh': self.conf_thresh,
            'nms_thresh': self.nms_thresh,
            'test_size': self.test_size,
            'input_name': self.input_name,
            'output_name': self.output_name,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
        }

