"""
TensorRT-optimized YOLOX Detector
Provides 3-5x faster inference than PyTorch
Compatible interface with YOLOXDetector
"""

import numpy as np
from loguru import logger


class TensorRTDetector:
    """
    TensorRT-optimized YOLOX detector
    ~3-5x faster than PyTorch implementation
    Compatible interface with YOLOXDetector
    """
    
    def __init__(self, engine_path, conf_thresh=0.5, nms_thresh=0.45, 
                 test_size=(640, 640), num_classes=1):
        """
        Args:
            engine_path: Path to TensorRT engine file (.trt)
            conf_thresh: Confidence threshold
            nms_thresh: NMS threshold
            test_size: Input size (height, width)
            num_classes: Number of classes (1 for person detection)
        """
        self.engine_path = engine_path
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.test_size = test_size
        self.num_classes = num_classes
        
        logger.info(f"Initializing TensorRT Detector...")
        logger.info(f"  Engine: {engine_path}")
        logger.info(f"  Conf threshold: {conf_thresh}")
        logger.info(f"  NMS threshold: {nms_thresh}")
        logger.info(f"  Test size: {test_size}")
        
        # Load TensorRT engine
        self.engine, self.context = self._load_engine(engine_path)
        
        # Get input/output info
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        
        # Get shapes
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        
        # Handle dynamic batch
        if -1 in self.input_shape:
            self.input_shape = [1, 3, test_size[0], test_size[1]]
            self.context.set_input_shape(self.input_name, self.input_shape)
        
        logger.info(f"  Input shape: {self.input_shape}")
        logger.info(f"  Output shape: {self.context.get_tensor_shape(self.output_name)}")
        
        # Allocate buffers
        self._allocate_buffers()
        
        # Preprocessing params (same as YOLOX)
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
        logger.info("✅ TensorRT Detector initialized successfully")
    
    def _load_engine(self, engine_path):
        """Load TensorRT engine from file"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(engine_data)
            context = engine.create_execution_context()
            
            return engine, context
            
        except ImportError as e:
            logger.error(f"❌ Failed to import TensorRT: {e}")
            logger.info("Install with: pip install tensorrt pycuda")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load TensorRT engine: {e}")
            raise
    
    def _allocate_buffers(self):
        """Allocate GPU buffers for input/output"""
        import pycuda.driver as cuda
        
        # Get output shape
        output_shape = self.context.get_tensor_shape(self.output_name)
        
        # Calculate sizes
        input_size = int(np.prod(self.input_shape))
        output_size = int(np.prod(output_shape))
        
        # Allocate host (CPU) buffers
        self.h_input = cuda.pagelocked_empty(input_size, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(output_size, dtype=np.float32)
        
        # Allocate device (GPU) buffers
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        
        # Create stream
        self.stream = cuda.Stream()
        
        # Store output shape for later use
        self.output_shape = output_shape
    
    def detect(self, frame):
        """
        Detect objects in frame using TensorRT
        
        Args:
            frame: Input frame (H, W, 3) BGR
            
        Returns:
            detections: (N, 7) - [x1, y1, x2, y2, conf, cls, -1]
        """
        import pycuda.driver as cuda
        from yolox.data.data_augment import preproc
        
        # Preprocess (same as PyTorch version)
        img, ratio = preproc(frame, self.test_size, self.rgb_means, self.std)
        
        # Copy input to host buffer
        np.copyto(self.h_input, img.ravel())
        
        # Copy input to GPU
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        
        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy output from GPU
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        
        # Reshape output
        output = self.h_output.reshape(self.output_shape)
        
        # Postprocess (NMS)
        detections = self._postprocess(output, ratio)
        
        return detections
    
    def _postprocess(self, output, ratio):
        """
        Apply NMS and scale back to original size
        
        Args:
            output: Raw model output
            ratio: Resize ratio from preprocessing
            
        Returns:
            detections: (N, 7) - [x1, y1, x2, y2, conf, cls, -1]
        """
        import torch
        from yolox.utils import postprocess
        
        # Convert to torch tensor
        output_tensor = torch.from_numpy(output).float()
        
        # Apply postprocess (NMS)
        processed = postprocess(
            output_tensor, 
            self.num_classes,
            self.conf_thresh, 
            self.nms_thresh
        )
        
        if processed[0] is None:
            return np.empty((0, 7))
        
        # Scale back to original size
        detections = processed[0].cpu().numpy()
        detections[:, :4] /= ratio
        
        return detections
    
    def get_info(self):
        """Get detector information"""
        return {
            'backend': 'TensorRT',
            'engine_path': self.engine_path,
            'conf_thresh': self.conf_thresh,
            'nms_thresh': self.nms_thresh,
            'test_size': self.test_size,
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
        }
    
    def __del__(self):
        """Cleanup GPU resources"""
        try:
            if hasattr(self, 'd_input'):
                self.d_input.free()
            if hasattr(self, 'd_output'):
                self.d_output.free()
        except:
            pass


# Alias for backward compatibility
YOLOXDetectorTRT = TensorRTDetector

