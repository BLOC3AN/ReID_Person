"""
YOLOX Detector for Person Detection
Supports both YOLOX-X and ByteTrack MOT17 models
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from loguru import logger

# Add ByteTrack to path
BYTETRACK_PATH = Path(__file__).parent.parent.parent / "ByteTrack_Predict"
sys.path.insert(0, str(BYTETRACK_PATH))

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import postprocess, fuse_model


class YOLOXDetector:
    """
    YOLOX detector wrapper
    Supports MOT17 and YOLOX-X models
    """
    
    def __init__(self, model_path, model_type='mot17', device='cuda', 
                 fp16=True, conf_thresh=0.5, nms_thresh=0.45, 
                 test_size=(640, 640)):
        """
        Args:
            model_path: Path to model weights
            model_type: 'mot17' or 'yolox'
            device: 'cuda' or 'cpu'
            fp16: Use FP16 precision
            conf_thresh: Confidence threshold
            nms_thresh: NMS threshold
            test_size: Input size (height, width)
        """
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.fp16 = fp16 and (device == 'cuda')
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.test_size = test_size
        
        logger.info(f"Initializing YOLOX Detector ({model_type.upper()})...")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  FP16: {self.fp16}")
        logger.info(f"  Conf threshold: {conf_thresh}")
        logger.info(f"  NMS threshold: {nms_thresh}")
        
        # Load experiment config
        exp_file = BYTETRACK_PATH / "exps/example/mot/yolox_x_mix_det.py"
        self.exp = get_exp(str(exp_file), None)
        self.exp.test_conf = conf_thresh
        self.exp.nmsthre = nms_thresh
        self.exp.test_size = test_size
        
        # Load model
        self.model = self.exp.get_model().to(self.device)
        self.model.eval()
        
        # Load weights
        logger.info(f"Loading weights from {model_path}...")
        ckpt = torch.load(model_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)
        
        # Fuse model for faster inference
        logger.info("Fusing model...")
        self.model = fuse_model(self.model)
        
        if self.fp16:
            self.model = self.model.half()
        
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.num_classes = self.exp.num_classes
        
        logger.info("✅ Detector initialized successfully")
    
    def detect(self, frame):
        """
        Detect objects in frame
        
        Args:
            frame: Input frame (H, W, 3) BGR
            
        Returns:
            detections: (N, 7) - [x1, y1, x2, y2, conf, cls, -1]
        """
        height, width = frame.shape[:2]
        
        # Preprocess
        img, ratio = preproc(frame, self.test_size, self.rgb_means, self.std)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        
        if self.fp16:
            img = img.half()
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, 
                                self.conf_thresh, self.nms_thresh)
        
        if outputs[0] is None:
            return np.empty((0, 7))
        
        # Scale back to original size
        detections = outputs[0].cpu().numpy()
        detections[:, :4] /= ratio
        
        return detections
    
    def get_info(self):
        """Get detector information"""
        return {
            'model_type': self.model_type,
            'device': str(self.device),
            'fp16': self.fp16,
            'conf_thresh': self.conf_thresh,
            'nms_thresh': self.nms_thresh,
            'test_size': self.test_size,
            'num_classes': self.num_classes,
        }

