"""
OSNet-based ReID Feature Extractor
Lightweight (~2-3MB), optimized for real-time multi-camera tracking
Embedding dimension: 512 (omni-scale features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as transforms
from pathlib import Path
from loguru import logger


class OSNetExtractor:
    """Extract ReID features using OSNet model"""
    
    def __init__(self, model_path=None, use_cuda=True, feature_dim=512):
        """
        Args:
            model_path: Path to pretrained weights (optional)
            use_cuda: Use GPU if available
            feature_dim: Feature embedding dimension (default: 512)
        """
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.feature_dim = feature_dim
        
        # Try to load from torchreid first
        try:
            import torchreid
            self.net = torchreid.models.build_model(
                name='osnet_x0_5',
                num_classes=1000,
                pretrained=True
            )
            self.net.classifier = nn.Identity()
            logger.info("✅ Loaded OSNet from torchreid")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load from torchreid: {e}")
            logger.info("Using fallback OSNet implementation")
            self.net = self._build_osnet()
        
        if model_path and Path(model_path).exists():
            self._load_weights(model_path)
            logger.info(f"✅ Loaded weights from {model_path}")
        
        self.net.to(self.device)
        self.net.eval()
        
        # Input size: 256x128 (standard for ReID)
        self.size = (256, 128)
        
        # ImageNet normalization
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def _build_osnet(self):
        """Build simple OSNet-like model as fallback"""
        return nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.feature_dim),
        )
    
    def _load_weights(self, model_path):
        """Load pretrained weights"""
        try:
            ckpt = torch.load(model_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                self.net.load_state_dict(ckpt['model'])
            else:
                self.net.load_state_dict(ckpt)
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}")
    
    def extract(self, frame, bbox):
        """
        Extract feature for single person
        Args:
            frame: Input frame (H, W, 3) BGR
            bbox: [x, y, w, h]
        Returns:
            Feature vector (512,) normalized
        """
        x, y, w, h = [int(v) for v in bbox]
        crop = frame[y:y+h, x:x+w]
        
        if crop.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        crop = cv2.resize(crop, (self.size[1], self.size[0]))
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = transforms.ToPILImage()(crop_rgb)
        tensor = self.norm(crop_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat = self.net(tensor)
            feat = F.normalize(feat, p=2, dim=1)
        
        return feat.cpu().numpy()[0].astype(np.float32)
    
    def extract_batch(self, frame, bboxes):
        """
        Extract features for multiple persons
        Args:
            frame: Input frame (H, W, 3) BGR
            bboxes: List of [x, y, w, h]
        Returns:
            Features (N, 512) normalized
        """
        if len(bboxes) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        
        tensors = []
        for bbox in bboxes:
            x, y, w, h = [int(v) for v in bbox]
            crop = frame[y:y+h, x:x+w]
            
            if crop.size == 0:
                crop = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
            else:
                crop = cv2.resize(crop, (self.size[1], self.size[0]))
            
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = transforms.ToPILImage()(crop_rgb)
            tensor = self.norm(crop_pil)
            tensors.append(tensor)
        
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            feats = self.net(batch)
            feats = F.normalize(feats, p=2, dim=1)
        
        return feats.cpu().numpy().astype(np.float32)


def build_osnet_extractor(model_path=None, use_cuda=True):
    """Factory function to build OSNet extractor"""
    return OSNetExtractor(model_path=model_path, use_cuda=use_cuda)

