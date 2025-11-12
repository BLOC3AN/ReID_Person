#!/usr/bin/env python3
"""
Model Preloading Script for Register Service
Preloads detector and extractor models at startup to avoid lazy loading delays
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from loguru import logger
from core.detector import YOLOXDetector
from core.feature_extractor import ArcFaceExtractor


def preload_models():
    """Preload all models needed for registration"""
    
    logger.info("=" * 80)
    logger.info("🚀 PRELOADING MODELS FOR REGISTER SERVICE")
    logger.info("=" * 80)
    
    try:
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA Available: {cuda_available}")
        if cuda_available:
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # 1. Preload YOLOX Detector (MOT17)
        logger.info("\n📦 Preloading YOLOX Detector (MOT17)...")
        model_path = Path(__file__).parent.parent / "models" / "bytetrack_x_mot17.pth.tar"
        
        if not model_path.exists():
            logger.error(f"❌ Model not found: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        detector = YOLOXDetector(
            model_path=str(model_path),
            model_type="mot17",
            device="cuda" if cuda_available else "cpu",
            fp16=cuda_available,
            conf_thresh=0.6,
            nms_thresh=0.45
        )
        logger.info("✅ YOLOX Detector loaded successfully")
        
        # 2. Preload ArcFace Extractor
        logger.info("\n📦 Preloading ArcFace Extractor...")
        extractor = ArcFaceExtractor(
            model_name='buffalo_l',
            use_cuda=cuda_available
        )
        logger.info("✅ ArcFace Extractor loaded successfully")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL MODELS PRELOADED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("🎯 Register service is ready for instant inference!")
        logger.info("=" * 80 + "\n")
        
        return detector, extractor
        
    except Exception as e:
        logger.error(f"❌ Failed to preload models: {e}")
        raise


if __name__ == "__main__":
    preload_models()

