#!/usr/bin/env python3
"""
Model Preloading Script for Register Service
Preloads detector and extractor models at startup to avoid lazy loading delays
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import yaml
import torch
from loguru import logger
from core.detector import YOLOXDetector
from core.feature_extractor import ArcFaceExtractor
from core.arcface_triton_client import ArcFaceTritonClient


def preload_models():
    """Preload all models needed for registration"""

    logger.info("=" * 80)
    logger.info("🚀 PRELOADING MODELS FOR REGISTER SERVICE")
    logger.info("=" * 80)

    try:
        # Load config to check backend settings
        # Priority: configs/config.yaml (for services), then .streamlit/configs/config.yaml (for UI)
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        if not config_path.exists():
            config_path = Path(__file__).parent.parent / ".streamlit" / "configs" / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at configs/config.yaml or .streamlit/configs/config.yaml")

        logger.info(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        reid_backend = config.get('reid', {}).get('backend', 'insightface')
        logger.info(f"ReID Backend: {reid_backend}")

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

        # 2. Preload ArcFace Extractor (Triton Pipeline, Triton, or InsightFace)
        if reid_backend == 'triton_pipeline':
            logger.info("\n📦 Preloading Face Recognition Pipeline (SCRFD + ArcFace)...")
            from core import FaceRecognitionTriton

            triton_cfg = config.get('reid', {}).get('triton', {})
            triton_url = triton_cfg.get('url', 'localhost:8101')
            arcface_model = triton_cfg.get('arcface_model', 'arcface_tensorrt')
            face_detector_model = triton_cfg.get('face_detector_model', 'scrfd_10g')
            feature_dim = triton_cfg.get('feature_dim', 512)
            face_conf_threshold = triton_cfg.get('face_conf_threshold', 0.5)

            extractor = FaceRecognitionTriton(
                triton_url=triton_url,
                face_detector_model=face_detector_model,
                arcface_model=arcface_model,
                feature_dim=feature_dim,
                face_conf_threshold=face_conf_threshold
            )
            logger.info(f"✅ Face Recognition Pipeline loaded ({triton_url})")
            logger.info(f"   Face Detector: {face_detector_model}")
            logger.info(f"   ArcFace: {arcface_model}")

        elif reid_backend == 'triton':
            logger.info("\n📦 Preloading ArcFace Triton Client...")
            logger.warning("⚠️  Using Triton ArcFace without face detection (DEPRECATED)")

            triton_cfg = config.get('reid', {}).get('triton', {})
            triton_url = triton_cfg.get('url', 'localhost:8101')
            model_name = triton_cfg.get('arcface_model', triton_cfg.get('model_name', 'arcface_tensorrt'))
            feature_dim = triton_cfg.get('feature_dim', 512)

            extractor = ArcFaceTritonClient(
                triton_url=triton_url,
                model_name=model_name,
                feature_dim=feature_dim
            )
            logger.info(f"✅ ArcFace Triton Client loaded successfully ({triton_url}/{model_name})")
        else:
            logger.info("\n📦 Preloading ArcFace Extractor (InsightFace)...")
            arcface_model = config.get('reid', {}).get('arcface_model_name', 'buffalo_l')
            extractor = ArcFaceExtractor(
                model_name=arcface_model,
                use_cuda=cuda_available
            )
            logger.info("✅ ArcFace Extractor (InsightFace) loaded successfully")

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

