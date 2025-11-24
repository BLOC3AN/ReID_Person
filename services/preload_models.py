#!/usr/bin/env python3
"""
Model Preloading Script
Preloads all pipeline components at startup to avoid lazy loading delays
Can be used for both Register Service and Detection Service
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import yaml
import torch
import argparse
from loguru import logger
from core.preloaded_manager import preloaded_manager


def preload_models(config_path=None):
    """
    Preload all models using PreloadedPipelineManager

    Args:
        config_path: Optional path to config file
    """

    logger.info("=" * 80)
    logger.info("🚀 PRELOADING MODELS")
    logger.info("=" * 80)

    try:
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA Available: {cuda_available}")
        if cuda_available:
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Use PreloadedPipelineManager to load all components
        logger.info("\n📦 Initializing PreloadedPipelineManager...")
        preloaded_manager.initialize(config_path=config_path)

        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL MODELS PRELOADED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("🎯 Services are ready for instant inference!")
        logger.info("=" * 80 + "\n")

        # Keep the process running (for service mode)
        logger.info("💡 Models are now loaded and ready.")
        logger.info("💡 You can now run detection/tracking scripts with instant startup.")
        logger.info("💡 Press Ctrl+C to exit.\n")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to preload models: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preload pipeline models")
    parser.add_argument("--config", type=str, default=None,
                       help="Config file path (optional)")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as daemon (keep process alive)")

    args = parser.parse_args()

    # Preload models
    preload_models(config_path=args.config)

    # Keep running if daemon mode
    if args.daemon:
        try:
            import signal
            import time

            def signal_handler(sig, frame):
                logger.info("\n👋 Shutting down preload service...")
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            logger.info("🔄 Running in daemon mode. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n👋 Shutting down preload service...")
    else:
        logger.info("✅ Preloading complete. Exiting.")

