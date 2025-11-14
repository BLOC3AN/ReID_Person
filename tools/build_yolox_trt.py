#!/usr/bin/env python3
"""
Build YOLOX TensorRT engine with dynamic batch support
"""

import tensorrt as trt
import argparse
from loguru import logger
import os


def build_engine(onnx_path, engine_path, min_batch=1, opt_batch=2, max_batch=4, fp16=True):
    """
    Build TensorRT engine from ONNX with dynamic batch
    
    Args:
        onnx_path: Input ONNX model path
        engine_path: Output TensorRT engine path
        min_batch: Minimum batch size
        opt_batch: Optimal batch size
        max_batch: Maximum batch size
        fp16: Enable FP16 precision
    """
    logger.info("=" * 80)
    logger.info("YOLOX TensorRT Engine Builder")
    logger.info("=" * 80)
    logger.info(f"ONNX: {onnx_path}")
    logger.info(f"Engine: {engine_path}")
    logger.info(f"Batch range: [{min_batch}, {opt_batch}, {max_batch}]")
    logger.info(f"FP16: {fp16}")
    logger.info(f"TensorRT version: {trt.__version__}")
    
    # Create builder
    logger.info("\n📦 Creating TensorRT builder...")
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    logger.info(f"📖 Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            logger.error("Failed to parse ONNX file")
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            return False
    
    logger.info("✅ ONNX parsed successfully")
    
    # Print network info
    logger.info(f"\n📊 Network info:")
    logger.info(f"  Inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        logger.info(f"    {inp.name}: {inp.shape}")
    
    logger.info(f"  Outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        logger.info(f"    {out.name}: {out.shape}")
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    
    if fp16:
        logger.info("✅ FP16 mode enabled")
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Set dynamic batch
    logger.info(f"\n⚙️  Configuring dynamic batch...")
    profile = builder.create_optimization_profile()
    
    # Input shape: [batch, 3, 640, 640]
    input_name = network.get_input(0).name
    profile.set_shape(
        input_name,
        (min_batch, 3, 640, 640),  # min
        (opt_batch, 3, 640, 640),  # opt
        (max_batch, 3, 640, 640)   # max
    )
    config.add_optimization_profile(profile)
    
    logger.info(f"  Input: {input_name}")
    logger.info(f"    Min: ({min_batch}, 3, 640, 640)")
    logger.info(f"    Opt: ({opt_batch}, 3, 640, 640)")
    logger.info(f"    Max: ({max_batch}, 3, 640, 640)")
    
    # Build engine
    logger.info(f"\n🔨 Building TensorRT engine...")
    logger.info("  This may take several minutes...")
    
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        logger.error("Failed to build engine")
        return False
    
    # Save engine
    logger.info(f"\n💾 Saving engine: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    # Get file size
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    logger.info(f"✅ Engine saved: {size_mb:.1f} MB")
    
    logger.info("=" * 80)
    logger.info("✅ TensorRT engine built successfully!")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build YOLOX TensorRT engine")
    parser.add_argument(
        "-i", "--onnx",
        type=str,
        default="models/yolox_x_dynamic.onnx",
        help="Input ONNX file"
    )
    parser.add_argument(
        "-o", "--engine",
        type=str,
        default="models/yolox_x_dynamic_fp16.trt",
        help="Output TensorRT engine file"
    )
    parser.add_argument(
        "--min-batch", type=int, default=1,
        help="Minimum batch size"
    )
    parser.add_argument(
        "--opt-batch", type=int, default=2,
        help="Optimal batch size"
    )
    parser.add_argument(
        "--max-batch", type=int, default=4,
        help="Maximum batch size"
    )
    parser.add_argument(
        "--fp32", action="store_true",
        help="Use FP32 instead of FP16"
    )
    
    args = parser.parse_args()
    
    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        fp16=not args.fp32
    )

