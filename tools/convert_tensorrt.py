#!/usr/bin/env python3
"""
Convert ONNX model to TensorRT engine
Supports FP32, FP16, and INT8 precision
"""

import sys
import argparse
from pathlib import Path
from loguru import logger


def convert_to_tensorrt(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    int8: bool = False,
    workspace: int = 2048,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 1,
    verbose: bool = False
):
    """
    Convert ONNX model to TensorRT engine
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Output TensorRT engine path
        fp16: Enable FP16 precision
        int8: Enable INT8 precision (requires calibration)
        workspace: Max workspace size in MB
        min_batch: Minimum batch size (for dynamic batch)
        opt_batch: Optimal batch size (for dynamic batch)
        max_batch: Maximum batch size (for dynamic batch)
        verbose: Verbose logging
    
    Returns:
        engine_path: Path to TensorRT engine
    """
    
    # Convert to absolute paths
    onnx_path = str(Path(onnx_path).resolve())
    engine_path = str(Path(engine_path).resolve())

    logger.info("=" * 80)
    logger.info("CONVERTING ONNX TO TENSORRT ENGINE")
    logger.info("=" * 80)
    logger.info(f"ONNX: {onnx_path}")
    logger.info(f"Engine: {engine_path}")
    logger.info(f"FP16: {fp16}")
    logger.info(f"INT8: {int8}")
    logger.info(f"Workspace: {workspace} MB")
    logger.info(f"Batch range: [{min_batch}, {opt_batch}, {max_batch}]")
    logger.info("=" * 80)

    # Check if ONNX file exists
    if not Path(onnx_path).exists():
        logger.error(f"❌ ONNX file not found: {onnx_path}")
        return None
    
    try:
        import tensorrt as trt

        logger.info(f"\n📦 TensorRT version: {trt.__version__}")
        
    except ImportError as e:
        logger.error(f"❌ Failed to import TensorRT: {e}")
        logger.info("\n💡 Install TensorRT:")
        logger.info("   1. Download from: https://developer.nvidia.com/tensorrt")
        logger.info("   2. Or use: pip install tensorrt")
        logger.info("   3. Install pycuda: pip install pycuda")
        return None
    
    # Create builder
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    # Create network
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    # Create ONNX parser
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    logger.info(f"\n📖 Parsing ONNX model...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            logger.error("❌ Failed to parse ONNX file")
            for error in range(parser.num_errors):
                logger.error(f"  Error {error}: {parser.get_error(error)}")
            return None
    
    logger.info("✅ ONNX parsed successfully")
    
    # Print network info
    logger.info(f"\n📊 Network Information:")
    logger.info(f"  Layers: {network.num_layers}")
    logger.info(f"  Inputs: {network.num_inputs}")
    logger.info(f"  Outputs: {network.num_outputs}")
    
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        logger.info(f"\n  Input {i}:")
        logger.info(f"    Name: {input_tensor.name}")
        logger.info(f"    Shape: {input_tensor.shape}")
        logger.info(f"    Type: {input_tensor.dtype}")
    
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        logger.info(f"\n  Output {i}:")
        logger.info(f"    Name: {output_tensor.name}")
        logger.info(f"    Shape: {output_tensor.shape}")
        logger.info(f"    Type: {output_tensor.dtype}")
    
    # Create builder config
    config = builder.create_builder_config()
    
    # Set workspace size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 20)  # MB to bytes
    logger.info(f"\n⚙️  Workspace size: {workspace} MB")
    
    # Enable FP16
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("✅ FP16 mode enabled")
    elif fp16:
        logger.warning("⚠️  FP16 not supported on this platform")
    
    # Enable INT8
    if int8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.warning("⚠️  INT8 enabled but no calibrator provided")
            logger.info("   INT8 will use default quantization (may reduce accuracy)")
        else:
            logger.warning("⚠️  INT8 not supported on this platform")
    
    # Handle dynamic batch
    input_tensor = network.get_input(0)
    if input_tensor.shape[0] == -1:  # Dynamic batch
        logger.info(f"\n🔄 Configuring dynamic batch size...")
        
        profile = builder.create_optimization_profile()
        
        # Get input shape
        input_shape = list(input_tensor.shape)
        
        # Set min/opt/max shapes
        min_shape = [min_batch] + input_shape[1:]
        opt_shape = [opt_batch] + input_shape[1:]
        max_shape = [max_batch] + input_shape[1:]
        
        logger.info(f"  Min shape: {min_shape}")
        logger.info(f"  Opt shape: {opt_shape}")
        logger.info(f"  Max shape: {max_shape}")
        
        profile.set_shape(
            input_tensor.name,
            min=min_shape,
            opt=opt_shape,
            max=max_shape
        )
        
        config.add_optimization_profile(profile)
    else:
        logger.info(f"\n📌 Using fixed batch size: {input_tensor.shape[0]}")
    
    # Build engine
    logger.info(f"\n🔨 Building TensorRT engine...")
    logger.info("   This may take several minutes...")
    
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            logger.error("❌ Failed to build TensorRT engine")
            return None
        
        logger.info("✅ Engine built successfully")
        
    except Exception as e:
        logger.error(f"❌ Engine build failed: {e}")
        return None
    
    # Save engine
    logger.info(f"\n💾 Saving engine to {engine_path}...")
    
    output_path = Path(engine_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    engine_size = Path(engine_path).stat().st_size / (1024 * 1024)  # MB
    logger.info(f"✅ Engine saved ({engine_size:.2f} MB)")
    
    # Test engine
    logger.info(f"\n🧪 Testing TensorRT engine...")
    try:
        test_tensorrt_engine(engine_path)
    except Exception as e:
        logger.warning(f"⚠️  Engine test failed: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ CONVERSION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"\n📁 TensorRT engine saved to: {engine_path}")
    logger.info("\n📝 Next steps:")
    logger.info("  1. Test inference: python tools/test_tensorrt.py --engine <engine_path>")
    logger.info("  2. Benchmark: python tools/benchmark.py --engine <engine_path>")
    logger.info("=" * 80)
    
    return engine_path


def test_tensorrt_engine(engine_path: str):
    """Test TensorRT engine inference"""

    import tensorrt as trt
    import numpy as np
    import time

    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
    except Exception as e:
        logger.warning(f"⚠️  PyCUDA not available for testing: {e}")
        return

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Load engine
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    
    logger.info(f"  Engine loaded successfully")
    
    # Get input/output info
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    
    # Handle dynamic shapes
    if -1 in input_shape:
        input_shape = [1, 3, 640, 640]  # Use default shape
        context.set_input_shape(input_name, input_shape)
    
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Output shape: {context.get_tensor_shape(output_name)}")
    
    # Allocate buffers
    input_size = int(np.prod(input_shape))
    output_shape_actual = context.get_tensor_shape(output_name)
    output_size = int(np.prod(output_shape_actual))
    
    h_input = cuda.pagelocked_empty(input_size, dtype=np.float32)
    h_output = cuda.pagelocked_empty(output_size, dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    stream = cuda.Stream()
    
    # Test inference
    np.copyto(h_input, np.random.randn(input_size).astype(np.float32))
    
    cuda.memcpy_htod_async(d_input, h_input, stream)
    
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    
    start = time.time()
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    inference_time = (time.time() - start) * 1000
    
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    
    logger.info(f"  Inference time: {inference_time:.2f} ms")
    logger.info("✅ Engine test passed!")


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT")
    parser.add_argument(
        "--onnx",
        required=True,
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--output",
        help="Output TensorRT engine path (default: <onnx_path>_fp16.trt)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 precision (default: True)"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 precision (requires calibration)"
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=2048,
        help="Max workspace size in MB (default: 2048)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if args.output is None:
        onnx_path = Path(args.onnx)
        precision = "int8" if args.int8 else ("fp16" if args.fp16 else "fp32")
        args.output = str(onnx_path.parent / f"{onnx_path.stem}_{precision}.trt")
    
    # Convert
    convert_to_tensorrt(
        onnx_path=args.onnx,
        engine_path=args.output,
        fp16=args.fp16,
        int8=args.int8,
        workspace=args.workspace,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()

