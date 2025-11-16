#!/usr/bin/env python3
"""
Build TensorRT engine for SCRFD Face Detector
Converts InsightFace SCRFD ONNX model to optimized TensorRT engine
"""

import os
import sys
import tensorrt as trt
from pathlib import Path

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_scrfd_engine(onnx_path, engine_path, max_batch_size=8, input_size=(640, 640)):
    """
    Build TensorRT engine for SCRFD face detector with dynamic batch size
    
    Args:
        onnx_path: Path to SCRFD ONNX model (det_10g.onnx)
        engine_path: Path to save TensorRT engine
        max_batch_size: Maximum batch size (1-8)
        input_size: Input image size (height, width)
    """
    print(f"🔧 Building SCRFD TensorRT engine...")
    print(f"   ONNX: {onnx_path}")
    print(f"   Engine: {engine_path}")
    print(f"   Max batch size: {max_batch_size}")
    print(f"   Input size: {input_size}")
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    print(f"\n📖 Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("❌ Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print(f"✅ ONNX parsed successfully")
    print(f"   Network inputs: {network.num_inputs}")
    print(f"   Network outputs: {network.num_outputs}")
    
    # Print input/output info
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"   Input {i}: {inp.name} {inp.shape}")
    
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"   Output {i}: {out.name} {out.shape}")
    
    # Create builder config
    config = builder.create_builder_config()
    
    # Set memory pool limit (4GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    
    # Enable FP16 if supported
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"\n✅ FP16 mode enabled")
    else:
        print(f"\n⚠️  FP16 not supported, using FP32")
    
    # Check input shape and create optimization profile
    input_shape = network.get_input(0).shape
    input_name = network.get_input(0).name
    h, w = input_size

    print(f"\n📊 Input shape from ONNX: {input_shape}")

    # Check if any dimension is dynamic
    has_dynamic = any(dim == -1 for dim in input_shape)

    if has_dynamic:
        print(f"✅ Model has dynamic dimensions, creating optimization profile")

        profile = builder.create_optimization_profile()

        # SCRFD ONNX has shape (1, 3, -1, -1) - fixed batch, dynamic H/W
        # We need to set profile for the dynamic dimensions
        min_shape = (1, 3, h, w)
        opt_shape = (1, 3, h, w)
        max_shape = (1, 3, h, w)

        print(f"\n🎯 Optimization Profile:")
        print(f"   Input: {input_name}")
        print(f"   Shape: {min_shape} (fixed)")

        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
    else:
        print(f"✅ Model has fixed shape, no optimization profile needed")
    
    # Build engine
    print(f"\n🔨 Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("❌ Failed to build engine")
        return False
    
    # Save engine
    print(f"\n💾 Saving engine to {engine_path}...")
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    # Get file size
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"✅ Engine saved successfully ({size_mb:.2f} MB)")
    
    return True


def verify_engine(engine_path):
    """Verify the built engine"""
    print(f"\n🔍 Verifying engine...")
    
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("❌ Failed to load engine")
        return False
    
    print(f"✅ Engine loaded successfully")
    print(f"   Max batch size: {engine.max_batch_size}")
    print(f"   Num bindings: {engine.num_bindings}")
    
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        dtype = engine.get_binding_dtype(i)
        is_input = engine.binding_is_input(i)
        print(f"   Binding {i}: {name} {'[INPUT]' if is_input else '[OUTPUT]'} {shape} {dtype}")
    
    return True


if __name__ == "__main__":
    # Paths
    onnx_path = os.path.expanduser("~/.insightface/models/buffalo_l/det_10g.onnx")
    engine_path = "models/scrfd_10g_fp16.trt"
    
    # Check ONNX exists
    if not os.path.exists(onnx_path):
        print(f"❌ SCRFD ONNX model not found: {onnx_path}")
        print(f"   Please run InsightFace first to download the model:")
        print(f"   python -c \"from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=0)\"")
        sys.exit(1)
    
    print(f"✅ Found SCRFD ONNX: {onnx_path}")

    # Build engine
    success = build_scrfd_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        max_batch_size=8,
        input_size=(640, 640)
    )

    if not success:
        print("❌ Failed to build engine")
        sys.exit(1)

    # Verify engine
    if not verify_engine(engine_path):
        print("❌ Engine verification failed")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✅ SCRFD TensorRT engine built successfully!")
    print("=" * 80)
    print(f"Engine: {engine_path}")
    print(f"Next steps:")
    print(f"  1. Copy engine to Triton model repository")
    print(f"  2. Create Triton config.pbtxt")
    print(f"  3. Test with Triton client")
    print("=" * 80)

