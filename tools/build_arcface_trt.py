#!/usr/bin/env python3
"""
Build TensorRT engine for ArcFace (w600k_r50.onnx)
With dynamic batch support for multi-face processing
"""

import sys
import os
import tensorrt as trt
import numpy as np

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_arcface_engine(onnx_path, engine_path, max_batch_size=16):
    """
    Build TensorRT engine for ArcFace with dynamic batch size
    
    Args:
        onnx_path: Path to ArcFace ONNX model (w600k_r50.onnx)
        engine_path: Path to save TensorRT engine
        max_batch_size: Maximum batch size (1-16)
    """
    print(f"🔧 Building ArcFace TensorRT engine...")
    print(f"   ONNX: {onnx_path}")
    print(f"   Engine: {engine_path}")
    print(f"   Max batch size: {max_batch_size}")
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    print(f"📖 Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print('❌ Failed to parse ONNX file')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print(f"✅ ONNX parsed successfully")
    
    # Print network info
    print(f"\n📊 Network Info:")
    print(f"   Inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"     {i}: {input_tensor.name} {input_tensor.shape} {input_tensor.dtype}")
    
    print(f"   Outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"     {i}: {output_tensor.name} {output_tensor.shape} {output_tensor.dtype}")
    
    # Create builder config
    config = builder.create_builder_config()
    
    # Set memory pool limit (4GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    
    # Disable FP16 for debugging
    # if builder.platform_has_fast_fp16:
    #     print(f"✅ Enabling FP16 precision")
    #     config.set_flag(trt.BuilderFlag.FP16)
    print(f"⚠️  Using FP32 precision (FP16 disabled for debugging)")
    
    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    
    # ArcFace input: [batch, 3, 112, 112]
    input_name = network.get_input(0).name
    
    # Min shape: batch=1 (single face)
    min_shape = (1, 3, 112, 112)
    # Optimal shape: batch=4 (typical multi-face scenario)
    opt_shape = (4, 3, 112, 112)
    # Max shape: batch=16 (many faces in frame)
    max_shape = (max_batch_size, 3, 112, 112)
    
    print(f"\n🎯 Dynamic Shape Profile:")
    print(f"   Input: {input_name}")
    print(f"   Min:  {min_shape}")
    print(f"   Opt:  {opt_shape}")
    print(f"   Max:  {max_shape}")
    
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    # Build engine
    print(f"\n🔨 Building TensorRT engine (this may take 3-5 minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print('❌ Failed to build engine')
        return False
    
    # Save engine
    print(f"💾 Saving engine to {engine_path}...")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    # Get file size
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"✅ Engine saved successfully ({size_mb:.1f} MB)")
    
    return True


def test_engine(engine_path):
    """Test the built engine"""
    print(f"\n🧪 Testing engine...")
    
    # Load engine
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("❌ Failed to load engine")
        return False
    
    print(f"✅ Engine loaded successfully")
    print(f"   Num bindings: {engine.num_bindings}")
    
    # Print bindings info
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        dtype = engine.get_binding_dtype(i)
        print(f"   Binding {i}: {name} {shape} {dtype}")
    
    return True


if __name__ == "__main__":
    # Paths
    onnx_path = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
    engine_path = "models/arcface_w600k_r50_fp32.trt"
    
    # Check ONNX exists
    if not os.path.exists(onnx_path):
        print(f"❌ ArcFace ONNX model not found: {onnx_path}")
        sys.exit(1)
    
    print(f"✅ Found ArcFace ONNX: {onnx_path}")
    
    # Build engine
    success = build_arcface_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        max_batch_size=16
    )
    
    if not success:
        print("❌ Failed to build engine")
        sys.exit(1)
    
    # Test engine
    test_engine(engine_path)
    
    print("\n" + "="*80)
    print("✅ SUCCESS!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Create Triton model repository:")
    print(f"   mkdir -p triton_model_repository/arcface_tensorrt/1")
    print(f"   cp {engine_path} triton_model_repository/arcface_tensorrt/1/model.plan")
    print(f"\n2. Create config.pbtxt")
    print(f"\n3. Restart Triton server")

