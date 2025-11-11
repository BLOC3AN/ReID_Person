#!/usr/bin/env python3
"""
Verify ONNX model - check shape, inference, and accuracy
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_onnx_model(onnx_path: str, test_image: str = None):
    """
    Verify ONNX model structure and inference
    
    Args:
        onnx_path: Path to ONNX model
        test_image: Optional test image path for accuracy comparison
    """
    
    logger.info("=" * 80)
    logger.info("VERIFYING ONNX MODEL")
    logger.info("=" * 80)
    logger.info(f"Model: {onnx_path}")
    logger.info("=" * 80)
    
    # 1. Check file exists
    if not Path(onnx_path).exists():
        logger.error(f"❌ ONNX file not found: {onnx_path}")
        return False
    
    file_size = Path(onnx_path).stat().st_size / (1024 * 1024)  # MB
    logger.info(f"\n📁 File size: {file_size:.2f} MB")
    
    # 2. Load and check ONNX model
    logger.info("\n🔍 Loading ONNX model...")
    try:
        import onnx
        
        onnx_model = onnx.load(onnx_path)
        logger.info("✅ ONNX model loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to load ONNX model: {e}")
        return False
    
    # 3. Check model validity
    logger.info("\n🔍 Checking model validity...")
    try:
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model is valid")
    except Exception as e:
        logger.error(f"❌ ONNX model validation failed: {e}")
        return False
    
    # 4. Print model information
    logger.info("\n📊 Model Information:")
    logger.info(f"  IR Version: {onnx_model.ir_version}")
    logger.info(f"  Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
    logger.info(f"  Opset Version: {onnx_model.opset_import[0].version}")
    logger.info(f"  Graph Name: {onnx_model.graph.name}")
    
    # 5. Print input information
    logger.info("\n📥 Input Information:")
    for i, input_tensor in enumerate(onnx_model.graph.input):
        shape = [d.dim_value if d.dim_value > 0 else f'dynamic({d.dim_param})' 
                for d in input_tensor.type.tensor_type.shape.dim]
        dtype = input_tensor.type.tensor_type.elem_type
        
        dtype_map = {1: 'float32', 2: 'uint8', 3: 'int8', 6: 'int32', 7: 'int64', 10: 'float16'}
        dtype_str = dtype_map.get(dtype, f'unknown({dtype})')
        
        logger.info(f"  Input {i}:")
        logger.info(f"    Name: {input_tensor.name}")
        logger.info(f"    Shape: {shape}")
        logger.info(f"    Type: {dtype_str}")
    
    # 6. Print output information
    logger.info("\n📤 Output Information:")
    for i, output_tensor in enumerate(onnx_model.graph.output):
        shape = [d.dim_value if d.dim_value > 0 else f'dynamic({d.dim_param})' 
                for d in output_tensor.type.tensor_type.shape.dim]
        dtype = output_tensor.type.tensor_type.elem_type
        
        dtype_map = {1: 'float32', 2: 'uint8', 3: 'int8', 6: 'int32', 7: 'int64', 10: 'float16'}
        dtype_str = dtype_map.get(dtype, f'unknown({dtype})')
        
        logger.info(f"  Output {i}:")
        logger.info(f"    Name: {output_tensor.name}")
        logger.info(f"    Shape: {shape}")
        logger.info(f"    Type: {dtype_str}")
    
    # 7. Test ONNX inference
    logger.info("\n🧪 Testing ONNX Inference...")
    try:
        import onnxruntime as ort
        
        # Create session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        active_provider = session.get_providers()[0]
        logger.info(f"  Active Provider: {active_provider}")
        
        # Get input shape
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # Handle dynamic dimensions
        test_shape = []
        for dim in input_shape:
            if isinstance(dim, str):  # Dynamic dimension
                test_shape.append(1)  # Use batch=1 for testing
            else:
                test_shape.append(dim)
        
        logger.info(f"  Test input shape: {test_shape}")
        
        # Create random input
        test_input = np.random.randn(*test_shape).astype(np.float32)
        
        # Run inference
        import time
        start = time.time()
        outputs = session.run(None, {input_name: test_input})
        inference_time = (time.time() - start) * 1000  # ms
        
        logger.info(f"  Output shape: {outputs[0].shape}")
        logger.info(f"  Inference time: {inference_time:.2f} ms")
        logger.info("✅ ONNX inference test passed!")
        
        # Benchmark
        logger.info("\n⏱️  Running benchmark (100 iterations)...")
        times = []
        for _ in range(100):
            start = time.time()
            _ = session.run(None, {input_name: test_input})
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000 / avg_time
        
        logger.info(f"  Average time: {avg_time:.2f} ± {std_time:.2f} ms")
        logger.info(f"  FPS: {fps:.2f}")
        
    except ImportError:
        logger.warning("⚠️  onnxruntime not installed")
        logger.info("   Install with: pip install onnxruntime-gpu")
        return True  # Model is valid, just can't test inference
    except Exception as e:
        logger.error(f"❌ ONNX inference test failed: {e}")
        return False
    
    # 8. Compare with PyTorch (if test image provided)
    if test_image and Path(test_image).exists():
        logger.info("\n🔬 Comparing ONNX vs PyTorch accuracy...")
        try:
            compare_with_pytorch(onnx_path, test_image, session)
        except Exception as e:
            logger.warning(f"⚠️  Accuracy comparison failed: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ VERIFICATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    
    return True


def compare_with_pytorch(onnx_path: str, test_image: str, ort_session):
    """Compare ONNX output with PyTorch output"""
    
    import torch
    import cv2
    from yolox.exp import get_exp
    from yolox.data.data_augment import preproc
    
    # Load PyTorch model
    exp_file = Path(__file__).parent.parent / "exps/example/mot/yolox_x_mix_det.py"
    exp = get_exp(str(exp_file), None)
    
    model = exp.get_model()
    model.eval()
    
    # Load weights (extract from ONNX path)
    model_path = str(Path(onnx_path).parent / "bytetrack_x_mot17.pth.tar")
    if Path(model_path).exists():
        ckpt = torch.load(model_path, map_location='cpu')
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    else:
        logger.warning(f"⚠️  PyTorch weights not found: {model_path}")
        return
    
    # Load and preprocess image
    frame = cv2.imread(test_image)
    if frame is None:
        logger.warning(f"⚠️  Failed to load image: {test_image}")
        return
    
    img, ratio = preproc(frame, (640, 640), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    # PyTorch inference
    with torch.no_grad():
        torch_input = torch.from_numpy(img).unsqueeze(0).float()
        torch_output = model(torch_input)
    
    # ONNX inference
    onnx_input = img[np.newaxis, :, :, :].astype(np.float32)
    input_name = ort_session.get_inputs()[0].name
    onnx_output = ort_session.run(None, {input_name: onnx_input})[0]
    
    # Compare outputs
    torch_out_np = torch_output.cpu().numpy()
    diff = np.abs(torch_out_np - onnx_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    logger.info(f"  PyTorch output shape: {torch_out_np.shape}")
    logger.info(f"  ONNX output shape: {onnx_output.shape}")
    logger.info(f"  Max difference: {max_diff:.6f}")
    logger.info(f"  Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-3:
        logger.info("✅ Outputs match perfectly!")
    elif max_diff < 1e-2:
        logger.info("✅ Outputs match well (acceptable difference)")
    else:
        logger.warning(f"⚠️  Large difference detected: {max_diff}")


def main():
    parser = argparse.ArgumentParser(description="Verify ONNX model")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--test-image",
        help="Optional test image for accuracy comparison"
    )
    
    args = parser.parse_args()
    
    success = verify_onnx_model(args.model, args.test_image)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

