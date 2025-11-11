#!/usr/bin/env python3
"""
Export YOLOX model to ONNX for TensorRT optimization
Best practices: FP32, no fuse_model, fixed shape [1,3,640,640]
"""

import sys
import torch
import argparse
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolox.exp import get_exp


def export_onnx(
    model_path: str,
    output_path: str,
    test_size: tuple = (640, 640),
    dynamic_batch: bool = False,
    opset_version: int = 11,
    simplify: bool = True
):
    """
    Export YOLOX to ONNX with proper configuration for TensorRT
    
    Args:
        model_path: Path to PyTorch weights (.pth.tar)
        output_path: Output ONNX path
        test_size: Input size (height, width)
        dynamic_batch: Enable dynamic batch size
        opset_version: ONNX opset version (11 or 12 recommended)
        simplify: Simplify ONNX model using onnx-simplifier
    
    Returns:
        output_path: Path to exported ONNX model
    """
    
    logger.info("=" * 80)
    logger.info("EXPORTING YOLOX TO ONNX FOR TENSORRT")
    logger.info("=" * 80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Input size: {test_size}")
    logger.info(f"Dynamic batch: {dynamic_batch}")
    logger.info(f"Opset version: {opset_version}")
    logger.info(f"Simplify: {simplify}")
    logger.info("=" * 80)
    
    # 1. Load experiment config
    exp_file = Path(__file__).parent.parent / "exps/example/mot/yolox_x_mix_det.py"
    logger.info(f"\n📋 Loading experiment config from {exp_file}")
    exp = get_exp(str(exp_file), None)
    exp.test_size = test_size
    
    # 2. Get model
    logger.info("🔨 Building model...")
    model = exp.get_model()
    model.eval()
    
    # 3. Load weights
    logger.info(f"📦 Loading weights from {model_path}...")
    ckpt = torch.load(model_path, map_location='cpu')
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    
    logger.info("✅ Model loaded successfully")
    
    # ⚠️ IMPORTANT: Do NOT fuse_model, do NOT .half()
    # TensorRT will optimize better from FP32 model!
    logger.info("\n⚠️  Export settings:")
    logger.info("  - Precision: FP32 (TensorRT will convert to FP16/INT8)")
    logger.info("  - Fuse model: NO (TensorRT will fuse better)")
    logger.info("  - Half precision: NO (Export FP32)")
    
    # 4. Prepare dummy input
    dummy_input = torch.randn(1, 3, test_size[0], test_size[1])
    logger.info(f"\n🎯 Dummy input shape: {list(dummy_input.shape)}")
    
    # 5. Configure dynamic axes
    if dynamic_batch:
        dynamic_axes = {
            'images': {0: 'batch'},
            'output': {0: 'batch'}
        }
        logger.info("📊 Using dynamic batch size")
    else:
        dynamic_axes = None
        logger.info("📊 Using fixed batch size = 1")
    
    # 6. Export to ONNX
    logger.info("\n🚀 Exporting to ONNX...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,  # Optimize constants
            verbose=False,
            export_params=True,
        )
        logger.info(f"✅ ONNX model exported to {output_path}")
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        raise
    
    # 7. Simplify ONNX (optional but recommended)
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            logger.info("\n🔧 Simplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            model_simplified, check = onnx_simplify(onnx_model)
            
            if check:
                onnx.save(model_simplified, output_path)
                logger.info("✅ ONNX model simplified successfully")
            else:
                logger.warning("⚠️  Simplification check failed, using original model")
        except ImportError:
            logger.warning("⚠️  onnx-simplifier not installed, skipping simplification")
            logger.info("   Install with: pip install onnx-simplifier")
        except Exception as e:
            logger.warning(f"⚠️  Simplification failed: {e}")
            logger.info("   Using original ONNX model")
    
    # 8. Verify ONNX model
    logger.info("\n🔍 Verifying ONNX model...")
    try:
        import onnx
        
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Print model info
        logger.info("\n📊 Model Information:")
        logger.info(f"  IR Version: {onnx_model.ir_version}")
        logger.info(f"  Opset Version: {onnx_model.opset_import[0].version}")
        
        # Input info
        input_tensor = onnx_model.graph.input[0]
        input_shape = [d.dim_value if d.dim_value > 0 else 'dynamic' 
                      for d in input_tensor.type.tensor_type.shape.dim]
        logger.info(f"\n  Input:")
        logger.info(f"    Name: {input_tensor.name}")
        logger.info(f"    Shape: {input_shape}")
        logger.info(f"    Type: {input_tensor.type.tensor_type.elem_type}")
        
        # Output info
        output_tensor = onnx_model.graph.output[0]
        output_shape = [d.dim_value if d.dim_value > 0 else 'dynamic' 
                       for d in output_tensor.type.tensor_type.shape.dim]
        logger.info(f"\n  Output:")
        logger.info(f"    Name: {output_tensor.name}")
        logger.info(f"    Shape: {output_shape}")
        logger.info(f"    Type: {output_tensor.type.tensor_type.elem_type}")
        
        logger.info("\n✅ ONNX model verification passed!")
        
    except Exception as e:
        logger.error(f"❌ ONNX verification failed: {e}")
        raise
    
    # 9. Test ONNX inference
    logger.info("\n🧪 Testing ONNX inference with onnxruntime...")
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Create session
        session = ort.InferenceSession(
            output_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        logger.info(f"  Provider: {session.get_providers()[0]}")
        
        # Test inference
        test_input = np.random.randn(1, 3, test_size[0], test_size[1]).astype(np.float32)
        outputs = session.run(None, {'images': test_input})
        
        logger.info(f"  Input shape: {test_input.shape}")
        logger.info(f"  Output shape: {outputs[0].shape}")
        logger.info("✅ ONNX inference test passed!")
        
    except ImportError:
        logger.warning("⚠️  onnxruntime not installed, skipping inference test")
        logger.info("   Install with: pip install onnxruntime-gpu")
    except Exception as e:
        logger.error(f"❌ ONNX inference test failed: {e}")
        raise
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ EXPORT COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"\n📁 ONNX model saved to: {output_path}")
    logger.info("\n📝 Next steps:")
    logger.info("  1. Verify ONNX: python tools/verify_onnx.py --model <onnx_path>")
    logger.info("  2. Convert to TensorRT: python tools/convert_tensorrt.py --onnx <onnx_path>")
    logger.info("=" * 80)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOX to ONNX for TensorRT optimization"
    )
    parser.add_argument(
        "--model",
        default="models/bytetrack_x_mot17.pth.tar",
        help="Path to PyTorch model (.pth.tar)"
    )
    parser.add_argument(
        "--output",
        default="models/bytetrack_x_mot17_fp32.onnx",
        help="Output ONNX path"
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Input size (height width), default: 640 640"
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Enable dynamic batch size (for batch inference)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        choices=[11, 12, 13, 17, 18],
        help="ONNX opset version (default: 11)"
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip ONNX simplification"
    )
    
    args = parser.parse_args()
    
    # Create output directory if not exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export
    export_onnx(
        model_path=args.model,
        output_path=str(output_path),
        test_size=tuple(args.size),
        dynamic_batch=args.dynamic_batch,
        opset_version=args.opset,
        simplify=not args.no_simplify
    )


if __name__ == "__main__":
    main()

