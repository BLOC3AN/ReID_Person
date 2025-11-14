#!/usr/bin/env python3
"""
Export YOLOX model to ONNX with dynamic batch support
"""

import torch
import argparse
from loguru import logger
import sys
import os

# Add YOLOX to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from yolox.exp import get_exp


def export_onnx(exp_file, ckpt_path, output_path, dynamic_batch=True):
    """
    Export YOLOX model to ONNX
    
    Args:
        exp_file: Experiment config file
        ckpt_path: Checkpoint path (.pth)
        output_path: Output ONNX path
        dynamic_batch: Enable dynamic batch dimension
    """
    logger.info("=" * 80)
    logger.info("YOLOX ONNX Export")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Dynamic batch: {dynamic_batch}")
    
    # Load experiment
    exp = get_exp(exp_file, None)
    exp.test_conf = 0.01  # Lower threshold for export
    exp.nmsthre = 0.65
    
    # Build model
    logger.info("Building model...")
    model = exp.get_model()
    model.eval()
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    
    # Prepare dummy input
    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])
    
    # Export to ONNX
    logger.info("Exporting to ONNX...")
    
    if dynamic_batch:
        dynamic_axes = {
            'images': {0: 'batch'},
            'output': {0: 'batch'}
        }
        logger.info("  Dynamic axes: batch dimension")
    else:
        dynamic_axes = None
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    
    logger.info(f"✅ ONNX model exported: {output_path}")
    
    # Verify ONNX model
    import onnx
    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Print model info
    logger.info("\nModel Info:")
    for inp in onnx_model.graph.input:
        logger.info(f"  Input: {inp.name}")
        shape = [dim.dim_value if dim.dim_value > 0 else f"dynamic({dim.dim_param})" 
                 for dim in inp.type.tensor_type.shape.dim]
        logger.info(f"    Shape: {shape}")
    
    for out in onnx_model.graph.output:
        logger.info(f"  Output: {out.name}")
        shape = [dim.dim_value if dim.dim_value > 0 else f"dynamic({dim.dim_param})" 
                 for dim in out.type.tensor_type.shape.dim]
        logger.info(f"    Shape: {shape}")
    
    logger.info("✅ ONNX model verified")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOX to ONNX")
    parser.add_argument(
        "-f", "--exp_file",
        type=str,
        default="exps/default/yolox_x.py",
        help="Experiment description file"
    )
    parser.add_argument(
        "-c", "--ckpt",
        type=str,
        default="models/yolox_x.pth",
        help="Checkpoint file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="models/yolox_x_dynamic.onnx",
        help="Output ONNX file"
    )
    parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Disable dynamic batch"
    )
    
    args = parser.parse_args()
    
    export_onnx(
        exp_file=args.exp_file,
        ckpt_path=args.ckpt,
        output_path=args.output,
        dynamic_batch=not args.no_dynamic
    )

