#!/usr/bin/env python3
"""
Simple ONNX export script for YOLOX model
Fixes bias shape mismatch issue for TensorRT
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolox.exp import get_exp

def export_onnx_simple():
    """Export YOLOX to ONNX with proper settings for TensorRT"""
    
    # 1. Load model
    exp_file = "exps/example/mot/yolox_x_mix_det.py"
    ckpt_file = "models/bytetrack_x_mot17.pth.tar"
    output_file = "models/bytetrack_x_mot17_fp32_single.onnx"
    
    print(f"Loading model from {ckpt_file}...")
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    
    # Load checkpoint
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.cuda()

    # Disable L1 loss for export
    model.head.use_l1 = False

    print("Model loaded successfully (NOT fused - will let TensorRT fuse)")
    
    # 2. Prepare dummy input
    dummy_input = torch.randn(1, 3, 640, 640).cuda()
    
    # 3. Export to ONNX
    print(f"Exporting to {output_file}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=None,  # Fixed batch
        opset_version=12,  # TensorRT works best with 12-13
        do_constant_folding=True,  # Enable for optimization
        verbose=False,
        export_params=True,
        keep_initializers_as_inputs=True,  # ✅ Keep bias in ONNX
        training=torch.onnx.TrainingMode.EVAL
    )
    
    print(f"✅ ONNX export completed: {output_file}")

    # 4. Merge external data into single file
    import onnx
    print("Merging external data into single file...")
    model = onnx.load(output_file, load_external_data=True)
    onnx.save(model, output_file)
    print("✅ Merged to single file")

    # 5. Verify
    model = onnx.load(output_file)
    onnx.checker.check_model(model)
    print("✅ ONNX model verified")
    
    # Print info
    print(f"\nModel info:")
    print(f"  Input: {model.graph.input[0].name} - {[d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]}")
    print(f"  Output: {model.graph.output[0].name} - {[d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]}")
    print(f"  Opset: {model.opset_import[0].version}")

if __name__ == "__main__":
    export_onnx_simple()

