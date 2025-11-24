#!/usr/bin/env python3
"""
Export ONNX with FP16 precision to match PyTorch inference
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolox.exp import get_exp

def export_onnx_fp16():
    """Export YOLOX to ONNX with FP16 precision"""
    
    # 1. Load model
    exp_file = "exps/example/mot/yolox_x_mix_det.py"
    ckpt_file = "models/bytetrack_x_mot17.pth.tar"
    output_file = "models/bytetrack_x_mot17_fp16.onnx"
    
    print(f"Loading model from {ckpt_file}...")
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    
    # Load checkpoint
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.cuda()

    # ✅ Convert model to FP16
    model = model.half()
    print("✅ Model converted to FP16")

    # Disable L1 loss for export
    model.head.use_l1 = False

    print("Model loaded successfully")
    
    # 2. Prepare dummy input - FP16
    dummy_input = torch.randn(1, 3, 640, 640).half().cuda()
    print(f"Dummy input shape: {dummy_input.shape}, dtype: {dummy_input.dtype}")
    
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
        keep_initializers_as_inputs=True,  # Keep bias in ONNX
        training=torch.onnx.TrainingMode.EVAL
    )
    
    print(f"✅ ONNX export completed: {output_file}")

    # 4. Merge external data into single file
    import onnx
    print("Merging external data into single file...")
    model_onnx = onnx.load(output_file, load_external_data=True)
    onnx.save(model_onnx, output_file)
    print("✅ Merged to single file")

    # 5. Verify
    model_onnx = onnx.load(output_file)
    onnx.checker.check_model(model_onnx)
    print("✅ ONNX model verified")
    
    # Print info
    print(f"\nModel info:")
    print(f"  Input: {model_onnx.graph.input[0].name} - {[d.dim_value for d in model_onnx.graph.input[0].type.tensor_type.shape.dim]}")
    print(f"  Output: {model_onnx.graph.output[0].name} - {[d.dim_value for d in model_onnx.graph.output[0].type.tensor_type.shape.dim]}")
    print(f"  Opset: {model_onnx.opset_import[0].version}")
    
    # Check data types
    print(f"\nData types:")
    for inp in model_onnx.graph.input:
        dtype = inp.type.tensor_type.elem_type
        dtype_map = {1: 'FP32', 10: 'FP16', 2: 'UINT8', 3: 'INT8', 6: 'INT32', 7: 'INT64'}
        print(f"  Input '{inp.name}': {dtype_map.get(dtype, f'Unknown({dtype})')}")
    
    for out in model_onnx.graph.output:
        dtype = out.type.tensor_type.elem_type
        dtype_map = {1: 'FP32', 10: 'FP16', 2: 'UINT8', 3: 'INT8', 6: 'INT32', 7: 'INT64'}
        print(f"  Output '{out.name}': {dtype_map.get(dtype, f'Unknown({dtype})')}")

if __name__ == "__main__":
    export_onnx_fp16()

