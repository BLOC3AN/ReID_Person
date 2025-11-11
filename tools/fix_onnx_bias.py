#!/usr/bin/env python3
"""
Fix ONNX model bias mismatch issue
Remove incorrect bias from Conv layers where bias shape != weight shape[0]
"""

import onnx
import numpy as np

def fix_onnx_bias(input_path: str, output_path: str):
    """Fix bias mismatch in ONNX model"""
    
    print(f"Loading ONNX model from {input_path}...")
    model = onnx.load(input_path)
    
    # Find all Conv nodes with bias mismatch
    nodes_to_fix = []
    
    for node in model.graph.node:
        if node.op_type == 'Conv' and len(node.input) == 3:  # Has bias
            weight_name = node.input[1]
            bias_name = node.input[2]
            
            # Get weight and bias shapes
            weight_shape = None
            bias_shape = None
            
            for init in model.graph.initializer:
                if init.name == weight_name:
                    weight_shape = list(init.dims)
                if init.name == bias_name:
                    bias_shape = list(init.dims)
            
            if weight_shape and bias_shape:
                expected_bias_size = weight_shape[0]  # Output channels
                actual_bias_size = bias_shape[0]
                
                if expected_bias_size != actual_bias_size:
                    print(f"\n❌ Found mismatch in node: {node.name}")
                    print(f"   Weight shape: {weight_shape}")
                    print(f"   Bias shape: {bias_shape}")
                    print(f"   Expected bias size: {expected_bias_size}, got: {actual_bias_size}")
                    nodes_to_fix.append((node, weight_name, bias_name, expected_bias_size))
    
    if not nodes_to_fix:
        print("\n✅ No bias mismatch found!")
        return
    
    print(f"\n🔧 Fixing {len(nodes_to_fix)} Conv nodes...")
    
    # Fix each node
    for node, weight_name, old_bias_name, expected_size in nodes_to_fix:
        # Create new zero bias with correct shape
        new_bias_name = f"{node.name}_fixed_bias"
        new_bias = onnx.numpy_helper.from_array(
            np.zeros(expected_size, dtype=np.float32),
            name=new_bias_name
        )
        
        # Add new bias to initializers
        model.graph.initializer.append(new_bias)
        
        # Update node to use new bias
        node.input[2] = new_bias_name
        
        print(f"   ✅ Fixed {node.name}: created zero bias [{expected_size}]")
    
    # Save fixed model
    print(f"\n💾 Saving fixed model to {output_path}...")
    onnx.save(model, output_path)
    
    # Verify
    onnx.checker.check_model(model)
    print("✅ Fixed model verified!")
    
    print(f"\n📊 Summary:")
    print(f"   Fixed {len(nodes_to_fix)} Conv nodes")
    print(f"   Output: {output_path}")

if __name__ == "__main__":
    import sys
    
    input_file = "models/bytetrack_x_mot17_fp32_single.onnx"
    output_file = "models/bytetrack_x_mot17_fp32_fixed.onnx"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    fix_onnx_bias(input_file, output_file)

