#!/bin/bash

# ============================================================================
# Setup Triton Model Repository for Deployment
# ============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
MODEL_NAME="bytetrack_tensorrt"
MODEL_REPO="../triton_model_repository"
TRT_ENGINE="../models/bytetrack_x_mot17_fp16.trt"

print_info "Setting up Triton model repository for deployment..."

# Check TensorRT engine
if [ ! -f "$TRT_ENGINE" ]; then
    print_error "TensorRT engine not found: $TRT_ENGINE"
    print_info "Please convert ONNX to TensorRT first:"
    print_info "  cd .. && python tools/convert_tensorrt.py --onnx models/bytetrack_x_mot17_fp32.onnx --output $TRT_ENGINE --fp16"
    exit 1
fi

print_success "Found TensorRT engine: $TRT_ENGINE ($(du -h $TRT_ENGINE | cut -f1))"

# Create model repository structure
print_info "Creating model repository structure..."
mkdir -p "$MODEL_REPO/$MODEL_NAME/1"

# Copy TensorRT engine
print_info "Copying TensorRT engine..."
cp "$TRT_ENGINE" "$MODEL_REPO/$MODEL_NAME/1/model.plan"

print_success "Model repository ready at: $MODEL_REPO"

# Display structure
print_info "Repository structure:"
ls -lh "$MODEL_REPO/$MODEL_NAME/1/"
ls -lh "$MODEL_REPO/$MODEL_NAME/config.pbtxt" 2>/dev/null || print_info "Config already exists"

print_success "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Start services: cd deployment && docker-compose up -d triton"
echo "  2. Check health: curl http://localhost:8000/v2/health/ready"
echo "  3. Update config.yaml: backend: triton"
echo "  4. Start detection service: docker-compose up -d detection"

