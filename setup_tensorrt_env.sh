#!/bin/bash

# ============================================================================
# TensorRT Environment Setup Script
# ============================================================================
# This script sets up a Python 3.10 virtual environment with all required
# dependencies for running TensorRT inference on Person ReID system.
#
# Requirements:
#   - Ubuntu 20.04/22.04
#   - NVIDIA GPU (Tesla V100 or equivalent)
#   - CUDA 12.x installed
#   - Python 3.10 available
#
# Usage:
#   bash setup_tensorrt_env.sh
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VENV_PATH="$HOME/data/hai_venv_py310"
PYTHON_VERSION="3.10"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed"
        return 1
    fi
    return 0
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

print_header "Pre-flight Checks"

# Check Python 3.10
if ! check_command python3.10; then
    print_error "Python 3.10 is required but not found"
    print_info "Install with: sudo apt install python3.10 python3.10-venv python3.10-dev"
    exit 1
fi
print_success "Python 3.10 found: $(python3.10 --version)"

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. NVIDIA GPU required"
    exit 1
fi
print_success "NVIDIA GPU found: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    print_error "CUDA not found. Please install CUDA 12.x"
    exit 1
fi
print_success "CUDA found: $(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"

# ============================================================================
# Create Virtual Environment
# ============================================================================

print_header "Creating Virtual Environment"

if [ -d "$VENV_PATH" ]; then
    print_info "Virtual environment already exists at $VENV_PATH"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing environment..."
        rm -rf "$VENV_PATH"
    else
        print_info "Using existing environment"
    fi
fi

if [ ! -d "$VENV_PATH" ]; then
    print_info "Creating new virtual environment at $VENV_PATH"
    python3.10 -m venv "$VENV_PATH"
    print_success "Virtual environment created"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"
print_success "Virtual environment activated"

# ============================================================================
# Upgrade pip
# ============================================================================

print_header "Upgrading pip"
pip install --upgrade pip
print_success "pip upgraded to $(pip --version | awk '{print $2}')"

# ============================================================================
# Install PyTorch 2.0.1 (CUDA 11.8)
# ============================================================================

print_header "Installing PyTorch 2.0.1 (CUDA 11.8)"
print_info "This may take several minutes..."

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

print_success "PyTorch installed"

# ============================================================================
# Install NumPy < 2.0
# ============================================================================

print_header "Installing NumPy < 2.0"
pip install "numpy<2"
print_success "NumPy installed: $(python -c 'import numpy; print(numpy.__version__)')"

# ============================================================================
# Install TensorRT 8.6.1
# ============================================================================

print_header "Installing TensorRT 8.6.1"
print_info "This may take a few minutes..."

pip install tensorrt==8.6.1.post1 --extra-index-url https://pypi.nvidia.com

print_success "TensorRT installed"

# ============================================================================
# Install CUDNN 8.9.6.50
# ============================================================================

print_header "Installing CUDNN 8.9.6.50"
print_info "This is a large download (700+ MB)..."

pip install nvidia-cudnn-cu12==8.9.6.50 --no-deps

print_success "CUDNN installed"

# ============================================================================
# Install PyCUDA
# ============================================================================

print_header "Installing PyCUDA"
pip install pycuda
print_success "PyCUDA installed"

# ============================================================================
# Install ONNX Stack
# ============================================================================

print_header "Installing ONNX Stack"
pip install onnx onnxruntime-gpu
print_success "ONNX stack installed"

# ============================================================================
# Install Computer Vision & ML Libraries
# ============================================================================

print_header "Installing CV & ML Libraries"
pip install opencv-python scipy pillow
print_success "CV & ML libraries installed"

# ============================================================================
# Install YOLOX Dependencies
# ============================================================================

print_header "Installing YOLOX Dependencies"
pip install thop tabulate tqdm pycocotools
print_success "YOLOX dependencies installed"

# ============================================================================
# Install Tracking Dependencies
# ============================================================================

print_header "Installing Tracking Dependencies"
pip install lap cython_bbox
print_success "Tracking dependencies installed"

# ============================================================================
# Install Utilities
# ============================================================================

print_header "Installing Utilities"
pip install loguru pyyaml python-dotenv
print_success "Utilities installed"

# ============================================================================
# Install Vector Database (Optional)
# ============================================================================

print_header "Installing Vector Database (Optional)"
pip install qdrant-client
print_success "Vector database client installed"

# ============================================================================
# Verification
# ============================================================================

print_header "Verification"

# Verify Python version
PYTHON_VER=$(python --version | awk '{print $2}')
print_info "Python version: $PYTHON_VER"

# Verify PyTorch
PYTORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "FAILED")
if [ "$PYTORCH_VER" = "FAILED" ]; then
    print_error "PyTorch verification failed"
else
    print_success "PyTorch: $PYTORCH_VER"
fi

# Verify CUDA availability
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "FAILED")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    print_success "CUDA available: True"
else
    print_error "CUDA not available in PyTorch"
fi

# Verify TensorRT
TRT_VER=$(python -c "import tensorrt as trt; print(trt.__version__)" 2>/dev/null || echo "FAILED")
if [ "$TRT_VER" = "FAILED" ]; then
    print_error "TensorRT verification failed"
else
    print_success "TensorRT: $TRT_VER"
fi

# Verify CUDNN
CUDNN_VER=$(python -c "import torch; print(torch.backends.cudnn.version())" 2>/dev/null || echo "FAILED")
if [ "$CUDNN_VER" = "FAILED" ]; then
    print_error "CUDNN verification failed"
else
    print_success "CUDNN: $CUDNN_VER"
fi

# Verify NumPy
NUMPY_VER=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "FAILED")
if [ "$NUMPY_VER" = "FAILED" ]; then
    print_error "NumPy verification failed"
else
    print_success "NumPy: $NUMPY_VER"
fi

# Verify PyCUDA
PYCUDA_OK=$(python -c "import pycuda.autoinit; import pycuda.driver as cuda; print(cuda.Device(0).name())" 2>/dev/null || echo "FAILED")
if [ "$PYCUDA_OK" = "FAILED" ]; then
    print_error "PyCUDA verification failed"
else
    print_success "PyCUDA: $PYCUDA_OK"
fi

# ============================================================================
# Save Requirements
# ============================================================================

print_header "Saving Requirements"
pip freeze > requirements_tensorrt_py310.txt
print_success "Requirements saved to requirements_tensorrt_py310.txt"

# ============================================================================
# Summary
# ============================================================================

print_header "Installation Summary"

cat << EOF
✅ Virtual environment created at: $VENV_PATH
✅ Python version: $PYTHON_VER
✅ PyTorch version: $PYTORCH_VER
✅ TensorRT version: $TRT_VER
✅ CUDNN version: $CUDNN_VER
✅ NumPy version: $NUMPY_VER
✅ CUDA available: $CUDA_AVAILABLE
✅ GPU: $PYCUDA_OK

📝 Next Steps:
   1. Activate environment: source $VENV_PATH/bin/activate
   2. Run benchmark: python tools/benchmark.py --warmup 10 --iterations 100
   3. See QUICK_START_TENSORRT.md for usage examples

📖 Documentation:
   - QUICK_START_TENSORRT.md - Quick start guide
   - TENSORRT_SETUP.md - Full documentation
   - requirements_tensorrt_py310.txt - Frozen dependencies

⚠️  Remember:
   - Always activate the virtual environment before running
   - Do NOT upgrade PyTorch, TensorRT, or NumPy
   - Use Python 3.10 only

EOF

print_success "Installation completed successfully!"

