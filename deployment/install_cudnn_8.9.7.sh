#!/bin/bash

# CUDNN 8.9.7 Installation Script for CUDA 12.x
# This script downloads and installs CUDNN 8.9.7 for CUDA 12.x

set -e

CUDNN_VERSION="8.9.7.29"
CUDA_VERSION="cuda12"
DOWNLOAD_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_${CUDA_VERSION}-archive.tar.xz"
CUDNN_FILE="cudnn-linux-x86_64-${CUDNN_VERSION}_${CUDA_VERSION}-archive.tar.xz"
CUDNN_DIR="cudnn-linux-x86_64-${CUDNN_VERSION}_${CUDA_VERSION}-archive"
INSTALL_DIR="/usr/local/cudnn"

echo "=========================================="
echo "CUDNN 8.9.7 Installation Script"
echo "=========================================="
echo ""

# Check if already installed
if [ -f "$INSTALL_DIR/include/cudnn.h" ]; then
    echo "✅ CUDNN already installed at $INSTALL_DIR"
    ldconfig -p | grep libcudnn || echo "⚠️  Warning: libcudnn not in ldconfig"
    exit 0
fi

# Download CUDNN
echo "📥 Downloading CUDNN 8.9.7..."
if [ ! -f "/tmp/$CUDNN_FILE" ]; then
    cd /tmp
    wget -q --show-progress "$DOWNLOAD_URL" || {
        echo "❌ Download failed. Please download manually from:"
        echo "https://developer.nvidia.com/cudnn"
        exit 1
    }
fi

# Extract
echo "📦 Extracting CUDNN..."
cd /tmp
tar -xf "$CUDNN_FILE" || { echo "❌ Extract failed"; exit 1; }

# Install
echo "🔧 Installing CUDNN..."
sudo mkdir -p "$INSTALL_DIR/include" "$INSTALL_DIR/lib"
sudo cp "$CUDNN_DIR/include"/* "$INSTALL_DIR/include/"
sudo cp "$CUDNN_DIR/lib"/* "$INSTALL_DIR/lib/"

# Create symlinks
echo "🔗 Creating symlinks..."
sudo mkdir -p /usr/local/cuda/include
sudo ln -sf "$INSTALL_DIR/include/cudnn.h" /usr/local/cuda/include/cudnn.h
sudo ln -sf "$INSTALL_DIR/lib/libcudnn.so.8" /usr/local/cuda/lib64/libcudnn.so.8
sudo ln -sf "$INSTALL_DIR/lib/libcudnn.so" /usr/local/cuda/lib64/libcudnn.so

# Update library cache
echo "📚 Updating library cache..."
sudo ldconfig

# Verify
echo ""
echo "✅ CUDNN 8.9.7 installation completed!"
echo ""
echo "Verification:"
ldconfig -p | grep libcudnn | head -2
echo ""
ls -lh "$INSTALL_DIR/lib/libcudnn.so.8.9.7"

