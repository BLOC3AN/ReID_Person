#!/bin/bash

# Complete Setup Script for Person ReID System
# This script sets up CUDNN and prepares the deployment environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Person ReID System - Complete Setup"
echo "=========================================="
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU drivers not found. Please install NVIDIA drivers first."
    exit 1
fi

echo "✅ Docker and NVIDIA drivers found"
echo ""

# Install CUDNN
echo "📦 Installing CUDNN 8.9.7..."
if [ -f "$SCRIPT_DIR/install_cudnn_8.9.7.sh" ]; then
    bash "$SCRIPT_DIR/install_cudnn_8.9.7.sh"
else
    echo "❌ install_cudnn_8.9.7.sh not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start deployment: cd $SCRIPT_DIR && docker compose up -d"
echo "2. Check status: docker compose ps"
echo "3. View logs: docker compose logs -f triton"

