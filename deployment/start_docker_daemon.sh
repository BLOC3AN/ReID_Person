#!/bin/bash

# Script to install, configure and start Docker daemon with NVIDIA runtime support
# Docker data will be stored at /home/ubuntu/data/docker/

set -e  # Exit on error

DOCKER_DATA_ROOT="/home/ubuntu/data/docker"

echo "=========================================="
echo "Docker Installation & Configuration Script"
echo "=========================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Install Docker if not exists
echo "[1/5] Checking Docker installation..."
if ! command_exists docker; then
    echo "Docker not found. Installing Docker..."

    # Update package index
    sudo apt-get update

    # Install prerequisites
    sudo apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    # Add Docker's official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    # Set up Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    echo "✅ Docker installed successfully"
else
    echo "✅ Docker already installed ($(docker --version))"
fi

# 2. Install NVIDIA Container Runtime if not exists (optional)
echo ""
echo "[2/5] Checking NVIDIA Container Runtime..."
if ! command_exists nvidia-container-runtime; then
    echo "NVIDIA Container Runtime not found. Attempting to install..."

    # Add NVIDIA Container Runtime repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - 2>/dev/null || true
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-docker.list > /dev/null 2>&1 || true

    # Install NVIDIA Container Runtime (non-blocking if fails)
    sudo apt-get update > /dev/null 2>&1 || true
    sudo apt-get install -y nvidia-container-runtime 2>/dev/null || {
        echo "⚠️  NVIDIA Container Runtime not available for this distribution (continuing without GPU support)"
    }

    if command_exists nvidia-container-runtime; then
        echo "✅ NVIDIA Container Runtime installed successfully"
    fi
else
    echo "✅ NVIDIA Container Runtime already installed"
fi

# 3. Create Docker data directory
echo ""
echo "[3/5] Setting up Docker data directory..."
sudo mkdir -p "$DOCKER_DATA_ROOT"
echo "✅ Created directory: $DOCKER_DATA_ROOT"

# 4. Stop existing Docker daemon
echo ""
echo "[4/5] Stopping existing Docker daemon..."
sudo pkill -f "dockerd --data-root" || true
sudo systemctl stop docker.service || true
sudo systemctl stop docker.socket || true
sleep 2
echo "✅ Stopped existing Docker processes"

# 5. Configure Docker with NVIDIA runtime
echo ""
echo "[5/5] Configuring Docker daemon..."
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null << 'EOF'
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF
echo "✅ Created /etc/docker/daemon.json"

# Start Docker daemon with custom data-root
echo ""
echo "=========================================="
echo "Starting Docker daemon..."
echo "=========================================="
sudo dockerd \
  --data-root "$DOCKER_DATA_ROOT" \
  --config-file /etc/docker/daemon.json \
  -H unix:///var/run/docker.sock \
  > /tmp/dockerd.log 2>&1 &

# Wait for Docker to be ready
echo "Waiting for Docker daemon to be ready..."
for i in {1..10}; do
    if sudo docker info > /dev/null 2>&1; then
        break
    fi
    echo "  Attempt $i/10..."
    sleep 2
done

# Verify Docker is running
echo ""
if sudo docker info > /dev/null 2>&1; then
    echo "=========================================="
    echo "✅ Docker daemon started successfully!"
    echo "=========================================="
    echo ""
    echo "📁 Docker data root: $DOCKER_DATA_ROOT"
    echo ""
    echo "🔧 Runtime information:"
    sudo docker info | grep -i runtime -A 3
    echo ""
    echo "🎮 GPU information:"
    if command_exists nvidia-smi; then
        nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
    else
        echo "  nvidia-smi not found"
    fi
    echo ""
    echo "📋 Docker logs: /tmp/dockerd.log"
    echo "=========================================="
else
    echo "=========================================="
    echo "❌ Failed to start Docker daemon"
    echo "=========================================="
    echo "Check logs: tail -f /tmp/dockerd.log"
    exit 1
fi

