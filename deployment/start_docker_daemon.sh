#!/bin/bash

# Script to start Docker daemon with NVIDIA runtime support
# This ensures GPU access for containers

# Kill existing dockerd processes
echo "Stopping existing Docker daemon..."
sudo pkill -f "dockerd --data-root"
sleep 2

# Start Docker daemon with config file
echo "Starting Docker daemon with NVIDIA runtime..."
sudo dockerd \
  --data-root /home/ubuntu/data/docker \
  --config-file /etc/docker/daemon.json \
  -H unix:///var/run/docker.sock \
  > /tmp/dockerd.log 2>&1 &

# Wait for Docker to be ready
echo "Waiting for Docker daemon to be ready..."
sleep 5

# Verify Docker is running
if sudo docker info > /dev/null 2>&1; then
    echo "✅ Docker daemon started successfully"
    echo ""
    echo "Runtime information:"
    sudo docker info | grep -i runtime -A 3
    echo ""
    echo "GPU information:"
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
else
    echo "❌ Failed to start Docker daemon"
    echo "Check logs: tail -f /tmp/dockerd.log"
    exit 1
fi

