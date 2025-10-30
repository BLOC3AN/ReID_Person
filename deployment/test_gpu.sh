#!/bin/bash

# Test GPU access in all services

echo "========================================="
echo "GPU Deployment Test"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test function
test_service_gpu() {
    local service=$1
    local port=$2
    
    echo -e "${YELLOW}Testing $service service...${NC}"
    
    # Test PyTorch CUDA
    result=$(sudo docker exec person_reid_$service python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>&1)
    
    if [[ $result == *"CUDA: True"* ]]; then
        echo -e "${GREEN}✅ $service: GPU Enabled${NC}"
        echo "   $result"
    else
        echo -e "${RED}❌ $service: GPU NOT Available${NC}"
        echo "   $result"
        return 1
    fi
    
    # Test API health
    health=$(curl -s http://localhost:$port/health)
    if [[ $health == *"healthy"* ]] || [[ $health == *"ok"* ]]; then
        echo -e "${GREEN}✅ $service: API Healthy${NC}"
    else
        echo -e "${RED}❌ $service: API Unhealthy${NC}"
        return 1
    fi
    
    echo ""
    return 0
}

# Check if services are running
echo "1. Checking service status..."
sudo docker compose ps
echo ""

# Test each service
echo "2. Testing GPU access in services..."
echo ""

test_service_gpu "extract" "8001"
extract_status=$?

test_service_gpu "register" "8002"
register_status=$?

test_service_gpu "detection" "8003"
detection_status=$?

# Test UI (no GPU needed)
echo -e "${YELLOW}Testing ui service...${NC}"
ui_health=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501)
if [[ $ui_health == "200" ]] || [[ $ui_health == "301" ]]; then
    echo -e "${GREEN}✅ ui: Web Interface Accessible${NC}"
    ui_status=0
else
    echo -e "${RED}❌ ui: Web Interface Not Accessible${NC}"
    ui_status=1
fi
echo ""

# Check GPU status
echo "3. GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv
echo ""

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="

total_tests=4
passed_tests=0

if [ $extract_status -eq 0 ]; then ((passed_tests++)); fi
if [ $register_status -eq 0 ]; then ((passed_tests++)); fi
if [ $detection_status -eq 0 ]; then ((passed_tests++)); fi
if [ $ui_status -eq 0 ]; then ((passed_tests++)); fi

echo "Passed: $passed_tests/$total_tests"
echo ""

if [ $passed_tests -eq $total_tests ]; then
    echo -e "${GREEN}✅ All tests passed! GPU deployment successful.${NC}"
    echo ""
    echo "Access points:"
    echo "  - Web UI: http://localhost:8501"
    echo "  - Extract API: http://localhost:8001/docs"
    echo "  - Register API: http://localhost:8002/docs"
    echo "  - Detection API: http://localhost:8003/docs"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Check logs:${NC}"
    echo "  sudo docker compose logs extract"
    echo "  sudo docker compose logs register"
    echo "  sudo docker compose logs detection"
    echo "  sudo docker compose logs ui"
    exit 1
fi

