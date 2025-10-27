#!/bin/bash
################################################################################
# Package Script for Person ReID System
################################################################################

echo "================================================================================"
echo "PACKAGING PERSON RE-IDENTIFICATION SYSTEM"
echo "================================================================================"
echo ""

# Get current directory
CURRENT_DIR=$(pwd)
PACKAGE_NAME="person_reid_system"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${PACKAGE_NAME}_${TIMESTAMP}.tar.gz"

echo "📦 Package name: ${OUTPUT_FILE}"
echo "📁 Source: ${CURRENT_DIR}"
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "core" ] || [ ! -d "scripts" ]; then
    echo "❌ ERROR: Must run this script from person_reid_system/ directory"
    exit 1
fi

echo "🔍 Checking package contents..."
echo ""

# Show what will be packaged
echo "Contents to package:"
echo "  ✅ Core modules ($(ls core/*.py | wc -l) files)"
echo "  ✅ Scripts ($(ls scripts/*.py | wc -l) files)"
echo "  ✅ Models ($(ls models/*.pth* 2>/dev/null | wc -l) files)"
echo "  ✅ Config files"
echo "  ✅ Documentation"
echo "  ✅ Sample data"
echo "  ✅ Demo outputs"
echo ""

# Calculate size
TOTAL_SIZE=$(du -sh . | cut -f1)
echo "📊 Total size: ${TOTAL_SIZE}"
echo ""

# Create package
echo "📦 Creating package..."
cd ..
tar -czf "${OUTPUT_FILE}" \
    --exclude="${PACKAGE_NAME}/.git" \
    --exclude="${PACKAGE_NAME}/__pycache__" \
    --exclude="${PACKAGE_NAME}/*/__pycache__" \
    --exclude="${PACKAGE_NAME}/*.pyc" \
    "${PACKAGE_NAME}/"

if [ $? -eq 0 ]; then
    PACKAGE_SIZE=$(du -sh "${OUTPUT_FILE}" | cut -f1)
    echo ""
    echo "================================================================================"
    echo "✅ PACKAGE CREATED SUCCESSFULLY"
    echo "================================================================================"
    echo ""
    echo "📦 Package: ${OUTPUT_FILE}"
    echo "📊 Size: ${PACKAGE_SIZE}"
    echo "📁 Location: $(pwd)/${OUTPUT_FILE}"
    echo ""
    echo "To extract:"
    echo "  tar -xzf ${OUTPUT_FILE}"
    echo ""
    echo "================================================================================"
else
    echo ""
    echo "❌ ERROR: Failed to create package"
    exit 1
fi

