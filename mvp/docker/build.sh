#!/bin/bash

# ============================================
# Vision Platform Docker Build Script
# ============================================

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Vision Platform Docker Build${NC}"
echo -e "${BLUE}======================================${NC}"

# Get project root (mvp/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${YELLOW}Project root: ${PROJECT_ROOT}${NC}"

# Check if we're in the right directory
if [ ! -d "$PROJECT_ROOT/training" ]; then
    echo -e "${RED}Error: training/ directory not found${NC}"
    exit 1
fi

# Build base image
echo -e "\n${GREEN}[1/3] Building base image...${NC}"
docker build \
    -f "$SCRIPT_DIR/Dockerfile.base" \
    -t vision-platform-base:latest \
    "$PROJECT_ROOT"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Base image build failed${NC}"
    exit 1
fi

# Build timm image
echo -e "\n${GREEN}[2/3] Building timm image...${NC}"
docker build \
    -f "$SCRIPT_DIR/Dockerfile.timm" \
    -t vision-platform-timm:latest \
    "$PROJECT_ROOT"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ timm image build failed${NC}"
    exit 1
fi

# Build ultralytics image
echo -e "\n${GREEN}[3/3] Building ultralytics image...${NC}"
docker build \
    -f "$SCRIPT_DIR/Dockerfile.ultralytics" \
    -t vision-platform-ultralytics:latest \
    "$PROJECT_ROOT"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ ultralytics image build failed${NC}"
    exit 1
fi

# Success
echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}✓ All images built successfully!${NC}"
echo -e "${GREEN}======================================${NC}"

# List images
echo -e "\n${BLUE}Built images:${NC}"
docker images | grep -E "REPOSITORY|vision-platform"

# Show image sizes
echo -e "\n${BLUE}Image sizes:${NC}"
docker images vision-platform-base:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
docker images vision-platform-timm:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
docker images vision-platform-ultralytics:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo -e "\n${GREEN}Ready to use!${NC}"
echo -e "${YELLOW}Test with: docker run --rm vision-platform-ultralytics:latest python -c 'from ultralytics import YOLOWorld; print(\"OK\")'${NC}"
