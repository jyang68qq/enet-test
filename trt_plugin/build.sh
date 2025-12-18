#!/bin/bash
# Build script for MultiscaleDeformableAttnPlugin

set -e

echo "=== Building MultiscaleDeformableAttnPlugin ==="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required tools
echo "Checking required tools..."
command -v cmake >/dev/null 2>&1 || { echo -e "${RED}ERROR: cmake is required but not installed.${NC}" >&2; exit 1; }
command -v nvcc >/dev/null 2>&1 || { echo -e "${RED}ERROR: CUDA (nvcc) is required but not installed.${NC}" >&2; exit 1; }

echo -e "${GREEN}✓ cmake found${NC}"
echo -e "${GREEN}✓ nvcc found${NC}"

# Try to find TensorRT
TENSORRT_PATH=""
if [ -d "/usr/local/tensorrt" ]; then
    TENSORRT_PATH="/usr/local/tensorrt"
elif [ -d "/usr/src/tensorrt" ]; then
    TENSORRT_PATH="/usr/src/tensorrt"
elif [ -n "$TENSORRT_ROOT" ]; then
    TENSORRT_PATH="$TENSORRT_ROOT"
fi

if [ -n "$TENSORRT_PATH" ]; then
    echo -e "${GREEN}✓ TensorRT found at: $TENSORRT_PATH${NC}"
else
    echo -e "${YELLOW}⚠ TensorRT not found. You may need to specify -DTENSORRT_ROOT=<path>${NC}"
fi

# Create build directory
rm -rf build
mkdir -p build
cd build

echo ""
echo "Running CMake..."
if [ -n "$TENSORRT_PATH" ]; then
    cmake -DTENSORRT_ROOT="$TENSORRT_PATH" ..
else
    cmake ..
fi

echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo -e "${GREEN}=== Build completed successfully! ===${NC}"
echo ""
echo "Output files:"
ls -lh lib/*.so 2>/dev/null || echo "  (plugin library not found)"
ls -lh bin/plugin_example 2>/dev/null || echo "  (example executable not found)"

echo ""
echo "To install the plugin system-wide, run:"
echo "  sudo make install"
