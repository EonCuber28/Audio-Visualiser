#!/bin/bash
# Build script for Audio Visualizer - Creates standalone Linux executable
# Usage: ./build_executable.sh

set -e  # Exit on error

echo "=================================="
echo "Audio Visualizer Build Script"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Not running in a virtual environment${NC}"
    echo "It's recommended to activate the venv first:"
    echo "  source bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if PyInstaller is installed
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo -e "${YELLOW}PyInstaller not found. Installing...${NC}"
    pip install pyinstaller
    echo -e "${GREEN}✓ PyInstaller installed${NC}"
else
    echo -e "${GREEN}✓ PyInstaller already installed${NC}"
fi

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.spec
echo -e "${GREEN}✓ Cleaned${NC}"

# Build the executable
echo ""
echo "Building executable..."
echo "This may take a few minutes..."
echo ""

pyinstaller \
    --onefile \
    --name "audio-visualizer" \
    --add-data "README.md:." \
    --hidden-import "numpy" \
    --hidden-import "numpy.core" \
    --hidden-import "numpy.fft" \
    --hidden-import "scipy" \
    --hidden-import "scipy.interpolate" \
    --hidden-import "PyQt5" \
    --hidden-import "PyQt5.QtCore" \
    --hidden-import "PyQt5.QtGui" \
    --hidden-import "PyQt5.QtWidgets" \
    --collect-all numpy \
    --collect-all scipy \
    --collect-all PyQt5 \
    --strip \
    --noupx \
    audio_visualizer.py

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================="
    echo "✓ Build successful!"
    echo "==================================${NC}"
    echo ""
    echo "Executable location: dist/audio-visualizer"
    echo ""
    
    # Show file size
    SIZE=$(du -h dist/audio-visualizer | cut -f1)
    echo "File size: $SIZE"
    echo ""
    
    # Make executable
    chmod +x dist/audio-visualizer
    
    echo "To run the visualizer:"
    echo "  ./dist/audio-visualizer"
    echo ""
    echo "To install system-wide (optional):"
    echo "  sudo cp dist/audio-visualizer /usr/local/bin/"
    echo ""
    
    # Test if it runs
    echo "Testing executable..."
    if ./dist/audio-visualizer --help > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Executable test passed${NC}"
    else
        echo -e "${YELLOW}⚠ Executable test failed - may still work but verify manually${NC}"
    fi
    
else
    echo ""
    echo -e "${RED}=================================="
    echo "✗ Build failed!"
    echo "==================================${NC}"
    echo ""
    echo "Check the error messages above for details."
    exit 1
fi

echo ""
echo "Build complete!"
