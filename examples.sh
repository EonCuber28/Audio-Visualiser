#!/bin/bash
# Example usage of the audio visualizer with random color palettes

echo "Audio Visualizer - Random Color Palette Examples"
echo "=================================================="
echo ""
echo "These examples demonstrate the --random-color feature:"
echo ""

# Example 1: Auto-generated palette
echo "1. Auto-generated palette (different every time):"
echo "   python3 audio_visualizer.py --random-color"
echo ""

# Example 2: Fixed seed for reproducible colors
echo "2. Reproducible palette with seed 42:"
echo "   python3 audio_visualizer.py --random-color --color-seed 42"
echo ""

# Example 3: Debug mode to see palette parameters
echo "3. Debug mode (shows palette generation details):"
echo "   python3 audio_visualizer.py --random-color --debug"
echo ""

# Example 4: High performance with random colors
echo "4. High performance mode (120 FPS):"
echo "   python3 audio_visualizer.py --random-color --update-rate 120"
echo ""

# Example 5: Various seeds for exploration
echo "5. Try different seeds to explore palettes:"
echo "   python3 audio_visualizer.py --random-color --color-seed 1"
echo "   python3 audio_visualizer.py --random-color --color-seed 100"
echo "   python3 audio_visualizer.py --random-color --color-seed 999"
echo "   python3 audio_visualizer.py --random-color --color-seed 12345"
echo ""

echo "=================================================="
echo "Features of Random Color Mode:"
echo "  • 6-14 harmonious colors per palette"
echo "  • Cosine-based color generation (designer quality)"
echo "  • Smooth transitions via stereo balance"
echo "  • Reproducible with --color-seed"
echo "  • Different palette each time (auto seed)"
echo "=================================================="
