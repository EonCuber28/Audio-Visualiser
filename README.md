# Audio Frequency Visualizer

Real-time audio frequency analyzer that captures system output audio and displays a dynamic frequency spectrum using FFT (Fourier Transform).

**Supports:** Linux (PulseAudio/PipeWire) and Windows 10/11 (WASAPI Loopback)

## Features

- **Real-time Visualization**: Updates at configurable FPS (default 60) for smooth animation
- **System Audio Capture**: Monitors your system's output audio (what you're listening to)
- **Cross-Platform**: Works on Linux and Windows with platform-specific audio backends
- **FFT Analysis**: Uses Fast Fourier Transform to analyze frequency content
- **Adaptive Window**: Fully resizable GUI that adapts to window size
- **Multiple Color Modes**: 
  - **Normal Mode**: Stereo balance-based colors (blue=left, white=center, red=right)
  - **Happy Mode**: Vibrant, joyful colors with enhanced saturation
  - **Random Color Mode**: Procedurally generated harmonious color palettes using cosine-based algorithms
- **Stereo Visualization**: Shows left/right channel balance through color
- **Logarithmic Scale**: Matches human hearing perception
- **ISO 226 Equal-Loudness**: Compensates for human hearing sensitivity curves

## Installation

### Quick Install (Recommended)

The easiest way to install is using the setup.py script in a Python virtual environment:

```bash
# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install the visualizer
pip install -e .

# Run the visualizer
audio-visualizer
```

This will automatically install all dependencies for your platform (including PyAudio on Windows).

### Manual Installation

#### Windows 10/11

**1. Install Python Dependencies**

```bash
# Install required packages - soundcard is RECOMMENDED for hassle-free setup
pip install numpy PyQt5 soundcard scipy

# soundcard uses native WASAPI loopback to capture ALL system audio
# without requiring Stereo Mix or any other manual configuration!
```

**That's it!** With `soundcard`, the visualizer automatically captures all audio playing through your speakers - no additional setup required.

**Alternative Libraries (only if soundcard doesn't work):**

```bash
# sounddevice - requires Stereo Mix to be enabled (see below)
pip install numpy PyQt5 sounddevice scipy

# pyaudio - also requires Stereo Mix to be enabled
pip install numpy PyQt5 pyaudio scipy
```

**If using sounddevice or pyaudio, you must enable Stereo Mix:**

1. Right-click the speaker icon in system tray → **Sounds**
2. Go to **Recording** tab
3. Right-click in empty space → **Show Disabled Devices**
4. Find **Stereo Mix** (or **Wave Out Mix**)
5. Right-click → **Enable**
6. Set as default recording device (optional)

**Note:** If Stereo Mix is not available, use `soundcard` instead - it doesn't require Stereo Mix.

#### Linux (Ubuntu/Debian)

**1. Install System Dependencies**

```bash
# Update package list
sudo apt update

# Install PyQt5 dependencies
sudo apt install -y python3-pyqt5

# Install PulseAudio utils (if not already installed)
sudo apt install -y pulseaudio pavucontrol
```

**2. Install Python Dependencies**

```bash
# Using pip
pip install numpy PyQt5 scipy

# Or using requirements.txt
pip install -r requirements.txt
```

**Note:** PyAudio is not required on Linux - the visualizer uses `parec` instead.

**3. Configure PulseAudio for Audio Monitoring**

To capture system output audio, you need to load the loopback module:

```bash
# Load loopback module (temporary - until reboot)
pactl load-module module-loopback latency_msec=1

# To make it permanent, add to PulseAudio config:
echo "load-module module-loopback latency_msec=1" >> ~/.config/pulse/default.pa
```

Alternatively, you can use `pavucontrol` to set the recording device:
```bash
pavucontrol
```
Go to the "Recording" tab while the visualizer is running and select "Monitor of [your audio device]".

## Usage

### Run the Visualizer

```bash
# Basic usage
python3 audio_visualizer.py

# With random color palette (auto-generated seed)
python3 audio_visualizer.py --random-color

# With fixed seed for reproducible colors
python3 audio_visualizer.py --random-color --color-seed 42

# Happy mode with vibrant colors
python3 audio_visualizer.py --happy

# High performance mode (120 FPS)
python3 audio_visualizer.py --update-rate 120

# Debug mode with performance stats
python3 audio_visualizer.py --debug
```

### Command-Line Arguments

- `--human-bias FACTOR`: ISO 226 equal-loudness influence (0.0-1.0, default: 0.5)
- `--buffer-size SAMPLES`: FFT buffer size (default: 2048, larger = better bass resolution)
- `--chunk-size SAMPLES`: Audio chunk size (default: 512, smaller = lower latency)
- `--update-rate HZ`: Display update rate (default: 60, higher = smoother animation)
- `--silent`: Suppress all output except errors
- `--debug`: Enable debug output (performance stats, audio info, etc.)
- `--happy`: Enable joyful color mode with vibrant, saturated colors
- `--random-color`: Enable random color palette generation with harmonious colors
- `--color-seed SEED`: Random seed for color palette (default: auto from timestamp)

### Controls

- **F11 / F**: Toggle fullscreen mode
- **Window Resize**: The visualization automatically adapts to window size
- **Close**: Click the X button or press Alt+F4

## Troubleshooting

### No Audio Being Captured

1. **Check PulseAudio is running:**
   ```bash
   pulseaudio --check
   ```

2. **List available audio devices:**
   ```bash
   pactl list sources short
   ```
   Look for devices with "monitor" in the name.

3. **Use pavucontrol to select the correct source:**
   ```bash
   pavucontrol
   ```
   While the visualizer is running, go to the "Recording" tab and select the monitor of your output device.

### Installation Issues

**PyAudio fails to install:**
```bash
# Try installing from apt first
sudo apt install python3-pyaudio
```

**PyQt5 issues:**
```bash
# Install from apt instead of pip
sudo apt install python3-pyqt5
```

### Permission Issues

If you get permission errors:
```bash
# Add your user to the audio group
sudo usermod -a -G audio $USER

# Log out and log back in for changes to take effect
```

## How It Works

1. **Audio Capture**: Captures audio from PulseAudio/PipeWire monitor using `parec` (system output)
2. **Signal Processing**: 
   - Applies Hamming window to reduce spectral leakage
   - Performs FFT (Fast Fourier Transform) to convert time-domain audio to frequency-domain
   - Separates left/right channels for stereo analysis
   - Applies ISO 226 equal-loudness correction for human hearing perception
   - Converts to decibel scale for better visualization
3. **Color Generation**:
   - **Normal Mode**: Stereo balance mapped to blue→white→red gradient
   - **Happy Mode**: Enhanced saturation with vibrant color palette
   - **Random Color Mode**: Procedurally generated using cosine palette function (Inigo Quilez style)
     - Generates 6-14 harmonious colors per palette
     - Uses hue, chroma, and luminance randomization
     - Maps stereo balance or frequency to palette index
     - Interpolates between colors for smooth transitions
4. **Smoothing**: Uses exponential moving average to smooth the display
5. **Visualization**: Draws frequency bars on a logarithmic scale with dynamic colors

## Random Color Palette Algorithm

The `--random-color` mode implements a sophisticated color generation pipeline:

1. **Palette Generation (on startup)**:
   - Random seed initialization (from timestamp or user-provided)
   - Generate base parameters: hue range, chroma (saturation), luminance
   - Create 6-14 colors using cosine palette function: `rgb = a + b * cos(2π(c*t + d))`
   - Apply chroma/luminance adjustments for designer-quality colors

2. **FFT Processing (per frame)**:
   - Compute FFT magnitudes and stereo balance
   - Map stereo balance to palette index: `index = (balance + 1) * 0.5 * (palette_size - 1)`
   - Interpolate between adjacent palette colors for smooth transitions
   - Apply brightness curve: `brightness = clamp((magnitude_db + 60) / 60, 0, 1)`

3. **Final Color**: `finalColor = paletteColor * brightness^0.4`

## Technical Details

- **Sample Rate**: 48,000 Hz (PipeWire default)
- **Chunk Size**: 512 samples (configurable, default for low latency)
- **FFT Buffer Size**: 2048 samples (configurable, default for good frequency resolution)
- **Update Rate**: 60 FPS (configurable)
- **Frequency Range**: 20 Hz - 20 kHz (human hearing range)
- **Window Function**: Hamming window with amplitude correction
- **Color Interpolation**: Linear interpolation between palette colors
- **Brightness Curve**: Gamma correction (γ = 0.4) for better contrast

## License

Free to use and modify.
