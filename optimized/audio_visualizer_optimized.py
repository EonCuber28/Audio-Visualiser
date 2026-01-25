#!/usr/bin/env python3
"""
Real-time Audio Frequency Visualizer (Cross-Platform)
Captures system output audio and displays frequency spectrum using FFT
Supports: Linux (PulseAudio/PipeWire), Windows 10/11 (WASAPI Loopback)
"""

import sys
import os
import numpy as np
import builtins

# NumPy 2.x compatibility: add fromstring shim BEFORE any other imports
# This must be done early so soundcard can use it
if not hasattr(np, 'fromstring'):
    def _fromstring_compat(string, dtype=float, count=-1, sep='', offset=0):
        # NumPy 2.x removed fromstring, redirect to frombuffer
        if sep != '':
            # Text mode - not supported in frombuffer
            raise NotImplementedError("Text mode fromstring not supported in NumPy 2.x")
        # Binary mode - frombuffer has different signature
        return np.frombuffer(string, dtype=dtype, count=count, offset=offset)
    
    np.fromstring = _fromstring_compat
else:
    # If fromstring somehow exists, still override with frombuffer version
    _orig_fromstring = np.fromstring
    def _fromstring_compat(string, dtype=float, count=-1, sep='', offset=0, like=None):
        # NumPy 2.x: handle text mode
        if sep != '':
            raise NotImplementedError("Text mode fromstring not supported in NumPy 2.x")
        # Use frombuffer for binary mode
        return np.frombuffer(string, dtype=dtype, count=count, offset=offset)
    
    np.fromstring = _fromstring_compat

# Import soundcard IMMEDIATELY after numpy patch is in place
try:
    import soundcard as sc
    SOUNDCARD_AVAILABLE = True
except ImportError:
    SOUNDCARD_AVAILABLE = False

import struct
import argparse
import platform
import importlib.util
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSpinBox, QSlider, QCheckBox, QLineEdit, QPushButton,
                             QComboBox, QGroupBox, QDialog, QMessageBox)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QPainterPath, QPolygonF
import subprocess
import threading
import queue
import time
import random
import json
from pathlib import Path

# Windows volume control for normalization
if platform.system() == 'Windows':
    try:
        # Import ctypes helpers now; delay importing comtypes/pycaw until runtime
        # to avoid COM initialization side-effects (CoInitializeEx) during module import.
        from ctypes import cast, POINTER
        # Only detect presence of pycaw here; don't import it yet.
        HAS_PYCAW = importlib.util.find_spec('pycaw') is not None
    except Exception:
        HAS_PYCAW = False
else:
    HAS_PYCAW = False

# Fix Windows console UTF-8 encoding for checkmark characters
if platform.system() == 'Windows':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

# Platform detection
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

# Detect availability without importing (avoid COM init before Qt)
# Check all audio libraries for Windows
HAS_SOUNDCARD = False
HAS_SOUNDDEVICE = False
HAS_PYAUDIO = False

if IS_WINDOWS:
    HAS_SOUNDCARD = importlib.util.find_spec('soundcard') is not None
    HAS_SOUNDDEVICE = importlib.util.find_spec('sounddevice') is not None
    HAS_PYAUDIO = (importlib.util.find_spec('pyaudiowpatch') is not None) or \
                  (importlib.util.find_spec('pyaudio') is not None)

# Try to import scipy for better interpolation, fall back to linear if not available
try:
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Module-wide print wrapper that respects --silent and --debug flags.
# We shadow the module-level name `print` so all subsequent unqualified
# calls to `print()` in this module go through our wrapper. Use
# `_set_print_flags(silent, debug)` to update behavior from the
# `AudioVisualizer` constructor.
_ORIG_PRINT = builtins.print
_GLOBAL_SILENT = False
_GLOBAL_DEBUG = False

def _set_print_flags(silent, debug):
    global _GLOBAL_SILENT, _GLOBAL_DEBUG
    _GLOBAL_SILENT = bool(silent)
    _GLOBAL_DEBUG = bool(debug)

def print(*args, debug_only=False, force=False, **kwargs):
    """Module-local print wrapper.

    - If `force=True`, always prints (bypasses silent).
    - If `debug_only=True`, prints only when `_GLOBAL_DEBUG` is True.
    - Otherwise prints unless `_GLOBAL_SILENT` is True.
    """
    if _GLOBAL_SILENT and not force:
        return
    if debug_only and not _GLOBAL_DEBUG:
        return
    return _ORIG_PRINT(*args, **kwargs)


# ===== Settings persistence (cross-platform) =====
def get_config_dir():
    """Get platform-appropriate config directory for settings persistence"""
    if IS_WINDOWS:
        # Windows: Use AppData/Roaming
        config_dir = Path.home() / 'AppData' / 'Roaming' / 'audio-visualizer'
    elif platform.system() == 'Darwin':
        # macOS: Use Library/Application Support
        config_dir = Path.home() / 'Library' / 'Application Support' / 'audio-visualizer'
    else:
        # Linux and others: Use ~/.config (XDG standard)
        config_dir = Path.home() / '.config' / 'audio-visualizer'
    
    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_settings_file():
    """Get path to settings JSON file"""
    return get_config_dir() / 'settings.json'


def load_settings():
    """Load settings from file, return dict or empty dict if file doesn't exist"""
    settings_file = get_settings_file()
    
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load settings file: {e}")
            return {}
    
    return {}


def save_settings(settings):
    """Save settings to file"""
    settings_file = get_settings_file()
    
    try:
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save settings file: {e}")


class AudioVisualizer(QMainWindow):
    # Signal for thread-safe visualization updates
    update_signal = pyqtSignal()
    
    def __init__(self, chunk_size=1024, buffer_size=8192, update_rate=60, human_bias=0.5, silent=False, debug=False, happy_mode=False, random_color=False, color_seed=None, num_bars=None, device_id=None, device_name=None):
        super().__init__()
        # Initialize module-level print flags early so any subsequent
        # prints (including those in load_settings()) respect the
        # --silent and --debug command-line flags.
        _set_print_flags(silent, debug)
        self.setWindowTitle("Autisum Frequency Visualizer")
        self.setGeometry(100, 100, 1200, 600)
        self.user_num_bars = num_bars  # User-specified bar count (None = auto)
        
        # Fullscreen state tracking
        self.is_fullscreen = False
        self.normal_geometry = None
        
        # Minimal UI - no menu bar, status bar, etc.
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        
        # Audio parameters (from command line arguments)
        self.CHUNK = chunk_size  # Small chunks for low latency
        self.FFT_SIZE = buffer_size  # Larger FFT buffer for better low frequency resolution
        self.FORMAT = 'int16'
        self.CHANNELS = 2
        self.RATE = 48000  # Sample rate (matching PipeWire default)
        self.HUMAN_BIAS = human_bias  # ISO 226 equal-loudness influence
        self.UPDATE_RATE = update_rate  # Display update rate in Hz
        self.SILENT = silent  # Suppress all output
        self.DEBUG = debug  # Enable debug output
        self.HAPPY_MODE = happy_mode  # Joyful color mode
        self.RANDOM_COLOR = random_color  # Random color palette mode
        # Optional user-specified device overrides (string id or name)
        self.user_device_id = device_id
        self.user_device_name = device_name
        
        # Load saved settings from file (command line args override saved settings)
        saved_settings = load_settings()
        
        # Track if we loaded any settings
        loaded_count = 0
        
        # Apply saved settings if no command line args were provided (all defaults)
        # Only load from file if user didn't explicitly set these values on command line
        if chunk_size == 1024:  # Default value
            self.CHUNK = saved_settings.get('chunk_size', chunk_size)
            if 'chunk_size' in saved_settings:
                loaded_count += 1
        if buffer_size == 8192:  # Default value
            self.FFT_SIZE = saved_settings.get('buffer_size', buffer_size)
            if 'buffer_size' in saved_settings:
                loaded_count += 1
        if update_rate == 60:  # Default value
            self.UPDATE_RATE = saved_settings.get('update_rate', update_rate)
            if 'update_rate' in saved_settings:
                loaded_count += 1
        if human_bias == 0.5:  # Default value
            self.HUMAN_BIAS = saved_settings.get('human_bias', human_bias)
            if 'human_bias' in saved_settings:
                loaded_count += 1
        if not silent:  # False is default
            self.SILENT = saved_settings.get('silent', silent)
            if 'silent' in saved_settings:
                loaded_count += 1
        if not debug:  # False is default
            self.DEBUG = saved_settings.get('debug', debug)
            if 'debug' in saved_settings:
                loaded_count += 1
        if not happy_mode:  # False is default
            self.HAPPY_MODE = saved_settings.get('happy_mode', happy_mode)
            if 'happy_mode' in saved_settings:
                loaded_count += 1
        if not random_color:  # False is default
            self.RANDOM_COLOR = saved_settings.get('random_color', random_color)
            if 'random_color' in saved_settings:
                loaded_count += 1
        if color_seed is None:  # None is default
            color_seed = saved_settings.get('color_seed', color_seed)
            if 'color_seed' in saved_settings:
                loaded_count += 1
        if num_bars is None:  # None is default
            self.user_num_bars = saved_settings.get('num_bars', num_bars)
            if 'num_bars' in saved_settings:
                loaded_count += 1
        
        # Notify user if settings were loaded
        if loaded_count > 0 and not self.SILENT:
            print(f"✓ Loaded {loaded_count} saved settings from {get_settings_file()}")
        
        # Random color palette generation
        if self.RANDOM_COLOR:
            self.color_palette = self._generate_color_palette(color_seed)
            if not self.SILENT:
                print(f"✓ Generated random color palette with {len(self.color_palette)} colors (seed: {color_seed if color_seed else 'auto'})")
        else:
            self.color_palette = None
        
        # Rolling buffer for FFT
        self.audio_buffer = np.zeros(self.FFT_SIZE, dtype=np.float32)
        self.audio_buffer_left = np.zeros(self.FFT_SIZE, dtype=np.float32)
        self.audio_buffer_right = np.zeros(self.FFT_SIZE, dtype=np.float32)
        
        # Pre-compute Hamming window (optimization: avoid recalculating every frame)
        self.hamming_window = np.hamming(self.FFT_SIZE).astype(np.float32)
        # Window correction factor to compensate for Hamming window attenuation
        self.window_correction = np.sqrt(self.FFT_SIZE / np.sum(self.hamming_window**2))
        
        # Pre-compute ISO 226:2003 equal-loudness correction (optimization)
        # Reference frequencies from ISO 226:2003 standard
        self.iso226_freqs = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 
                                     315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 
                                     3150, 4000, 5000, 6300, 8000, 10000, 12500])
        # Sound pressure levels (dB SPL) for 60 phon equal-loudness contour from ISO 226:2003
        iso226_60phon = np.array([109.5, 104.2, 99.1, 94.2, 89.9, 85.8, 81.9, 78.5, 75.4, 
                                 72.3, 69.7, 67.4, 65.4, 63.5, 62.1, 60.8, 59.8, 60.0, 
                                 62.1, 63.3, 60.0, 57.4, 56.5, 57.7, 61.0, 66.5, 71.6, 
                                 73.3, 68.7])
        # Boost = SPL - 60: frequencies needing more SPL get positive boost to compensate
        self.iso226_boost = iso226_60phon - 60.0
        
        # Frequency analysis - rfft returns FFT_SIZE//2 + 1 values
        fft_size = self.FFT_SIZE // 2 + 1
        self.fft_data = np.full(fft_size, -80.0)  # Initialize to minimum dB instead of 0
        self.freqs = np.fft.rfftfreq(self.FFT_SIZE, 1.0 / self.RATE)
        self.stereo_balance = np.zeros(fft_size)
        self.stereo_balance_smoothed = np.zeros(fft_size)  # Smoothed stereo balance for color transitions
        
        # Oscilloscope waveform buffer (store recent samples for display)
        self.oscilloscope_samples = 1024  # Number of samples to display (reduced for performance)
        self.waveform_buffer = np.zeros(self.oscilloscope_samples, dtype=np.float32)
        # Per-platform oscilloscope gain (increase on Linux where levels are lower)
        self.oscilloscope_gain = 2.5 if IS_LINUX else 1.0
        
        # Audio buffer queue
        self.audio_queue = queue.Queue(maxsize=3)  # Smaller queue to minimize latency
        self.running = True
        self.frames_without_data = 0  # Track consecutive frames without audio
        
        # Windows system volume tracking for normalization
        self.system_volume = 1.0  # Default to 100% if we can't get it
        self.volume_interface = None
        if IS_WINDOWS and HAS_PYCAW:
            try:
                # Lazy import pycaw/comtypes here to avoid CoInitializeEx running
                # during module import which can conflict with Qt (leading to
                # "Cannot change thread mode after it is set")
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
                self.system_volume = self.volume_interface.GetMasterVolumeLevelScalar()
                if not self.SILENT:
                    print(f"✓ Windows system volume detection enabled (current: {self.system_volume*100:.0f}%)")
            except OSError as ose:
                # COM initialization failed due to threading model mismatch (common with Qt)
                if self.DEBUG:
                    print("pycaw/comtypes OSError during COM init:", ose)
                if not self.SILENT:
                    print("Note: pycaw is installed but could not initialize due to COM threading mode.")
                    print("System volume compensation disabled.")
                self.volume_interface = None
                # Do not reassign global HAS_PYCAW here (would shadow module variable)
            except Exception as e:
                if self.DEBUG:
                    print(f"Could not initialize volume tracking: {e}")
                self.volume_interface = None
        elif IS_WINDOWS and not HAS_PYCAW:
            if not self.SILENT:
                print("Note: Install pycaw for automatic system volume compensation:")
                print("  pip install pycaw")
        
        # Performance monitoring
        self.frame_times = []
        self.last_frame_time = time.perf_counter()
        self.fps_report_interval = 2.0  # Report every 2 seconds
        self.last_fps_report = time.perf_counter()
        self.target_frame_time = 1.0 / 60.0  # Target 60 FPS (16.67ms)
        
        # Start audio capture thread
        self.audio_thread = threading.Thread(target=self.capture_audio, daemon=True)
        self.audio_thread.start()
        
        # Setup UI
        self.setup_ui()
        
        # Qt timers are broken on this system - use Python threading with sleep
        self.timer_interval = 1.0 / self.UPDATE_RATE
        
        # Timing tracking
        self.last_viz_time = time.perf_counter()
        
        # Connect signal to slot for thread-safe updates
        self.update_signal.connect(self.update_visualization)
        
        if not self.SILENT:
            print(f"Timer interval: {self.timer_interval*1000:.2f}ms (target {self.UPDATE_RATE} FPS)")
        
        # Start visualization thread with proper sleep-based timing
        self.viz_running = True
        self.viz_thread = threading.Thread(target=self._visualization_loop, daemon=True)
        self.viz_thread.start()
    
    def _generate_color_palette(self, seed=None):
        """Generate random color palette using cosine palette function
        
        This implements the end-to-end pipeline:
        1. Generate base parameters (hue, chroma, luminance)
        2. Use HSL color space for 3-color palette (Left, Center, Right)
        3. Return array of 3 RGB colors for stereo balance interpolation
        """
        # Initialize random seed with high precision
        if seed is None:
            # Use time with nanosecond precision to ensure different seeds
            seed = random.randint(0, (2**32)-1)%(2**32)
        else:
            seed = int(seed) % (2**32)
        
        # Store seed for display
        self._palette_seed = seed
        
        # Use a local RNG instead of global state
        rng = np.random.RandomState(seed)
        
        if self.DEBUG:
            print(f"Palette seed: {seed}")
        
        # Step 1: Generate base hue for center color
        hue_center = rng.uniform(0.0, 1.0)
        
        # Step 2: Generate complementary/analogous hues for left and right
        # Use 120-degree separation for triadic harmony
        hue_separation = 0.25 + 0.1 * rng.uniform(-1.0, 1.0)  # ~90-150 degrees
        hue_left = (hue_center - hue_separation) % 1.0
        hue_right = (hue_center + hue_separation) % 1.0
        
        # Step 3: Generate saturation and luminance
        saturation = 0.6 + 0.3 * rng.uniform(0.0, 1.0)  # 0.6-0.9
        luminance = 0.65 + 0.25 * rng.uniform(0.0, 1.0)  # 0.65-0.9
        
        if self.DEBUG:
            print(f"Palette: L={hue_left:.3f}, C={hue_center:.3f}, R={hue_right:.3f}, sat={saturation:.3f}, lum={luminance:.3f}")
        
        # Step 4: Convert HSL to RGB for each color
        def hsl_to_rgb(h, s, l):
            """Convert HSL to RGB (all values in [0,1])"""
            c = (1 - abs(2 * l - 1)) * s
            x = c * (1 - abs((h * 6) % 2 - 1))
            m = l - c / 2
            
            h6 = h * 6
            if h6 < 1:
                r, g, b = c, x, 0
            elif h6 < 2:
                r, g, b = x, c, 0
            elif h6 < 3:
                r, g, b = 0, c, x
            elif h6 < 4:
                r, g, b = 0, x, c
            elif h6 < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
        
        palette = [
            hsl_to_rgb(hue_left, saturation, luminance),    # Left
            hsl_to_rgb(hue_center, saturation, luminance),  # Center
            hsl_to_rgb(hue_right, saturation, luminance)    # Right
        ]
        
        return palette
    
    def regenerate_palette(self):
        """Regenerate random color palette with new seed"""
        if self.RANDOM_COLOR:
            self.color_palette = self._generate_color_palette()
            # Clear the canvas color cache to force re-generation with new palette
            if hasattr(self, 'canvas'):
                if hasattr(self.canvas, 'cached_colors'):
                    self.canvas.cached_colors.clear()
                # Invalidate color LUT to force rebuild with new palette
                if hasattr(self.canvas, '_color_lut'):
                    self.canvas._color_lut = None
                    self.canvas._color_lut_mode = None
            if not self.SILENT:
                print(f"✓ Regenerated palette (seed: {self._palette_seed})")
        
    def capture_audio(self):
        """Capture audio using platform-specific method"""
        if self.DEBUG:
            print(f"Audio capture thread started (platform: {'Windows' if IS_WINDOWS else 'Linux'})")
        if IS_WINDOWS:
            self._capture_audio_windows()
        else:
            self._capture_audio_linux()
    
    def _capture_audio_windows(self):
        """Capture audio on Windows using soundcard, sounddevice, or PyAudio WASAPI loopback"""
        if self.DEBUG:
            print(f"Windows capture: HAS_SOUNDCARD={HAS_SOUNDCARD}, HAS_SOUNDDEVICE={HAS_SOUNDDEVICE}, HAS_PYAUDIO={HAS_PYAUDIO}")
        # Try soundcard first - it handles multi-channel WASAPI loopback better
        if HAS_SOUNDCARD:
            if self.DEBUG:
                print("Calling _capture_audio_windows_soundcard()...")
            self._capture_audio_windows_soundcard()
        elif HAS_PYAUDIO:
            if self.DEBUG:
                print("Calling _capture_audio_windows_pyaudio()...")
            self._capture_audio_windows_pyaudio()
        elif HAS_SOUNDDEVICE:
            self._capture_audio_windows_sounddevice()
        elif HAS_PYAUDIO:
            self._capture_audio_windows_pyaudio()
        else:
            if not self.SILENT:
                print("✗ ERROR: No audio library installed for Windows.")
                print("Install soundcard for automatic system audio capture:")
                print("  pip install soundcard")
                print("")
                print("soundcard uses native WASAPI loopback to capture ALL audio")
                print("playing through your speakers - no extra setup required!")
    
    def _capture_audio_windows_soundcard(self):
        """Capture audio using soundcard (native WASAPI loopback - best for Windows)
        
        This method captures system audio output directly using WASAPI loopback,
        which requires NO extra setup like enabling Stereo Mix. It automatically
        captures all audio playing through the default speaker.
        """
        global SOUNDCARD_AVAILABLE
        
        if not SOUNDCARD_AVAILABLE:
            if not self.SILENT:
                print("soundcard not available, falling back to pyaudiowpatch...")
            return self._capture_audio_windows_pyaudiowpatch()
        
        try:
            if not self.SILENT:
                print("Using soundcard for audio capture (native WASAPI loopback)...")
            
            # Get default speaker and its loopback - this is the key to capturing
            # system audio without any extra setup on Windows
            loopback = None
            default_speaker = None
            
            try:
                # Get default speaker (output device)
                default_speaker = sc.default_speaker()
                default_name = getattr(default_speaker, 'name', None)
                default_id = getattr(default_speaker, 'id', None)
                if not self.SILENT:
                    print(f"Default speaker: {default_name} (id: {default_id})")

                # Get all microphones - prefer loopback device that matches the default speaker
                all_mics = sc.all_microphones(include_loopback=True)

                # If user requested a specific device, try that first
                if getattr(self, 'user_device_id', None):
                    try:
                        loopback = sc.get_microphone(id=str(self.user_device_id), include_loopback=True)
                        if not self.SILENT:
                            print(f"✓ Using user-specified device id: {getattr(loopback,'name',None)} (id: {getattr(loopback,'id',None)})")
                    except Exception:
                        if self.DEBUG:
                            print(f"Could not open user device id {self.user_device_id}")

                if loopback is None and getattr(self, 'user_device_name', None):
                    # Try to match user-specified name substring
                    uname = self.user_device_name.lower()
                    for mic in all_mics:
                        if uname in getattr(mic, 'name', '').lower():
                            loopback = mic
                            if not self.SILENT:
                                print(f"✓ Using user-specified device name match: {mic.name} (id: {mic.id})")
                            break

                # First: try to find an exact loopback match for default speaker by name or id
                if default_name or default_id:
                    for mic in all_mics:
                        if not getattr(mic, 'isloopback', False):
                            continue
                        mic_name = getattr(mic, 'name', '')
                        mic_id = getattr(mic, 'id', None)
                        name_match = default_name and (default_name in mic_name or mic_name in default_name)
                        id_match = default_id and mic_id and (str(default_id) in str(mic_id) or str(mic_id) in str(default_id))
                        if name_match or id_match:
                            loopback = mic
                            if not self.SILENT:
                                print(f"✓ Found matching loopback: {mic_name} (id: {mic_id})")
                            break

                # Second: look for explicit loopback devices, prefer ones with many channels (often outputs)
                if loopback is None:
                    for mic in all_mics:
                        if getattr(mic, 'isloopback', False):
                            # prefer devices with >2 channels (likely the main speaker device)
                            if getattr(mic, 'channels', 0) > 2:
                                loopback = mic
                                if not self.SILENT:
                                    print(f"✓ Selected high-channel loopback device: {getattr(mic,'name',None)} (channels: {getattr(mic,'channels',None)})")
                                break

                # Third: consider Stereo Mix ONLY if it appears as loopback or if no loopback found
                if loopback is None:
                    for mic in all_mics:
                        mic_name = getattr(mic, 'name', '').lower()
                        if 'stereo mix' in mic_name:
                            # Accept stereo mix only if it's a loopback or as a last resort
                            if getattr(mic, 'isloopback', False):
                                loopback = mic
                                if not self.SILENT:
                                    print(f"✓ Using Stereo Mix loopback: {mic.name}")
                                break

                # If no exact match, try to get loopback by speaker ID using soundcard API
                if loopback is None and default_id is not None:
                    try:
                        # soundcard expects the device identifier string; try both id and name
                        loopback = sc.get_microphone(id=str(default_id), include_loopback=True)
                        if not self.SILENT:
                            print(f"✓ Got loopback via speaker ID: {getattr(loopback, 'name', '')} (id: {getattr(loopback, 'id', None)})")
                    except Exception:
                        try:
                            loopback = sc.get_microphone(id=str(default_name), include_loopback=True)
                            if not self.SILENT:
                                print(f"✓ Got loopback via speaker name: {getattr(loopback, 'name', '')} (id: {getattr(loopback, 'id', None)})")
                        except Exception:
                            pass

                # As a last effort, if pycaw is available try to query the default render device friendly name
                if loopback is None and importlib.util.find_spec('pycaw') is not None:
                    try:
                        from comtypes import CLSCTX_ALL
                        from pycaw.pycaw import AudioUtilities
                        speakers = AudioUtilities.GetSpeakers()
                        props = speakers.GetId()
                        # Try to fetch a friendly name if available
                        try:
                            friendly = speakers.FriendlyName
                        except Exception:
                            friendly = None
                        if friendly and not self.SILENT:
                            print(f"pycaw reports default render device: {friendly}")
                        # Try to match friendly name against loopback devices
                        if friendly:
                            for mic in all_mics:
                                if mic.isloopback and friendly in getattr(mic, 'name', ''):
                                    loopback = mic
                                    if not self.SILENT:
                                        print(f"✓ Matched loopback by pycaw friendly name: {mic.name}")
                                    break
                    except OSError:
                        # COM init failed; skip pycaw device matching
                        if self.DEBUG:
                            print("pycaw/comtypes unavailable for device matching due to COM init")
                    except Exception:
                        pass
                
            except Exception as e:
                if self.DEBUG:
                    print(f"Could not get default speaker: {e}")
            
            # Fallback: find any loopback device
            if loopback is None:
                if self.DEBUG:
                    print("Searching for any loopback device...")
                
                all_mics = sc.all_microphones(include_loopback=True)
                
                if self.DEBUG or not self.SILENT:
                    print("Available loopback devices:")
                    for mic in all_mics:
                        if mic.isloopback:
                            print(f"  - {mic.name} (channels: {mic.channels})")
                
                # Pick the first loopback device (usually the default speaker's loopback)
                for mic in all_mics:
                    if mic.isloopback:
                        loopback = mic
                        if not self.SILENT:
                            print(f"✓ Using loopback device: {mic.name}")
                        break
            
            if loopback is None:
                if not self.SILENT:
                    print("✗ ERROR: No loopback device found!")
                    print("This is unusual - WASAPI loopback should always be available on Windows.")
                    print("Try restarting your audio service or computer.")
                return
            else:
                # Clear, unique selection log to make debugging easier
                if not self.SILENT:
                    print("SELECTED_LOOPBACK:", getattr(loopback, 'name', None), "id=", getattr(loopback, 'id', None), "channels=", getattr(loopback, 'channels', None))
            
            # Determine channels - use all available channels from the device
            device_channels = loopback.channels
            if device_channels >= 2:
                channels = min(device_channels, 2)  # Use stereo (2 channels max for our processing)
            elif device_channels == 1:
                channels = 1
            else:
                channels = 2
            
            if channels != self.CHANNELS:
                if not self.SILENT:
                    channel_type = "mono" if channels == 1 else "stereo"
                    print(f"Note: Using {channels} channel{'s' if channels > 1 else ''} ({channel_type})")
                self.CHANNELS = channels
            
            # Use our configured sample rate (soundcard will handle resampling if needed)
            sample_rate = self.RATE
            
            if not self.SILENT:
                print(f"  Sample rate: {sample_rate} Hz")
                print(f"  Capturing ALL audio from: {loopback.name}")
                print(f"  Channels: {channels}")
                print("✓ Audio capture started (soundcard WASAPI loopback)")
                print(f"  Thread starting capture loop...")
            
            read_count = 0
            
            # Record in a loop with automatic stream recreation
            last_err_msg = None
            last_err_time = 0.0
            suppressed = 0
            consecutive_errors = 0
            chunk_size_idx = 0
            # Try different chunk sizes (10ms, 20ms, 5ms) to find one that works
            chunk_sizes_ms = [10, 20, 5]
            
            while self.running:
                # Align chunk to WASAPI period
                chunk_ms = chunk_sizes_ms[chunk_size_idx % len(chunk_sizes_ms)]
                chunk_frames = max(1, int(sample_rate * chunk_ms / 1000))
                
                if self.DEBUG and chunk_size_idx > 0:
                    print(f"Trying chunk size: {chunk_ms}ms ({chunk_frames} frames)")
                
                try:
                    with loopback.recorder(samplerate=sample_rate, channels=channels) as rec:
                        if self.DEBUG:
                            print(f"[SOUNDCARD] Recorder opened, entering read loop...")
                        
                        while self.running:
                            try:
                                # Read audio chunk
                                if self.DEBUG and read_count == 0:
                                    print(f"[SOUNDCARD] About to read {chunk_frames} frames...")
                                
                                data = rec.record(numframes=chunk_frames)
                                
                                if self.DEBUG and read_count == 0:
                                    print(f"[SOUNDCARD] ✓ Read succeeded, got {len(data)} samples")
                                
                                # Convert float32 data to int16 for compatibility
                                data_int16 = (data * 32767).astype(np.int16)
                                data_bytes = data_int16.tobytes()
                                
                                read_count += 1
                                consecutive_errors = 0
                                
                                # Put data in queue (drop old data if queue is full)
                                if self.audio_queue.full():
                                    try:
                                        self.audio_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                self.audio_queue.put(data_bytes)
                                
                                if self.DEBUG and read_count % 30 == 1:  # Every ~0.5s
                                    data_int = np.frombuffer(data_bytes, dtype=np.int16)
                                    rms = np.sqrt(np.mean(data_int.astype(np.float32)**2))
                                    print(f"[SOUNDCARD] chunk {read_count}: queue={self.audio_queue.qsize()}, RMS={rms:.1f}")
                                    print(f"📊 soundcard chunk {read_count}: queue={self.audio_queue.qsize()}, size={len(data_bytes)}B, RMS={rms:.1f}")
                                elif self.DEBUG and read_count == 1:
                                    print(f"First chunk captured! Size: {len(data_bytes)} bytes, channels: {channels}")
                                # Reset error throttling on successful read
                                suppressed = 0
                                last_err_msg = None
                                last_err_time = 0.0
                            
                            except Exception as e:
                                consecutive_errors += 1
                                # Throttle repeated identical errors to avoid CLI flood
                                msg = f"Read error: {e}"
                                now = time.perf_counter()
                                if last_err_msg == msg and (now - last_err_time) < 5.0:
                                    suppressed += 1
                                else:
                                    if self.DEBUG and suppressed > 0 and last_err_msg:
                                        print(f"{last_err_msg} (suppressed {suppressed} repeats)")
                                    if self.DEBUG:
                                        print(msg)
                                    last_err_msg = msg
                                    last_err_time = now
                                    suppressed = 0
                                # Back off briefly to let WASAPI recover
                                time.sleep(0.02)
                                
                                # If errors persist, break to recreate recorder
                                if consecutive_errors >= 30:
                                    if self.DEBUG:
                                        print(f"Recreating recorder after {consecutive_errors} errors...")
                                    break
                
                except Exception as open_err:
                    if self.DEBUG:
                        print(f"Recorder creation error: {open_err}")
                    time.sleep(0.2)
                
                # Try next chunk size if we had persistent errors
                if consecutive_errors >= 30:
                    chunk_size_idx += 1
                    consecutive_errors = 0
                    time.sleep(0.1)
                else:
                    # Normal exit, don't cycle chunk sizes
                    break
        
        except Exception as e:
            if not self.SILENT:
                print(f"Error in soundcard audio capture: {e}")
            if self.DEBUG:
                import traceback
                traceback.print_exc()
        finally:
            if self.DEBUG:
                print("soundcard audio capture thread exiting")
    
    def _capture_audio_windows_sounddevice(self):
        """Capture audio using sounddevice (preferred for Windows)"""
        try:
            import sounddevice as sd
            
            if not self.SILENT:
                print("Using sounddevice for audio capture...")
            
            # List all devices to find loopback
            devices = sd.query_devices()
            if not self.SILENT or self.DEBUG:
                print("\nAvailable audio devices:")
                for i, dev in enumerate(devices):
                    hostapi_info = sd.query_hostapis(dev['hostapi'])
                    print(f"  {i}: {dev['name']}")
                    print(f"      API: {hostapi_info['name']}, In:{dev['max_input_channels']}, Out:{dev['max_output_channels']}, SR:{dev['default_samplerate']}Hz")
            
            # Find loopback device - prioritize WASAPI devices
            loopback_device = None
            loopback_index = None
            
            # Strategy 1: Look for explicit loopback/stereo mix devices
            for i, dev in enumerate(devices):
                name_lower = dev['name'].lower()
                if dev['max_input_channels'] > 0 and (
                    'stereo mix' in name_lower or
                    'wave out' in name_lower or
                    'loopback' in name_lower or
                    'what u hear' in name_lower or
                    'what you hear' in name_lower):
                    loopback_device = dev
                    loopback_index = i
                    if not self.SILENT:
                        print(f"✓ Found loopback device: {dev['name']}")
                    break
            
            # Strategy 2: If no explicit loopback, suggest better options
            # Note: sounddevice doesn't support true WASAPI loopback
            if loopback_device is None:
                if not self.SILENT:
                    print("\n✗ No loopback device found with sounddevice!")
                    print("\nTo capture system audio on Windows, you have 3 options:")
                    print("\nOption 1 - Use soundcard (BEST - no Stereo Mix needed!):")
                    print("  1. Uninstall sounddevice: pip uninstall sounddevice")
                    print("  2. Install soundcard: pip install soundcard")
                    print("  3. Run the program again")
                    print("  → This uses native WASAPI loopback, captures directly from your speakers")
                    print("\nOption 2 - Enable Stereo Mix (works with sounddevice):")
                    print("  1. Right-click speaker icon in taskbar → Sounds")
                    print("  2. Go to 'Recording' tab")
                    print("  3. Right-click in empty area → 'Show Disabled Devices'")
                    print("  4. Right-click 'Stereo Mix' → Enable")
                    print("  5. Set it as Default Device (optional)")
                    print("\nOption 3 - Use PyAudio (also requires Stereo Mix):")
                    print("  1. Uninstall sounddevice: pip uninstall sounddevice")
                    print("  2. Install PyAudio: pip install pyaudio")
                    print("  3. Enable Stereo Mix (see Option 2)")
                    print("  4. Run the program again")
                    print("\n→ RECOMMENDED: Use Option 1 (soundcard) for hassle-free audio capture!")
                return
            
            # Use device's native sample rate
            sample_rate = int(loopback_device['default_samplerate'])
            
            # Use device's available channels (prefer stereo but accept mono)
            max_input_channels = loopback_device['max_input_channels']
            if max_input_channels >= 2:
                channels = 2  # Use stereo if available
            elif max_input_channels == 1:
                channels = 1  # Use mono if that's all we have
            else:
                channels = 2  # Fallback to stereo
            
            if not self.SILENT:
                print(f"Opening stream: {sample_rate}Hz, {channels} channel{'s' if channels > 1 else ''}, chunk={self.CHUNK}")
            
            # Update settings if different
            if sample_rate != self.RATE:
                if not self.SILENT:
                    print(f"Note: Using {sample_rate}Hz (device default)")
                self.RATE = sample_rate
                self.freqs = np.fft.rfftfreq(self.FFT_SIZE, 1.0 / self.RATE)
            
            if channels != self.CHANNELS:
                if not self.SILENT:
                    channel_type = "mono" if channels == 1 else "stereo"
                    print(f"Note: Using {channels} channel{'s' if channels > 1 else ''} ({channel_type})")
                self.CHANNELS = channels
            
            if not self.SILENT:
                print("✓ Audio capture started (sounddevice)")
            
            read_count = 0
            
            # Define callback function for non-blocking audio capture
            def audio_callback(indata, frames, time_info, status):
                nonlocal read_count
                if status and self.DEBUG:
                    print(f"Audio callback status: {status}")
                
                # Convert to bytes for compatibility with existing code
                data_bytes = indata.tobytes()
                read_count += 1
                
                # Put data in queue (drop old data if queue is full)
                if self.audio_queue.full():
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.audio_queue.put(data_bytes)
                
                if self.DEBUG and read_count % 100 == 0:
                    print(f"Audio thread: {read_count} chunks, queue: {self.audio_queue.qsize()}")
            
            # Open input stream with callback (non-blocking mode)
            with sd.InputStream(
                device=loopback_index,
                channels=channels,
                samplerate=sample_rate,
                blocksize=self.CHUNK,
                dtype='int16',
                callback=audio_callback
            ) as stream:
                # Keep thread alive while running
                while self.running:
                    time.sleep(0.1)
            
        except Exception as e:
            if not self.SILENT:
                print(f"Error in sounddevice audio capture: {e}")
            if self.DEBUG:
                import traceback
                traceback.print_exc()
        finally:
            if self.DEBUG:
                print("sounddevice audio capture thread exiting")
    
    def _capture_audio_windows_pyaudio(self):
        """Capture audio on Windows using PyAudio WASAPI loopback (fallback)"""
        if not HAS_PYAUDIO:
            if not self.SILENT:
                print("✗ ERROR: PyAudio not installed. Install with: pip install pyaudio")
            return
        
        try:
            # Prefer pyaudiowpatch if available (WASAPI loopback support)
            using_pyaudiowpatch = False
            try:
                import pyaudiowpatch as pyaudio
                using_pyaudiowpatch = True
                if self.DEBUG:
                    print("Using pyaudiowpatch for WASAPI loopback")
            except ImportError:
                import pyaudio
                if self.DEBUG:
                    print("Using standard PyAudio")
            p = pyaudio.PyAudio()
            
            # Find WASAPI host API
            wasapi_host_api = None
            for i in range(p.get_host_api_count()):
                api_info = p.get_host_api_info_by_index(i)
                if self.DEBUG:
                    print(f"Host API {i}: {api_info['name']}")
                if 'WASAPI' in api_info['name']:
                    wasapi_host_api = i
                    if self.DEBUG:
                        print(f"✓ Found WASAPI host API: index {i}")
                    break
            
            # Find WASAPI loopback device
            # Strategy: Use the default OUTPUT device as INPUT for WASAPI loopback
            wasapi_info = None
            loopback_device_index = None
            default_output_index = None
            
            if not self.SILENT:
                print("Searching for WASAPI loopback device...")
            
            # First, find the default output device using WASAPI
            if wasapi_host_api is not None:
                wasapi_api_info = p.get_host_api_info_by_index(wasapi_host_api)
                default_output_index = wasapi_api_info.get('defaultOutputDevice')
                
                if default_output_index is not None and default_output_index >= 0:
                    output_dev = p.get_device_info_by_index(default_output_index)
                    if not self.SILENT:
                        print(f"Default output device: {output_dev['name']}")
            
            # Look for devices - prioritize in this order:
            # 1. Explicit loopback/stereo mix devices
            # 2. Default output device (for WASAPI loopback)
            # 3. Any output device as fallback
            
            candidates = []
            for i in range(p.get_device_count()):
                dev_info = p.get_device_info_by_index(i)
                if self.DEBUG:
                    print(f"Device {i}: {dev_info['name']} (In:{dev_info['maxInputChannels']}, Out:{dev_info['maxOutputChannels']}, API:{dev_info['hostApi']})")
                
                name_lower = dev_info['name'].lower()
                is_wasapi = wasapi_host_api is not None and dev_info['hostApi'] == wasapi_host_api
                
                # Priority 0: WASAPI [Loopback] matching default speaker (HIGHEST)
                # Prefer the default output device's loopback
                if is_wasapi and '[loopback]' in name_lower and dev_info['maxInputChannels'] > 0:
                    # Check if this is the default speaker's loopback (prioritize even if multi-channel)
                    is_default_loopback = False
                    if default_output_index is not None:
                        default_name = p.get_device_info_by_index(default_output_index)['name']
                        loopback_base = dev_info['name'].replace(' [Loopback]', '')
                        if default_name == loopback_base:
                            is_default_loopback = True
                    
                    # Default speaker loopback always gets priority 0
                    if is_default_loopback:
                        candidates.append((0, i, dev_info, f"WASAPI [Loopback] (default speaker, {dev_info['maxInputChannels']}ch)"))
                    # Other stereo loopbacks get priority 1
                    elif dev_info['maxInputChannels'] == 2:
                        candidates.append((1, i, dev_info, "WASAPI [Loopback] (stereo)"))
                    else:
                        # Skip non-default multi-channel devices
                        if self.DEBUG:
                            print(f"Skipping {dev_info['name']}: {dev_info['maxInputChannels']} channels (not stereo, not default)")
                
                # Priority 2: Explicit loopback devices like Stereo Mix
                elif dev_info['maxInputChannels'] > 0 and ('stereo mix' in name_lower or 
                    'wave out' in name_lower or 
                    'what u hear' in name_lower or 'what you hear' in name_lower):
                    # Exclude [Loopback] devices from this category (already handled above)
                    if '[loopback]' not in name_lower:
                        candidates.append((2, i, dev_info, "Stereo Mix"))
                
                # Priority 2: Default WASAPI output device (for loopback capture with as_loopback=True)
                elif is_wasapi and i == default_output_index and dev_info['maxOutputChannels'] > 0:
                    candidates.append((3, i, dev_info, "default WASAPI output"))
                
                # Priority 3: Any WASAPI output device (can be used for loopback)
                elif is_wasapi and dev_info['maxOutputChannels'] > 0:
                    candidates.append((4, i, dev_info, "WASAPI output"))
            
            # Sort by priority and select best candidate
            if candidates:
                candidates.sort(key=lambda x: x[0])
                priority, loopback_device_index, wasapi_info, device_type = candidates[0]
                
                if not self.SILENT:
                    print(f"✓ Found {device_type} device: {wasapi_info['name']}")
                    if priority > 1:
                        print(f"  (Using WASAPI loopback mode - will capture audio playing on this device)")
            else:
                if not self.SILENT:
                    print("✗ ERROR: No suitable audio device found!")
                    print("\nTroubleshooting:")
                    print("  1. Make sure your audio device is working")
                    print("  2. Try enabling 'Stereo Mix' in Windows Sound settings:")
                    print("     - Right-click speaker icon → Sounds")
                    print("     - Recording tab → Right-click → Show Disabled Devices")
                    print("     - Enable 'Stereo Mix' or 'Wave Out Mix'")
                return
            
            # Get device capabilities
            device_info = p.get_device_info_by_index(loopback_device_index)
            device_name = device_info.get('name', '')
            
            # Use device's default sample rate
            default_sample_rate = int(device_info.get('defaultSampleRate', 44100))
            
            # Use the device's native sample rate for best compatibility
            # Don't force resampling - let Windows handle it at the native rate
            sample_rate = default_sample_rate
            
            # Check if this is a WASAPI [Loopback] device (already configured for loopback)
            is_wasapi_loopback = '[Loopback]' in device_name and device_info.get('maxInputChannels', 0) > 0
            
            # For WASAPI [Loopback] devices, they're already configured as input devices
            # For other output devices, we need to use loopback mode
            if is_wasapi_loopback:
                max_channels = int(device_info.get('maxInputChannels', 2))
                use_loopback = False  # Already in loopback mode
            else:
                # Determine if we're using loopback mode (output device as input)
                is_output_device = device_info.get('maxOutputChannels', 0) > 0 and device_info.get('maxInputChannels', 0) == 0
                use_loopback = is_output_device and wasapi_host_api is not None
                
                # For WASAPI loopback, use output channel count; otherwise use input channels
                if use_loopback:
                    max_channels = int(device_info.get('maxOutputChannels', 2))
                else:
                    max_channels = int(device_info.get('maxInputChannels', 2))
            
            # Use device's available channels - capture ALL channels and downmix ourselves
            # This preserves all audio data instead of discarding channels
            if max_channels >= 1:
                channels = max_channels  # Use ALL available channels
            else:
                channels = 2  # Fallback to stereo
            
            if not self.SILENT:
                if max_channels > 2:
                    print(f"Device info: {default_sample_rate}Hz, {max_channels} channels (will downmix to stereo)")
                elif max_channels == 2:
                    print(f"Device info: {default_sample_rate}Hz, 2 channels (stereo)")
                else:
                    print(f"Device info: {default_sample_rate}Hz, {channels} channel (mono)")
                if is_wasapi_loopback:
                    print("Using WASAPI [Loopback] device (pre-configured for system audio)")
                elif use_loopback:
                    print("Using WASAPI loopback mode (capturing system audio output)")
                if self.DEBUG:
                    print(f"Device host API: {device_info['hostApi']}")
                    print(f"Device name: {device_name}")
            
            # Build stream parameters
            # Align buffer to ~10ms frames to match common WASAPI periods
            chunk_frames = max(1, int(sample_rate / 100))
            stream_params = {
                'format': pyaudio.paInt16,
                'channels': channels,
                'rate': sample_rate,
                'input': True,
                'input_device_index': loopback_device_index,
                'frames_per_buffer': chunk_frames,
            }
            # For pyaudiowpatch + WASAPI output devices, enable loopback capture
            if using_pyaudiowpatch and use_loopback:
                stream_params['as_loopback'] = True
            
            if not self.SILENT:
                print(f"Opening audio stream: {sample_rate}Hz, {channels} channels, chunk={chunk_frames}")
                if use_loopback:
                    print("Note: Attempting to use output device for system audio capture")
                    print("This requires either Stereo Mix or a PyAudio build with WASAPI loopback support")
            
            # Try to open stream (validate format if possible)
            stream = None
            last_error = None
            try:
                # Check format support to catch invalid configurations early
                if self.DEBUG:
                    supported = p.is_format_supported(sample_rate,
                        input_device=loopback_device_index,
                        input_channels=channels,
                        input_format=pyaudio.paInt16)
                    print(f"Format supported: {supported}")
            except Exception:
                # Some builds don't implement is_format_supported for WASAPI; continue
                pass
            
            # Try with current settings
            try:
                stream = p.open(**stream_params)
                if not self.SILENT:
                    print(f"✓ Opened with {sample_rate}Hz")
            except (OSError, ValueError) as e:
                last_error = e
                if self.DEBUG:
                    print(f"Failed to open stream: {e}")
            
            # If failed, try common sample rates
            if stream is None:
                if self.DEBUG:
                    print("Trying common sample rates...")
                
                for try_rate in [44100, 48000, 96000, 192000, 22050, 16000]:
                    if try_rate == sample_rate:
                        continue  # Already tried
                    
                    stream_params['rate'] = try_rate
                    try:
                        if self.DEBUG:
                            print(f"Trying {try_rate}Hz...")
                        stream = p.open(**stream_params)
                        sample_rate = try_rate
                        if not self.SILENT:
                            print(f"✓ Success with {try_rate}Hz")
                        break
                    except (OSError, ValueError) as e:
                        last_error = e
                        if self.DEBUG:
                            print(f"Failed: {e}")
                        continue
            
            if stream is None:
                # Still failed - provide detailed error
                if not self.SILENT:
                    print(f"\n✗ ERROR: Could not open audio device")
                    print(f"Last error: {last_error}\n")
                    print("=" * 70)
                    print("TO CAPTURE SYSTEM AUDIO ON WINDOWS, YOU MUST ENABLE STEREO MIX:")
                    print("=" * 70)
                    print("\n1. Right-click the speaker icon in your taskbar")
                    print("2. Select 'Sounds' (or 'Sound settings', then click 'Sound Control Panel')")
                    print("3. Go to the 'Recording' tab")
                    print("4. Right-click in an empty area and select 'Show Disabled Devices'")
                    print("5. You should see 'Stereo Mix' appear")
                    print("6. Right-click 'Stereo Mix' and select 'Enable'")
                    print("7. Right-click 'Stereo Mix' again and select 'Set as Default Device'")
                    print("8. Click 'OK' and restart this program\n")
                    print("NOTE: If Stereo Mix doesn't appear:")
                    print("  - Your audio driver may not support it")
                    print("  - Try updating your audio drivers")
                    print("  - Some Realtek drivers have it disabled by default")
                    print("  - Virtual audio cables like VB-Cable can work as an alternative")
                    print("=" * 70)
                return
            
            # Update RATE if we had to use a different one
            if sample_rate != self.RATE:
                if not self.SILENT:
                    print(f"Note: Using {sample_rate}Hz instead of requested {self.RATE}Hz")
                self.RATE = sample_rate
                # Recalculate frequency bins
                self.freqs = np.fft.rfftfreq(self.FFT_SIZE, 1.0 / self.RATE)
            
            # Update channels - if we're capturing multi-channel, we'll downmix to stereo
            actual_channels = 2 if channels > 2 else channels
            if actual_channels != self.CHANNELS:
                if not self.SILENT:
                    if channels > 2:
                        print(f"Note: Capturing {channels} channels, downmixing to stereo")
                    else:
                        channel_type = "mono" if actual_channels == 1 else "stereo"
                        print(f"Note: Using {actual_channels} channel{'s' if actual_channels > 1 else ''} ({channel_type}) instead of requested {self.CHANNELS}")
                self.CHANNELS = actual_channels
            
            if not self.SILENT:
                print("✓ Audio capture started (Windows WASAPI)")
            
            if self.DEBUG:
                print(f"Starting read loop: chunk_frames={chunk_frames}, running={self.running}")
            
            read_count = 0
            consecutive_errors = 0
            last_err_msg = None
            last_err_time = 0.0
            suppressed = 0
            
            while self.running:
                try:
                    # Read audio chunk with timeout handling
                    # stream.read() can hang forever on WASAPI if device isn't producing audio
                    data = stream.read(chunk_frames, exception_on_overflow=False)
                    
                    # If we get here, read succeeded
                    if read_count == 0 and self.DEBUG:
                        print("✓ First audio chunk read successfully!")
                    
                    read_count += 1
                    consecutive_errors = 0
                    
                    # Downmix multi-channel audio to stereo if needed
                    if channels > 2:
                        # Parse multi-channel data
                        data_int = np.frombuffer(data, dtype=np.int16)
                        multi_channel = data_int.reshape(-1, channels)
                        
                        # Standard 5.1/7.1 downmix to stereo:
                        # Front L/R get full weight, center gets 0.707, surround gets 0.707
                        # This preserves all audio information
                        if channels == 6:  # 5.1: FL, FR, C, LFE, SL, SR
                            stereo = np.zeros((len(multi_channel), 2), dtype=np.float32)
                            stereo[:, 0] = multi_channel[:, 0] + 0.707*multi_channel[:, 2] + 0.707*multi_channel[:, 4]  # Left
                            stereo[:, 1] = multi_channel[:, 1] + 0.707*multi_channel[:, 2] + 0.707*multi_channel[:, 5]  # Right
                            # Add LFE to both channels
                            stereo[:, 0] += 0.5 * multi_channel[:, 3]
                            stereo[:, 1] += 0.5 * multi_channel[:, 3]
                        elif channels == 8:  # 7.1: FL, FR, C, LFE, SL, SR, BL, BR
                            stereo = np.zeros((len(multi_channel), 2), dtype=np.float32)
                            stereo[:, 0] = multi_channel[:, 0] + 0.707*multi_channel[:, 2] + 0.5*multi_channel[:, 4] + 0.5*multi_channel[:, 6]
                            stereo[:, 1] = multi_channel[:, 1] + 0.707*multi_channel[:, 2] + 0.5*multi_channel[:, 5] + 0.5*multi_channel[:, 7]
                            stereo[:, 0] += 0.5 * multi_channel[:, 3]
                            stereo[:, 1] += 0.5 * multi_channel[:, 3]
                        else:  # Unknown layout - average all channels
                            stereo = np.mean(multi_channel.astype(np.float32), axis=1, keepdims=True)
                            stereo = np.tile(stereo, (1, 2))
                        
                        # Clip and convert back to int16
                        stereo = np.clip(stereo, -32768, 32767).astype(np.int16)
                        data = stereo.tobytes()
                        
                        if read_count == 1 and self.DEBUG:
                            print(f"Downmixed {channels} channels to stereo using proper surround mixing")
                    
                    # Put data in queue (drop old data if queue is full)
                    if self.audio_queue.full():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.audio_queue.put(data)
                    
                    if self.DEBUG and read_count % 30 == 1:  # Print every 30 chunks (~0.5s)
                        data_int = np.frombuffer(data, dtype=np.int16)
                        rms = np.sqrt(np.mean(data_int.astype(np.float32)**2))
                        print(f"📊 Audio thread: chunk {read_count}, queue: {self.audio_queue.qsize()}, RMS: {rms:.1f}")
                        
                except Exception as e:
                    consecutive_errors += 1
                    # Throttle repeated identical errors to avoid CLI flood
                    msg = f"Read error: {e}"
                    now = time.perf_counter()
                    # Only print once every 5s for identical errors; summarize repeats
                    if last_err_msg == msg and (now - last_err_time) < 5.0:
                        suppressed += 1
                    else:
                        if self.DEBUG and suppressed > 0 and last_err_msg:
                            print(f"{last_err_msg} (suppressed {suppressed} repeats)")
                        if self.DEBUG:
                            print(msg)
                        last_err_msg = msg
                        last_err_time = now
                        suppressed = 0
                    
                    # If this is the common WASAPI error (e.g., 0x88890007), back off a bit
                    err_text = str(e)
                    if '0x88890007' in err_text or 'AUDCLNT_E' in err_text:
                        time.sleep(0.02)
                    else:
                        time.sleep(0.005)
                    
                    # If errors keep happening, try restarting the stream
                    if consecutive_errors >= 50:
                        if self.DEBUG:
                            print("Restarting WASAPI stream due to repeated read errors...")
                        try:
                            stream.stop_stream()
                            stream.close()
                        except Exception:
                            pass
                        try:
                            stream = p.open(**stream_params)
                            consecutive_errors = 0
                            last_err_msg = None
                            suppressed = 0
                            last_err_time = 0.0
                            if self.DEBUG:
                                print("WASAPI stream restarted successfully")
                            continue
                        except Exception as e2:
                            if self.DEBUG:
                                print(f"Stream restart failed: {e2}")
                            time.sleep(0.5)
                            continue
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            if not self.SILENT:
                print(f"Error in Windows audio capture: {e}")
            if self.DEBUG:
                import traceback
                traceback.print_exc()
        finally:
            if self.DEBUG:
                print("Windows audio capture thread exiting")
    
    def _capture_audio_linux(self):
        """Capture audio on Linux using PulseAudio/PipeWire parec
        
        This captures ALL audio from the default audio output sink's monitor.
        The monitor source captures everything going to that output device,
        including audio from all applications mixed together.
        """
        try:
            # First, get the default sink (output device)
            default_sink = None
            try:
                result = subprocess.run(
                    ['pactl', 'get-default-sink'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                default_sink = result.stdout.strip()
                if not self.SILENT:
                    print(f"Default audio sink: {default_sink}")
            except subprocess.CalledProcessError:
                # Older pactl versions may not support get-default-sink
                if self.DEBUG:
                    print("pactl get-default-sink not available, will search for monitor")
            
            # Get all available sources
            result = subprocess.run(
                ['pactl', 'list', 'sources', 'short'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse available monitor sources
            monitor_sources = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    source_name = parts[1]
                    if 'monitor' in source_name.lower():
                        monitor_sources.append(source_name)
                        if self.DEBUG:
                            print(f"Found monitor source: {source_name}")
            
            if not monitor_sources:
                if not self.SILENT:
                    print("✗ ERROR: No monitor source found!")
                    print("Make sure PulseAudio or PipeWire is running.")
                return
            
            # Select the best monitor source:
            # 1. If we know the default sink, use its monitor
            # 2. Otherwise use the first available monitor (typically the default)
            monitor_source = None
            
            if default_sink:
                # Look for the monitor of the default sink
                # Monitor sources are typically named: <sink_name>.monitor
                expected_monitor = f"{default_sink}.monitor"
                for source in monitor_sources:
                    if source == expected_monitor or default_sink in source:
                        monitor_source = source
                        if not self.SILENT:
                            print(f"✓ Using default sink monitor: {monitor_source}")
                        break
            
            # Fallback to first monitor source
            if not monitor_source and monitor_sources:
                monitor_source = monitor_sources[0]
                if not self.SILENT:
                    print(f"✓ Using monitor source: {monitor_source}")
            
            if not monitor_source:
                if not self.SILENT:
                    print("✗ ERROR: No monitor source found!")
                return
            
            if not self.SILENT:
                print(f"  Capturing ALL audio from this output device")
                if len(monitor_sources) > 1:
                    print(f"  (Other available monitors: {', '.join(m for m in monitor_sources if m != monitor_source)})")
            
            # Start parec to capture audio - use raw format to avoid conversion issues
            bytes_per_sample = 2  # int16
            bytes_per_frame = self.CHUNK * self.CHANNELS * bytes_per_sample
            
            cmd = [
                'parec',
                '--device=' + monitor_source,
                '--format=s16le',
                '--rate=' + str(self.RATE),
                '--channels=' + str(self.CHANNELS),
                '--latency-msec=1',
                '--raw'  # Use raw output
            ]
            print(f"[AUDIO_CAPTURE] Will run: {' '.join(cmd)}")
            print(f"[AUDIO_CAPTURE] Requested CHANNELS={self.CHANNELS}, RATE={self.RATE}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Capture errors instead of suppressing
                bufsize=0  # No buffering - immediate data
            )
            
            if not self.SILENT:
                print("✓ Audio capture started (Linux PulseAudio/PipeWire)")
            
            # Monitor stderr in background
            def read_stderr():
                for line in process.stderr:
                    if self.DEBUG:
                        print(f"parec stderr: {line.decode().strip()}")
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()
            
            read_count = 0
            buffer = b''  # Buffer for accumulating partial reads
            
            while self.running:
                # Read available data - use a smaller read size to avoid blocking
                read_size = min(4096, bytes_per_frame - len(buffer)) if len(buffer) < bytes_per_frame else 4096
                try:
                    chunk = process.stdout.read(read_size)
                except Exception as e:
                    print(f"Read error: {e}")
                    break
                    
                if len(chunk) > 0:
                    buffer += chunk
                    
                    # Do we have a complete frame?
                    if len(buffer) >= bytes_per_frame:
                        data = buffer[:bytes_per_frame]
                        buffer = buffer[bytes_per_frame:]  # Keep any extra
                        read_count += 1
                        
                        # Put data in queue (drop old data if queue is full)
                        if self.audio_queue.full():
                            try:
                                self.audio_queue.get_nowait()  # Remove old data
                            except queue.Empty:
                                pass
                        self.audio_queue.put(data)
                        
                        if self.DEBUG and read_count % 100 == 0:
                            print(f"Audio thread: {read_count} chunks, queue: {self.audio_queue.qsize()}")
                else:
                    # No data available, sleep briefly and continue
                    time.sleep(0.001)
                    continue
                        
        except Exception as e:
            if not self.SILENT:
                print(f"Error in Linux audio capture: {e}")
            if self.DEBUG:
                import traceback
                traceback.print_exc()
        finally:
            if self.DEBUG:
                print("Linux audio capture thread exiting")
    
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove all margins
        layout.setSpacing(0)  # Remove spacing
        central_widget.setLayout(layout)
        
        # Set black background everywhere to avoid white borders
        self.setStyleSheet("QMainWindow { background-color: rgb(20, 20, 30); }")
        central_widget.setStyleSheet("background-color: rgb(20, 20, 30);")
        
        # Set palette for Qt theme elements
        from PyQt5.QtGui import QPalette
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(20, 20, 30))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(20, 20, 30))
        self.setPalette(palette)
        central_widget.setPalette(palette)
        
        # Visualization canvas
        self.canvas = VisualizerCanvas(self)
        self.canvas.setStyleSheet("")  # Clear any inherited styles
        self.canvas.setAutoFillBackground(False)  # Disable auto-fill, we paint manually
        layout.addWidget(self.canvas)
        
    def update_visualization(self):
        """Read audio data and update FFT visualization"""
        frame_start = time.perf_counter()
        
        # AGGRESSIVE DEBUG
        if not hasattr(self, '_update_count'):
            self._update_count = 0
        self._update_count += 1
        
        if self._update_count == 1:
            print(f"[VIZ] First update called!")
        if self._update_count % 60 == 0:
            print(f"[VIZ] Update #{self._update_count}, queue size: {self.audio_queue.qsize()}")
        
        # Profiling timestamps
        prof = {} if self.DEBUG else None
        
        data_processed = False
        try:
            # Get one audio chunk from queue (most recent)
            data = None
            
            # Get latest chunk (non-blocking)
            try:
                data = self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            
            if self.DEBUG and data is not None and self._update_count % 60 == 0:
                print(f"Viz: Got chunk, queue size: {self.audio_queue.qsize()}, data size: {len(data)} bytes")
            elif self.DEBUG and self._update_count == 1:
                print(f"Viz: First update, queue size: {self.audio_queue.qsize()}")
            
            if data is not None:
                print(f"[AUDIO] Got data from queue: {len(data)} bytes, CHANNELS={self.CHANNELS}")
                # Convert byte data to numpy array
                data_int = np.frombuffer(data, dtype=np.int16)
                print(f"[AUDIO] data_int.shape={data_int.shape}, dtype={data_int.dtype}, min={data_int.min()}, max={data_int.max()}")
                # Debug: Check RMS amplitude
                if self.DEBUG or True:
                    rms = np.sqrt(np.mean(data_int.astype(np.float32)**2))
                    print(f"[AUDIO] RMS: {rms:.1f} (max: {np.abs(data_int).max()})")
                # Infer actual channel count from the received data (defensive)
                channels_used = self.CHANNELS
                # If the data length doesn't match expected per self.CHANNELS, try to infer
                if data_int.size == self.CHUNK * self.CHANNELS:
                    channels_used = self.CHANNELS
                elif data_int.size == self.CHUNK * 1:
                    channels_used = 1
                elif data_int.size == self.CHUNK * 2:
                    channels_used = 2
                else:
                    # Try to infer by dividing by CHUNK (if CHUNK matches frames)
                    inferred = data_int.size // self.CHUNK if self.CHUNK > 0 else 0
                    if inferred in (1, 2):
                        channels_used = inferred
                    else:
                        # Fallback: if data length is even, prefer stereo for processing convenience
                        channels_used = 2 if data_int.size % 2 == 0 else 1
                if channels_used != self.CHANNELS:
                    print(f"[AUDIO] WARNING: expected {self.CHANNELS} channels but inferred {channels_used} channels for this chunk (data_int.size={data_int.size}). Using {channels_used} for processing this frame.")

                if channels_used == 2:
                    # Defensive: only reshape if data length is even and compatible
                    if data_int.size % 2 == 0:
                        data_stereo = data_int.reshape(-1, 2)
                        print(f"[AUDIO] data_stereo.shape={data_stereo.shape}")
                        data_left = data_stereo[:, 0]
                        data_right = data_stereo[:, 1]
                        print(f"[AUDIO] data_left[0:5]={data_left[:5]}, data_right[0:5]={data_right[:5]}")
                        data_float = (data_left.astype(np.float32) + data_right.astype(np.float32)) * 0.5
                        # Shift buffers using slicing (much faster than np.roll)
                        shift = len(data_left)
                        print(f"[AUDIO] Shifting audio_buffer_left/right by {shift}")
                        self.audio_buffer_left[:-shift] = self.audio_buffer_left[shift:]
                        self.audio_buffer_left[-shift:] = data_left
                        self.audio_buffer_right[:-shift] = self.audio_buffer_right[shift:]
                        self.audio_buffer_right[-shift:] = data_right
                        print(f"[AUDIO] audio_buffer_left shape: {self.audio_buffer_left.shape}, audio_buffer_right shape: {self.audio_buffer_right.shape}")
                    else:
                        print(f"[AUDIO] Stereo expected but data_int.size % 2 != 0, treating as mono")
                        data_float = data_int.astype(np.float32)
                else:
                    print(f"[AUDIO] Mono processing (channels_used={channels_used})")
                    data_float = data_int.astype(np.float32)
                # Shift the main buffer using slicing (much faster than np.roll)
                shift = len(data_float)
                print(f"[AUDIO] Shifting audio_buffer by {shift}")
                self.audio_buffer[:-shift] = self.audio_buffer[shift:]
                self.audio_buffer[-shift:] = data_float
                # Record which channel count was used for this frame (for FFT stage)
                self._last_channels_used = channels_used
                data_processed = True
            
            if self.DEBUG and prof is not None:
                prof['audio_process'] = time.perf_counter() - frame_start
            
            if data_processed:
                # Step 2 & 3: Window and FFT per channel (use pre-computed window)
                channels_for_fft = getattr(self, '_last_channels_used', self.CHANNELS)
                print(f"[FFT] Using {channels_for_fft} channel(s) for FFT computation")
                if channels_for_fft == 2:
                    windowed_left = self.audio_buffer_left * self.hamming_window
                    windowed_right = self.audio_buffer_right * self.hamming_window
                    
                    FL = np.fft.rfft(windowed_left)
                    FR = np.fft.rfft(windowed_right)
                    
                    # Step 4 & 5: Magnitude and stereo balance (optimized: combined operations)
                    ML = np.abs(FL)
                    MR = np.abs(FR)
                    fft_magnitude = np.sqrt(ML**2 + MR**2) * self.window_correction
                    
                    # Step 8: Stereo balance per bin
                    sum_mag = MR + ML + 1e-6
                    self.stereo_balance = np.clip((MR - ML) / sum_mag, -1.0, 1.0)
                    # Smooth stereo balance over time for stable color transitions
                    alpha_balance = 0.25
                    # Ensure smoothed buffer exists and matches length
                    if not hasattr(self, 'stereo_balance_smoothed') or self.stereo_balance_smoothed.shape != self.stereo_balance.shape:
                        self.stereo_balance_smoothed = self.stereo_balance.copy()
                    else:
                        self.stereo_balance_smoothed = (alpha_balance * self.stereo_balance +
                                                        (1.0 - alpha_balance) * self.stereo_balance_smoothed)
                else:
                    # Mono: just compute FFT
                    windowed_data = self.audio_buffer * self.hamming_window
                    fft_result = np.fft.rfft(windowed_data)
                    fft_magnitude = np.abs(fft_result) * self.window_correction
                    self.stereo_balance.fill(0)
                    self.stereo_balance_smoothed.fill(0)  # No smoothing needed for mono
                
                if self.DEBUG and prof is not None:
                    prof['fft_compute'] = time.perf_counter() - frame_start - prof['audio_process']
                
                # Get current system volume for normalization (Windows only)
                if IS_WINDOWS and self.volume_interface is not None:
                    try:
                        current_volume = self.volume_interface.GetMasterVolumeLevelScalar()
                        # Smooth volume changes to avoid jarring transitions
                        self.system_volume = 0.9 * self.system_volume + 0.1 * current_volume
                    except Exception:
                        pass  # Keep using last known volume
                
                # Step 6: Convert to dB (log loudness) - optimized with pre-computed constants
                # Apply system volume compensation to normalize the display
                volume_compensation = max(0.01, self.system_volume)  # Avoid division by zero
                norm_factor = self.FFT_SIZE * 32768.0 * volume_compensation
                fft_db = 20 * np.log10(fft_magnitude / norm_factor + 1e-10)
                
                if not self.HAPPY_MODE:
                    # NORMAL MODE: Apply equal-loudness contour (ISO 226, 60 phon) BEFORE clamping
                    # This adjusts for human hearing sensitivity
                    if self.HUMAN_BIAS > 0:
                        hearing_correction = np.interp(self.freqs, self.iso226_freqs, self.iso226_boost, 
                                                      left=self.iso226_boost[0], right=self.iso226_boost[-1])
                        # Apply ISO 226 correction scaled by HUMAN_BIAS (0.0 = flat, 1.0 = full correction)
                        fft_db += hearing_correction * self.HUMAN_BIAS
                    
                    # Clamp dynamic range to [-80, 0] after all corrections
                    fft_db = np.clip(fft_db, -80, 0)
                else:
                    # HAPPY MODE: Compressed dynamic range (lighter response)
                    # Clamp to [-60, 0] for no deep blacks
                    fft_db = np.clip(fft_db, -60, 0)
                
                # Smoothing: reduce noise while maintaining responsiveness
                alpha = 0.3  # Reduced for better performance
                self.fft_data = alpha * fft_db + (1 - alpha) * self.fft_data
                self.frames_without_data = 0  # Reset counter
                
                if self.DEBUG and prof is not None:
                    prof['db_smooth'] = time.perf_counter() - frame_start - prof['fft_compute'] - prof['audio_process']
            else:
                # No audio data - wait 150ms before setting to silence to avoid flicker
                self.frames_without_data += 1
                # Calculate frames needed for 150ms timeout based on update rate
                timeout_frames = int(0.150 * self.UPDATE_RATE)
                if self.frames_without_data > timeout_frames:
                    self.fft_data = np.full(len(self.fft_data), -80.0)
            
            # Copy recent samples for oscilloscope (skip every other frame for performance)
            if data_processed and hasattr(self, '_frame_counter'):
                if self._frame_counter % 2 == 0:  # Only update every 2nd frame
                    recent_samples = self.audio_buffer[-self.oscilloscope_samples:].copy()
                    normalized = recent_samples / 32768.0
                    # Apply platform-specific oscilloscope gain and clip
                    normalized = np.clip(normalized * self.oscilloscope_gain, -1.0, 1.0)
                    self.waveform_buffer = normalized.astype(np.float32)
                self._frame_counter += 1
            elif data_processed:
                self._frame_counter = 0
                recent_samples = self.audio_buffer[-self.oscilloscope_samples:].copy()
                normalized = recent_samples / 32768.0
                normalized = np.clip(normalized * self.oscilloscope_gain, -1.0, 1.0)
                self.waveform_buffer = normalized.astype(np.float32)
            
            if self.DEBUG and prof is not None:
                prof['oscilloscope'] = time.perf_counter() - frame_start - prof.get('db_smooth', 0) - prof.get('fft_compute', 0) - prof.get('audio_process', 0)
            
            # Always update canvas (use smoothed stereo balance for color transitions)
            render_start = time.perf_counter()
            self.canvas.update_data(self.fft_data, self.freqs, self.stereo_balance_smoothed, self.waveform_buffer)
            
            if self.DEBUG and prof is not None:
                prof['render'] = time.perf_counter() - render_start
            
            # Performance monitoring
            frame_end = time.perf_counter()
            frame_time = frame_end - frame_start
            self.frame_times.append(frame_time)
            
            # Report FPS periodically
            if frame_end - self.last_fps_report >= self.fps_report_interval:
                if len(self.frame_times) > 0:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                    min_frame_time = min(self.frame_times) * 1000  # ms
                    max_frame_time = max(self.frame_times) * 1000  # ms
                    avg_frame_time_ms = avg_frame_time * 1000  # ms
                    if self.DEBUG:
                        actual_fps = len(self.frame_times) / self.fps_report_interval
                        timer_interval_ms = 1000.0 / self.UPDATE_RATE
                        
                        print(f"Performance: {fps:.1f} FPS (actual render: {actual_fps:.1f}) | Frame time: avg={avg_frame_time_ms:.2f}ms min={min_frame_time:.2f}ms max={max_frame_time:.2f}ms")
                        print(f"  Target: {self.UPDATE_RATE} FPS ({timer_interval_ms:.2f}ms interval)")
                        if prof:
                            print(f"  Breakdown: audio={prof.get('audio_process', 0)*1000:.2f}ms fft={prof.get('fft_compute', 0)*1000:.2f}ms " +
                                  f"db/smooth={prof.get('db_smooth', 0)*1000:.2f}ms scope={prof.get('oscilloscope', 0)*1000:.2f}ms render={prof.get('render', 0)*1000:.2f}ms")
                    self.frame_times.clear()
                self.last_fps_report = frame_end
                
        except Exception as e:
            if not self.SILENT:
                print(f"Error reading audio: {e}")
            if self.DEBUG:
                import traceback
                traceback.print_exc()
    
    def run_audio_diagnostics(self, duration=5, out_file='/tmp/aVis_diagnostics.log'):
        """Run a short audio capture diagnostic: read raw chunks from audio_queue and log details."""
        start = time.time()
        chunks = []
        inferred_channels = {}
        total_bytes = 0
        print(f"[DIAG] Starting audio diagnostics for {duration}s, logging to {out_file}")
        with open(out_file, 'w') as fh:
            fh.write(f"Audio diagnostics - start={time.ctime()}\n")
            fh.write(f"Requested CHANNELS={self.CHANNELS}, CHUNK={self.CHUNK}, RATE={self.RATE}\n")
            while time.time() - start < duration:
                try:
                    data = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    fh.write("[DIAG] queue empty - no data available in this interval\n")
                    continue
                if not data:
                    fh.write("[DIAG] got empty chunk\n")
                    continue
                total_bytes += len(data)
                data_int = np.frombuffer(data, dtype=np.int16)
                # Infer channels
                ch = None
                if self.CHUNK > 0 and data_int.size % self.CHUNK == 0:
                    frames = data_int.size // self.CHUNK
                    if frames in (1,2):
                        ch = frames
                if ch is None:
                    ch = 2 if data_int.size % 2 == 0 else 1
                inferred_channels[ch] = inferred_channels.get(ch, 0) + 1
                rms = float(np.sqrt(np.mean(data_int.astype(np.float32)**2))) if data_int.size>0 else 0.0
                fh.write(f"[DIAG] chunk_bytes={len(data)}, data_int.size={data_int.size}, inferred_ch={ch}, rms={rms:.2f}\n")
                if data_int.size >= 8:
                    fh.write(f"[DIAG] samples(0:8)={data_int[:8].tolist()}\n")
                chunks.append((len(data), data_int.shape))
            fh.write(f"[DIAG] finished - total_chunks={len(chunks)}, total_bytes={total_bytes}\n")
            fh.write(f"[DIAG] inferred_channels_summary={inferred_channels}\n")
            fh.write(f"Audio diagnostics - end={time.ctime()}\n")
        print(f"[DIAG] Wrote diagnostics to {out_file}")
        # After diagnostics, attempt graceful shutdown of worker threads and Qt
        print("[DIAG] Exiting after diagnostics - attempting graceful shutdown")
        # Signal threads to stop
        try:
            self.viz_running = False
            self.running = False
        except Exception:
            pass
        # Drain queue to avoid blocking
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
        except Exception:
            pass
        # Join threads briefly
        try:
            if hasattr(self, 'audio_thread') and getattr(self, 'audio_thread') is not None and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=1.0)
            if hasattr(self, 'viz_thread') and getattr(self, 'viz_thread') is not None and self.viz_thread.is_alive():
                self.viz_thread.join(timeout=1.0)
        except Exception:
            pass
        # Close the window and quit Qt event loop if running
        try:
            self.close()
        except Exception:
            pass
        try:
            from PyQt5.QtWidgets import QApplication
            QApplication.quit()
        except Exception:
            pass
        # Finally, exit
        raise SystemExit(0)
    
    def _visualization_loop(self):
        """Thread loop that calls update_visualization at precise intervals"""
        next_frame_time = time.perf_counter()
        
        while self.viz_running and self.running:
            # Wait until it's time for the next frame
            current_time = time.perf_counter()
            sleep_time = next_frame_time - current_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Emit signal to trigger update in main thread (thread-safe)
            self.update_signal.emit()
            
            # Schedule next frame
            next_frame_time += self.timer_interval
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode with no window borders"""
        if not self.is_fullscreen:
            # Save current geometry and window flags
            self.normal_geometry = self.geometry()
            # Enter fullscreen with absolutely no borders, covering everything
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.showFullScreen()
            # Ensure it covers everything including taskbars/sidebars
            from PyQt5.QtWidgets import QApplication
            screen = QApplication.primaryScreen().geometry()
            self.setGeometry(screen)
            self.is_fullscreen = True
        else:
            # Exit fullscreen and restore borders
            self.setWindowState(Qt.WindowNoState)  # Clear fullscreen state first
            self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
            self.showNormal()  # Show before setting geometry
            if self.normal_geometry:
                self.setGeometry(self.normal_geometry)
            self.is_fullscreen = False
    
    def open_settings_dialog(self):
        """Open the settings dialog (modeless - doesn't block)"""
        from PyQt5.QtCore import QTimer
        def show_dialog():
            if not hasattr(self, 'settings_dialog') or self.settings_dialog is None:
                self.settings_dialog = SettingsDialog(self, self)
                # Connect signal for settings updates
                self.settings_dialog.settings_applied.connect(self.apply_settings)
                self.settings_dialog.show()
            else:
                # Ensure existing dialog is visible and bring to front
                try:
                    self.settings_dialog.show()
                    self.settings_dialog.raise_()
                    self.settings_dialog.activateWindow()
                except Exception:
                    # If the existing instance is invalid, recreate it
                    self.settings_dialog = SettingsDialog(self, self)
                    self.settings_dialog.settings_applied.connect(self.apply_settings)
                    self.settings_dialog.show()
        QTimer.singleShot(0, show_dialog)
    
    def apply_settings(self, settings):
        """Apply settings from the dialog
        
        Args:
            settings: dict with keys like 'chunk_size', 'buffer_size', etc.
        """
        try:
            # Apply visual settings immediately (no thread safety needed)
            if 'human_bias' in settings:
                self.HUMAN_BIAS = settings['human_bias']
            
            if 'happy_mode' in settings:
                self.HAPPY_MODE = settings['happy_mode']
            
            if 'random_color' in settings:
                self.RANDOM_COLOR = settings['random_color']
                if self.RANDOM_COLOR and 'color_seed' in settings:
                    seed = settings['color_seed']
                    self.color_palette = self._generate_color_palette(seed)
            
            if 'num_bars' in settings:
                self.user_num_bars = settings['num_bars']
            
            if 'update_rate' in settings:
                self.UPDATE_RATE = settings['update_rate']
                self.timer_interval = 1.0 / self.UPDATE_RATE
            
            if 'silent' in settings:
                self.SILENT = settings['silent']
            
            if 'debug' in settings:
                self.DEBUG = settings['debug']
            
            # Audio settings that require restart of audio thread
            if 'chunk_size' in settings and settings['chunk_size'] != self.CHUNK:
                self.CHUNK = settings['chunk_size']
                if not self.SILENT:
                    print(f"Chunk size changed to {self.CHUNK} (takes effect on next audio restart)")
            
            if 'buffer_size' in settings and settings['buffer_size'] != self.FFT_SIZE:
                self.FFT_SIZE = settings['buffer_size']
                # Recalculate FFT-related buffers
                fft_size = self.FFT_SIZE // 2 + 1
                self.fft_data = np.full(fft_size, -80.0)
                self.freqs = np.fft.rfftfreq(self.FFT_SIZE, 1.0 / self.RATE)
                self.stereo_balance = np.zeros(fft_size)
                self.stereo_balance_smoothed = np.zeros(fft_size)
                self.hamming_window = np.hamming(self.FFT_SIZE).astype(np.float32)
                self.window_correction = np.sqrt(self.FFT_SIZE / np.sum(self.hamming_window**2))
                if not self.SILENT:
                    print(f"Buffer size changed to {self.FFT_SIZE} (takes effect on next audio restart)")
            
            # Device selection
            if 'device_id' in settings and settings['device_id'] != 'default':
                self.user_device_id = settings['device_id']
                if not self.SILENT:
                    print(f"Device changed to {settings['device_id']} (takes effect on next audio restart)")
            
            # Save updated settings to file
            save_settings(settings)
            
        except Exception as e:
            if not self.SILENT:
                print(f"✗ Error applying settings: {e}")
            if self.DEBUG:
                import traceback
                traceback.print_exc()
    
    def keyPressEvent(self, event):
        """Handle keyboard events"""
        # F11 or F to toggle fullscreen
        if event.key() in (Qt.Key_F11, Qt.Key_F):
            self.toggle_fullscreen()
        # Escape to exit fullscreen
        elif event.key() == Qt.Key_Escape and self.is_fullscreen:
            self.toggle_fullscreen()
        # C to regenerate color palette (only in random color mode)
        elif event.key() == Qt.Key_C and self.RANDOM_COLOR:
            self.regenerate_palette()
        # S to open settings dialog
        elif event.key() == Qt.Key_S:
            self.open_settings_dialog()
        # Q to quit
        elif event.key() == Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Clean up when closing the application"""
        self.viz_running = False
        self.running = False
        if hasattr(self, 'audio_thread') and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1)
        if hasattr(self, 'viz_thread') and self.viz_thread.is_alive():
            self.viz_thread.join(timeout=1)
        event.accept()


class VisualizerCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_visualizer = parent  # Store reference to parent
        self.fft_data = np.zeros(1024)
        self.freqs = np.zeros(1024)
        self.stereo_balance = np.zeros(1024)  # Stereo balance data
        self.waveform_data = np.zeros(1024)  # Oscilloscope waveform data (reduced for performance)
        self.setMinimumSize(200, 200)  # Reduced minimum to allow narrower windows
        
        # Cache for performance
        self.cached_layout = None  # Cache computed layout
        self.last_dimensions = (0, 0)  # Track size changes
        self.cached_bar_freqs = None  # Cache bar frequencies
        self.cached_num_bars = 0  # Cache number of bars
        self.actual_min_freq = 20.0  # Actual minimum frequency being displayed
        self.actual_max_freq = 10000.0  # Actual maximum frequency being displayed
        
        # Pre-computed interpolation cache for cubic smoothing
        self.cached_cubic_interp = None  # scipy interp1d object
        self.cached_freqs_hash = None  # Hash of freqs to detect changes
        
        # Pre-computed brightness/color arrays (avoid per-bar computation)
        self.brightness_array = np.zeros(512)
        self.stereo_balance_array = np.zeros(512)
        
        # Peak hold lines
        self.peak_hold = np.zeros(2048)  # Peak positions for each bar
        self.peak_fall_rate = 0.006  # How fast peaks fall (per frame)
        
        # Frame rate throttling (slightly higher threshold to ensure 60 FPS)
        self.last_paint_time = 0.0
        self.min_paint_interval = 0.0165  # ~60 FPS max (16.5ms)
        
        # Pre-compute color LUT for fast lookups (100 brightness x 100 balance levels)
        self._color_lut = None
        self._color_lut_mode = None  # Track which mode LUT was built for
        
        # Persistent worker thread for interpolation (no reallocation)
        self._interp_lock = threading.Lock()
        self._interp_data = None  # Double-buffered interpolation results
        self._interp_queue = queue.Queue(maxsize=1)  # Work queue (size 1 = only latest)
        self._interp_stop_event = threading.Event()
        self._interp_thread = threading.Thread(target=self._interpolation_worker, daemon=True)
        self._interp_thread.start()  # Start once, runs for lifetime
        
        # Enable double buffering to prevent tearing
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)  # Let Qt handle background
        self.setAttribute(Qt.WA_NoSystemBackground, False)  # Allow system background initially
    
    # Pre-compute color anchors (class-level optimization)
    _normal_anchors = np.array([
        [44, 123, 229],    # -1.0 (full left) - blue
        [121, 183, 234],   # -0.5 (left) - light blue
        [229, 229, 229],   # 0.0 (center) - white
        [234, 183, 155],   # +0.5 (right) - light orange
        [229, 83, 61]      # +1.0 (full right) - red
    ])
    _anchor_positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    def _get_normal_color(self, brightness, stereo_balance, cache_key):
        """Normal color mode - original pipeline"""
        # Step 9: Prepare LUT input (normalize balance to [0, 1])
        lut_x = (stereo_balance + 1.0) * 0.5
        
        # Interpolate RGB values using pre-computed anchors
        r = np.interp(lut_x, self._anchor_positions, self._normal_anchors[:, 0])
        g = np.interp(lut_x, self._anchor_positions, self._normal_anchors[:, 1])
        b = np.interp(lut_x, self._anchor_positions, self._normal_anchors[:, 2])
        
        # Step 12: Apply brightness (multiply by brightness)
        # Scale brightness to create more contrast and boost overall brightness
        brightness_scaled = brightness ** 0.4  # Gamma correction for better contrast and brightness
        r = int(r * brightness_scaled)
        g = int(g * brightness_scaled)
        b = int(b * brightness_scaled)
        
        color = QColor(r, g, b)
        self.cached_colors[cache_key] = color
        return color
    
    # Pre-compute happy mode color anchors
    _happy_anchors = np.array([
        [70, 200, 255],    # -1.0 (full left) - Sky cyan
        [120, 225, 255],   # -0.5 (left) - Light aqua
        [120, 255, 200],   # 0.0 (center) - Mint green
        [255, 160, 220],   # +0.5 (right) - Soft pink
        [255, 90, 200]     # +1.0 (full right) - Magenta
    ])
    
    def _get_happy_color(self, brightness, stereo_balance, freq_ratio, cache_key):
        """Happy mode - joyful color pipeline with vibrant colors"""
        # Joy curve for brightness - lift minimum light (no blacks ever)
        brightness_joy = brightness ** 0.55
        brightness_lifted = 0.30 + (brightness_joy * 0.70)  # Mix(0.30, 1.0, brightness_joy)
        
        # Step 5: Joyful Stereo Color LUT (sRGB)
        lut_x = (stereo_balance + 1.0) * 0.5
        
        # Interpolate base color using pre-computed anchors
        r = np.interp(lut_x, self._anchor_positions, self._happy_anchors[:, 0])
        g = np.interp(lut_x, self._anchor_positions, self._happy_anchors[:, 1])
        b = np.interp(lut_x, self._anchor_positions, self._happy_anchors[:, 2])
        
        # Apply brightness
        r = r * brightness_lifted
        g = g * brightness_lifted
        b = b * brightness_lifted
        
        # Subtle glow (not EDM cringe)
        glow = 0.10 * brightness_lifted
        r = min(255, r + glow * 255)
        g = min(255, g + glow * 255)
        b = min(255, b + glow * 255)
        
        return QColor(int(r), int(g), int(b))
    
    def _get_random_palette_color(self, brightness, stereo_balance, freq_ratio, cache_key):
        """Random palette mode - interpolate between 3 colors (L/C/R) based on stereo balance
        
        Palette structure:
        - Index 0: Left channel color
        - Index 1: Center (mono) color
        - Index 2: Right channel color
        
        Processing:
        1. Map stereo balance [-1,1] to palette interpolation
        2. Apply brightness curve
        """
        palette = self.parent_visualizer.color_palette
        
        # Stereo balance: -1.0 = full left, 0.0 = center, +1.0 = full right
        # Map to palette indices: -1 -> 0 (left), 0 -> 1 (center), +1 -> 2 (right)
        
        if stereo_balance <= 0:
            # Left side: interpolate between left (0) and center (1)
            t = (stereo_balance + 1.0)  # [-1,0] -> [0,1]
            color_left = palette[0]
            color_center = palette[1]
            
            r = color_left[0] * (1 - t) + color_center[0] * t
            g = color_left[1] * (1 - t) + color_center[1] * t
            b = color_left[2] * (1 - t) + color_center[2] * t
        else:
            # Right side: interpolate between center (1) and right (2)
            t = stereo_balance  # [0,1]
            color_center = palette[1]
            color_right = palette[2]
            
            r = color_center[0] * (1 - t) + color_right[0] * t
            g = color_center[1] * (1 - t) + color_right[1] * t
            b = color_center[2] * (1 - t) + color_right[2] * t
        
        # Apply brightness curve (gamma correction for better contrast)
        brightness_scaled = brightness ** 0.4
        r = r * brightness_scaled
        g = g * brightness_scaled
        b = b * brightness_scaled
        
        return QColor(int(r), int(g), int(b))
    
    def _build_color_lut(self, num_brightness=64, num_balance=64):
        """Pre-build color lookup table for fast color computation
        
        Creates a 2D array of QColor objects indexed by:
        - brightness (0-63 -> 0.0-1.0)
        - stereo balance (0-63 -> -1.0 to +1.0)
        """
        mode_key = (self.parent_visualizer.HAPPY_MODE, 
                    self.parent_visualizer.RANDOM_COLOR,
                    id(self.parent_visualizer.color_palette) if self.parent_visualizer.color_palette else None)
        
        # Only rebuild if mode changed
        if self._color_lut is not None and self._color_lut_mode == mode_key:
            return
        
        self._color_lut = [[None for _ in range(num_balance)] for _ in range(num_brightness)]
        self._color_lut_mode = mode_key
        
        for bi in range(num_brightness):
            brightness = bi / (num_brightness - 1)
            for si in range(num_balance):
                balance = (si / (num_balance - 1)) * 2.0 - 1.0  # Map to [-1, 1]
                
                # Compute color based on mode
                if self.parent_visualizer.RANDOM_COLOR and self.parent_visualizer.color_palette:
                    color = self._compute_random_palette_color(brightness, balance)
                elif self.parent_visualizer.HAPPY_MODE:
                    color = self._compute_happy_color(brightness, balance)
                else:
                    color = self._compute_normal_color(brightness, balance)
                
                self._color_lut[bi][si] = color
    
    def _compute_normal_color(self, brightness, stereo_balance):
        """Compute normal mode color without caching"""
        lut_x = (stereo_balance + 1.0) * 0.5
        r = np.interp(lut_x, self._anchor_positions, self._normal_anchors[:, 0])
        g = np.interp(lut_x, self._anchor_positions, self._normal_anchors[:, 1])
        b = np.interp(lut_x, self._anchor_positions, self._normal_anchors[:, 2])
        brightness_scaled = brightness ** 0.4
        return QColor(int(r * brightness_scaled), int(g * brightness_scaled), int(b * brightness_scaled))
    
    def _compute_happy_color(self, brightness, stereo_balance):
        """Compute happy mode color without caching"""
        brightness_joy = brightness ** 0.55
        brightness_lifted = 0.30 + (brightness_joy * 0.70)
        lut_x = (stereo_balance + 1.0) * 0.5
        r = np.interp(lut_x, self._anchor_positions, self._happy_anchors[:, 0])
        g = np.interp(lut_x, self._anchor_positions, self._happy_anchors[:, 1])
        b = np.interp(lut_x, self._anchor_positions, self._happy_anchors[:, 2])
        r = r * brightness_lifted
        g = g * brightness_lifted
        b = b * brightness_lifted
        glow = 0.10 * brightness_lifted
        r = min(255, r + glow * 255)
        g = min(255, g + glow * 255)
        b = min(255, b + glow * 255)
        return QColor(int(r), int(g), int(b))
    
    def _compute_random_palette_color(self, brightness, stereo_balance):
        """Compute random palette color without caching"""
        palette = self.parent_visualizer.color_palette
        if stereo_balance <= 0:
            t = (stereo_balance + 1.0)
            color_left = palette[0]
            color_center = palette[1]
            r = color_left[0] * (1 - t) + color_center[0] * t
            g = color_left[1] * (1 - t) + color_center[1] * t
            b = color_left[2] * (1 - t) + color_center[2] * t
        else:
            t = stereo_balance
            color_center = palette[1]
            color_right = palette[2]
            r = color_center[0] * (1 - t) + color_right[0] * t
            g = color_center[1] * (1 - t) + color_right[1] * t
            b = color_center[2] * (1 - t) + color_right[2] * t
        brightness_scaled = brightness ** 0.4
        return QColor(int(r * brightness_scaled), int(g * brightness_scaled), int(b * brightness_scaled))
    
    def get_color_from_lut(self, brightness, stereo_balance):
        """Fast color lookup from pre-computed LUT"""
        if self._color_lut is None:
            self._build_color_lut()
        
        # Quantize to LUT indices
        bi = min(63, max(0, int(brightness * 63)))
        si = min(63, max(0, int((stereo_balance + 1.0) * 31.5)))
        
        return self._color_lut[bi][si]
    
    def _compute_interpolation(self, freqs_filtered, data_filtered, bar_freqs):
        """Compute interpolation (can be called from worker thread)"""
        smooth_mask = bar_freqs <= 1000
        data_interpolated = np.zeros_like(bar_freqs)
        freqs_hash = self.cached_freqs_hash
        
        if HAS_SCIPY and np.any(smooth_mask):
            smooth_indices = np.where(smooth_mask)[0]
            high_indices = np.where(~smooth_mask)[0]
            
            try:
                # Rebuild cubic interpolator only when frequency bins change
                if self.cached_cubic_interp is None or self.cached_freqs_hash != freqs_hash:
                    self.cached_cubic_interp = interp1d(freqs_filtered, data_filtered, 
                                                        kind='cubic', fill_value='extrapolate',
                                                        assume_sorted=True)
                    self.cached_freqs_hash = freqs_hash
                else:
                    # Update y-values
                    self.cached_cubic_interp = interp1d(freqs_filtered, data_filtered,
                                                        kind='cubic', fill_value='extrapolate',
                                                        assume_sorted=True)
                
                # Cubic interpolation for bass/mids (20-1000Hz)
                data_interpolated[smooth_indices] = self.cached_cubic_interp(bar_freqs[smooth_indices])
                
                # Linear interpolation for high frequencies (1000Hz+)
                if len(high_indices) > 0:
                    data_interpolated[high_indices] = np.interp(bar_freqs[high_indices], 
                                                                freqs_filtered, data_filtered)
            except:
                # Fallback to linear if cubic fails
                data_interpolated = np.interp(bar_freqs, freqs_filtered, data_filtered)
        else:
            # No scipy or no smooth range - use linear
            data_interpolated = np.interp(bar_freqs, freqs_filtered, data_filtered)
        
        return data_interpolated
    
    def _interpolation_worker(self):
        """Persistent worker thread - runs for application lifetime, no reallocation"""
        while not self._interp_stop_event.is_set():
            try:
                # Wait for work with timeout to check stop event periodically
                try:
                    work = self._interp_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                freqs_filtered, data_filtered, bar_freqs = work
                
                # Compute interpolation
                result = self._compute_interpolation(freqs_filtered, data_filtered, bar_freqs)
                
                # Store result atomically
                with self._interp_lock:
                    self._interp_data = result
                    
            except Exception:
                # Silently continue on errors - will use previous frame's data
                pass
    
    def __del__(self):
        """Cleanup: Stop worker thread gracefully"""
        try:
            self._interp_stop_event.set()
            if hasattr(self, '_interp_thread') and self._interp_thread.is_alive():
                self._interp_thread.join(timeout=1.0)
        except:
            pass
        
    def update_data(self, fft_data, freqs, stereo_balance=None, waveform=None):
        """Update the FFT data and trigger repaint"""
        # Throttle updates to prevent excessive CPU usage
        current_time = time.perf_counter()
        if current_time - self.last_paint_time < self.min_paint_interval:
            return  # Skip this update, too soon
        
        self.last_paint_time = current_time
        self.fft_data = fft_data
        self.freqs = freqs
        self.stereo_balance = stereo_balance if stereo_balance is not None else np.zeros(len(fft_data))
        self.waveform_data = waveform if waveform is not None else np.zeros(len(self.waveform_data))
        self.update()  # Use update() to let Qt batch repaints (lower CPU)
        
    def paintEvent(self, event):
        """Draw the frequency spectrum"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)  # Disabled for performance
        painter.setRenderHint(QPainter.TextAntialiasing, False)  # Disable text AA too
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)  # Disable smooth transforms
        
        # Background - always paint this first
        bg_color = QColor(20, 20, 30)
        painter.fillRect(self.rect(), bg_color)
        
        width = self.width()
        height = self.height()
        # Debug: announce paint events when debug mode is enabled
        if hasattr(self, 'parent_visualizer') and getattr(self.parent_visualizer, 'DEBUG', False):
            if not hasattr(self, '_paint_count'):
                self._paint_count = 0
            self._paint_count += 1
            try:
                print(f"[PAINT] paint #{self._paint_count} size={width}x{height} fft_len={len(self.fft_data)}")
            except Exception:
                pass
        
        # Check if dimensions changed
        dimensions_changed = (width, height) != self.last_dimensions
        if dimensions_changed:
            self.last_dimensions = (width, height)
        
        # Reserve space for oscilloscope at bottom
        oscilloscope_height = 70  # Total height including padding
        
        # Determine layout
        use_vertical = width < height
        
        # Calculate space needed for spectrum indicator in vertical mode
        spectrum_indicator_height = 0
        if self.parent_visualizer.RANDOM_COLOR and self.parent_visualizer.color_palette and use_vertical:
            spectrum_indicator_height = 50  # 15 (spectrum) + 20 (gap) + 12 (seed text) + 3 (tight margin)
        
        if use_vertical:
            margin_left, margin_right, margin_top, margin_bottom = 40, 10, 3, 5  # Minimal margins for max vertical space
            margin_top += spectrum_indicator_height  # Add space for spectrum indicator
        else:
            margin_left, margin_right, margin_top, margin_bottom = 60, 10, 10, 20  # Increased bottom from 15 to 20 for dB text
        
        # Add oscilloscope space to bottom margin
        margin_bottom += oscilloscope_height
        
        graph_width = width - margin_left - margin_right
        graph_height = height - margin_top - margin_bottom
        
        if graph_width <= 0 or graph_height <= 0:
            return
        
        # Draw frequency labels (reduced count for performance)
        painter.setFont(QFont('Arial', 8))
        painter.setPen(QColor(150, 150, 170))
        freq_labels = [20, 100, 500, 1000, 5000, 15000]  # Fewer labels for less drawing
        
        if use_vertical:
            # Vertical layout - labels on Y axis (logarithmic scale, high freq at top)
            for freq in freq_labels:
                if freq >= self.actual_min_freq and freq <= self.actual_max_freq:
                    # Use logarithmic positioning matching actual bar distribution
                    freq_position = (np.log10(freq) - np.log10(self.actual_min_freq)) / (np.log10(self.actual_max_freq) - np.log10(self.actual_min_freq))
                    y_pos = margin_top + int((1.0 - freq_position) * graph_height)
                    painter.drawLine(margin_left - 5, y_pos, margin_left, y_pos)
                    if freq >= 1000:
                        label = f"{freq//1000}k"
                    else:
                        label = str(freq)
                    painter.drawText(5, y_pos + 4, label)
            
            # Draw dB scale on X-axis (reduced for performance)
            db_levels = [-80, -40, 0]
            for db in db_levels:
                x_pos = margin_left + int((db - (-80)) / 80 * graph_width)
                if x_pos >= margin_left and x_pos <= width - margin_right:
                    painter.drawLine(x_pos, height - margin_bottom, x_pos, height - margin_bottom + 5)
                    painter.drawText(x_pos - 15, height - margin_bottom + 18, f"{db}dB")
        else:
            # Horizontal layout - labels on X axis
            for freq in freq_labels:
                if freq >= self.actual_min_freq and freq <= self.actual_max_freq:
                    x_pos = margin_left + graph_width * (np.log10(freq) - np.log10(self.actual_min_freq)) / (np.log10(self.actual_max_freq) - np.log10(self.actual_min_freq))
                    painter.drawLine(int(x_pos), height - margin_bottom, int(x_pos), height - margin_bottom + 5)
                    if freq >= 1000:
                        label = f"{freq//1000}k"
                    else:
                        label = str(freq)
                    painter.drawText(int(x_pos - 15), height - margin_bottom + 18, label)
            
            # Draw dB scale on Y-axis (reduced for performance)
            db_levels = [-80, -40, 0]
            for db in db_levels:
                y_pos = margin_top + graph_height - int((db - (-80)) / 80 * graph_height)
                if y_pos >= margin_top and y_pos <= height - margin_bottom:
                    painter.drawLine(margin_left - 5, y_pos, margin_left, y_pos)
                    painter.drawText(5, y_pos + 4, f"{db}dB")        # Draw frequency bars - EVENLY SPACED in screen space
        if len(self.fft_data) > 0 and len(self.freqs) > 0:
            # Focus on audible range - cut off sub-bass noise below 10 Hz
            mask = (self.freqs >= 10.0) & (self.freqs <= 15000)  # Start at 10 Hz to eliminate low-frequency noise
            freqs_filtered = self.freqs[mask]
            data_filtered = self.fft_data[mask]
            
            if len(data_filtered) > 0:
                # Use realistic dB range: -80 to 0 dB (typical audio range)
                db_min = -80
                db_max = 0
                db_range = db_max - db_min
                
                # Number of bars based on layout (reduced for performance)
                if self.parent_visualizer.user_num_bars is not None:
                    # User specified bar count
                    num_bars = max(8, min(16384, self.parent_visualizer.user_num_bars))  # Allow 8-16384 bars
                else:
                    # Auto-calculate based on window size
                    if use_vertical:
                        num_bars = min(256, graph_height // 3)  # Cap at 256 bars
                    else:
                        num_bars = min(512, graph_width // 3)  # Cap at 512 bars
                
                # Cache bar frequencies if dimensions changed
                if dimensions_changed or self.cached_num_bars != num_bars:
                    if self.parent_visualizer.DEBUG:
                        print(f"[DEBUG] Recalculating bars: dimensions_changed={dimensions_changed}, cached={self.cached_num_bars}, current={num_bars}, user_num_bars={self.parent_visualizer.user_num_bars}")
                    # Start from first available frequency bin to avoid extrapolation issues
                    min_freq = max(freqs_filtered[0], 10.0)  # Use actual first bin or 10 Hz minimum
                    self.actual_min_freq = min_freq
                    self.actual_max_freq = 10000.0
                    self.cached_bar_freqs = np.logspace(np.log10(min_freq), np.log10(10000), num_bars)
                    self.cached_num_bars = num_bars
                    # Invalidate cubic interpolation cache when bar freqs change
                    self.cached_cubic_interp = None
                    # Resize peak_hold array when bar count changes
                    self.peak_hold = np.zeros(num_bars)
                    # Cache frequency hash for interpolation
                    self.cached_freqs_hash = (len(freqs_filtered), freqs_filtered[0], freqs_filtered[-1])
                    
                    # Debug: Show bass region detail
                    if self.parent_visualizer.DEBUG:
                        print(f"\n=== BASS DETAIL DEBUG ===")
                        print(f"Total bars: {num_bars}")
                        print(f"FFT bins available: {len(freqs_filtered)}")
                        print(f"FFT frequency resolution: {self.parent_visualizer.RATE / self.parent_visualizer.FFT_SIZE:.2f} Hz/bin")
                        print(f"\nFirst 30 FFT bins:")
                        for i in range(min(30, len(freqs_filtered))):
                            print(f"  Bin {i}: {freqs_filtered[i]:.2f} Hz = {data_filtered[i]:.1f} dB")
                        print(f"\nFirst 30 bars (target frequencies):")
                        for i in range(min(30, num_bars)):
                            print(f"  Bar {i}: {self.cached_bar_freqs[i]:.2f} Hz")
                        print(f"========================\n")
                
                bar_freqs = self.cached_bar_freqs
                
                # Multi-threaded interpolation: Submit work to persistent worker thread
                # Use cached results from previous frame for rendering (non-blocking)
                with self._interp_lock:
                    # Use cached results if available, otherwise compute synchronously
                    if self._interp_data is not None:
                        data_interpolated = self._interp_data.copy()
                    else:
                        # First frame - compute synchronously
                        data_interpolated = self._compute_interpolation(freqs_filtered, data_filtered, bar_freqs)
                
                # Submit next interpolation to worker queue (non-blocking, drops old work if queue full)
                try:
                    # Use put_nowait to avoid blocking - if queue full, skip (worker still processing previous)
                    self._interp_queue.put_nowait((freqs_filtered.copy(), data_filtered.copy(), bar_freqs.copy()))
                except queue.Full:
                    pass  # Worker busy with previous frame, will use current cache next time
                
                log_freqs = bar_freqs
                
                # Vectorized stereo balance interpolation
                stereo_balance_interpolated = np.interp(bar_freqs, freqs_filtered, self.stereo_balance[mask])
                
                # Step 7: Vectorized brightness computation
                if self.parent_visualizer.HAPPY_MODE:
                    brightness_interpolated = np.clip((data_interpolated + 60.0) * (1.0/60.0), 0.0, 1.0)
                else:
                    brightness_interpolated = np.clip((data_interpolated + 80.0) * 0.0125, 0.0, 1.0)
                
                # Ensure color LUT is built
                self._build_color_lut()
                
                # Ensure peak_hold matches the number of bars/data points before updating peaks
                if self.peak_hold.shape[0] != brightness_interpolated.shape[0]:
                    self.peak_hold = np.zeros(brightness_interpolated.shape[0])
                # Update peaks: keep max of current brightness and decayed peak
                self.peak_hold = np.maximum(brightness_interpolated, 
                                            np.maximum(0, self.peak_hold - self.peak_fall_rate))
                
                # Pre-compute all bar positions and dimensions
                num_freqs = len(log_freqs)
                is_mono = self.parent_visualizer.CHANNELS == 1
                
                # Mono brightness adjustment (vectorized)
                if is_mono:
                    freq_ratios = np.arange(num_freqs) / num_freqs
                    freq_brightness_boost = 0.4 + (freq_ratios * 0.6)
                    brightness_interpolated = np.clip(brightness_interpolated * freq_brightness_boost, 0.0, 1.0)
                
                # Get all colors at once using LUT (vectorized index computation)
                brightness_indices = ((brightness_interpolated * 63).astype(np.int32)) % 64
                balance_indices = (((stereo_balance_interpolated + 1.0) * 31.5).astype(np.int32)) % 64
                
                # Draw bars - orientation depends on layout
                painter.setPen(Qt.NoPen)
                
                if use_vertical:
                    # Vertical layout - bars linearly spaced, representing logarithmic frequencies
                    bar_height = graph_height / num_freqs
                    bar_h = max(1, int(bar_height) + 1)
                    # Pre-compute y positions for all bars
                    y_positions = margin_top + (np.arange(num_freqs) * bar_height).astype(np.int32)
                    bar_widths = (brightness_interpolated[num_freqs-1::-1] * graph_width + 0.5).astype(np.int32)
                    peak_xs = (self.peak_hold[num_freqs-1::-1] * graph_width + 0.5).astype(np.int32) + margin_left
                    # BATCHED DRAWING: Group bars by color for fewer setBrush calls
                    color_groups = {}  # color_key -> list of (y_pos, width, idx, peak_x)
                    for i in range(num_freqs):
                        idx = num_freqs - 1 - i
                        # Skip if both bar and peak are negligible
                        if idx >= brightness_interpolated.shape[0] or i >= self.peak_hold.shape[0]:
                            continue
                        if brightness_interpolated[idx] < 0.01 and self.peak_hold[i] < 0.01:
                            continue
                        y_pos = y_positions[i]
                        bar_width_val = bar_widths[i]
                        peak_x = peak_xs[i] if self.peak_hold[i] > 0.001 else None
                        if bar_width_val > 0 or peak_x is not None:
                            # Use color indices as key for grouping
                            color_key = (brightness_indices[idx], balance_indices[idx])
                            if color_key not in color_groups:
                                color_groups[color_key] = []
                            color_groups[color_key].append((y_pos, bar_width_val, peak_x))
                    # Draw all bars grouped by color (reduces setBrush calls from 500+ to ~20-30)
                    for color_key, bars in color_groups.items():
                        color = self._color_lut[color_key[0]][color_key[1]]
                        painter.setBrush(color)
                        for y_pos, bar_width_val, peak_x in bars:
                            if bar_width_val > 0:
                                painter.drawRect(margin_left, y_pos, bar_width_val, bar_h)
                            if peak_x is not None:
                                painter.setPen(QPen(color, 1))
                                painter.drawLine(peak_x, y_pos, peak_x, y_pos + bar_h)
                                painter.setPen(Qt.NoPen)
                else:
                    # Horizontal layout - bars linearly spaced, representing logarithmic frequencies
                    bar_width = graph_width / num_freqs
                    bar_w = max(1, int(bar_width) + 1)
                    # Pre-compute x positions for all bars
                    x_positions = margin_left + (np.arange(num_freqs) * bar_width).astype(np.int32)
                    bar_heights = (brightness_interpolated * graph_height + 0.5).astype(np.int32)
                    peak_heights = (self.peak_hold * graph_height + 0.5).astype(np.int32)
                    # BATCHED DRAWING: Group bars by color for fewer setBrush calls
                    color_groups = {}  # color_key -> list of (x_pos, height, peak_h)
                    for i in range(num_freqs):
                        if i >= brightness_interpolated.shape[0] or i >= self.peak_hold.shape[0]:
                            continue
                        # Skip if both bar and peak are negligible
                        if brightness_interpolated[i] < 0.01 and self.peak_hold[i] < 0.01:
                            continue
                        x_pos = x_positions[i]
                        bar_height_val = bar_heights[i]
                        peak_h = peak_heights[i] if self.peak_hold[i] > 0.001 else None
                        if bar_height_val > 0 or peak_h is not None:
                            # Use color indices as key for grouping
                            color_key = (brightness_indices[i], balance_indices[i])
                            if color_key not in color_groups:
                                color_groups[color_key] = []
                            color_groups[color_key].append((x_pos, bar_height_val, peak_h))
                    # Draw all bars grouped by color (reduces setBrush calls from 500+ to ~20-30)
                    for color_key, bars in color_groups.items():
                        color = self._color_lut[color_key[0]][color_key[1]]
                        painter.setBrush(color)
                        for x_pos, bar_height_val, peak_h in bars:
                            if bar_height_val > 0:
                                y_pos = margin_top + (graph_height - bar_height_val)
                                painter.drawRect(x_pos, y_pos, bar_w, bar_height_val)
                            if peak_h is not None:
                                peak_y = margin_top + (graph_height - peak_h)
                                painter.setPen(QPen(color, 1))
                                painter.drawLine(x_pos, peak_y, x_pos + bar_w, peak_y)
                                painter.setPen(Qt.NoPen)
        
        # Draw oscilloscope at bottom (in reserved space)
        if len(self.waveform_data) > 0:
            scope_height = 50  # Height of oscilloscope waveform area
            scope_top = height - oscilloscope_height + 15  # Start 15px from reserved bottom space (increased from 10)
            # Use full width in both layouts for consistency
            scope_left = 10
            scope_width = width - 20
            
            # Background for oscilloscope
            painter.fillRect(scope_left, scope_top, scope_width, scope_height, QColor(15, 15, 25))
            
            # Set clipping to prevent overdraw
            painter.setClipRect(scope_left, scope_top, scope_width, scope_height)
            
            # Draw center line
            center_y = scope_top + scope_height // 2
            painter.setPen(QPen(QColor(40, 40, 50), 1))
            painter.drawLine(scope_left, center_y, scope_left + scope_width, center_y)
            
            # Draw waveform using QPainterPath for batch rendering (much faster)
            if len(self.waveform_data) > 1:
                num_points = min(len(self.waveform_data), scope_width) // 2  # Half resolution for performance
                step = len(self.waveform_data) / num_points
                scale = scope_height * 0.45
                x_scale = scope_width / num_points
                
                # Pre-compute all points using vectorized operations
                indices = (np.arange(num_points) * step).astype(np.int32)
                indices = np.clip(indices, 0, len(self.waveform_data) - 1)
                y_values = center_y - (np.clip(self.waveform_data[indices], -1.0, 1.0) * scale).astype(np.int32)
                x_values = scope_left + (np.arange(num_points) * x_scale).astype(np.int32)
                
                # Build polygon from pre-computed points (faster than QPainterPath with loop)
                points = [QPointF(x_values[i], y_values[i]) for i in range(num_points)]
                polygon = QPolygonF(points)
                
                # Draw as a single polyline (no fill)
                painter.setPen(QPen(QColor(100, 200, 255), 1))
                painter.setBrush(Qt.NoBrush)  # Ensure no fill
                painter.drawPolyline(polygon)
            
            # Remove clipping
            painter.setClipping(False)
            painter.setPen(Qt.NoPen)
        
        # Draw color spectrum indicator and seed
        if self.parent_visualizer.RANDOM_COLOR and self.parent_visualizer.color_palette:
            spectrum_width = 100  # Reduced from 150
            spectrum_height = 15  # Reduced from 20
            margin = 10
            
            # Get seed text
            seed_text = f"Seed: {self.parent_visualizer._palette_seed}" if hasattr(self.parent_visualizer, '_palette_seed') else ""
            painter.setFont(QFont('Arial', 8))  # Reduced from 9
            seed_text_width = painter.fontMetrics().horizontalAdvance(seed_text) if seed_text else 0
            
            # Layout depends on use_vertical flag
            label_offset = 10  # Space for 'L' label on the left
            if use_vertical:
                # Vertical mode: stack spectrum and seed
                info_x = margin + label_offset
                info_y = margin
            else:
                # Horizontal mode: inline from right edge
                total_width = spectrum_width + label_offset + (20 + seed_text_width if seed_text else 0)
                info_x = width - margin - total_width + label_offset
                info_y = margin
            
            # Draw labels: L (left), C (center), R (right) - draw L first before spectrum
            painter.setPen(QColor(200, 200, 220))
            painter.setFont(QFont('Arial', 7, QFont.Bold))
            painter.drawText(info_x - label_offset, int(info_y + spectrum_height / 2 + 3), "L")
            
            # Draw color spectrum bar showing L/C/R colors
            painter.setPen(Qt.NoPen)
            num_segments = len(self.parent_visualizer.color_palette)
            segment_width = spectrum_width / num_segments
            
            for i, color_tuple in enumerate(self.parent_visualizer.color_palette):
                x = info_x + int(i * segment_width)
                painter.fillRect(int(x), info_y, int(segment_width) + 1, spectrum_height, 
                               QColor(color_tuple[0], color_tuple[1], color_tuple[2]))
            
            # Draw border around spectrum
            painter.setPen(QPen(QColor(150, 150, 170), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(info_x, info_y, spectrum_width, spectrum_height)
            
            # Draw remaining labels: C (center), R (right)
            # Draw C with background to ensure visibility
            painter.setPen(QColor(200, 200, 220))
            painter.setFont(QFont('Arial', 7, QFont.Bold))
            
            # Draw C with semi-transparent background for visibility
            c_x = int(info_x + spectrum_width / 2 - 3)
            c_y = int(info_y + spectrum_height / 2 + 3)
            painter.fillRect(c_x - 2, c_y - 10, 10, 12, QColor(20, 20, 30, 200))  # Dark background
            painter.drawText(c_x, c_y, "C")
            
            painter.drawText(info_x + spectrum_width + 5, int(info_y + spectrum_height / 2 + 3), "R")
            
            # Draw seed info: stacked below in vertical mode, inline in horizontal mode
            if seed_text:
                painter.setFont(QFont('Arial', 8))
                painter.setPen(QColor(180, 180, 200))
                if use_vertical:
                    # Stack below spectrum with larger gap
                    painter.drawText(info_x, info_y + spectrum_height + 20, seed_text)
                else:
                    # Inline to the right
                    painter.drawText(info_x + spectrum_width + 15, int(info_y + spectrum_height / 2 + 3), seed_text)


class SettingsDialog(QDialog):
    """Settings configuration dialog for audio and visual parameters"""
    
    # Signal to notify parent of settings changes
    settings_applied = pyqtSignal(dict)
    
    def __init__(self, parent, visualizer):
        super().__init__(parent)
        self.visualizer = visualizer
        self.setWindowTitle("Settings")
        self.setGeometry(200, 200, 500, 650)
        self.setStyleSheet("QDialog { background-color: rgb(40, 40, 50); }")
        
        # Store default values
        self.defaults = {
            'chunk_size': 512,
            'buffer_size': 2048,
            'update_rate': 144,
            'human_bias': 0.5,
            'num_bars': None,
            'happy_mode': False,
            'random_color': False,
            'color_seed': None,
            'silent': False,
            'debug': False,
        }
        
        self.setup_ui()
        self.load_current_values()
        
    def setup_ui(self):
        """Build the settings dialog UI"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # ===== Audio Settings Group =====
        audio_group = QGroupBox("Audio Settings")
        audio_layout = QVBoxLayout()
        
        # Device selection
        device_layout = QHBoxLayout()
        device_lbl = QLabel("Device:")
        device_lbl.setToolTip("Select the audio output device to capture from")
        device_layout.addWidget(device_lbl)
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(250)
        self.device_combo.setToolTip("Choose which audio device to monitor")
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Scan for available audio devices")
        refresh_btn.clicked.connect(self.refresh_devices)
        device_layout.addWidget(refresh_btn)
        audio_layout.addLayout(device_layout)
        
        # Chunk size
        chunk_layout = QHBoxLayout()
        chunk_lbl = QLabel("Chunk Size (samples):")
        chunk_lbl.setToolTip("Smaller = lower latency, higher CPU. Must be power of 2.")
        chunk_layout.addWidget(chunk_lbl)
        self.chunk_spinbox = QSpinBox()
        self.chunk_spinbox.setMinimum(256)
        self.chunk_spinbox.setMaximum(16384)
        self.chunk_spinbox.setSingleStep(256)
        self.chunk_spinbox.setToolTip("Audio buffer chunk size in samples (256-16384, power of 2)")
        chunk_layout.addWidget(self.chunk_spinbox)
        chunk_layout.addStretch()
        audio_layout.addLayout(chunk_layout)
        
        # Buffer size
        buffer_layout = QHBoxLayout()
        buffer_lbl = QLabel("Buffer Size (samples):")
        buffer_lbl.setToolTip("Larger = better bass resolution, higher latency. Must be power of 2.")
        buffer_layout.addWidget(buffer_lbl)
        self.buffer_spinbox = QSpinBox()
        self.buffer_spinbox.setMinimum(512)
        self.buffer_spinbox.setMaximum(32768)
        self.buffer_spinbox.setSingleStep(512)
        self.buffer_spinbox.setToolTip("FFT buffer size in samples (512-32768, power of 2)")
        buffer_layout.addWidget(self.buffer_spinbox)
        buffer_layout.addStretch()
        audio_layout.addLayout(buffer_layout)
        
        # Update rate
        update_layout = QHBoxLayout()
        update_lbl = QLabel("Update Rate (Hz):")
        update_lbl.setToolTip("Display refresh rate - higher = smoother but uses more CPU")
        update_layout.addWidget(update_lbl)
        self.update_spinbox = QSpinBox()
        self.update_spinbox.setMinimum(1)
        self.update_spinbox.setMaximum(240)
        self.update_spinbox.setSingleStep(1)
        self.update_spinbox.setToolTip("Target display refresh rate in Hz (1-240)")
        update_layout.addWidget(self.update_spinbox)
        update_layout.addStretch()
        audio_layout.addLayout(update_layout)
        
        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)
        
        # ===== Visual Settings Group =====
        visual_group = QGroupBox("Visual Settings")
        visual_layout = QVBoxLayout()
        
        # Human bias slider
        bias_layout = QHBoxLayout()
        bias_lbl = QLabel("Human Bias:")
        bias_lbl.setToolTip("Apply ISO 226 equal-loudness curve (matches human hearing)")
        bias_layout.addWidget(bias_lbl)
        self.bias_slider = QSlider(Qt.Horizontal)
        self.bias_slider.setMinimum(-50)
        self.bias_slider.setMaximum(100)
        self.bias_slider.setTickPosition(QSlider.TicksBelow)
        self.bias_slider.setTickInterval(10)
        self.bias_slider.setToolTip("-50% = inverse curve, 0% = linear, 100% = full ISO 226 curve")
        self.bias_label = QLabel("50%")
        self.bias_slider.valueChanged.connect(self.update_bias_label)
        bias_layout.addWidget(self.bias_slider)
        bias_layout.addWidget(self.bias_label)
        visual_layout.addLayout(bias_layout)
        
        # Number of bars
        bars_layout = QHBoxLayout()
        bars_lbl = QLabel("Number of Bars:")
        bars_lbl.setToolTip("More bars = more detail, lower performance")
        bars_layout.addWidget(bars_lbl)
        self.bars_spinbox = QSpinBox()
        self.bars_spinbox.setMinimum(8)
        self.bars_spinbox.setMaximum(16384)
        self.bars_spinbox.setSingleStep(1)
        self.bars_spinbox.setToolTip("Frequency bars to display (8-16384)")
        bars_layout.addWidget(self.bars_spinbox)
        auto_btn = QPushButton("Auto")
        auto_btn.setToolTip("Auto-calculate bars based on window size")
        auto_btn.clicked.connect(self.set_bars_auto)
        bars_layout.addWidget(auto_btn)
        bars_layout.addStretch()
        visual_layout.addLayout(bars_layout)
        
        # Happy mode checkbox
        self.happy_checkbox = QCheckBox("Happy Mode (Vibrant Colors)")
        self.happy_checkbox.setToolTip("Bright, joyful colors with improved contrast")
        visual_layout.addWidget(self.happy_checkbox)
        
        # Random color checkbox
        self.random_color_checkbox = QCheckBox("Random Color Palette")
        self.random_color_checkbox.setToolTip("Generate harmonious random colors based on stereo balance")
        visual_layout.addWidget(self.random_color_checkbox)
        
        # Color seed
        seed_layout = QHBoxLayout()
        seed_lbl = QLabel("Color Seed:")
        seed_lbl.setToolTip("Seed for reproducible color generation")
        seed_layout.addWidget(seed_lbl)
        self.seed_input = QLineEdit()
        self.seed_input.setMaximumWidth(150)
        self.seed_input.setToolTip("Leave empty for auto seed, or enter a number")
        seed_layout.addWidget(self.seed_input)
        regen_btn = QPushButton("Regenerate")
        regen_btn.setToolTip("Generate a new random seed")
        regen_btn.clicked.connect(self.regenerate_color_seed)
        seed_layout.addWidget(regen_btn)
        seed_layout.addStretch()
        visual_layout.addLayout(seed_layout)
        
        visual_group.setLayout(visual_layout)
        main_layout.addWidget(visual_group)
        
        # ===== Application Settings Group =====
        app_group = QGroupBox("Application Settings")
        app_layout = QVBoxLayout()
        
        self.silent_checkbox = QCheckBox("Silent Mode")
        self.silent_checkbox.setToolTip("Suppress all console output")
        app_layout.addWidget(self.silent_checkbox)
        
        self.debug_checkbox = QCheckBox("Debug Mode")
        self.debug_checkbox.setToolTip("Print detailed debugging information")
        app_layout.addWidget(self.debug_checkbox)
        
        app_group.setLayout(app_layout)
        main_layout.addWidget(app_group)
        
        # ===== Buttons =====
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setToolTip("Restore all settings to default values")
        reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(reset_btn)
        
        close_btn = QPushButton("Close")
        close_btn.setToolTip("Close this dialog (press S to reopen)")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        apply_btn = QPushButton("Apply")
        apply_btn.setToolTip("Apply all changes")
        apply_btn.clicked.connect(self.apply_settings)
        apply_btn.setStyleSheet("background-color: rgb(70, 120, 180);")
        button_layout.addWidget(apply_btn)
        
        main_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
    
    def load_current_values(self):
        """Load current values from visualizer"""
        self.chunk_spinbox.setValue(self.visualizer.CHUNK)
        self.buffer_spinbox.setValue(self.visualizer.FFT_SIZE)
        self.update_spinbox.setValue(self.visualizer.UPDATE_RATE)
        self.bias_slider.setValue(int((self.visualizer.HUMAN_BIAS - 0.5) * 200))
        self.happy_checkbox.setChecked(self.visualizer.HAPPY_MODE)
        self.random_color_checkbox.setChecked(self.visualizer.RANDOM_COLOR)
        self.silent_checkbox.setChecked(self.visualizer.SILENT)
        self.debug_checkbox.setChecked(self.visualizer.DEBUG)
        
        # Bars
        if self.visualizer.user_num_bars is not None:
            self.bars_spinbox.setValue(self.visualizer.user_num_bars)
        else:
            self.bars_spinbox.setValue(64)  # Show a reasonable default
        
        # Color seed
        if hasattr(self.visualizer, '_palette_seed'):
            self.seed_input.setText(str(self.visualizer._palette_seed))
        
        # Device selection
        self.refresh_devices()
    
    def refresh_devices(self):
        """Scan and populate audio devices"""
        self.device_combo.clear()
        devices = []
        
        # Try soundcard first
        if HAS_SOUNDCARD:
            try:
                for dev in sc.all_speakers():
                    dev_name = getattr(dev, 'name', 'Unknown')
                    dev_id = getattr(dev, 'id', None)
                    devices.append((dev_name, str(dev_id) if dev_id else dev_name))
            except Exception as e:
                if self.visualizer.DEBUG:
                    print(f"Error scanning soundcard devices: {e}")
        
        # Try sounddevice
        if HAS_SOUNDDEVICE:
            try:
                for i, dev in enumerate(sd.query_devices()):
                    if dev['max_input_channels'] > 0 or dev['max_output_channels'] > 0:
                        dev_name = dev.get('name', f'Device {i}')
                        devices.append((dev_name, f"sd:{i}"))
            except Exception as e:
                if self.visualizer.DEBUG:
                    print(f"Error scanning sounddevice devices: {e}")
        
        # Add "Default" option
        devices.insert(0, ("Default", "default"))
        # Populate combo box
        for name, device_id in devices:
            self.device_combo.addItem(name, device_id)
        
        # Select current device if available
        if self.visualizer.user_device_name:
            idx = self.device_combo.findText(self.visualizer.user_device_name)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)
    
    def update_bias_label(self):
        """Update bias percentage display"""
        value = self.bias_slider.value()
        self.bias_label.setText(f"{value:+d}%")
    
    def set_bars_auto(self):
        """Set bars to auto mode"""
        self.bars_spinbox.setValue(64)  # Show default, visualizer will auto-calculate
    
    def regenerate_color_seed(self):
        """Generate new random seed"""
        new_seed = random.randint(0, (2**32)-1) % (2**32)
        self.seed_input.setText(str(new_seed))
    
    def validate_inputs(self):
        """Validate all settings before applying"""
        errors = []
        
        chunk = self.chunk_spinbox.value()
        buffer = self.buffer_spinbox.value()
        update_rate = self.update_spinbox.value()
        
        # Check power of 2
        if chunk & (chunk - 1) != 0:
            errors.append(f"Chunk size {chunk} is not a power of 2")
        if buffer & (buffer - 1) != 0:
            errors.append(f"Buffer size {buffer} is not a power of 2")
        
        # Check ranges
        if buffer < chunk:
            errors.append("Buffer size must be >= chunk size")
        
        # Show errors if any
        if errors:
            error_msg = "Validation errors:\n" + "\n".join(f"• {e}" for e in errors)
            QMessageBox.warning(self, "Validation Error", error_msg)
            return False
        
        return True
    
    def apply_settings(self):
        """Apply settings and emit signal to parent"""
        if not self.validate_inputs():
            return
        
        settings = {
            'chunk_size': self.chunk_spinbox.value(),
            'buffer_size': self.buffer_spinbox.value(),
            'update_rate': self.update_spinbox.value(),
            'human_bias': (self.bias_slider.value() + 50) / 200.0,
            'num_bars': self.bars_spinbox.value() if self.bars_spinbox.value() != 64 else None,
            'happy_mode': self.happy_checkbox.isChecked(),
            'random_color': self.random_color_checkbox.isChecked(),
            'color_seed': int(self.seed_input.text()) if self.seed_input.text() else None,
            'silent': self.silent_checkbox.isChecked(),
            'debug': self.debug_checkbox.isChecked(),
            'device_id': self.device_combo.currentData(),
        }
        
        # Save settings to file
        save_settings(settings)
        
        # Emit signal to update visualizer
        self.settings_applied.emit(settings)
        
        if not self.visualizer.SILENT:
            print("✓ Settings applied and saved successfully")
    
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        self.chunk_spinbox.setValue(self.defaults['chunk_size'])
        self.buffer_spinbox.setValue(self.defaults['buffer_size'])
        self.update_spinbox.setValue(self.defaults['update_rate'])
        self.bias_slider.setValue(int((self.defaults['human_bias'] - 0.5) * 200))
        self.happy_checkbox.setChecked(self.defaults['happy_mode'])
        self.random_color_checkbox.setChecked(self.defaults['random_color'])
        self.silent_checkbox.setChecked(self.defaults['silent'])
        self.debug_checkbox.setChecked(self.defaults['debug'])
        self.seed_input.clear()
        self.bars_spinbox.setValue(64)


def main():
    parser = argparse.ArgumentParser(
        description='Real-time Audio Frequency Visualizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                                    # Run with default settings
  %(prog)s --human-bias 0.8                  # Increase ISO 226 influence
  %(prog)s --buffer-size 16384 --chunk-size 2048  # Higher quality, more latency
  %(prog)s --update-rate 120                 # Smoother display at 120 FPS
        '''
    )
    
    parser.add_argument('--human-bias', type=float, default=0.5, metavar='FACTOR',
                        help='ISO 226 equal-loudness influence (0.0-1.0, default: 0.5). '
                             'Higher values make the visualization match human hearing more closely.')
    
    parser.add_argument('--buffer-size', type=int, default=2048, metavar='SAMPLES',
                        help='FFT buffer size in samples (default: 2048). '
                             'Larger values give better bass resolution but more latency.')
    
    parser.add_argument('--chunk-size', type=int, default=512, metavar='SAMPLES',
                        help='Audio chunk size in samples (default: 512). '
                             'Smaller values reduce latency but may increase CPU usage.')
    
    parser.add_argument('--update-rate', type=int, default=144, metavar='HZ',
                        help='Display update rate in Hz (default: 144). '
                             'Higher values make animation smoother but use more CPU.')
    
    parser.add_argument('--silent', action='store_true',
                        help='Suppress all output except errors.')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output (performance stats, audio info, etc.).')
    
    parser.add_argument('--happy', action='store_true',
                        help='Enable joyful color mode with vibrant, saturated colors.')
    
    parser.add_argument('--random-color', action='store_true',
                        help='Enable random color palette generation with harmonious, designer-quality colors.')
    
    parser.add_argument('--color-seed', type=int, default=None, metavar='SEED',
                        help='Random seed for color palette generation (default: auto from timestamp). '
                             'Only used with --random-color.')
    
    parser.add_argument('--num-bars', type=int, default=None, metavar='COUNT',
                        help='Number of frequency bars to display (default: auto-calculated based on window size). '
                             'Range: 8-16384 bars.')

    parser.add_argument('--device-id', type=str, default=None,
                        help='Force capture from this output device ID (use with soundcard device id).')

    parser.add_argument('--device-name', type=str, default=None,
                        help='Force capture from an output device matching this name (substring match).')

    parser.add_argument('--diagnose-audio', action='store_true',
                        help='Run a short audio capture diagnostic and write logs to /tmp/aVis_diagnostics.log then exit.')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.human_bias < 0 or args.human_bias > 1:
        parser.error('--human-bias must be between 0.0 and 1.0')
    if args.buffer_size < args.chunk_size:
        parser.error('--buffer-size must be >= --chunk-size')
    if args.chunk_size < 256 or args.chunk_size > 16384:
        parser.error('--chunk-size must be between 256 and 16384')
    if args.buffer_size < 512 or args.buffer_size > 32768:
        parser.error('--buffer-size must be between 512 and 32768')
    if args.update_rate < 1 or args.update_rate > 240:
        parser.error('--update-rate must be between 1 and 240 Hz')
    if args.num_bars is not None and (args.num_bars < 8 or args.num_bars > 16384):
        parser.error('--num-bars must be between 8 and 16384')
    
    # Check if buffer_size and chunk_size are powers of 2 for optimal FFT performance
    if not args.silent:
        if args.buffer_size & (args.buffer_size - 1) != 0:
            print(f"Warning: --buffer-size {args.buffer_size} is not a power of 2. FFT may be slower.", file=sys.stderr)
        if args.chunk_size & (args.chunk_size - 1) != 0:
            print(f"Warning: --chunk-size {args.chunk_size} is not a power of 2. FFT may be slower.", file=sys.stderr)
    
    app = QApplication(sys.argv)
    visualizer = AudioVisualizer(
        chunk_size=args.chunk_size,
        buffer_size=args.buffer_size,
        update_rate=args.update_rate,
        human_bias=args.human_bias,
        silent=args.silent,
        debug=args.debug,
        happy_mode=args.happy,
        random_color=args.random_color,
        color_seed=args.color_seed,
        num_bars=args.num_bars,
        device_id=args.device_id,
        device_name=args.device_name
    )
    visualizer.show()
    # If user requested diagnostics, run and exit before entering Qt event loop
    if args.diagnose_audio:
        # Run 8 seconds of diagnostics to give time for stereo capture
        visualizer.run_audio_diagnostics(duration=8, out_file='/tmp/aVis_diagnostics.log')
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
