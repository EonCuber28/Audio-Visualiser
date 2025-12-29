#!/usr/bin/env python3
"""
Real-time Audio Frequency Visualizer (Cross-Platform)
Captures system output audio and displays frequency spectrum using FFT
Supports: Linux (PulseAudio/PipeWire), Windows 10/11 (WASAPI Loopback)
"""

import sys
import numpy as np
import struct
import argparse
import platform
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
import subprocess
import threading
import queue
import time
import random

# Platform detection
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

# Try to import soundcard for Windows (native WASAPI loopback - best option)
HAS_SOUNDCARD = False
if IS_WINDOWS:
    try:
        import soundcard as sc
        HAS_SOUNDCARD = True
    except ImportError:
        pass

# Fallback: Try to import sounddevice for Windows
HAS_SOUNDDEVICE = False
if IS_WINDOWS and not HAS_SOUNDCARD:
    try:
        import sounddevice as sd
        HAS_SOUNDDEVICE = True
    except ImportError:
        pass

# Fallback: Try to import PyAudio for Windows WASAPI loabback
HAS_PYAUDIO = False
if IS_WINDOWS and not HAS_SOUNDCARD and not HAS_SOUNDDEVICE:
    try:
        import pyaudio
        HAS_PYAUDIO = True
    except ImportError:
        pass

# Try to import scipy for better interpolation, fall back to linear if not available
try:
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AudioVisualizer(QMainWindow):
    # Signal for thread-safe visualization updates
    update_signal = pyqtSignal()
    
    def __init__(self, chunk_size=1024, buffer_size=8192, update_rate=60, human_bias=0.5, silent=False, debug=False, happy_mode=False, random_color=False, color_seed=None):
        super().__init__()
        self.setWindowTitle("Audio Frequency Visualizer")
        self.setGeometry(100, 100, 1200, 600)
        
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
        
        # Audio buffer queue
        self.audio_queue = queue.Queue(maxsize=3)  # Smaller queue to minimize latency
        self.running = True
        self.frames_without_data = 0  # Track consecutive frames without audio
        
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
            if hasattr(self, 'canvas') and hasattr(self.canvas, 'cached_colors'):
                self.canvas.cached_colors.clear()
            if not self.SILENT:
                print(f"✓ Regenerated palette (seed: {self._palette_seed})")
        
    def capture_audio(self):
        """Capture audio using platform-specific method"""
        if IS_WINDOWS:
            self._capture_audio_windows()
        else:
            self._capture_audio_linux()
    
    def _capture_audio_windows(self):
        """Capture audio on Windows using soundcard, sounddevice, or PyAudio WASAPI loopback"""
        # Try soundcard first (native WASAPI loopback - no Stereo Mix needed!)
        if HAS_SOUNDCARD:
            self._capture_audio_windows_soundcard()
        elif HAS_SOUNDDEVICE:
            self._capture_audio_windows_sounddevice()
        elif HAS_PYAUDIO:
            self._capture_audio_windows_pyaudio()
        else:
            if not self.SILENT:
                print("✗ ERROR: No audio library installed for Windows.")
                print("Install one of these:")
                print("  pip install soundcard  (recommended - native WASAPI loopback)")
                print("  pip install sounddevice")
                print("  pip install pyaudio")
    
    def _capture_audio_windows_soundcard(self):
        """Capture audio using soundcard (native WASAPI loopback - best for Windows)"""
        try:
            import soundcard as sc
            
            if not self.SILENT:
                print("Using soundcard for audio capture (native WASAPI loopback)...")
            
            # Get default speaker/loopback
            try:
                default_speaker = sc.default_speaker()
                loopback = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)
                
                if not self.SILENT:
                    print(f"✓ Capturing from: {default_speaker.name}")
                    print(f"  Channels: {loopback.channels}")
                    print(f"  Sample rate: {loopback.samplerate} Hz")
                
            except Exception as e:
                # Fallback: try to find any loopback device
                if self.DEBUG:
                    print(f"Could not get default speaker loopback: {e}")
                    print("Searching for loopback devices...")
                
                mics = sc.all_microphones(include_loopback=True)
                loopback = None
                
                for mic in mics:
                    if mic.isloopback:
                        loopback = mic
                        if not self.SILENT:
                            print(f"✓ Found loopback device: {mic.name}")
                        break
                
                if loopback is None:
                    if not self.SILENT:
                        print("✗ ERROR: No loopback device found!")
                    return
            
            # Update settings based on device
            if loopback.samplerate != self.RATE:
                if not self.SILENT:
                    print(f"Note: Using {loopback.samplerate}Hz (device default)")
                self.RATE = int(loopback.samplerate)
                self.freqs = np.fft.rfftfreq(self.FFT_SIZE, 1.0 / self.RATE)
            
            # Determine channels
            device_channels = loopback.channels
            if device_channels >= 2:
                channels = 2
            elif device_channels == 1:
                channels = 1
            else:
                channels = 2
            
            if channels != self.CHANNELS:
                if not self.SILENT:
                    channel_type = "mono" if channels == 1 else "stereo"
                    print(f"Note: Using {channels} channel{'s' if channels > 1 else ''} ({channel_type})")
                self.CHANNELS = channels
            
            if not self.SILENT:
                print("✓ Audio capture started (soundcard WASAPI loopback)")
            
            read_count = 0
            
            # Record in a loop
            with loopback.recorder(samplerate=self.RATE, channels=channels, blocksize=self.CHUNK) as rec:
                while self.running:
                    try:
                        # Read audio chunk
                        data = rec.record(numframes=self.CHUNK)
                        
                        # Convert float32 data to int16 for compatibility
                        data_int16 = (data * 32767).astype(np.int16)
                        data_bytes = data_int16.tobytes()
                        
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
                    
                    except Exception as e:
                        if self.DEBUG:
                            print(f"Read error: {e}")
                        time.sleep(0.001)
                        continue
        
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
                
                # Priority 1: Explicit loopback devices
                if dev_info['maxInputChannels'] > 0 and ('stereo mix' in name_lower or 
                    'wave out' in name_lower or 'loopback' in name_lower or
                    'what u hear' in name_lower or 'what you hear' in name_lower):
                    candidates.append((1, i, dev_info, "explicit loopback"))
                
                # Priority 2: Default WASAPI output device (for loopback capture)
                elif is_wasapi and i == default_output_index and dev_info['maxOutputChannels'] > 0:
                    candidates.append((2, i, dev_info, "default WASAPI output"))
                
                # Priority 3: Any WASAPI output device (can be used for loopback)
                elif is_wasapi and dev_info['maxOutputChannels'] > 0:
                    candidates.append((3, i, dev_info, "WASAPI output"))
            
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
            
            # Use device's default sample rate
            default_sample_rate = int(device_info.get('defaultSampleRate', 44100))
            sample_rate = default_sample_rate
            
            # Determine if we're using loopback mode (output device as input)
            is_output_device = device_info.get('maxOutputChannels', 0) > 0 and device_info.get('maxInputChannels', 0) == 0
            use_loopback = is_output_device and wasapi_host_api is not None
            
            # For WASAPI loopback, use output channel count; otherwise use input channels
            if use_loopback:
                max_channels = int(device_info.get('maxOutputChannels', 2))
            else:
                max_channels = int(device_info.get('maxInputChannels', 2))
            
            # Use device's available channels (prefer stereo but accept mono)
            if max_channels >= 2:
                channels = 2  # Use stereo if available
            elif max_channels == 1:
                channels = 1  # Use mono if that's all we have
            else:
                channels = 2  # Fallback to stereo
            
            if not self.SILENT:
                channel_type = "mono" if channels == 1 else "stereo"
                print(f"Device info: {default_sample_rate}Hz, {channels} channel{'s' if channels > 1 else ''} ({channel_type})")
                if use_loopback:
                    print("Using WASAPI loopback mode (capturing system audio output)")
                if self.DEBUG:
                    print(f"Device host API: {device_info['hostApi']}")
                    print(f"Is output device: {is_output_device}")
            
            # Build stream parameters
            stream_params = {
                'format': pyaudio.paInt16,
                'channels': channels,
                'rate': sample_rate,
                'input': True,
                'input_device_index': loopback_device_index,
                'frames_per_buffer': self.CHUNK,
            }
            
            if not self.SILENT:
                print(f"Opening audio stream: {sample_rate}Hz, {channels} channels, chunk={self.CHUNK}")
                if use_loopback:
                    print("Note: Attempting to use output device for system audio capture")
                    print("This requires either Stereo Mix or a PyAudio build with WASAPI loopback support")
            
            # Try to open stream
            stream = None
            last_error = None
            
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
            
            # Update channels if different
            if channels != self.CHANNELS:
                if not self.SILENT:
                    channel_type = "mono" if channels == 1 else "stereo"
                    print(f"Note: Using {channels} channel{'s' if channels > 1 else ''} ({channel_type}) instead of requested {self.CHANNELS}")
                self.CHANNELS = channels
            
            if not self.SILENT:
                print("✓ Audio capture started (Windows WASAPI)")
            
            read_count = 0
            
            while self.running:
                try:
                    # Read audio chunk
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    read_count += 1
                    
                    # Put data in queue (drop old data if queue is full)
                    if self.audio_queue.full():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.audio_queue.put(data)
                    
                    if self.DEBUG and read_count % 100 == 0:
                        print(f"Audio thread: {read_count} chunks, queue: {self.audio_queue.qsize()}")
                        
                except Exception as e:
                    if self.DEBUG:
                        print(f"Read error: {e}")
                    time.sleep(0.001)
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
        """Capture audio on Linux using PulseAudio/PipeWire parec"""
        try:
            # Get the monitor source name
            result = subprocess.run(
                ['pactl', 'list', 'sources', 'short'],
                capture_output=True,
                text=True,
                check=True
            )
            
            monitor_source = None
            for line in result.stdout.split('\n'):
                if 'monitor' in line.lower():
                    monitor_source = line.split()[1]
                    if not self.SILENT:
                        print(f"✓ Using monitor source: {monitor_source}")
                    break
            
            if not monitor_source:
                if not self.SILENT:
                    print("✗ ERROR: No monitor source found!")
                return
            
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
            
            if not self.SILENT:
                print(f"Starting parec with command: {' '.join(cmd)}")
            
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
        
        # Track timing
        self.last_viz_time = frame_start
        
        # Debug: count calls to see if timer is misbehaving
        if not hasattr(self, '_viz_call_count'):
            self._viz_call_count = 0
            self._viz_call_start = frame_start
        self._viz_call_count += 1
        
        # Profiling timestamps
        prof = {} if self.DEBUG else None
        
        data_processed = False
        try:
            # Process all available audio chunks and update rolling buffer
            chunks_processed = 0
            last_data = None
            
            # Get all chunks but only process the last one if multiple available
            while not self.audio_queue.empty():
                try:
                    last_data = self.audio_queue.get_nowait()
                    chunks_processed += 1
                except queue.Empty:
                    break
            
            if last_data is not None:
                data = last_data
                
                # Convert byte data to numpy array
                data_int = np.frombuffer(data, dtype=np.int16)
                
                # If stereo, process channels separately for stereo balance
                if self.CHANNELS == 2:
                    data_stereo = data_int.reshape(-1, 2)
                    data_left = data_stereo[:, 0]
                    data_right = data_stereo[:, 1]
                    data_float = (data_left.astype(np.float32) + data_right.astype(np.float32)) * 0.5
                    
                    # Roll buffers and add new data
                    self.audio_buffer_left = np.roll(self.audio_buffer_left, -len(data_left))
                    self.audio_buffer_left[-len(data_left):] = data_left
                    self.audio_buffer_right = np.roll(self.audio_buffer_right, -len(data_right))
                    self.audio_buffer_right[-len(data_right):] = data_right
                else:
                    # Mono: convert to float
                    data_float = data_int.astype(np.float32)
                
                # Roll the main buffer and add new data
                self.audio_buffer = np.roll(self.audio_buffer, -len(data_float))
                self.audio_buffer[-len(data_float):] = data_float
                
                data_processed = True
            
            if self.DEBUG and prof is not None:
                prof['audio_process'] = time.perf_counter() - frame_start
            
            if data_processed:
                # Step 2 & 3: Window and FFT per channel (use pre-computed window)
                if self.CHANNELS == 2:
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
                    
                    # Apply minimal smoothing to stereo balance for color transitions
                    alpha_balance = 0.4
                    self.stereo_balance_smoothed = alpha_balance * self.stereo_balance + (1 - alpha_balance) * self.stereo_balance_smoothed
                else:
                    # Mono: just compute FFT
                    windowed_data = self.audio_buffer * self.hamming_window
                    fft_result = np.fft.rfft(windowed_data)
                    fft_magnitude = np.abs(fft_result) * self.window_correction
                    self.stereo_balance.fill(0)
                    self.stereo_balance_smoothed.fill(0)  # No smoothing needed for mono
                
                if self.DEBUG and prof is not None:
                    prof['fft_compute'] = time.perf_counter() - frame_start - prof['audio_process']
                
                # Step 6: Convert to dB (log loudness) - optimized with pre-computed constants
                norm_factor = self.FFT_SIZE * 32768.0  # Pre-compute normalization
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
                    self.waveform_buffer = recent_samples / 32768.0
                self._frame_counter += 1
            elif data_processed:
                self._frame_counter = 0
                recent_samples = self.audio_buffer[-self.oscilloscope_samples:].copy()
                self.waveform_buffer = recent_samples / 32768.0
            
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
                        
                        # Report call rate vs render rate
                        call_duration = frame_end - self._viz_call_start
                        call_rate = self._viz_call_count / call_duration if call_duration > 0 else 0
                        
                        print(f"Performance: {fps:.1f} FPS (actual render: {actual_fps:.1f}, calls: {call_rate:.1f}) | Frame time: avg={avg_frame_time_ms:.2f}ms min={min_frame_time:.2f}ms max={max_frame_time:.2f}ms")
                        print(f"  Target: {self.UPDATE_RATE} FPS ({timer_interval_ms:.2f}ms interval)")
                        if prof:
                            print(f"  Breakdown: audio={prof.get('audio_process', 0)*1000:.2f}ms fft={prof.get('fft_compute', 0)*1000:.2f}ms " +
                                  f"db/smooth={prof.get('db_smooth', 0)*1000:.2f}ms scope={prof.get('oscilloscope', 0)*1000:.2f}ms render={prof.get('render', 0)*1000:.2f}ms")
                        
                        # Reset call counter
                        self._viz_call_count = 0
                        self._viz_call_start = frame_end
                    self.frame_times.clear()
                self.last_fps_report = frame_end
                
        except Exception as e:
            if not self.SILENT:
                print(f"Error reading audio: {e}")
            if self.DEBUG:
                import traceback
                traceback.print_exc()
    
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
        self.cached_colors = {}  # Color lookup table
        self.cached_layout = None  # Cache computed layout
        self.last_dimensions = (0, 0)  # Track size changes
        self.cached_bar_freqs = None  # Cache bar frequencies
        self.cached_num_bars = 0  # Cache number of bars
        self.actual_min_freq = 20.0  # Actual minimum frequency being displayed
        self.actual_max_freq = 10000.0  # Actual maximum frequency being displayed
        
        # Peak hold lines
        self.peak_hold = np.zeros(2048)  # Peak positions for each bar
        self.peak_fall_rate = 0.006  # How fast peaks fall (per frame)
        
        # Frame rate throttling (slightly higher threshold to ensure 60 FPS)
        self.last_paint_time = 0.0
        self.min_paint_interval = 0.0165  # ~60 FPS max (16.5ms)
        
        # Enable double buffering to prevent tearing
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)  # Let Qt handle background
        self.setAttribute(Qt.WA_NoSystemBackground, False)  # Allow system background initially
        
    def get_color_for_brightness(self, brightness, stereo_balance=0.0, freq_ratio=0.0):
        """Get color using stereo balance LUT and brightness scaling
        brightness: 0-1 (from dB loudness)
        stereo_balance: -1 (left) to +1 (right)
        freq_ratio: 0-1 (position in frequency range, for happy mode enhancements)
        """
        # Cache colors with limited size (optimization: prevent unbounded memory growth)
        brightness_key = int(brightness * 100)
        balance_key = int((stereo_balance + 1) * 50)
        freq_key = int(freq_ratio * 100) if self.parent_visualizer.HAPPY_MODE else 0
        cache_key = (brightness_key, balance_key, freq_key)
        
        if cache_key in self.cached_colors:
            return self.cached_colors[cache_key]
        
        # Limit cache size to prevent memory bloat
        if len(self.cached_colors) > 10000:
            self.cached_colors.clear()
        
        # RANDOM COLOR MODE: Use generated palette
        if self.parent_visualizer.RANDOM_COLOR and self.parent_visualizer.color_palette:
            return self._get_random_palette_color(brightness, stereo_balance, freq_ratio, cache_key)
        elif self.parent_visualizer.HAPPY_MODE:
            # HAPPY MODE: Joyful color pipeline
            return self._get_happy_color(brightness, stereo_balance, freq_ratio, cache_key)
        else:
            # NORMAL MODE: Original pipeline
            return self._get_normal_color(brightness, stereo_balance, cache_key)
    
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
        
        color = QColor(int(r), int(g), int(b))
        self.cached_colors[cache_key] = color
        return color
    
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
        
        color = QColor(int(r), int(g), int(b))
        self.cached_colors[cache_key] = color
        return color
        
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
        
        # Check if dimensions changed
        dimensions_changed = (width, height) != self.last_dimensions
        if dimensions_changed:
            self.last_dimensions = (width, height)
        
        # Reserve space for oscilloscope at bottom
        oscilloscope_height = 70  # Total height including padding
        
        # Determine layout
        use_vertical = width < 400
        
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
                if use_vertical:
                    num_bars = min(256, graph_height // 3)  # Cap at 256 bars
                else:
                    num_bars = min(512, graph_width // 3)  # Cap at 512 bars
                
                # Cache bar frequencies if dimensions changed
                if dimensions_changed or self.cached_num_bars != num_bars:
                    # Start from first available frequency bin to avoid extrapolation issues
                    min_freq = max(freqs_filtered[0], 10.0)  # Use actual first bin or 10 Hz minimum
                    self.actual_min_freq = min_freq
                    self.actual_max_freq = 10000.0
                    self.cached_bar_freqs = np.logspace(np.log10(min_freq), np.log10(10000), num_bars)
                    self.cached_num_bars = num_bars
                    
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
                
                # Interpolate FFT data to display bars
                # Use cubic interpolation for bass and mids (20-1000Hz), linear for higher frequencies
                smooth_mask = bar_freqs <= 1000
                data_interpolated = np.zeros_like(bar_freqs)
                
                if HAS_SCIPY and np.any(smooth_mask):
                    # Smooth region: fancy cubic spline interpolation for smooth curves (20-1000Hz)
                    smooth_indices = np.where(smooth_mask)[0]
                    high_indices = np.where(~smooth_mask)[0]
                    
                    try:
                        # Cubic interpolation for bass/mids (20-1000Hz)
                        smooth_interp = interp1d(freqs_filtered, data_filtered, kind='cubic', fill_value='extrapolate')
                        data_interpolated[smooth_indices] = smooth_interp(bar_freqs[smooth_indices])
                        
                        # Linear interpolation for high frequencies (1000Hz+)
                        if len(high_indices) > 0:
                            data_interpolated[high_indices] = np.interp(bar_freqs[high_indices], freqs_filtered, data_filtered)
                    except:
                        # Fallback to linear if cubic fails
                        data_interpolated = np.interp(bar_freqs, freqs_filtered, data_filtered)
                else:
                    # No scipy or no smooth range - use linear for everything
                    data_interpolated = np.interp(bar_freqs, freqs_filtered, data_filtered)
                
                log_freqs = bar_freqs
                
                # Interpolate stereo balance data for each bar
                stereo_balance_interpolated = np.interp(bar_freqs, freqs_filtered, self.stereo_balance[mask])
                
                # Step 7: Loudness → Brightness (optimized: vectorized operations)
                if self.parent_visualizer.HAPPY_MODE:
                    brightness_interpolated = np.clip((data_interpolated + 60.0) * (1.0/60.0), 0.0, 1.0)
                else:
                    brightness_interpolated = np.clip((data_interpolated + 80.0) * 0.0125, 0.0, 1.0)
                
                # Pre-compute colors for this frame to avoid repeated lookups
                frame_colors = []
                is_mono = self.parent_visualizer.CHANNELS == 1
                
                for i in range(len(log_freqs)):
                    brightness = brightness_interpolated[i]
                    balance = stereo_balance_interpolated[i]
                    freq_ratio = i / len(log_freqs)
                    
                    # Mono mode: apply frequency-based brightness gradient
                    # High frequencies (right side) = brighter, Low frequencies (left side) = dimmer
                    if is_mono:
                        # Create gradient: 0.4 (low freq/dim) to 1.0 (high freq/bright)
                        freq_brightness_boost = 0.4 + (freq_ratio * 0.6)
                        brightness = brightness * freq_brightness_boost
                        brightness = min(1.0, brightness)  # Clamp to max 1.0
                    
                    frame_colors.append(self.get_color_for_brightness(brightness, balance, freq_ratio))
                
                # Draw bars - orientation depends on layout
                painter.setPen(Qt.NoPen)
                
                # Resize peak hold array if needed
                if len(self.peak_hold) != num_bars:
                    self.peak_hold = np.zeros(num_bars)
                
                if use_vertical:
                    # Vertical layout - bars linearly spaced, representing logarithmic frequencies
                    bar_height = graph_height / num_bars
                    num_freqs = len(log_freqs)
                    peak_fall = self.peak_fall_rate
                    
                    for i in range(num_freqs):
                        idx = num_freqs - 1 - i
                        brightness = brightness_interpolated[idx]
                        
                        # Update peak hold always
                        self.peak_hold[i] = brightness if brightness > self.peak_hold[i] else max(0, self.peak_hold[i] - peak_fall)
                        
                        # Skip drawing if brightness too low (optimization)
                        if brightness < 0.01 and self.peak_hold[i] < 0.01:
                            continue
                        
                        # Linear position from top to bottom with overlap to prevent gaps
                        y_pos = margin_top + int(i * bar_height)
                        # Add 1px to height to ensure no gaps between bars
                        bar_h = max(1, int(bar_height) + 1)
                        bar_width_val = int(brightness * graph_width + 0.5)
                        
                        if bar_width_val > 0:
                            painter.setBrush(frame_colors[idx])
                            painter.drawRect(margin_left, y_pos, bar_width_val, bar_h)
                        
                        # Draw peak hold line (simplified - no color lookup)
                        if self.peak_hold[i] > 0.001:
                            peak_x = margin_left + int(self.peak_hold[i] * graph_width + 0.5)
                            painter.setPen(QPen(frame_colors[idx], 1))
                            painter.drawLine(peak_x, y_pos, peak_x, y_pos + bar_h)
                            painter.setPen(Qt.NoPen)

                else:
                    # Horizontal layout - bars linearly spaced, representing logarithmic frequencies
                    bar_width_val = graph_width / num_bars
                    num_freqs = len(log_freqs)
                    peak_fall = self.peak_fall_rate
                    
                    for i in range(num_freqs):
                        brightness = brightness_interpolated[i]
                        
                        # Update peak hold always
                        self.peak_hold[i] = brightness if brightness > self.peak_hold[i] else max(0, self.peak_hold[i] - peak_fall)
                        
                        # Skip drawing if brightness too low (optimization)
                        if brightness < 0.01 and self.peak_hold[i] < 0.01:
                            continue
                        
                        # Linear position from left to right with overlap to prevent gaps
                        x_pos = margin_left + int(i * bar_width_val)
                        # Add 1px to width to ensure no gaps between bars
                        bar_w = max(1, int(bar_width_val) + 1)
                        bar_height = int(brightness * graph_height + 0.5)
                        
                        if bar_height > 0:
                            painter.setBrush(frame_colors[i])
                            y_pos = margin_top + (graph_height - bar_height)
                            painter.drawRect(x_pos, y_pos, bar_w, bar_height)
                        
                        # Draw peak hold line (simplified - no color lookup)
                        if self.peak_hold[i] > 0.001:
                            peak_height = int(self.peak_hold[i] * graph_height + 0.5)
                            peak_y = margin_top + (graph_height - peak_height)
                            painter.setPen(QPen(frame_colors[i], 1))
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
            
            # Draw waveform (optimized: reduced resolution)
            painter.setPen(QPen(QColor(100, 200, 255), 1))
            if len(self.waveform_data) > 1:
                num_points = min(len(self.waveform_data), scope_width) // 2  # Half resolution for performance
                step = len(self.waveform_data) / num_points
                scale = scope_height * 0.45
                inv_num_points = scope_width / num_points
                
                for i in range(num_points - 1):
                    idx1 = int(i * step)
                    idx2 = int((i + 1) * step)
                    
                    # Normalize waveform to oscilloscope height
                    y1 = center_y - int(np.clip(self.waveform_data[idx1], -1.0, 1.0) * scale)
                    y2 = center_y - int(np.clip(self.waveform_data[idx2], -1.0, 1.0) * scale)
                    x1 = scope_left + int(i * inv_num_points)
                    x2 = scope_left + int((i + 1) * inv_num_points)
                    
                    painter.drawLine(x1, y1, x2, y2)
            
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
    
    parser.add_argument('--human-bias', type=float, default=0, metavar='FACTOR',
                        help='ISO 226 equal-loudness influence (0.0-1.0, default: 0.5). '
                             'Higher values make the visualization match human hearing more closely.')
    
    parser.add_argument('--buffer-size', type=int, default=2048, metavar='SAMPLES',
                        help='FFT buffer size in samples (default: 2048). '
                             'Larger values give better bass resolution but more latency.')
    
    parser.add_argument('--chunk-size', type=int, default=512, metavar='SAMPLES',
                        help='Audio chunk size in samples (default: 512). '
                             'Smaller values reduce latency but may increase CPU usage.')
    
    parser.add_argument('--update-rate', type=int, default=60, metavar='HZ',
                        help='Display update rate in Hz (default: 60). '
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
        color_seed=args.color_seed
    )
    visualizer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
