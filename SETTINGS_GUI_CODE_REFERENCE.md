# Settings GUI Implementation - Code Reference

## Quick Start for Users

### Accessing Settings
```
Button: Click ⚙ Settings in top-right corner
Keyboard: Press S key
```

## Key Code Components

### 1. Opening the Settings Dialog

**In AudioVisualizer:**
```python
def open_settings_dialog(self):
    """Open the settings dialog (modeless - doesn't block)"""
    if not hasattr(self, 'settings_dialog') or self.settings_dialog is None:
        self.settings_dialog = SettingsDialog(self, self)
        self.settings_dialog.settings_applied.connect(self.apply_settings)
        self.settings_dialog.show()
    else:
        # Bring existing dialog to front if already open
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()
```

**Keyboard Binding:**
```python
def keyPressEvent(self, event):
    # ... other handlers ...
    elif event.key() == Qt.Key_S:
        self.open_settings_dialog()
```

### 2. Settings Button in Main Window

**In setup_ui():**
```python
# Top bar with settings button
top_bar = QWidget()
top_bar_layout = QHBoxLayout()

settings_btn = QPushButton("⚙ Settings")
settings_btn.clicked.connect(self.open_settings_dialog)
settings_btn.setStyleSheet("""
    QPushButton {
        background-color: rgb(50, 50, 60);
        border: 1px solid rgb(80, 80, 100);
        border-radius: 3px;
        padding: 5px 10px;
    }
    QPushButton:hover {
        background-color: rgb(70, 70, 85);
    }
""")
top_bar_layout.addStretch()
top_bar_layout.addWidget(settings_btn)
```

### 3. SettingsDialog Initialization

```python
class SettingsDialog(QDialog):
    settings_applied = pyqtSignal(dict)  # Thread-safe communication
    
    def __init__(self, parent, visualizer):
        super().__init__(parent)
        self.visualizer = visualizer
        self.setWindowTitle("Settings")
        self.setGeometry(200, 200, 500, 650)
        
        self.defaults = {
            'chunk_size': 512,
            'buffer_size': 2048,
            'update_rate': 144,
            'human_bias': 0.5,
            # ... more defaults ...
        }
        
        self.setup_ui()
        self.load_current_values()
```

### 4. Audio Settings Group Example

```python
# Device selection with tooltips
device_layout = QHBoxLayout()
device_lbl = QLabel("Device:")
device_lbl.setToolTip("Select the audio output device to capture from")
device_layout.addWidget(device_lbl)

self.device_combo = QComboBox()
self.device_combo.setToolTip("Choose which audio device to monitor")
device_layout.addWidget(self.device_combo)

refresh_btn = QPushButton("Refresh")
refresh_btn.setToolTip("Scan for available audio devices")
refresh_btn.clicked.connect(self.refresh_devices)
device_layout.addWidget(refresh_btn)
```

### 5. Validation Logic

```python
def validate_inputs(self):
    """Validate all settings before applying"""
    errors = []
    
    chunk = self.chunk_spinbox.value()
    buffer = self.buffer_spinbox.value()
    
    # Check power of 2
    if chunk & (chunk - 1) != 0:
        errors.append(f"Chunk size {chunk} is not a power of 2")
    if buffer & (buffer - 1) != 0:
        errors.append(f"Buffer size {buffer} is not a power of 2")
    
    # Check relationships
    if buffer < chunk:
        errors.append("Buffer size must be >= chunk size")
    
    if errors:
        error_msg = "Validation errors:\n" + "\n".join(f"• {e}" for e in errors)
        QMessageBox.warning(self, "Validation Error", error_msg)
        return False
    
    return True
```

### 6. Applying Settings

```python
def apply_settings(self, settings):
    """Apply settings from the dialog"""
    try:
        # Visual settings (apply immediately)
        if 'human_bias' in settings:
            self.HUMAN_BIAS = settings['human_bias']
        
        if 'update_rate' in settings:
            self.UPDATE_RATE = settings['update_rate']
            self.timer_interval = 1.0 / self.UPDATE_RATE
        
        if 'happy_mode' in settings:
            self.HAPPY_MODE = settings['happy_mode']
        
        # Audio settings (take effect next cycle)
        if 'chunk_size' in settings and settings['chunk_size'] != self.CHUNK:
            self.CHUNK = settings['chunk_size']
            print(f"Chunk size changed to {self.CHUNK}")
        
        if 'buffer_size' in settings and settings['buffer_size'] != self.FFT_SIZE:
            self.FFT_SIZE = settings['buffer_size']
            # Recalculate FFT buffers
            fft_size = self.FFT_SIZE // 2 + 1
            self.fft_data = np.full(fft_size, -80.0)
            # ... more buffer setup ...
        
    except Exception as e:
        print(f"✗ Error applying settings: {e}")
```

### 7. Device Detection

```python
def refresh_devices(self):
    """Scan and populate audio devices"""
    self.device_combo.clear()
    devices = []
    
    # Try soundcard
    if HAS_SOUNDCARD:
        try:
            for dev in sc.all_speakers():
                dev_name = getattr(dev, 'name', 'Unknown')
                dev_id = getattr(dev, 'id', None)
                devices.append((dev_name, str(dev_id) if dev_id else dev_name))
        except Exception as e:
            print(f"Error scanning soundcard devices: {e}")
    
    # Try sounddevice
    if HAS_SOUNDDEVICE:
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0 or dev['max_output_channels'] > 0:
                    dev_name = dev.get('name', f'Device {i}')
                    devices.append((dev_name, f"sd:{i}"))
        except Exception as e:
            pass
    
    # Populate combo with device (name, id) pairs
    devices.insert(0, ("Default", "default"))
    for name, device_id in devices:
        self.device_combo.addItem(name, device_id)
```

### 8. Color Palette Regeneration

```python
def regenerate_color_seed(self):
    """Generate new random seed"""
    new_seed = random.randint(0, (2**32)-1) % (2**32)
    self.seed_input.setText(str(new_seed))
```

### 9. Sending Settings to Parent

```python
def apply_settings(self):
    """Apply settings and emit signal to parent"""
    if not self.validate_inputs():
        return  # Don't apply if validation fails
    
    settings = {
        'chunk_size': self.chunk_spinbox.value(),
        'buffer_size': self.buffer_spinbox.value(),
        'update_rate': self.update_spinbox.value(),
        'human_bias': self.bias_slider.value() / 100.0,
        'happy_mode': self.happy_checkbox.isChecked(),
        'random_color': self.random_color_checkbox.isChecked(),
        'color_seed': int(self.seed_input.text()) if self.seed_input.text() else None,
        'silent': self.silent_checkbox.isChecked(),
        'debug': self.debug_checkbox.isChecked(),
        'device_id': self.device_combo.currentData(),
    }
    
    # Emit signal (thread-safe)
    self.settings_applied.emit(settings)
    
    print("✓ Settings applied successfully")
```

## Dialog Layout Structure

```
┌─────────────────────────────────────┐
│ Settings                        [X] │
├─────────────────────────────────────┤
│ ┌─ Audio Settings ────────────────┐ │
│ │ Device:        [Dropdown] [↻]  │ │
│ │ Chunk Size:    [512  ] samples │ │
│ │ Buffer Size:   [2048 ] samples │ │
│ │ Update Rate:   [144  ] Hz      │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─ Visual Settings ───────────────┐ │
│ │ Human Bias:  [slider] 50%       │ │
│ │ Num Bars:    [64] [Auto]        │ │
│ │ ☐ Happy Mode (Vibrant Colors)   │ │
│ │ ☐ Random Color Palette          │ │
│ │ Color Seed: [12345] [Regenerate]│ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─ Application Settings ──────────┐ │
│ │ ☐ Silent Mode                   │ │
│ │ ☐ Debug Mode                    │ │
│ └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│ [Reset] [Close]      [Apply]       │
└─────────────────────────────────────┘
```

## Thread Safety

The implementation uses PyQt's signal/slot mechanism for thread-safe communication:

```
User adjusts settings in dialog
    ↓
User clicks "Apply"
    ↓
Dialog validates inputs
    ↓
Dialog emits settings_applied signal
    ↓
AudioVisualizer receives signal in main thread
    ↓
apply_settings() updates visualizer attributes
    ↓
Visualization updates on next frame
```

No locks needed - Qt handles synchronization automatically.

## Error Handling Examples

### Validation Error
```
If user enters:
- Chunk size not power of 2
- Buffer < chunk
- Invalid number ranges

Dialog shows:
┌──────────────────────┐
│ Validation Error     │
├──────────────────────┤
│ Validation errors:   │
│ • Chunk size 500 is  │
│   not a power of 2   │
│ • Buffer size must   │
│   be >= chunk size   │
└──────────────────────┘
```

### Runtime Error Handling
```python
except Exception as e:
    if not self.SILENT:
        print(f"✗ Error applying settings: {e}")
    if self.DEBUG:
        import traceback
        traceback.print_exc()
```

## Settings Summary

| Setting | Type | Range | Default | Effect |
|---------|------|-------|---------|--------|
| Device | ComboBox | auto-detected | Default | Changes audio source |
| Chunk Size | SpinBox | 256-16384 | 512 | Audio latency |
| Buffer Size | SpinBox | 512-32768 | 2048 | FFT resolution |
| Update Rate | SpinBox | 1-240 Hz | 144 | Display smoothness |
| Human Bias | Slider | 0-100% | 50% | ISO 226 curve |
| Num Bars | SpinBox | 8-16384 | Auto | Spectrum detail |
| Happy Mode | Checkbox | On/Off | Off | Color scheme |
| Random Color | Checkbox | On/Off | Off | Color generation |
| Color Seed | Text | 0-2³² | Auto | Reproducibility |
| Silent Mode | Checkbox | On/Off | Off | Output suppression |
| Debug Mode | Checkbox | On/Off | Off | Debug output |
