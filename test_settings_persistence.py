#!/usr/bin/env python3
"""Test script to verify settings persistence"""

from audio_visualizer import get_config_dir, get_settings_file, load_settings, save_settings
import json

print("=" * 60)
print("Settings Persistence Test")
print("=" * 60)

# Show config location
config_dir = get_config_dir()
settings_file = get_settings_file()

print(f"\n✓ Config directory: {config_dir}")
print(f"✓ Settings file: {settings_file}")

# Test saving settings
test_settings = {
    'chunk_size': 1024,
    'buffer_size': 4096,
    'update_rate': 120,
    'human_bias': 0.75,
    'num_bars': 128,
    'happy_mode': True,
    'random_color': False,
    'color_seed': 12345,
    'silent': False,
    'debug': True,
    'device_id': 'default'
}

print("\n--- Saving test settings ---")
save_settings(test_settings)
print("✓ Settings saved")

# Verify file exists
if settings_file.exists():
    print(f"✓ Settings file exists: {settings_file}")
    
    # Read file directly to verify content
    with open(settings_file, 'r') as f:
        file_content = json.load(f)
    
    print("\n--- File content ---")
    print(json.dumps(file_content, indent=2))
else:
    print(f"✗ Settings file NOT found: {settings_file}")

# Test loading settings
print("\n--- Loading settings ---")
loaded_settings = load_settings()
print(f"✓ Loaded {len(loaded_settings)} settings")

# Verify settings match
print("\n--- Verification ---")
all_match = True
for key, value in test_settings.items():
    loaded_value = loaded_settings.get(key)
    match = loaded_value == value
    symbol = "✓" if match else "✗"
    print(f"{symbol} {key}: saved={value}, loaded={loaded_value}")
    if not match:
        all_match = False

if all_match:
    print("\n✓✓✓ All settings saved and loaded correctly! ✓✓✓")
else:
    print("\n✗✗✗ Some settings did not match! ✗✗✗")

print("\n" + "=" * 60)
