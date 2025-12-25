#!/usr/bin/env python3
"""
Setup script for Audio Frequency Visualizer
"""

from setuptools import setup, find_packages
import sys
import platform

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Base requirements (cross-platform)
install_requires = [
    "numpy>=1.21.0",
    "PyQt5>=5.15.0",
]

# Optional requirements
extras_require = {
    'scipy': ['scipy>=1.7.0'],  # Better interpolation for bass frequencies
    'windows': ['pyaudio>=0.2.11'] if platform.system() == 'Windows' else [],
}

# Add platform-specific requirements to base install
if platform.system() == 'Windows':
    install_requires.append('pyaudio>=0.2.11')

setup(
    name="audio-frequency-visualizer",
    version="1.0.0",
    author="Isaac Clark + MS Copilot",
    description="Real-time audio frequency analyzer with FFT visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/audio-visualizer",
    py_modules=["audio_visualizer"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Microsoft :: Windows :: Windows 11",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "audio-visualizer=audio_visualizer:main",
        ],
    },
    keywords="audio visualizer fft frequency spectrum analyzer real-time",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/audio-visualizer/issues",
        "Source": "https://github.com/yourusername/audio-visualizer",
    },
)
