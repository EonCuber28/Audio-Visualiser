#!/usr/bin/env python3
"""
GPU-accelerated offline renderer using an OpenGL shader pipeline.
This keeps the original scripts untouched and provides an alternate renderer.

Notes:
- Draws spectrum bars, peak holds, and oscilloscope in a shader.
- Labels/axis text are not rendered (GPU path keeps visuals clean).
- Background video is uploaded as a texture each frame for GPU composition.
"""

import argparse
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile
import ctypes

# If running headless, use offscreen rendering
if os.environ.get("DISPLAY", "") == "":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import (
    QOffscreenSurface,
    QOpenGLContext,
    QSurfaceFormat,
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLShaderProgram,
    QOpenGLShader,
    QImage,
    QVector4D,
)
from PyQt5.QtCore import Qt, QCoreApplication

from render_video import (
    ensure_wav,
    compute_iso226_boost,
    generate_color_palette,
    _extract_lab_palette,
    _fallback_palette_lab,
    _lab_to_srgb,
    _srgb_to_lab,
    _LabPaletteState,
)

try:
    import OpenGL.GL as _gl
    _HAS_PYOPENGL = True
except Exception:
    _gl = None
    _HAS_PYOPENGL = False

# Try to import scipy for cubic interpolation (matching base visualizer)
try:
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


class _InterpCache:
    def __init__(self):
        self.interp = None
        self.freqs_hash = None


class ColorLUTBuilder:
    _normal_anchors = np.array([
        [44, 123, 229],
        [121, 183, 234],
        [229, 229, 229],
        [234, 183, 155],
        [229, 83, 61],
    ], dtype=np.float32)
    _happy_anchors = np.array([
        [70, 200, 255],
        [120, 225, 255],
        [120, 255, 200],
        [255, 160, 220],
        [255, 90, 200],
    ], dtype=np.float32)
    _anchor_positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

    @staticmethod
    def _interp_anchors(lut_x, anchors):
        r = np.interp(lut_x, ColorLUTBuilder._anchor_positions, anchors[:, 0])
        g = np.interp(lut_x, ColorLUTBuilder._anchor_positions, anchors[:, 1])
        b = np.interp(lut_x, ColorLUTBuilder._anchor_positions, anchors[:, 2])
        return r, g, b

    @staticmethod
    def build_lut(mode, palette=None, size=64):
        """
        Build LUT as uint8 array [size, size, 3], indexed by:
        brightness [0..1] -> Y, balance [0..1] -> X.
        """
        lut = np.zeros((size, size, 3), dtype=np.uint8)
        for bi in range(size):
            brightness = bi / (size - 1)
            for si in range(size):
                balance = (si / (size - 1)) * 2.0 - 1.0

                if mode == "random" and palette is not None:
                    # Palette interpolation
                    if balance <= 0:
                        t = balance + 1.0
                        c0 = palette[0]
                        c1 = palette[1]
                        r = c0[0] * (1 - t) + c1[0] * t
                        g = c0[1] * (1 - t) + c1[1] * t
                        b = c0[2] * (1 - t) + c1[2] * t
                    else:
                        t = balance
                        c1 = palette[1]
                        c2 = palette[2]
                        r = c1[0] * (1 - t) + c2[0] * t
                        g = c1[1] * (1 - t) + c2[1] * t
                        b = c1[2] * (1 - t) + c2[2] * t
                    brightness_scaled = brightness ** 0.4
                    r *= brightness_scaled
                    g *= brightness_scaled
                    b *= brightness_scaled
                elif mode == "happy":
                    brightness_joy = brightness ** 0.55
                    brightness_lifted = 0.30 + (brightness_joy * 0.70)
                    lut_x = (balance + 1.0) * 0.5
                    r, g, b = ColorLUTBuilder._interp_anchors(lut_x, ColorLUTBuilder._happy_anchors)
                    r *= brightness_lifted
                    g *= brightness_lifted
                    b *= brightness_lifted
                    glow = 0.10 * brightness_lifted
                    r = min(255, r + glow * 255)
                    g = min(255, g + glow * 255)
                    b = min(255, b + glow * 255)
                else:
                    lut_x = (balance + 1.0) * 0.5
                    r, g, b = ColorLUTBuilder._interp_anchors(lut_x, ColorLUTBuilder._normal_anchors)
                    brightness_scaled = brightness ** 0.4
                    r *= brightness_scaled
                    g *= brightness_scaled
                    b *= brightness_scaled

                lut[bi, si, 0] = np.clip(r, 0, 255)
                lut[bi, si, 1] = np.clip(g, 0, 255)
                lut[bi, si, 2] = np.clip(b, 0, 255)
        return lut


class GLShaderRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setVersion(3, 3)

        self.surface = QOffscreenSurface()
        self.surface.setFormat(fmt)
        self.surface.create()

        self.context = QOpenGLContext()
        self.context.setFormat(fmt)
        self.context.create()
        self.context.makeCurrent(self.surface)

        if not _HAS_PYOPENGL:
            raise RuntimeError(
                "PyOpenGL is required for the GPU renderer in this environment. "
                "Install it with: pip install PyOpenGL"
            )
        # Use PyOpenGL bindings for GL calls
        self.gl = _gl
        self.gl.glPixelStorei(self.gl.GL_UNPACK_ALIGNMENT, 1)

        fbo_fmt = QOpenGLFramebufferObjectFormat()
        fbo_fmt.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        fbo_fmt.setTextureTarget(self.gl.GL_TEXTURE_2D)
        self.fbo = QOpenGLFramebufferObject(width, height, fbo_fmt)

        self.program = QOpenGLShaderProgram()
        self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, self._vertex_shader())
        self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, self._fragment_shader())
        if not self.program.link():
            raise RuntimeError(f"Shader link failed: {self.program.log()}")

        self._setup_quad()
        self._setup_textures()

    def _vertex_shader(self):
        return """
        #version 330 core
        layout (location = 0) in vec2 a_pos;
        layout (location = 1) in vec2 a_uv;
        out vec2 v_uv;
        void main() {
            v_uv = a_uv;
            gl_Position = vec4(a_pos, 0.0, 1.0);
        }
        """

    def _fragment_shader(self):
        return """
        #version 330 core
        in vec2 v_uv;
        out vec4 FragColor;

        uniform sampler2D u_background;
        uniform sampler2D u_bars;
        uniform sampler2D u_balance;
        uniform sampler2D u_peaks;
        uniform sampler2D u_lut;
        uniform sampler2D u_wave;

        uniform int u_numBars;
        uniform int u_barTexWidth;
        uniform int u_useVertical;
        uniform vec4 u_graphRect;
        uniform vec4 u_scopeRect;
        uniform float u_scopeCenter;
        uniform float u_opacity;
        uniform float u_peakThickness;
        uniform float u_waveThickness;
        uniform int u_debugMode;

        vec3 lutColor(float brightness, float balance01) {
            float bx = clamp(balance01, 0.0, 1.0);
            float by = clamp(brightness, 0.0, 1.0);
            float x = (bx * 63.0 + 0.5) / 64.0;
            float y = (by * 63.0 + 0.5) / 64.0;
            return texture(u_lut, vec2(x, y)).rgb;
        }

        void main() {
            vec2 uv = v_uv; // OpenGL-style bottom-left coordinates
            vec3 bg = texture(u_background, uv).rgb;
            vec3 viz = vec3(0.0);
            float alpha = 0.0;

            // Bars
            if (uv.x >= u_graphRect.x && uv.x <= (u_graphRect.x + u_graphRect.z) &&
                uv.y >= u_graphRect.y && uv.y <= (u_graphRect.y + u_graphRect.w)) {
                float gx = (uv.x - u_graphRect.x) / u_graphRect.z;
                float gy = (uv.y - u_graphRect.y) / u_graphRect.w;

                int idx;
                if (u_useVertical == 1) {
                    idx = int(floor((1.0 - gy) * float(u_numBars)));
                    idx = clamp(idx, 0, u_numBars - 1);
                } else {
                    idx = int(floor(gx * float(u_numBars)));
                    idx = clamp(idx, 0, u_numBars - 1);
                }

                float tex_x = (float(idx) + 0.5) / float(u_barTexWidth);
                float brightness = texture(u_bars, vec2(tex_x, 0.5)).r;
                float balance01 = texture(u_balance, vec2(tex_x, 0.5)).r;
                float peak = texture(u_peaks, vec2(tex_x, 0.5)).r;

                if (u_useVertical == 1) {
                    float bar_w = brightness;
                    if (gx <= bar_w) {
                        viz = lutColor(brightness, balance01);
                        alpha = 1.0;
                    }
                    float px = peak;
                    if (abs(gx - px) < u_peakThickness) {
                        viz = lutColor(brightness, balance01);
                        alpha = 1.0;
                    }
                } else {
                    float bar_h = brightness;
                    if (gy <= bar_h) {
                        viz = lutColor(brightness, balance01);
                        alpha = 1.0;
                    }
                    float py = peak;
                    if (abs(gy - py) < u_peakThickness) {
                        viz = lutColor(brightness, balance01);
                        alpha = 1.0;
                    }
                }

                // Debug mode: draw bar boundaries with thin lines
                if (u_debugMode == 1) {
                    float bar_width = 1.0 / float(u_numBars);
                    float edge_thickness = min(bar_width * 0.2, 0.0025);
                    float bar_edge_x = mod(gx, bar_width);

                    // Vertical lines at bar boundaries
                    if (bar_edge_x < edge_thickness || (bar_width - bar_edge_x) < edge_thickness) {
                        viz = vec3(1.0, 0.0, 0.0); // Red
                        alpha = 0.8;
                    }

                    // Optional horizontal guide at 50%
                    if (abs(gy - 0.5) < edge_thickness) {
                        viz = mix(viz, vec3(1.0, 0.0, 0.0), 0.5);
                        alpha = max(alpha, 0.6);
                    }
                }
            }

            // Oscilloscope (simple line)
            if (uv.x >= u_scopeRect.x && uv.x <= (u_scopeRect.x + u_scopeRect.z) &&
                uv.y >= u_scopeRect.y && uv.y <= (u_scopeRect.y + u_scopeRect.w)) {
                float sx = (uv.x - u_scopeRect.x) / u_scopeRect.z;
                int sidx = int(floor(sx * 1024.0));
                sidx = clamp(sidx, 0, 1023);
                float wx = (float(sidx) + 0.5) / 1024.0;
                float w = texture(u_wave, vec2(wx, 0.5)).r * 2.0 - 1.0;
                float y = u_scopeCenter + (w * u_scopeRect.w * 0.45);
                if (abs(uv.y - y) < u_waveThickness) {
                    viz = vec3(0.39, 0.78, 1.0);
                    alpha = 1.0;
                }
                // center line
                if (abs(uv.y - u_scopeCenter) < u_waveThickness * 0.5) {
                    viz = vec3(0.16, 0.16, 0.20);
                    alpha = 1.0;
                }
            }

            vec3 outColor = mix(bg, viz, u_opacity * alpha);
            FragColor = vec4(outColor, 1.0);
        }
        """

    def _setup_quad(self):
        self._ensure_current()
        self.vao = self.gl.glGenVertexArrays(1)
        self.vbo = self.gl.glGenBuffers(1)

        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype=np.float32)

        self.gl.glBindVertexArray(self.vao)
        self.gl.glBindBuffer(self.gl.GL_ARRAY_BUFFER, self.vbo)
        self.gl.glBufferData(self.gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, self.gl.GL_STATIC_DRAW)

        self.gl.glEnableVertexAttribArray(0)
        self.gl.glVertexAttribPointer(0, 2, self.gl.GL_FLOAT, self.gl.GL_FALSE, 4 * 4, None)
        self.gl.glEnableVertexAttribArray(1)
        self.gl.glVertexAttribPointer(1, 2, self.gl.GL_FLOAT, self.gl.GL_FALSE, 4 * 4, self._voidp(2 * 4))

        self.gl.glBindBuffer(self.gl.GL_ARRAY_BUFFER, 0)
        self.gl.glBindVertexArray(0)

    def _setup_textures(self):
        self._ensure_current()
        self.bg_tex = self._create_texture(self.width, self.height, self.gl.GL_RGB8, self.gl.GL_RGB, self.gl.GL_UNSIGNED_BYTE)
        self.bar_tex_width = 1
        self.bars_tex = self._create_texture(1, 1, self.gl.GL_R8, self.gl.GL_RED, self.gl.GL_UNSIGNED_BYTE)
        self.balance_tex = self._create_texture(1, 1, self.gl.GL_R8, self.gl.GL_RED, self.gl.GL_UNSIGNED_BYTE)
        self.peaks_tex = self._create_texture(1, 1, self.gl.GL_R8, self.gl.GL_RED, self.gl.GL_UNSIGNED_BYTE)
        self.wave_tex = self._create_texture(1024, 1, self.gl.GL_R8, self.gl.GL_RED, self.gl.GL_UNSIGNED_BYTE)
        self.lut_tex = self._create_texture(64, 64, self.gl.GL_RGB8, self.gl.GL_RGB, self.gl.GL_UNSIGNED_BYTE)

        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.lut_tex)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MIN_FILTER, self.gl.GL_NEAREST)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MAG_FILTER, self.gl.GL_NEAREST)

        # Initialize background to dark blue to avoid undefined texture data
        solid = np.full((self.height, self.width, 3), [20, 20, 30], dtype=np.uint8)
        self.update_background(solid.tobytes())

    def _create_texture(self, w, h, internal_format, fmt, dtype):
        self._ensure_current()
        tex = self.gl.glGenTextures(1)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, tex)
        self.gl.glTexImage2D(self.gl.GL_TEXTURE_2D, 0, internal_format, w, h, 0, fmt, dtype, None)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MIN_FILTER, self.gl.GL_LINEAR)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MAG_FILTER, self.gl.GL_LINEAR)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_WRAP_S, self.gl.GL_CLAMP_TO_EDGE)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_WRAP_T, self.gl.GL_CLAMP_TO_EDGE)
        return tex

    def _voidp(self, offset):
        return ctypes.c_void_p(offset)

    def _ensure_current(self):
        if self.context is not None and self.surface is not None:
            self.context.makeCurrent(self.surface)

    def update_background(self, rgb_bytes):
        self._ensure_current()
        self.gl.glPixelStorei(self.gl.GL_UNPACK_ALIGNMENT, 1)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.bg_tex)
        self.gl.glTexSubImage2D(self.gl.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, self.gl.GL_RGB, self.gl.GL_UNSIGNED_BYTE, rgb_bytes)

    def update_bars(self, brightness, balance01, peaks, num_bars):
        self._ensure_current()
        self.gl.glPixelStorei(self.gl.GL_UNPACK_ALIGNMENT, 1)
        self.num_bars = num_bars
        b_u8 = np.clip(brightness * 255.0, 0, 255).astype(np.uint8)
        bal_u8 = np.clip(balance01 * 255.0, 0, 255).astype(np.uint8)
        p_u8 = np.clip(peaks * 255.0, 0, 255).astype(np.uint8)
        import sys
        if hasattr(sys, '_debug_frame_idx') and sys._debug_frame_idx in [0, 1, 2, 3]:
            print(
                f"[GL DEBUG BARS] Frame {sys._debug_frame_idx}: "
                f"b_array_len={len(b_u8)} b0-20={b_u8[0:20].tolist()} "
                f"b_minmax=({b_u8.min()}, {b_u8.max()}) b_nz={np.count_nonzero(b_u8)} "
                f"bar_tex_width_before={self.bar_tex_width} num_bars={num_bars}"
            )
        if self.bar_tex_width != num_bars:
            self.bar_tex_width = num_bars
            self.bars_tex = self._create_texture(num_bars, 1, self.gl.GL_R8, self.gl.GL_RED, self.gl.GL_UNSIGNED_BYTE)
            self.balance_tex = self._create_texture(num_bars, 1, self.gl.GL_R8, self.gl.GL_RED, self.gl.GL_UNSIGNED_BYTE)
            self.peaks_tex = self._create_texture(num_bars, 1, self.gl.GL_R8, self.gl.GL_RED, self.gl.GL_UNSIGNED_BYTE)
            # For 1D bar textures, use nearest filtering to avoid horizontal interpolation/smear
            for tex in (self.bars_tex, self.balance_tex, self.peaks_tex):
                self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, tex)
                self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MIN_FILTER, self.gl.GL_NEAREST)
                self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MAG_FILTER, self.gl.GL_NEAREST)
        b_full = b_u8
        bal_full = bal_u8
        p_full = p_u8
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.bars_tex)
        self.gl.glTexSubImage2D(self.gl.GL_TEXTURE_2D, 0, 0, 0, self.bar_tex_width, 1, self.gl.GL_RED, self.gl.GL_UNSIGNED_BYTE, b_full)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.balance_tex)
        self.gl.glTexSubImage2D(self.gl.GL_TEXTURE_2D, 0, 0, 0, self.bar_tex_width, 1, self.gl.GL_RED, self.gl.GL_UNSIGNED_BYTE, bal_full)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.peaks_tex)
        self.gl.glTexSubImage2D(self.gl.GL_TEXTURE_2D, 0, 0, 0, self.bar_tex_width, 1, self.gl.GL_RED, self.gl.GL_UNSIGNED_BYTE, p_full)

    def update_wave(self, wave):
        self._ensure_current()
        self.gl.glPixelStorei(self.gl.GL_UNPACK_ALIGNMENT, 1)
        w_u8 = np.clip((wave * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.wave_tex)
        self.gl.glTexSubImage2D(self.gl.GL_TEXTURE_2D, 0, 0, 0, 1024, 1, self.gl.GL_RED, self.gl.GL_UNSIGNED_BYTE, w_u8)

    def update_lut(self, lut):
        self._ensure_current()
        self.gl.glPixelStorei(self.gl.GL_UNPACK_ALIGNMENT, 1)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.lut_tex)
        self.gl.glTexSubImage2D(self.gl.GL_TEXTURE_2D, 0, 0, 0, 64, 64, self.gl.GL_RGB, self.gl.GL_UNSIGNED_BYTE, lut)

    def render(self, graph_rect, scope_rect, scope_center, opacity, use_vertical, debug_mode=False):
        self._ensure_current()
        self.fbo.bind()
        self.gl.glViewport(0, 0, self.width, self.height)
        self.gl.glDisable(self.gl.GL_DEPTH_TEST)
        self.gl.glDisable(self.gl.GL_BLEND)
        self.gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT)

        self.program.bind()
        self.program.setUniformValue("u_background", 0)
        self.program.setUniformValue("u_bars", 1)
        self.program.setUniformValue("u_balance", 2)
        self.program.setUniformValue("u_peaks", 3)
        self.program.setUniformValue("u_lut", 4)
        self.program.setUniformValue("u_wave", 5)
        self.program.setUniformValue("u_numBars", int(self.num_bars))
        self.program.setUniformValue("u_barTexWidth", int(self.bar_tex_width))
        self.program.setUniformValue("u_useVertical", int(use_vertical))
        self.program.setUniformValue("u_graphRect", QVector4D(*graph_rect))
        self.program.setUniformValue("u_scopeRect", QVector4D(*scope_rect))
        self.program.setUniformValue("u_scopeCenter", float(scope_center))
        self.program.setUniformValue("u_opacity", float(opacity))
        self.program.setUniformValue("u_peakThickness", float(1.0 / max(1.0, self.height)))
        self.program.setUniformValue("u_waveThickness", float(1.0 / max(1.0, self.height)))
        self.program.setUniformValue("u_debugMode", int(1 if debug_mode else 0))

        self.gl.glActiveTexture(self.gl.GL_TEXTURE0)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.bg_tex)
        self.gl.glActiveTexture(self.gl.GL_TEXTURE1)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.bars_tex)
        self.gl.glActiveTexture(self.gl.GL_TEXTURE2)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.balance_tex)
        self.gl.glActiveTexture(self.gl.GL_TEXTURE3)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.peaks_tex)
        self.gl.glActiveTexture(self.gl.GL_TEXTURE4)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.lut_tex)
        self.gl.glActiveTexture(self.gl.GL_TEXTURE5)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.wave_tex)

        self.gl.glBindVertexArray(self.vao)
        self.gl.glDrawArrays(self.gl.GL_TRIANGLE_STRIP, 0, 4)
        self.gl.glBindVertexArray(0)

        self.program.release()
        self.fbo.release()

        self.gl.glFinish()

        return self.fbo.toImage()


def compute_bar_data(freqs, fft_db, stereo_balance, width, height, interp_cache, peak_hold, happy_mode, channels, bar_cap=None, tight_layout=True):
    # Layout + margins (match base visualizer proportions)
    use_vertical = width < 400
    oscilloscope_height = 70
    if tight_layout:
        margin_left, margin_right, margin_top, margin_bottom = 10, 10, 10, 10
    else:
        if use_vertical:
            margin_left, margin_right, margin_top, margin_bottom = 40, 10, 3, 5
        else:
            margin_left, margin_right, margin_top, margin_bottom = 60, 10, 10, 20
    margin_bottom += oscilloscope_height

    graph_width = max(1, width - margin_left - margin_right)
    graph_height = max(1, height - margin_top - margin_bottom)

    if use_vertical:
        num_bars = min(256, graph_height // 3)
    else:
        num_bars = min(512, graph_width // 3)
    if bar_cap is not None:
        num_bars = min(num_bars, int(bar_cap))
    num_bars = max(1, int(num_bars))

    mask = (freqs >= 10.0) & (freqs <= 15000)
    freqs_filtered = freqs[mask]
    data_filtered = fft_db[mask]

    if len(data_filtered) == 0:
        return None

    min_freq = max(freqs_filtered[0], 10.0)
    bar_freqs = np.logspace(np.log10(min_freq), np.log10(10000.0), num_bars)

    smooth_mask = bar_freqs <= 1000
    data_interpolated = np.zeros_like(bar_freqs)

    if HAS_SCIPY and np.any(smooth_mask):
        freqs_hash = (len(freqs_filtered), freqs_filtered[0], freqs_filtered[-1])
        if interp_cache.interp is None or interp_cache.freqs_hash != freqs_hash:
            interp_cache.interp = interp1d(
                freqs_filtered, data_filtered, kind='cubic', fill_value='extrapolate', assume_sorted=True
            )
            interp_cache.freqs_hash = freqs_hash
        data_interpolated[smooth_mask] = interp_cache.interp(bar_freqs[smooth_mask])
        if np.any(~smooth_mask):
            data_interpolated[~smooth_mask] = np.interp(bar_freqs[~smooth_mask], freqs_filtered, data_filtered)
    else:
        data_interpolated = np.interp(bar_freqs, freqs_filtered, data_filtered)

    stereo_balance_interpolated = np.interp(bar_freqs, freqs_filtered, stereo_balance[mask])
    import sys
    if hasattr(sys, "_debug_frame_idx") and sys._debug_frame_idx in [0, 1, 2]:
        print(f"[FFT DEBUG] Frame {sys._debug_frame_idx}: data_interp min={data_interpolated.min():.1f}, max={data_interpolated.max():.1f}, median={np.median(data_interpolated):.1f}")

    # Brightness mapping: softer knee to avoid overfilling (clipping) while keeping low bars visible.
    if happy_mode:
        brightness = np.clip((data_interpolated + 70.0) / 50.0, 0.0, 1.0)
    else:
        brightness = np.clip((data_interpolated + 100.0) / 80.0, 0.0, 1.0)
    # Soft-compress highs so bars don’t shoot to the top; keep a small floor for visibility.
    brightness = np.power(brightness, 0.65)
    brightness = np.maximum(brightness, 0.03)

    if channels == 1:
        freq_ratios = np.arange(num_bars) / max(1, num_bars)
        freq_brightness_boost = 0.4 + (freq_ratios * 0.6)
        brightness = np.clip(brightness * freq_brightness_boost, 0.0, 1.0)

    if peak_hold is None or len(peak_hold) != num_bars:
        peak_hold = np.zeros(num_bars, dtype=np.float32)

    peak_hold = np.maximum(brightness, np.maximum(0, peak_hold - 0.006))

    # Fallback disabled: brightness scaling should handle this
    # if np.max(brightness) <= 1e-6:
    #     brightness[:] = 0.02
    #     peak_hold[:] = np.maximum(peak_hold, brightness)

    balance01 = np.clip((stereo_balance_interpolated + 1.0) * 0.5, 0.0, 1.0)
    brightness = np.nan_to_num(brightness, nan=0.0, posinf=1.0, neginf=0.0)
    balance01 = np.nan_to_num(balance01, nan=0.5, posinf=1.0, neginf=0.0)
    peak_hold = np.nan_to_num(peak_hold, nan=0.0, posinf=1.0, neginf=0.0)

    graph_x = margin_left / width
    graph_y_top = margin_top / height
    graph_h = graph_height / height
    graph_y = 1.0 - graph_y_top - graph_h
    graph_rect = (
        graph_x,
        graph_y,
        graph_width / width,
        graph_h,
    )

    scope_height = 50
    scope_left = 10
    scope_width = width - 20
    scope_top = height - 70 + 15
    scope_y_top = scope_top / height
    scope_h = scope_height / height
    scope_y = 1.0 - scope_y_top - scope_h
    scope_rect = (
        scope_left / width,
        scope_y,
        scope_width / width,
        scope_h,
    )
    scope_center = scope_rect[1] + scope_rect[3] * 0.5

    return {
        "use_vertical": use_vertical,
        "num_bars": int(num_bars),
        "brightness": brightness.astype(np.float32),
        "balance01": balance01.astype(np.float32),
        "peaks": peak_hold.astype(np.float32),
        "graph_rect": graph_rect,
        "scope_rect": scope_rect,
        "scope_center": scope_center,
        "peak_hold": peak_hold,
    }


def render_video_gl(
    input_audio,
    output_path,
    width,
    height,
    fps,
    fft_size,
    chunk_size,
    human_bias,
    happy_mode,
    random_color,
    color_seed,
    debug,
    silent,
    video_encoder,
    video_preset,
    video_crf,
    video_bitrate,
    use_opengl,
    bar_cap,
    nvenc_cq,
    nvenc_rc,
    background_video,
    viz_opacity,
    bg_color_smooth,
    bg_color_sample_step,
):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to render video. Install ffmpeg first.")

    if use_opengl and QApplication.instance() is None:
        QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL)

    app = QApplication.instance() or QApplication([])

    renderer = GLShaderRenderer(width, height)

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = ensure_wav(input_audio, temp_dir)
        rate, data = wavfile.read(audio_path)

        if data.ndim == 1:
            channels = 1
            data = data.astype(np.int16)
        else:
            channels = data.shape[1]
            if channels > 2:
                data = data[:, :2]
                channels = 2

        if data.dtype != np.int16:
            data = data.astype(np.int16)

        total_samples = data.shape[0]
        duration = total_samples / rate
        total_frames = int(math.ceil(duration * fps))

        if not silent:
            print(f"Audio: {rate} Hz, channels={channels}, duration={duration:.2f}s")
            print(f"Rendering {total_frames} frames at {fps} FPS -> {width}x{height}")

        freqs = np.fft.rfftfreq(fft_size, 1.0 / rate)
        hamming = np.hamming(fft_size).astype(np.float32)
        window_correction = np.sqrt(fft_size / np.sum(hamming**2))
        # Normalize FFT magnitude: buffer is int16, so scale reference to fft_size instead of fft_size*32768
        norm_factor = float(fft_size)

        use_hearing_bias = (not happy_mode) and (human_bias > 0)
        if use_hearing_bias:
            iso226_freqs, iso226_boost = compute_iso226_boost()
            hearing_correction = np.interp(
                freqs,
                iso226_freqs,
                iso226_boost,
                left=iso226_boost[0],
                right=iso226_boost[-1],
            )

        fft_db_state = np.full(fft_size // 2 + 1, -80.0, dtype=np.float32)
        stereo_balance_state = np.zeros(fft_size // 2 + 1, dtype=np.float32)

        # Background video decoder
        bg_process = None
        bg_frame_size = width * height * 3
        palette_state = _LabPaletteState()
        bg_luma_smoothed = None
        interp_cache = _InterpCache()
        peak_hold = None
        lut_mode = None
        lut_palette_key = None

        if background_video:
            if not os.path.isfile(background_video):
                raise RuntimeError(f"Background video not found: {background_video}")
            bg_cmd = [
                "ffmpeg",
                "-stream_loop",
                "-1",
                "-i",
                background_video,
                "-an",
                "-vf",
                f"scale={width}:{height},fps={fps}",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            ]
            bg_process = subprocess.Popen(bg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # ffmpeg output
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-i",
            audio_path,
        ]

        encoder = video_encoder or "libx264"
        if encoder == "h264_nvenc" and not video_preset:
            video_preset = "p4"
        if encoder == "libx264" and not video_preset:
            video_preset = "medium"

        if encoder == "libx264":
            cmd += ["-c:v", "libx264", "-preset", str(video_preset)]
            if video_crf is not None:
                cmd += ["-crf", str(video_crf)]
        elif encoder == "h264_nvenc":
            cmd += ["-c:v", "h264_nvenc", "-preset", str(video_preset)]
            if video_bitrate:
                cmd += ["-b:v", str(video_bitrate)]
            else:
                rc_mode = nvenc_rc or "vbr_hq"
                cq_value = nvenc_cq if nvenc_cq is not None else 19
                cmd += ["-rc", str(rc_mode), "-cq", str(cq_value), "-b:v", "0"]
        else:
            raise RuntimeError(f"Unsupported video encoder: {encoder}")

        cmd += [
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-c:a",
            "aac",
            "-shortest",
            output_path,
        ]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        samples_per_frame = int(rate / fps)
        oscilloscope_samples = 1024

        # Initialize LUT
        last_bg_bytes = None

        if random_color:
            lut_mode = "random"
        elif happy_mode:
            lut_mode = "happy"
        else:
            lut_mode = "normal"
        if lut_mode == "random" and not background_video:
            palette = generate_color_palette(color_seed, debug)
            lut = ColorLUTBuilder.build_lut(lut_mode, palette=palette)
            lut_palette_key = tuple(map(tuple, palette))
        else:
            lut = ColorLUTBuilder.build_lut(lut_mode, palette=None)
        renderer.update_lut(lut)

        for frame_idx in range(total_frames):
            import sys
            sys._debug_frame_idx = frame_idx
            end = min(total_samples, (frame_idx + 1) * samples_per_frame)
            start = end - fft_size
            if start < 0:
                pad = -start
                start = 0
            else:
                pad = 0

            if channels == 1:
                frame_samples = data[start:end]
                if pad > 0:
                    frame_samples = np.pad(frame_samples, (pad, 0), mode="constant")
                buffer = frame_samples.astype(np.float32) / 32768.0

                windowed = buffer * hamming
                fft_result = np.fft.rfft(windowed)
                fft_magnitude = np.abs(fft_result) * window_correction
                stereo_balance = np.zeros_like(fft_db_state)
            else:
                frame_samples = data[start:end, :2]
                if pad > 0:
                    frame_samples = np.pad(frame_samples, ((pad, 0), (0, 0)), mode="constant")
                left = frame_samples[:, 0].astype(np.float32) / 32768.0
                right = frame_samples[:, 1].astype(np.float32) / 32768.0
                buffer = (left + right) * 0.5

                windowed_left = left * hamming
                windowed_right = right * hamming
                FL = np.fft.rfft(windowed_left)
                FR = np.fft.rfft(windowed_right)
                ML = np.abs(FL)
                MR = np.abs(FR)
                fft_magnitude = np.hypot(ML, MR) * window_correction

                sum_mag = MR + ML + 1e-6
                stereo_balance = np.clip((MR - ML) / sum_mag, -1.0, 1.0)

            fft_db = 20 * np.log10(fft_magnitude / norm_factor + 1e-10)
            if not happy_mode:
                if use_hearing_bias:
                    fft_db -= hearing_correction * human_bias
                fft_db = np.clip(fft_db, -80, 0)
            else:
                fft_db = np.clip(fft_db, -60, 0)

            # Smoothing
            alpha = 0.3
            fft_db_state = alpha * fft_db + (1 - alpha) * fft_db_state
            alpha_balance = 0.4
            stereo_balance_state = alpha_balance * stereo_balance + (1 - alpha_balance) * stereo_balance_state

            # Oscilloscope
            wave = buffer[-oscilloscope_samples:] / 32768.0
            if len(wave) < oscilloscope_samples:
                wave = np.pad(wave, (oscilloscope_samples - len(wave), 0), mode="constant")

            bar_data = compute_bar_data(
                freqs,
                fft_db_state,
                stereo_balance_state,
                width,
                height,
                interp_cache,
                peak_hold,
                happy_mode,
                channels,
                bar_cap=bar_cap,
                tight_layout=True,
            )
            if bar_data is None:
                continue

            # Update bar textures immediately after computing bar data, before any rendering
            renderer.update_bars(
                bar_data["brightness"],
                bar_data["balance01"],
                bar_data["peaks"],
                bar_data["num_bars"],
            )
            if debug and frame_idx <= 8:
                gx, gy, gw, gh = bar_data["graph_rect"]
                b = bar_data["brightness"]
                print(
                    f"[GL DEBUG FRAME {frame_idx}] num_bars={bar_data['num_bars']} bar_tex_width={renderer.bar_tex_width} "
                    f"b_len={len(b)} b_minmax=({b.min():.3f},{b.max():.3f}) b_nz={np.count_nonzero(b)} "
                    f"graph_rect=({gx:.6f},{gy:.6f},{gw:.6f},{gh:.6f})"
                )

            if debug and frame_idx == 0:
                print(
                    "[GL DEBUG] size=", (width, height),
                    "num_bars=", bar_data["num_bars"],
                    "graph_rect=", bar_data["graph_rect"],
                    "scope_rect=", bar_data["scope_rect"],
                )
                print(f"[GL DEBUG] bar_tex_width={renderer.bar_tex_width}, u_numBars={bar_data['num_bars']}")
                graph_x, graph_y, graph_w, graph_h = bar_data["graph_rect"]
                print(f"[GL DEBUG] graph normalized coords: x={graph_x:.6f}, y={graph_y:.6f}, w={graph_w:.6f}, h={graph_h:.6f}")
                print(f"[GL DEBUG] graph pixel coords: x={graph_x*width:.1f}, y={graph_y*height:.1f}, w={graph_w*width:.1f}, h={graph_h*height:.1f}")
                print(f"[GL DEBUG] bar index calculation: gx*num_bars samples {bar_data['num_bars']} bars across graph width")
                print(f"[GL DEBUG] expected bar width pixels: {(graph_w*width)/bar_data['num_bars']:.2f}px per bar")
                
                # Sample some bar calculations
                print("[GL DEBUG] Sample bar mappings (pixel x -> bar index):")
                for px in [0, width//4, width//2, 3*width//4, width-1]:
                    norm_x = px / width
                    if norm_x >= graph_x and norm_x <= (graph_x + graph_w):
                        gx = (norm_x - graph_x) / graph_w
                        bar_idx = int(gx * bar_data['num_bars'])
                        bar_idx = min(bar_idx, bar_data['num_bars']-1)
                        print(f"  x={px}px (norm={norm_x:.4f}) -> gx={gx:.4f} -> bar_idx={bar_idx}")


            peak_hold = bar_data["peak_hold"]

            # Background frame
            if bg_process is not None and bg_process.stdout is not None:
                bg_bytes = bg_process.stdout.read(bg_frame_size)
                if len(bg_bytes) != bg_frame_size:
                    try:
                        bg_process.kill()
                    except Exception:
                        pass
                    bg_process = subprocess.Popen(bg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    bg_bytes = bg_process.stdout.read(bg_frame_size)

                if len(bg_bytes) == bg_frame_size:
                    last_bg_bytes = bg_bytes
                    renderer.update_background(bg_bytes)
                    if random_color:
                        try:
                            arr = np.frombuffer(bg_bytes, dtype=np.uint8).reshape((height, width, 3))
                            palette_lab, luma = _extract_lab_palette(
                                arr,
                                palette_state,
                                sample_step=bg_color_sample_step,
                                smooth_strength=bg_color_smooth,
                            )
                        except Exception:
                            luma = 0.5
                            palette_lab = None

                        if palette_lab is None:
                            mean_rgb = arr.mean(axis=(0, 1)) if "arr" in locals() else np.array([96, 96, 112])
                            bg_mean_lab = _srgb_to_lab(mean_rgb.astype(np.float32) / 255.0)
                            palette_lab = _fallback_palette_lab(bg_mean_lab, prev_palette=None)

                        if palette_lab is not None:
                            rgb_palette = []
                            for lab in palette_lab:
                                srgb = _lab_to_srgb(lab)
                                rgb = np.clip(srgb * 255.0, 0, 255).astype(np.int32)
                                rgb_palette.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))

                            palette_key = tuple(map(tuple, rgb_palette))
                            if lut_palette_key != palette_key:
                                lut = ColorLUTBuilder.build_lut("random", palette=rgb_palette)
                                renderer.update_lut(lut)
                                lut_palette_key = palette_key

                        if bg_luma_smoothed is None:
                            bg_luma_smoothed = luma
                        else:
                            bg_luma_smoothed = 0.9 * bg_luma_smoothed + 0.1 * luma
                else:
                    # If frame read fails, keep last good background to avoid flashes
                    if last_bg_bytes is not None:
                        renderer.update_background(last_bg_bytes)
            else:
                solid = np.full((height, width, 3), [20, 20, 30], dtype=np.uint8)
                renderer.update_background(solid.tobytes())

            adaptive_factor = 1.0
            if bg_process is not None and bg_luma_smoothed is not None:
                adaptive_factor = 0.6 + 0.8 * bg_luma_smoothed

            opacity = float(viz_opacity) * adaptive_factor
            opacity = max(0.0, min(1.0, opacity))

            renderer.update_wave(wave.astype(np.float32))

            image = renderer.render(
                bar_data["graph_rect"],
                bar_data["scope_rect"],
                bar_data["scope_center"],
                opacity,
                bar_data["use_vertical"],
                debug_mode=(debug and frame_idx <= 5),
            )

            if image.width() != width or image.height() != height:
                image = image.scaled(width, height)

            image_rgb = image.convertToFormat(QImage.Format_RGB888)
            ptr = image_rgb.bits()
            ptr.setsize(image_rgb.byteCount())
            process.stdin.write(ptr)

            if not silent and frame_idx % max(1, int(fps)) == 0:
                print(f"Rendered {frame_idx + 1}/{total_frames} frames {frame_idx / total_frames * 100:.1f}% ")

        process.stdin.close()
        process.wait()

        if bg_process is not None:
            try:
                bg_process.kill()
            except Exception:
                pass

    if process.returncode != 0:
        raise RuntimeError("ffmpeg failed to encode video")


def main():
    parser = argparse.ArgumentParser(description="Render audio visualizer to video (GPU shader pipeline)")
    parser.add_argument("--audio", required=True, help="Input audio file (.wav recommended)")
    parser.add_argument(
        "--background-video",
        default=None,
        help="Optional background video file (visualizer will be composited on top)",
    )
    parser.add_argument("--output", required=True, help="Output video file (e.g., out.mp4)")
    parser.add_argument("--width", type=int, default=1920, help="Video width")
    parser.add_argument("--height", type=int, default=1080, help="Video height")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    parser.add_argument(
        "--update-rate",
        type=int,
        default=None,
        help="Visualizer update rate in Hz (alias for --fps in offline render).",
    )
    parser.add_argument("--buffer-size", type=int, default=2048, help="FFT buffer size")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size (unused for offline render)")
    parser.add_argument(
        "--human-bias",
        type=float,
        default=0.5,
        help="ISO 226 equal-loudness influence (0-1)",
    )
    parser.add_argument("--happy", action="store_true", help="Enable happy color mode")
    parser.add_argument("--random-color", action="store_true", help="Enable random palette")
    parser.add_argument("--color-seed", type=int, default=None, help="Random palette seed")
    parser.add_argument(
        "--video-encoder",
        default="libx264",
        help="Video encoder (libx264 or h264_nvenc). NVENC uses CQ 1-51 when bitrate is omitted (lower = higher quality).",
    )
    parser.add_argument(
        "--video-preset",
        default=None,
        help="Encoder preset (x264: ultrafast..veryslow, nvenc: p1..p7)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="Constant quality for libx264: 0-51 (lower = higher quality, typical 16-23)",
    )
    parser.add_argument(
        "--video-bitrate",
        default=None,
        help="Target bitrate for NVENC (e.g. 8M). If omitted, uses CQ mode.",
    )
    parser.add_argument(
        "--nvenc-cq",
        type=int,
        default=None,
        help="NVENC constant quality value (lower = higher quality).",
    )
    parser.add_argument(
        "--nvenc-rc",
        default=None,
        help="NVENC rate control mode (e.g., vbr_hq, vbr, cbr).",
    )
    parser.add_argument(
        "--use-opengl",
        action="store_true",
        help="Use a desktop OpenGL context when available.",
    )
    parser.add_argument(
        "--bar-cap",
        type=int,
        default=None,
        help="Max number of bars to render (lower = faster).",
    )
    parser.add_argument(
        "--viz-opacity",
        type=float,
        default=0.3,
        help="Visualizer opacity (0-1, default 0.3).",
    )
    parser.add_argument(
        "--bg-color-smooth",
        type=int,
        default=10,
        help="Frames to smooth background color over (default 10).",
    )
    parser.add_argument(
        "--bg-color-sample-step",
        type=int,
        default=4,
        help="Pixel sampling step for background color (lower = higher sampling rate, default 4).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--silent", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    if args.human_bias < 0 or args.human_bias > 1:
        parser.error("--human-bias must be between 0.0 and 1.0")
    if args.buffer_size < 512 or args.buffer_size > 32768:
        parser.error("--buffer-size must be between 512 and 32768")
    if args.fps < 1 or args.fps > 240:
        parser.error("--fps must be between 1 and 240")
    if args.update_rate is not None and (args.update_rate < 1 or args.update_rate > 240):
        parser.error("--update-rate must be between 1 and 240")
    if args.viz_opacity < 0 or args.viz_opacity > 1:
        parser.error("--viz-opacity must be between 0 and 1")
    if args.bg_color_smooth < 1 or args.bg_color_smooth > 120:
        parser.error("--bg-color-smooth must be between 1 and 120")
    if args.bg_color_sample_step < 1 or args.bg_color_sample_step > 64:
        parser.error("--bg-color-sample-step must be between 1 and 64")

    resolved_fps = args.fps
    if args.update_rate is not None:
        resolved_fps = args.update_rate

    render_video_gl(
        input_audio=args.audio,
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=resolved_fps,
        fft_size=args.buffer_size,
        chunk_size=args.chunk_size,
        human_bias=args.human_bias,
        happy_mode=args.happy,
        random_color=args.random_color,
        color_seed=args.color_seed,
        debug=args.debug,
        silent=args.silent,
        video_encoder=args.video_encoder,
        video_preset=args.video_preset,
        video_crf=args.crf if args.video_encoder == "libx264" else None,
        video_bitrate=args.video_bitrate,
        use_opengl=args.use_opengl,
        bar_cap=args.bar_cap,
        nvenc_cq=args.nvenc_cq,
        nvenc_rc=args.nvenc_rc,
        background_video=args.background_video,
        viz_opacity=args.viz_opacity,
        bg_color_smooth=args.bg_color_smooth,
        bg_color_sample_step=args.bg_color_sample_step,
    )


if __name__ == "__main__":
    main()
