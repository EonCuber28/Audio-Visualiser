#!/usr/bin/env python3
"""
Offline renderer: render the visualizer to a video file using a source audio file.
Requires ffmpeg in PATH.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile

# If running headless, use offscreen rendering
if os.environ.get("DISPLAY", "") == "":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QImage, QPainter, QColor, QSurfaceFormat
from PyQt5.QtCore import Qt, QCoreApplication

from audio_visualiser.app import VisualizerCanvas


class RenderContext:
    def __init__(
        self,
        rate,
        fft_size,
        channels,
        happy_mode=False,
        random_color=False,
        color_seed=None,
        debug=False,
    ):
        self.RATE = rate
        self.FFT_SIZE = fft_size
        self.CHANNELS = channels
        self.HAPPY_MODE = happy_mode
        self.RANDOM_COLOR = random_color
        self.DEBUG = debug
        self.color_palette = None
        self.color_palette_oklab = None
        self.palette_mode = "rgb"
        self._palette_seed = None
        if random_color:
            self.color_palette = generate_color_palette(color_seed, debug)
            self._palette_seed = _PALETTE_SEED


_PALETTE_SEED = None

# QUICK WIN PARAMETERS - color energy tuning
COLOR_BOOST_PARAMS = {
    "saturation_boost": 1.8,
    "center_chroma_boost": 1.4,
    "pan_hue_range": 0.4,
    "pan_chroma_boost": 1.6,
    "loudness_impact": 0.5,
}


def _hsl_to_rgb(h, s, l):
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


def _rgb_to_hsl(r, g, b):
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    max_v = max(r, g, b)
    min_v = min(r, g, b)
    l = (max_v + min_v) * 0.5
    if max_v == min_v:
        return 0.0, 0.0, l
    d = max_v - min_v
    s = d / (2.0 - max_v - min_v) if l > 0.5 else d / (max_v + min_v)
    if max_v == r:
        h = ((g - b) / d + (6 if g < b else 0)) / 6.0
    elif max_v == g:
        h = ((b - r) / d + 2) / 6.0
    else:
        h = ((r - g) / d + 4) / 6.0
    return h % 1.0, s, l


def generate_color_palette(seed=None, debug=False):
    """Generate random color palette using cosine-like HSL palette."""
    global _PALETTE_SEED
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    else:
        seed = int(seed) % (2**32)

    _PALETTE_SEED = seed
    rng = np.random.RandomState(seed)

    hue_center = rng.uniform(0.0, 1.0)
    hue_separation = 0.25 + 0.1 * rng.uniform(-1.0, 1.0)
    hue_left = (hue_center - hue_separation) % 1.0
    hue_right = (hue_center + hue_separation) % 1.0

    saturation = 0.6 + 0.3 * rng.uniform(0.0, 1.0)
    luminance = 0.65 + 0.25 * rng.uniform(0.0, 1.0)

    if debug:
        print(
            f"Palette: L={hue_left:.3f}, C={hue_center:.3f}, R={hue_right:.3f}, "
            f"sat={saturation:.3f}, lum={luminance:.3f}"
        )

    return [
        _hsl_to_rgb(hue_left, saturation, luminance),
        _hsl_to_rgb(hue_center, saturation, luminance),
        _hsl_to_rgb(hue_right, saturation, luminance),
    ]


def generate_palette_from_center_rgb(center_rgb, seed=None, debug=False):
    """Generate L/C/R palette using the same color generator logic, seeded by a base color."""
    global _PALETTE_SEED
    if seed is None:
        seed = int(center_rgb[0]) << 16 | int(center_rgb[1]) << 8 | int(center_rgb[2])
    else:
        seed = int(seed) % (2**32)

    _PALETTE_SEED = seed
    rng = np.random.RandomState(seed)

    hue_center, saturation, luminance = _rgb_to_hsl(*center_rgb)
    hue_separation = 0.25 + 0.1 * rng.uniform(-1.0, 1.0)
    hue_left = (hue_center - hue_separation) % 1.0
    hue_right = (hue_center + hue_separation) % 1.0

    # Nudge saturation/luminance to keep vibrancy without washing out the base color
    saturation = float(np.clip(saturation + 0.1 * rng.uniform(-0.6, 0.6), 0.4, 0.95))
    luminance = float(np.clip(luminance + 0.08 * rng.uniform(-0.6, 0.6), 0.35, 0.9))

    if debug:
        print(
            f"BG Palette: L={hue_left:.3f}, C={hue_center:.3f}, R={hue_right:.3f}, "
            f"sat={saturation:.3f}, lum={luminance:.3f}"
        )

    return [
        _hsl_to_rgb(hue_left, saturation, luminance),
        _hsl_to_rgb(hue_center, saturation, luminance),
        _hsl_to_rgb(hue_right, saturation, luminance),
    ]


def _srgb_to_linear(srgb):
    srgb = np.clip(srgb, 0.0, 1.0)
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * (linear ** (1 / 2.4)) - 0.055)


def _linear_to_oklab(rgb):
    # rgb in linear [0,1]
    M1 = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    M2 = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    lms = rgb @ M1.T
    lms_cbrt = np.cbrt(np.clip(lms, 1e-12, None))
    return lms_cbrt @ M2.T


def _oklab_to_linear(lab):
    M1 = np.array(
        [
            [1.0, 0.3963377774, 0.2158037573],
            [1.0, -0.1055613458, -0.0638541728],
            [1.0, -0.0894841775, -1.2914855480],
        ]
    )
    M2 = np.array(
        [
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010],
        ]
    )
    lms = lab @ M1.T
    lms3 = lms ** 3
    return lms3 @ M2.T


def _fallback_oklab_palette_from_rgb(center_rgb, axis=None):
    srgb = np.array(center_rgb, dtype=np.float32) / 255.0
    linear = _srgb_to_linear(srgb)
    lab = _linear_to_oklab(linear)
    base_L = float(np.clip(lab[0], 0.30, 0.85))
    a = float(lab[1])
    b = float(lab[2])

    axis_vec = np.array([a, b], dtype=np.float32)
    axis_norm = float(np.sqrt(axis_vec[0] ** 2 + axis_vec[1] ** 2))
    if axis is not None:
        axis_vec = np.array(axis, dtype=np.float32)
        axis_norm = float(np.sqrt(axis_vec[0] ** 2 + axis_vec[1] ** 2))
    if axis_norm < 1e-4:
        axis_vec = np.array([1.0, 0.0], dtype=np.float32)
        axis_norm = 1.0
    axis_vec = axis_vec / axis_norm

    base_chroma = float(np.sqrt(a * a + b * b))
    center_chroma = min(base_chroma * 0.35, 0.06)
    offset_chroma = float(np.clip(base_chroma * 0.75, 0.06, 0.22))

    center_ab = axis_vec * center_chroma
    left_ab = center_ab - axis_vec * offset_chroma
    right_ab = center_ab + axis_vec * offset_chroma

    left = np.array([base_L, left_ab[0], left_ab[1]], dtype=np.float32)
    center = np.array([base_L, center_ab[0], center_ab[1]], dtype=np.float32)
    right = np.array([base_L, right_ab[0], right_ab[1]], dtype=np.float32)
    return [left, center, right]


def _local_mean(arr, k=3):
    pad = k // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="edge")
    cumsum = padded.cumsum(axis=0).cumsum(axis=1)
    h, w = arr.shape
    y0 = np.arange(h)
    x0 = np.arange(w)
    y1 = y0 + k
    x1 = x0 + k
    sum_area = (
        cumsum[y1[:, None], x1[None, :]]
        - cumsum[y0[:, None], x1[None, :]]
        - cumsum[y1[:, None], x0[None, :]]
        + cumsum[y0[:, None], x0[None, :]]
    )
    return sum_area / (k * k)


def _wrap_angle(theta):
    return (theta + 2 * np.pi) % (2 * np.pi)


def _angle_diff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def _smooth_angle(prev_angle, new_angle, alpha):
    prev_vec = np.array([np.cos(prev_angle), np.sin(prev_angle)], dtype=np.float32)
    new_vec = np.array([np.cos(new_angle), np.sin(new_angle)], dtype=np.float32)
    vec = (1 - alpha) * prev_vec + alpha * new_vec
    return float(np.arctan2(vec[1], vec[0]))


class _HeroColorState:
    def __init__(self):
        self.hero_h = None
        self.hero_C = 0.08
        self.hero_conf = 0.0
        self.age = 0
        self.low_conf_frames = 0
        self.prev_palette = None
        self.prev_axis = None
        self.prev_median_L = None
        self.prev_L_grid = None

    def reset(self):
        self.hero_h = None
        self.hero_C = 0.08
        self.hero_conf = 0.0
        self.age = 0
        self.low_conf_frames = 0
        self.prev_palette = None
        self.prev_axis = None

    def update(self, candidate_h, candidate_C, conf, scene_dark, scene_gray, red_bias, smooth_strength):
        if red_bias:
            conf *= 0.6
        if scene_dark:
            conf *= 0.75
        if scene_gray:
            conf *= 0.9
        conf = float(np.clip(conf, 0.0, 1.0))

        min_duration = max(5, int(smooth_strength * 0.5))
        if self.hero_h is None:
            self.hero_h = float(candidate_h)
            self.hero_conf = conf
            self.age = 0
            self.low_conf_frames = 0
        else:
            prev_h = self.hero_h
            prev_conf = self.hero_conf
            hue_delta = abs(_angle_diff(candidate_h, prev_h))
            lock_gain = 1.5 if scene_dark else 1.3
            lock_thresh = np.deg2rad(18 if scene_dark else 12)
            hold_thresh = np.deg2rad(10 if scene_dark else 14)

            if conf > prev_conf * lock_gain and (hue_delta > lock_thresh or conf > 0.4):
                self.hero_h = float(candidate_h)
                self.hero_conf = conf
                self.age = 0
                self.low_conf_frames = 0
            elif hue_delta < hold_thresh and conf >= prev_conf * 0.75:
                self.hero_h = _smooth_angle(prev_h, candidate_h, 0.08)
                self.hero_conf = max(prev_conf * 0.97, conf)
                self.age += 1
                self.low_conf_frames = 0
            else:
                self.low_conf_frames += 1
                self.age += 1
                if self.low_conf_frames > min_duration and conf < prev_conf * 0.6:
                    self.hero_h = float(candidate_h)
                    self.hero_conf = conf
                    self.age = 0
                    self.low_conf_frames = 0

        if conf < 0.12:
            target_C = self.hero_C * 0.9
        else:
            target_C = candidate_C
        self.hero_C = float(np.clip(0.85 * self.hero_C + 0.15 * target_C, 0.02, 0.35))
        return self.hero_h, self.hero_C, self.hero_conf


def _extract_oklab_palette(frame_rgb, state, sample_step=6, smooth_strength=10):
    # Downsample for speed
    step = max(1, int(sample_step))
    small = frame_rgb[::step, ::step, :]
    if small.size == 0:
        return None, None

    # 1) Perceptual normalization (no clipping)
    srgb = small.astype(np.float32) / 255.0
    linear = _srgb_to_linear(srgb)
    lab = _linear_to_oklab(linear)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    chroma = np.sqrt(a * a + b * b)
    hue = _wrap_angle(np.arctan2(b, a))

    # 2) Scene characterization
    median_L = float(np.median(L))
    if state.prev_median_L is not None and abs(median_L - state.prev_median_L) > 0.22:
        state.reset()
    state.prev_median_L = median_L

    L_thresh = float(np.clip(0.12 + 0.6 * median_L, 0.08, 0.45))
    reliable = L >= L_thresh

    C_med = float(np.percentile(chroma, 50))
    C_gray = float(np.clip(C_med * 0.8, 0.02, 0.07))
    achromatic = chroma < C_gray
    R_gray = float(np.mean(achromatic))

    C_max = float(np.max(chroma)) if chroma.size else 0.0
    mu_C = float(np.mean(chroma))
    sigma_C = float(np.std(chroma))
    chroma_snr = float(mu_C / (sigma_C + 1e-6))

    scene_dark = median_L < 0.3
    scene_gray = R_gray > 0.65

    red_bias = False
    if scene_dark:
        chroma_mask = chroma >= C_gray
        if np.any(chroma_mask):
            red_ratio = float(np.mean(np.abs(_angle_diff(hue[chroma_mask], 0.0)) < np.deg2rad(20)))
            red_bias = red_ratio > 0.35

    # 3) Chromatic identity dataset (two-stage filter)
    C_hi = float(np.clip(np.percentile(chroma, 70), C_gray * 1.3, 0.2))
    valid = reliable & (chroma >= C_gray) & (chroma >= C_hi)
    if np.sum(valid) < 40:
        valid = reliable & (chroma >= C_gray)
    if np.sum(valid) < 40:
        return None, L

    # 4) Chromatic saliency weighting
    chroma_v = chroma[valid]
    hue_v = hue[valid]
    a_v = a[valid]
    b_v = b[valid]
    L_v = L[valid]

    C_mid = max(C_gray * 1.5, 0.03)
    wC = 1.0 / (1.0 + np.exp(-18.0 * (chroma_v - C_mid)))

    # Rarity in (a,b) space
    ab_bins = 24
    a_range = (-0.5, 0.5)
    b_range = (-0.5, 0.5)
    hist2d, _, _ = np.histogram2d(a_v, b_v, bins=ab_bins, range=[a_range, b_range])
    hist2d = hist2d.astype(np.float32) + 1.0
    a_idx = np.clip(((a_v - a_range[0]) / (a_range[1] - a_range[0]) * ab_bins).astype(int), 0, ab_bins - 1)
    b_idx = np.clip(((b_v - b_range[0]) / (b_range[1] - b_range[0]) * ab_bins).astype(int), 0, ab_bins - 1)
    density = hist2d[a_idx, b_idx]
    wR = 1.0 / np.sqrt(density)
    wR = wR / (np.max(wR) + 1e-6)

    # Spatial coherence via local circular variance in hue
    mean_sin = _local_mean(np.sin(hue), k=3)
    mean_cos = _local_mean(np.cos(hue), k=3)
    R = np.sqrt(mean_sin * mean_sin + mean_cos * mean_cos)
    wS = np.clip(R, 0.0, 1.0) ** 2
    wS = wS[valid]

    w = wC * wR * wS
    if np.sum(w) <= 1e-6:
        return None, L

    # 5) Full hero color extraction (energy-preserving)
    saturation_boost = float(COLOR_BOOST_PARAMS.get("saturation_boost", 1.8))
    weights = L_v * (chroma_v * saturation_boost)
    weights = np.clip(weights, 0.0, 1.0) * w
    total_weight = float(np.sum(weights))
    if total_weight > 0:
        hero_L = float(np.sum(L_v * weights) / total_weight)
        hero_a = float(np.sum(a_v * weights) / total_weight)
        hero_b = float(np.sum(b_v * weights) / total_weight)
    else:
        hero_L, hero_a, hero_b = 0.7, 0.1, 0.1

    hero_h = float(np.arctan2(hero_b, hero_a))
    hero_C = float(np.sqrt(hero_a * hero_a + hero_b * hero_b))

    # Confidence from weighted hue concentration
    u_x = np.cos(hue_v)
    u_y = np.sin(hue_v)
    weight_mag = w * chroma_v
    Hx = float(np.sum(weight_mag * u_x))
    Hy = float(np.sum(weight_mag * u_y))
    w_sum = float(np.sum(weight_mag) + 1e-6)
    conf = float(np.sqrt(Hx * Hx + Hy * Hy) / w_sum)

    candidate_C = float(np.clip(hero_C, 0.04, 0.32))
    if conf < 0.08 or chroma_snr < 0.6:
        return None, L

    # 6) Hero color state management
    hero_h, hero_C, hero_conf = state.update(
        hero_h,
        candidate_C,
        conf,
        scene_dark=scene_dark,
        scene_gray=scene_gray,
        red_bias=red_bias,
        smooth_strength=smooth_strength,
    )

    # 7) Perceptual exaggeration
    chroma_amp = 1.15 + 1.1 * R_gray
    hero_C = float(np.clip(hero_C * chroma_amp, 0.05, 0.34))

    if R_gray > 0.6:
        primaries = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])
        diffs = np.array([abs(_angle_diff(hero_h, p)) for p in primaries])
        target = primaries[int(np.argmin(diffs))]
        hero_h = _smooth_angle(hero_h, target, 0.10)

    # 8) LUT palette construction in OKLab
    base_L = float(np.clip(0.55 * median_L + 0.45 * hero_L, 0.15, 0.90))
    if median_L < 0.35:
        base_L = min(0.90, base_L + 0.18)
    elif median_L > 0.70:
        base_L = max(0.15, base_L - 0.18)

    delta_h = np.deg2rad(24)
    center_C = min(hero_C * 0.50, 0.12)
    side_C = min(hero_C * 1.10, 0.36)

    def lab_from_hc(Lv, Cc, hh):
        return np.array([Lv, Cc * np.cos(hh), Cc * np.sin(hh)], dtype=np.float32)

    left = lab_from_hc(base_L, side_C, hero_h - delta_h)
    center = lab_from_hc(base_L, center_C, hero_h)
    right = lab_from_hc(base_L, side_C, hero_h + delta_h)
    selected = [left, center, right]

    # 9) Temporal LUT interpolation
    alpha = 1.0 / max(1, int(smooth_strength))
    if state.prev_palette is None:
        smoothed = selected
    else:
        smoothed = []
        for i in range(3):
            prev = state.prev_palette[i]
            cur = selected[i]
            smoothed.append((1 - alpha) * prev + alpha * cur)

    state.prev_palette = smoothed
    state.prev_axis = np.array([np.cos(hero_h), np.sin(hero_h)], dtype=np.float32)
    state.prev_L_grid = L

    return smoothed, L


def ensure_wav(input_path, temp_dir):
    """Ensure audio is WAV. If not, try ffmpeg to convert."""
    if input_path.lower().endswith(".wav"):
        return input_path

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "Input is not WAV and ffmpeg is not available. "
            "Install ffmpeg or provide a .wav file."
        )

    out_wav = os.path.join(temp_dir, "_render_audio.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-acodec",
        "pcm_s16le",
        "-ac",
        "2",
        "-ar",
        "48000",
        out_wav,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_wav


def compute_iso226_boost():
    iso226_freqs = np.array(
        [
            20,
            25,
            31.5,
            40,
            50,
            63,
            80,
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
            4000,
            5000,
            6300,
            8000,
            10000,
            12500,
        ]
    )
    iso226_60phon = np.array(
        [
            109.5,
            104.2,
            99.1,
            94.2,
            89.9,
            85.8,
            81.9,
            78.5,
            75.4,
            72.3,
            69.7,
            67.4,
            65.4,
            63.5,
            62.1,
            60.8,
            59.8,
            60.0,
            62.1,
            63.3,
            60.0,
            57.4,
            56.5,
            57.7,
            61.0,
            66.5,
            71.6,
            73.3,
            68.7,
        ]
    )
    return iso226_freqs, iso226_60phon - 60.0


def render_video(
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

        # Prepare FFT / analysis
        freqs = np.fft.rfftfreq(fft_size, 1.0 / rate)
        hamming = np.hamming(fft_size).astype(np.float32)
        window_correction = np.sqrt(fft_size / np.sum(hamming**2))
        iso226_freqs, iso226_boost = compute_iso226_boost()

        # State for smoothing
        fft_db_state = np.full(fft_size // 2 + 1, -80.0, dtype=np.float32)
        stereo_balance_state = np.zeros(fft_size // 2 + 1, dtype=np.float32)

        # Qt setup
        if use_opengl and QApplication.instance() is None:
            QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
            fmt = QSurfaceFormat()
            fmt.setRenderableType(QSurfaceFormat.OpenGL)
            fmt.setProfile(QSurfaceFormat.CoreProfile)
            fmt.setVersion(3, 3)
            QSurfaceFormat.setDefaultFormat(fmt)

        app = QApplication.instance() or QApplication([])
        parent_ctx = RenderContext(
            rate=rate,
            fft_size=fft_size,
            channels=channels,
            happy_mode=happy_mode,
            random_color=random_color,
            color_seed=color_seed,
            debug=debug,
        )
        # VisualizerCanvas expects a QWidget parent. Create a dummy widget
        # and then override the visualizer context reference.
        dummy_parent = QWidget()
        canvas = VisualizerCanvas(dummy_parent)
        canvas.parent_visualizer = parent_ctx
        canvas.resize(width, height)
        canvas.min_paint_interval = 0.0
        canvas.last_paint_time = 0.0
        canvas.transparent_background = True
        canvas.setAttribute(Qt.WA_TranslucentBackground, True)
        canvas.setAttribute(Qt.WA_NoSystemBackground, True)
        if bar_cap is not None:
            canvas.bar_cap = bar_cap
        if background_video:
            canvas.minimal_overlay = True
            parent_ctx.RANDOM_COLOR = True
            parent_ctx.palette_mode = "oklab"
            # Initialize with a neutral palette until first frame
            parent_ctx.color_palette = None
            parent_ctx.color_palette_oklab = None
            parent_ctx._palette_seed = None

        # Prepare background video decoder (optional)
        bg_process = None
        bg_frame_size = width * height * 3
        bg_luma_smoothed = None
        palette_state = _HeroColorState()

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

        # Prepare ffmpeg process
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

        # Video encoder selection
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
                # NVENC constant-quality mode
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

        for frame_idx in range(total_frames):
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
                buffer = frame_samples.astype(np.float32)

                windowed = buffer * hamming
                fft_result = np.fft.rfft(windowed)
                fft_magnitude = np.abs(fft_result) * window_correction
                stereo_balance = np.zeros_like(fft_db_state)
            else:
                frame_samples = data[start:end, :2]
                if pad > 0:
                    frame_samples = np.pad(frame_samples, ((pad, 0), (0, 0)), mode="constant")

                left = frame_samples[:, 0].astype(np.float32)
                right = frame_samples[:, 1].astype(np.float32)
                buffer = (left + right) * 0.5

                windowed_left = left * hamming
                windowed_right = right * hamming
                FL = np.fft.rfft(windowed_left)
                FR = np.fft.rfft(windowed_right)
                ML = np.abs(FL)
                MR = np.abs(FR)
                fft_magnitude = np.sqrt(ML**2 + MR**2) * window_correction

                sum_mag = MR + ML + 1e-6
                stereo_balance = np.clip((MR - ML) / sum_mag, -1.0, 1.0)

            norm_factor = fft_size * 32768.0
            fft_db = 20 * np.log10(fft_magnitude / norm_factor + 1e-10)

            if not happy_mode:
                if human_bias > 0:
                    hearing_correction = np.interp(
                        freqs,
                        iso226_freqs,
                        iso226_boost,
                        left=iso226_boost[0],
                        right=iso226_boost[-1],
                    )
                    # Apply equal-loudness weighting: attenuate lows/highs vs 1 kHz
                    fft_db -= hearing_correction * human_bias
                fft_db = np.clip(fft_db, -80, 0)
            else:
                fft_db = np.clip(fft_db, -60, 0)

            # Smoothing
            alpha = 0.3
            fft_db_state = alpha * fft_db + (1 - alpha) * fft_db_state
            alpha_balance = 0.4
            stereo_balance_state = (
                alpha_balance * stereo_balance + (1 - alpha_balance) * stereo_balance_state
            )

            # Oscilloscope
            wave = buffer[-oscilloscope_samples:] / 32768.0

            canvas.update_data(fft_db_state, freqs, stereo_balance_state, wave)

            # Prepare background frame
            if bg_process is not None and bg_process.stdout is not None:
                bg_bytes = bg_process.stdout.read(bg_frame_size)
                if len(bg_bytes) != bg_frame_size:
                    # Restart decoder if stream ended unexpectedly
                    try:
                        bg_process.kill()
                    except Exception:
                        pass
                    bg_process = subprocess.Popen(bg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    bg_bytes = bg_process.stdout.read(bg_frame_size)

                if len(bg_bytes) == bg_frame_size:
                    bg_image = QImage(bg_bytes, width, height, QImage.Format_RGB888).copy()

                    # Compute background luminance (downsampled for speed)
                    try:
                        arr = np.frombuffer(bg_bytes, dtype=np.uint8)
                        arr = arr.reshape((height, width, 3))
                        step = max(1, int(bg_color_sample_step))
                        sample = arr[::step, ::step, :]
                        luma = (
                            0.2126 * sample[:, :, 0]
                            + 0.7152 * sample[:, :, 1]
                            + 0.0722 * sample[:, :, 2]
                        ).mean() / 255.0

                        # Primary palette (OKLab saliency pipeline)
                        palette_oklab, _ = _extract_oklab_palette(
                            sample,
                            palette_state,
                            sample_step=1,
                            smooth_strength=bg_color_smooth,
                        )
                    except Exception:
                        luma = 0.5
                        palette_oklab = None

                    if palette_oklab is None:
                        if palette_state.prev_palette is not None:
                            palette_oklab = palette_state.prev_palette
                        else:
                            mean_rgb = sample.mean(axis=(0, 1)) if "sample" in locals() else np.array([96, 96, 112])
                            palette_oklab = _fallback_oklab_palette_from_rgb(
                                mean_rgb,
                                axis=palette_state.prev_axis,
                            )

                    if bg_luma_smoothed is None:
                        bg_luma_smoothed = luma
                    else:
                        bg_luma_smoothed = 0.9 * bg_luma_smoothed + 0.1 * luma

                    if palette_oklab is not None:
                        rgb_palette = []
                        for lab in palette_oklab:
                            linear = _oklab_to_linear(lab)
                            srgb = _linear_to_srgb(linear)
                            rgb = np.clip(srgb * 255.0, 0, 255).astype(np.int32)
                            rgb_palette.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
                        parent_ctx.color_palette = rgb_palette
                        parent_ctx.color_palette_oklab = palette_oklab

                    # Invalidate caches to reflect new palette
                    canvas.cached_colors.clear()
                    canvas._color_lut = None
                    canvas._color_lut_mode = None
                else:
                    bg_image = QImage(width, height, QImage.Format_RGB32)
                    bg_image.fill(QColor(20, 20, 30))
            else:
                bg_image = QImage(width, height, QImage.Format_RGB32)
                bg_image.fill(QColor(20, 20, 30))

            # Render visualizer onto transparent surface
            viz_image = QImage(width, height, QImage.Format_ARGB32)
            viz_image.fill(QColor(0, 0, 0, 0))
            viz_painter = QPainter(viz_image)
            canvas.render(viz_painter)
            viz_painter.end()

            # Composite background + visualizer with adaptive opacity
            image = QImage(width, height, QImage.Format_RGB32)
            painter = QPainter(image)
            painter.drawImage(0, 0, bg_image)

            adaptive_factor = 1.0
            if bg_process is not None and bg_luma_smoothed is not None:
                adaptive_factor = 0.6 + 0.8 * bg_luma_smoothed

            opacity = float(viz_opacity) * adaptive_factor
            opacity = max(0.0, min(1.0, opacity))
            painter.setOpacity(opacity)
            painter.drawImage(0, 0, viz_image)
            painter.end()

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
    parser = argparse.ArgumentParser(description="Render audio visualizer to video")
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
        help="Use an OpenGL-backed Qt surface (may improve render throughput).",
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
    if args.viz_opacity < 0 or args.viz_opacity > 1:
        parser.error("--viz-opacity must be between 0 and 1")
    if args.bg_color_smooth < 1 or args.bg_color_smooth > 120:
        parser.error("--bg-color-smooth must be between 1 and 120")
    if args.bg_color_sample_step < 1 or args.bg_color_sample_step > 64:
        parser.error("--bg-color-sample-step must be between 1 and 64")

    render_video(
        input_audio=args.audio,
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
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
