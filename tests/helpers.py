"""numpy-only measurement helpers shared by the fast tests."""

from __future__ import annotations

import numpy as np


def welch(x: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed Welch PSD along the last axis, 50 % overlap. Returns (f, psd)."""
    x = np.atleast_2d(x)
    nperseg = int(min(nperseg, x.shape[-1]))  # a short input becomes one segment
    step = nperseg // 2
    win = np.hanning(nperseg)
    scale = 1.0 / (fs * (win**2).sum())
    segs = [x[:, s : s + nperseg] * win for s in range(0, x.shape[1] - nperseg + 1, step)]
    p = np.mean([np.abs(np.fft.rfft(s, axis=-1)) ** 2 for s in segs], axis=0) * scale
    p[:, 1:-1] *= 2.0
    return np.fft.rfftfreq(nperseg, 1.0 / fs), p


def band_power(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    f, p = welch(x, fs, int(4 * fs))
    m = (f >= lo) & (f < hi)
    return p[:, m].sum(axis=-1) * (f[1] - f[0])


def psd_slope(x: np.ndarray, fs: float, f_lo=1.0, f_hi=40.0, exclude=(7.0, 14.0)) -> float:
    """Negative of the log-log slope of the median PSD: ~beta for 1/f^beta noise."""
    f, p = welch(x, fs, int(4 * fs))
    fit = (f >= f_lo) & (f <= f_hi) & ~((f >= exclude[0]) & (f <= exclude[1]))
    return float(-np.polyfit(np.log10(f[fit]), np.log10(np.median(p[:, fit], axis=0)), 1)[0])


def render_whole_and_chunked(make_layer, n_total: int, rng: np.random.Generator):
    """Render one layer in one call and a fresh, identically-seeded one in random chunks."""
    whole = make_layer().render(0, n_total)
    layer = make_layer()
    parts, t0 = [], 0
    while t0 < n_total:
        n = int(min(n_total - t0, rng.integers(1, 700)))
        parts.append(layer.render(t0, n))
        t0 += n
    return whole, np.concatenate(parts, axis=1)
