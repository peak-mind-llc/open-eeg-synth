"""Streaming signal primitives: exact OU, a 1/f^beta cascade, decimation (DESIGN §4, §8)."""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly


class OU:
    """Exact-discretisation Ornstein-Uhlenbeck processes, one per row.

    x[n] = mu + a (x[n-1] - mu) + sigma sqrt(1 - a^2) w[n],  a = exp(-1 / (tau_s fs)).
    Starts from the stationary distribution so there is no warm-up transient.
    """

    def __init__(
        self,
        n_series: int,
        fs: float,
        *,
        mu: float = 0.0,
        sigma: float = 1.0,
        tau_s: float = 1.0,
        rng: np.random.Generator,
    ) -> None:
        self.n = int(n_series)
        self.mu = float(mu)
        self.a = float(np.exp(-1.0 / (tau_s * fs)))
        self.g = float(sigma) * float(np.sqrt(1.0 - self.a**2))
        x0 = float(sigma) * rng.standard_normal(self.n)
        self.zi = (self.a * x0).reshape(self.n, 1)  # transposed DF-II state: a * y[-1]

    def step(self, white: np.ndarray, mu: np.ndarray | float | None = None) -> np.ndarray:
        """Advance by ``white.shape[1]`` samples.

        ``mu`` may be a per-sample array overriding ``self.mu`` for this call; it shifts the mean
        immediately, sample by sample — it is added on top of the relaxing deviation, not a new
        stationary target the process relaxes *towards* over ``tau_s``.
        """
        dev, self.zi = lfilter([self.g], [1.0, -self.a], white, axis=-1, zi=self.zi)
        return dev + (self.mu if mu is None else mu)


class PinkCascade:
    """1/f^beta noise via first-order pole-zero sections (Corsini & Saletti 1988), unit variance."""

    def __init__(
        self,
        n_series: int,
        fs: float,
        *,
        beta: float = 1.2,
        f_lo: float = 0.03,
        n_per_decade: float = 2.0,
    ) -> None:
        self.n = int(n_series)
        self.fs = float(fs)
        f_hi = 0.45 * self.fs
        n_sec = int(np.ceil(n_per_decade * np.log10(f_hi / f_lo)))
        r = (f_hi / f_lo) ** (1.0 / n_sec)
        c = 2.0 * self.fs
        self.sections: list[tuple[np.ndarray, np.ndarray]] = []
        for k in range(n_sec):
            fp = f_lo * r**k
            fz = fp * r ** (beta / 2.0)
            wp = c * np.tan(np.pi * fp / self.fs)  # bilinear pre-warp, rad/s
            wz = c * np.tan(np.pi * fz / self.fs)
            k0 = wp / wz
            b = np.array([k0 * (c + wz) / (c + wp), k0 * (wz - c) / (c + wp)])
            a = np.array([1.0, (wp - c) / (c + wp)])
            self.sections.append((b, a))
        self.zi = [np.zeros((self.n, 1)) for _ in self.sections]
        # Exact calibration: for white input of unit variance, Var(output) = sum(h**2) where h is
        # the cascade's impulse response. int(40 * fs) samples is enough for the lowest pole (f_lo)
        # to ring out (confirmed convergent to 5 significant figures by int(20 * fs)); no random
        # draw is involved, so this does not go through seeds.stream_rng.
        h = np.zeros(int(40 * self.fs))
        h[0] = 1.0
        for b, a in self.sections:
            h = lfilter(b, a, h)
        self.scale = 1.0 / float(np.sqrt(np.sum(h**2)))

    def warm_up(self, rng: np.random.Generator, seconds: float) -> None:
        self.process(rng.standard_normal((int(seconds * self.fs), self.n)).T)

    def process(self, white: np.ndarray) -> np.ndarray:
        y = white
        for i, (b, a) in enumerate(self.sections):
            y, self.zi[i] = lfilter(b, a, y, axis=-1, zi=self.zi[i])
        return y * self.scale


def lowpass_decimate(x: np.ndarray, factor: int) -> np.ndarray:
    """Anti-alias low-pass and decimate along the last axis, as an amplifier front end would.

    Filters with a Kaiser-windowed FIR (passband to 0.9 of the *new* Nyquist, stopband from the new
    Nyquist on) before down-sampling by ``factor``, so nothing above the new Nyquist survives to
    fold back into the passband. Runs over one whole finite block: ``x``'s edges are zero-padded,
    there is no carried filter state between calls, so a streaming caller must render extra margin
    on each side and discard it rather than decimate consecutive short chunks back to back.
    """
    factor = int(factor)
    taps = firwin(50 * factor + 1, 0.9 / factor, window=("kaiser", 8.0))
    return resample_poly(x, up=1, down=factor, axis=-1, window=taps)


def raised_cosine_envelope(n: int, fs: float, rise_s: float, fall_s: float) -> np.ndarray:
    """A unit-peak envelope: raised-cosine rise, flat top, raised-cosine fall.

    If ``rise_s + fall_s`` exceeds ``n / fs`` the two ramps are clipped to fit and meet without a
    flat top; the peak then stays below 1.0 (the fall ramp starts before the rise ramp reaches it).
    """
    env = np.ones(int(n))
    nr = min(n // 2, int(round(rise_s * fs)))
    nf = min(n - nr, int(round(fall_s * fs)))
    if nr > 0:
        env[:nr] = 0.5 * (1.0 - np.cos(np.pi * np.arange(nr) / nr))
    if nf > 0:
        env[n - nf :] = 0.5 * (1.0 + np.cos(np.pi * (np.arange(nf) + 1) / nf))
    return env
