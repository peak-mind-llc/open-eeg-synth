"""Classic cardiac and respiration mocks (moved from Coherence Recorder).

Deterministic synthetic RR intervals, ECG and chest accelerometer signals for
mock Polar H10 sessions. Kept sample-identical to the Coherence Recorder
original (see ``tests/classic/test_golden.py``).
"""

from __future__ import annotations

import math

import numpy as np


class MockRRSource:
    """Yields RR intervals (ms): baseline + RSA oscillation + light noise.

    Advances an internal breathing-phase clock by each emitted RR interval, so
    successive ``next_rr()`` calls trace a continuous respiratory oscillation.
    """

    def __init__(
        self,
        *,
        baseline_ms: float = 850.0,
        breath_bpm: float = 5.5,
        rsa_amplitude_ms: float = 40.0,
        noise_ms: float = 4.0,
        seed: int | None = None,
    ) -> None:
        self.baseline_ms = baseline_ms
        self.breath_hz = breath_bpm / 60.0
        self.rsa_amplitude_ms = rsa_amplitude_ms
        self.noise_ms = noise_ms
        self._rng = np.random.default_rng(seed)
        self._t_s = 0.0

    def next_rr(self) -> float:
        osc = self.rsa_amplitude_ms * math.sin(2 * math.pi * self.breath_hz * self._t_s)
        noise = float(self._rng.normal(0.0, self.noise_ms))
        rr = self.baseline_ms + osc + noise
        rr = max(300.0, min(2000.0, rr))
        self._t_s += rr / 1000.0
        return rr


class MockEcgSource:
    """Synthetic ECG (µV) for mock Polar H10 sessions: a QRS spike per beat + a
    small T-wave + baseline noise. Breathing modulates BOTH the beat timing
    (RSA → the RR oscillation) and the R-amplitude (the EDR effect), phase-locked
    — so the live respiration pipeline recovers a real breathing rate AND a high
    breath↔HR coherence on the mock, not just a flat trace. Not physiological,
    just a recognizable, demo-faithful waveform without hardware.
    """

    def __init__(
        self,
        *,
        bpm: float = 70.0,
        fs: float = 130.0,
        breath_bpm: float = 5.5,
        breath_mod: float = 0.18,
        rsa_ms: float = 45.0,
        seed: int | None = None,
    ) -> None:
        self.base_period_s = 60.0 / bpm
        self.fs = fs
        self.breath_hz = breath_bpm / 60.0
        self.breath_mod = breath_mod  # R-amplitude modulation depth (EDR)
        self.rsa_ms = rsa_ms  # beat-timing modulation depth (RSA)
        self._t = 0.0
        self._last_beat_t = -10.0
        self._next_beat_t = 0.0
        self._beat_amp = 900.0
        self._rng = np.random.default_rng(seed)

    def next_chunk(self, n: int) -> list[float]:
        dt = 1.0 / self.fs
        out: list[float] = []
        for _ in range(n):
            if self._t >= self._next_beat_t:
                bphase = 2 * math.pi * self.breath_hz * self._t
                rr = self.base_period_s + (self.rsa_ms / 1000.0) * math.sin(bphase)  # RSA
                self._last_beat_t = self._next_beat_t
                self._next_beat_t = self._last_beat_t + max(0.3, rr)
                self._beat_amp = 900.0 * (1.0 + self.breath_mod * math.sin(bphase))  # EDR
            d = self._t - self._last_beat_t
            qrs = math.exp(-(d**2) / (2 * 0.012**2)) * self._beat_amp
            dt_t = d - 0.32 * self.base_period_s
            twave = math.exp(-(dt_t**2) / (2 * 0.04**2)) * 120.0
            out.append(qrs + twave + float(self._rng.normal(0.0, 8.0)))
            self._t += dt
        return out


class MockAccSource:
    """Synthetic 3-axis accelerometer for mock Polar H10 sessions: a breathing
    oscillation on one axis (chest-wall movement) with a LONG-EXHALE shape (so
    the inhale:exhale mechanics read realistically), gravity on another axis, and
    noise. Lets the accelerometer-respiration path run without hardware.
    """

    def __init__(
        self,
        *,
        fs: float = 25.0,
        breath_bpm: float = 5.5,
        inhale_frac: float = 0.34,
        amp: float = 180.0,
        seed: int | None = None,
    ) -> None:
        self.fs = fs
        self.breath_hz = breath_bpm / 60.0
        self.inhale_frac = inhale_frac  # < 0.5 → exhale longer than inhale
        self.amp = amp
        self._t = 0.0
        self._rng = np.random.default_rng(seed)

    def next_chunk(self, n: int) -> list[tuple[float, float, float]]:
        dt = 1.0 / self.fs
        inh = self.inhale_frac
        out: list[tuple[float, float, float]] = []
        for _ in range(n):
            ph = (self._t * self.breath_hz) % 1.0
            # Asymmetric breath: rise over the inhale fraction, fall over the
            # (longer) exhale — the chest-expansion waveform.
            lvl = ph / inh if ph < inh else 1.0 - (ph - inh) / (1.0 - inh)
            x = self.amp * (lvl - 0.5) + float(self._rng.normal(0.0, 4.0))
            y = float(self._rng.normal(0.0, 4.0))
            z = 1000.0 + float(self._rng.normal(0.0, 4.0))  # ~1 g gravity axis
            out.append((x, y, z))
            self._t += dt
        return out
