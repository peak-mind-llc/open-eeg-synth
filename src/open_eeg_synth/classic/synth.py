"""Classic streaming EEG synthesizer (moved from Coherence Recorder).

Realistic, device-agnostic EEG for mock device sources: 1/f pink noise with
strong common-mode coupling, posterior-dominant alpha (per-electrode weights),
frontal eye blinks, and synthetic ECG on HR/ECG channels. Stateful and
phase-continuous across ``next_chunk()``; deterministic under ``seed``.

This module is kept sample-identical to the Coherence Recorder original (see
``tests/classic/test_golden.py``). Its constants were calibrated against a real
eyes-open 19-channel recording made on a NeuroField Q21 amplifier. New work
belongs in the layered engine, not here.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np

# Per-electrode alpha amplitude weight (occipital >> frontal); from the Q21
# calibration in mock_lsl_stream.py. Unknown labels use _DEFAULT_ALPHA_WEIGHT.
_ALPHA_WEIGHTS: dict[str, float] = {
    "Fp1": 0.15,
    "Fp2": 0.15,
    "F3": 0.25,
    "F4": 0.25,
    "C3": 0.50,
    "C4": 0.50,
    "P3": 0.85,
    "P4": 0.85,
    "O1": 1.00,
    "O2": 1.00,
    "F7": 0.20,
    "F8": 0.20,
    "T3": 0.30,
    "T4": 0.30,
    "T5": 0.75,
    "T6": 0.75,
    "Fz": 0.30,
    "Cz": 0.55,
    "Pz": 0.95,
    "Oz": 1.00,
    "POz": 0.95,
    "CPz": 0.70,
    "FCz": 0.40,
}
_DEFAULT_ALPHA_WEIGHT = 0.40
_HR_LABELS = {"HR", "ECG", "EKG"}

# Calibrated signal constants (verbatim from mock_lsl_stream.py).
PINK_NOISE_RMS = 65.0
COMMON_MODE_WEIGHT = 0.97
ALPHA_AMPLITUDE = 18.0
ALPHA_CENTER_FREQ = 9.5
ALPHA_FREQ_SIGMA = 0.35
ALPHA_FREQ_TAU = 4.0
ALPHA_ENV_SIGMA = 0.5
ALPHA_ENV_TAU = 1.5
ALPHA_ENV_MIN = 0.25
ALPHA_ENV_MAX = 1.8
BLINK_RATE_HZ = 0.16  # ~0.02 per 0.125 s chunk in the original
BLINK_AMPLITUDE = 100.0
BLINK_DURATION_S = 0.2
HR_MEAN_BPM = 72.0
HR_BPM_STD = 2.0
HR_AMPLITUDE = 200.0
PINK_BUFFER_SECONDS = 30.0
ECG_BUFFER_SECONDS = 60.0


def alpha_weight(label: str) -> float:
    return _ALPHA_WEIGHTS.get(label.strip(), _DEFAULT_ALPHA_WEIGHT)


def _is_hr(label: str) -> bool:
    return label.strip().upper() in _HR_LABELS


def _is_frontal(label: str) -> bool:
    return label.strip().upper().startswith("F") and not _is_hr(label)


@dataclasses.dataclass(frozen=True)
class MarkerSchedule:
    kind: str = "none"  # "none" | "periodic" | "oddball"
    period_s: float = 2.0
    p_target: float = 0.2
    labels: tuple[str, str] = ("standard", "target")

    def __post_init__(self) -> None:
        if self.period_s <= 0:
            raise ValueError("period_s must be positive")


class RealisticEEGSynthesizer:
    def __init__(
        self,
        channel_labels: Sequence[str],
        srate: float,
        *,
        seed: int | None = None,
        markers: MarkerSchedule | None = None,
    ) -> None:
        if not channel_labels:
            raise ValueError("channel_labels must be non-empty")
        if srate <= 0:
            raise ValueError("srate must be positive")
        self.labels = list(channel_labels)
        self.srate = float(srate)
        self.markers = markers or MarkerSchedule()
        self._rng = np.random.default_rng(seed)
        self._n = len(self.labels)
        self._uw = math.sqrt(max(0.0, 1.0 - COMMON_MODE_WEIGHT**2))
        self._hr_idx = [i for i, lbl in enumerate(self.labels) if _is_hr(lbl)]
        self._frontal_idx = [i for i, lbl in enumerate(self.labels) if _is_frontal(lbl)]
        self._alpha_w = np.array([alpha_weight(lbl) for lbl in self.labels], dtype=np.float32)
        for i in self._hr_idx:
            self._alpha_w[i] = 0.0
        self._alpha_phase = 0.0
        self._alpha_freq = ALPHA_CENTER_FREQ
        self._alpha_log_env = 0.0
        self._regen_pink()
        self._regen_ecg()
        self._marker_clock_s = 0.0
        self._next_marker_s = self.markers.period_s if self.markers.kind != "none" else math.inf

    # ---- pink / ecg buffers (rolling, regenerated on wrap) ----
    def _make_pink(self, n: int) -> np.ndarray:
        white = self._rng.standard_normal(n)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.srate)
        spectrum = np.fft.rfft(white)
        freqs[0] = 1.0
        spectrum *= 1.0 / np.sqrt(freqs)
        pink = np.fft.irfft(spectrum, n=n)
        rms = float(np.sqrt(np.mean(pink**2)))
        return (pink / rms).astype(np.float32) if rms > 0 else pink.astype(np.float32)

    def _regen_pink(self) -> None:
        self._pink_n = int(PINK_BUFFER_SECONDS * self.srate)
        self._common_buf = self._make_pink(self._pink_n)
        self._unique_buf = np.stack([self._make_pink(self._pink_n) for _ in range(self._n)], axis=0)
        self._pink_pos = 0

    def _next_pink(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        if self._pink_pos + n > self._pink_n:
            self._regen_pink()
        s = slice(self._pink_pos, self._pink_pos + n)
        self._pink_pos += n
        return self._common_buf[s], self._unique_buf[:, s]

    def _regen_ecg(self) -> None:
        self._ecg_buf = self._synthesize_ecg(ECG_BUFFER_SECONDS) if self._hr_idx else None
        self._ecg_pos = 0

    def _next_ecg(self, n: int) -> np.ndarray:
        if self._ecg_buf is None:
            return np.zeros(n, dtype=np.float32)
        if self._ecg_pos + n > len(self._ecg_buf):
            self._regen_ecg()
        s = slice(self._ecg_pos, self._ecg_pos + n)
        self._ecg_pos += n
        return self._ecg_buf[s]

    def _synthesize_ecg(self, duration_s: float) -> np.ndarray:
        n = int(duration_s * self.srate)
        rr_mean = 60.0 / HR_MEAN_BPM
        rr_std = rr_mean * (HR_BPM_STD / HR_MEAN_BPM)
        beat_times = []
        t = float(self._rng.uniform(0.0, rr_mean))
        while t < duration_s:
            beat_times.append(round(t * self.srate) / self.srate)
            mayer = 0.02 * math.sin(2 * math.pi * 0.1 * t)
            rr = rr_mean * (1.0 + mayer) + float(self._rng.normal(0.0, rr_std))
            t += max(0.4, rr)
        sig = np.zeros(n, dtype=np.float32)
        t_axis = np.arange(n, dtype=np.float32) / self.srate
        waves = (
            (-0.20, 0.040, 0.10),
            (-0.04, 0.018, -0.10),
            (0.00, 0.025, 1.00),
            (0.04, 0.025, -0.30),
            (0.30, 0.080, 0.30),
        )
        for bt in beat_times:
            for offset, width, amp in waves:
                tc = bt + offset
                lo = max(0, int((tc - 4.0 * width) * self.srate))
                hi = min(n, int((tc + 4.0 * width) * self.srate))
                if lo >= hi:
                    continue
                w = t_axis[lo:hi]
                sig[lo:hi] += amp * np.exp(-((w - tc) ** 2) / (2.0 * width * width))
        sig += self._rng.standard_normal(n).astype(np.float32) * 0.005
        peak = float(np.max(np.abs(sig)))
        if peak > 0:
            sig *= HR_AMPLITUDE / peak
        return sig

    def _ou_step(self, x, target, ss_sigma, tau, dt):
        theta = 1.0 / max(tau, 1e-3)
        sigma_ou = ss_sigma * math.sqrt(2.0 * theta)
        return (
            x
            + theta * (target - x) * dt
            + sigma_ou * math.sqrt(dt) * float(self._rng.standard_normal())
        )

    # ---- public ----
    def next_chunk(self, n_samples: int) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if n_samples > self._pink_n:
            raise ValueError(f"n_samples ({n_samples}) exceeds the {self._pink_n}-sample buffer")
        dt = n_samples / self.srate
        common, unique = self._next_pink(n_samples)
        out = (COMMON_MODE_WEIGHT * PINK_NOISE_RMS) * common[None, :] + (
            self._uw * PINK_NOISE_RMS
        ) * unique
        out = out.astype(np.float32)

        self._alpha_freq = self._ou_step(
            self._alpha_freq, ALPHA_CENTER_FREQ, ALPHA_FREQ_SIGMA, ALPHA_FREQ_TAU, dt
        )
        self._alpha_log_env = self._ou_step(
            self._alpha_log_env, 0.0, ALPHA_ENV_SIGMA, ALPHA_ENV_TAU, dt
        )
        env = min(ALPHA_ENV_MAX, max(ALPHA_ENV_MIN, math.exp(self._alpha_log_env)))
        phase_inc = 2.0 * math.pi * self._alpha_freq / self.srate
        phases = self._alpha_phase + phase_inc * np.arange(n_samples, dtype=np.float64)
        alpha_wave = ((ALPHA_AMPLITUDE * env) * np.sin(phases)).astype(np.float32)
        self._alpha_phase = (self._alpha_phase + phase_inc * n_samples) % (2.0 * math.pi)
        out += self._alpha_w[:, None] * alpha_wave[None, :]

        if self._frontal_idx and self._rng.random() < BLINK_RATE_HZ * dt:
            blen = min(n_samples, max(1, int(BLINK_DURATION_S * self.srate)))
            t = np.linspace(0.0, math.pi, blen, dtype=np.float32)
            blink = (BLINK_AMPLITUDE * np.sin(t)).astype(np.float32)
            for idx in self._frontal_idx:
                out[idx, :blen] += blink

        if self._hr_idx:
            ecg = self._next_ecg(n_samples)
            for idx in self._hr_idx:
                out[idx, :] = ecg

        return out

    def due_markers(self, n_samples: int) -> list[tuple[float, str]]:
        """Markers (onset_seconds, label) whose scheduled time falls within the
        next ``n_samples`` worth of stream time. Advances the marker clock.

        Call ``due_markers(n)`` with the SAME ``n`` you pass to
        ``next_chunk(n)``, in lockstep, so the marker clock tracks the signal.
        """
        end = self._marker_clock_s + n_samples / self.srate
        if self.markers.kind == "none":
            self._marker_clock_s = end
            return []
        out: list[tuple[float, str]] = []
        while self._next_marker_s < end:
            if self.markers.kind == "oddball":
                is_target = float(self._rng.random()) < self.markers.p_target
                label = self.markers.labels[1] if is_target else self.markers.labels[0]
            else:
                label = self.markers.labels[0]
            out.append((self._next_marker_s, label))
            self._next_marker_s += self.markers.period_s
        self._marker_clock_s = end
        return out
