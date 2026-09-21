"""Continuous power-line contamination with unequal channel coupling."""

from __future__ import annotations

import math

import numpy as np

from open_eeg_synth.artifacts.base import ContinuousArtifact, Remedy, TruthRecord
from open_eeg_synth.artifacts.registry import register
from open_eeg_synth.headmodel import load_head_model


@register
class MainsHum(ContinuousArtifact):
    kind = "mains_hum"

    def __init__(
        self,
        *,
        freq_hz: float,
        base_rms_uv: float = 8.0,
        gain_sigma: float = 0.35,
        am_depth: float = 0.25,
        am_hz: float = 0.035,
        harmonic_ratio: float = 0.1,
    ) -> None:
        if not math.isfinite(freq_hz) or float(freq_hz) not in {50.0, 60.0}:
            raise ValueError("freq_hz must be 50 or 60")
        if not math.isfinite(base_rms_uv) or base_rms_uv <= 0:
            raise ValueError("base_rms_uv must be finite and positive")
        if not math.isfinite(gain_sigma) or gain_sigma < 0:
            raise ValueError("gain_sigma must be finite and nonnegative")
        if not math.isfinite(am_depth) or not 0 <= am_depth < 1:
            raise ValueError("am_depth must be in [0, 1)")
        if not math.isfinite(am_hz) or am_hz < 0:
            raise ValueError("am_hz must be finite and nonnegative")
        if not math.isfinite(harmonic_ratio) or not 0 <= harmonic_ratio <= 1:
            raise ValueError("harmonic_ratio must be in [0, 1]")
        self.freq_hz = float(freq_hz)
        self.base_rms_uv = float(base_rms_uv)
        self.gain_sigma = float(gain_sigma)
        self.am_depth = float(am_depth)
        self.am_hz = float(am_hz)
        self.harmonic_ratio = float(harmonic_ratio)

    def _bind_continuous(self) -> None:
        nyquist = self.fs / 2
        if self.freq_hz >= nyquist:
            raise ValueError("mains fundamental must be below Nyquist")
        if self.am_depth > 0 and self.freq_hz + self.am_hz >= nyquist:
            raise ValueError("mains AM upper sideband must be below Nyquist")

        template_channels = load_head_model().channels
        unknown = set(self.ctx.channels) - set(template_channels)
        if unknown:
            raise ValueError(
                f"mains_hum channels are absent from the template head: {sorted(unknown)}"
            )
        full_gains = self.ctx.subject_rng.lognormal(0.0, self.gain_sigma, len(template_channels))
        full_gains /= np.median(full_gains)
        gain_by_channel = dict(zip(template_channels, full_gains, strict=True))
        self.gains = np.asarray([gain_by_channel[channel] for channel in self.ctx.channels])
        self.harmonics = tuple(
            multiple * self.freq_hz
            for multiple in range(1, 4)
            if multiple * self.freq_hz + (self.am_hz if self.am_depth > 0 else 0.0) < nyquist
        )
        self.phases = tuple(
            float(self.ctx.subject_rng.uniform(0, 2 * np.pi)) for _ in self.harmonics
        )
        self.am_phase = float(self.ctx.subject_rng.uniform(0, 2 * np.pi))

    def _render_block(self, t0: int, n: int) -> np.ndarray:
        times = np.arange(t0, t0 + n, dtype=float) / self.fs
        waveform = np.zeros(n, dtype=float)
        for index, (frequency, phase) in enumerate(zip(self.harmonics, self.phases, strict=True)):
            ratio = self.harmonic_ratio**index
            waveform += ratio * np.sin(2 * np.pi * frequency * times + phase)
        modulation = 1.0 + self.am_depth * np.sin(2 * np.pi * self.am_hz * times + self.am_phase)
        waveform *= np.sqrt(2.0) * self.base_rms_uv * modulation
        return np.outer(self.gains, waveform)

    def _truth_record(self) -> TruthRecord:
        return TruthRecord(
            self.kind,
            f"{self.freq_hz:g}hz",
            None,
            tuple(self.ctx.channels),
            0.0,
            None,
            self._peak_uv,
            (Remedy.NOTCH, Remedy.RE_REFERENCE),
            self.layer_name,
            {
                "freq_hz": self.freq_hz,
                "base_rms_uv": self.base_rms_uv,
                "gain_sigma": self.gain_sigma,
                "am_depth": self.am_depth,
                "am_hz": self.am_hz,
                "harmonic_ratio": self.harmonic_ratio,
                "harmonics_hz": list(self.harmonics),
                "channel_gains": self.gains.tolist(),
                "engineering_range": True,
            },
        )
