"""Persistent channel-local broadband acquisition noise."""

from __future__ import annotations

import math

import numpy as np

from open_eeg_synth.artifacts.base import ContinuousArtifact, Remedy, TruthRecord
from open_eeg_synth.artifacts.registry import register


@register
class NoisyChannel(ContinuousArtifact):
    kind = "noisy_channel"

    def __init__(self, *, channel: str = "random", rms_uv: float = 25.0) -> None:
        if not math.isfinite(rms_uv) or rms_uv <= 0:
            raise ValueError("rms_uv must be finite and positive")
        self.channel = channel
        self.rms_uv = float(rms_uv)

    def _bind_continuous(self) -> None:
        if self.channel == "random":
            self.bound_channel = str(self.ctx.subject_rng.choice(self.ctx.channels))
        elif self.channel in self.ctx.channels:
            self.bound_channel = self.channel
        else:
            raise ValueError(f"noisy_channel channel {self.channel!r} is not in the recording")
        self.channel_index = self.ctx.channels.index(self.bound_channel)
        self.previous_white = float(self.ctx.rng.standard_normal())

    def _render_block(self, t0: int, n: int) -> np.ndarray:
        white = self.ctx.rng.standard_normal(n)
        previous = np.concatenate(([self.previous_white], white[:-1]))
        waveform = (white - previous) / np.sqrt(2.0)
        self.previous_white = float(white[-1])
        out = np.zeros((self.n_ch, n), dtype=float)
        out[self.channel_index] = self.rms_uv * waveform
        return out

    def _truth_record(self) -> TruthRecord:
        return TruthRecord(
            self.kind,
            "broadband",
            None,
            (self.bound_channel,),
            0.0,
            None,
            self._peak_uv,
            (Remedy.MARK_BAD_CHANNEL, Remedy.INTERPOLATE),
            self.layer_name,
            {"rms_uv": self.rms_uv, "engineering_range": True},
        )
