"""Slow galvanic/sweat drift with a focal frontotemporal acquisition field."""

from __future__ import annotations

import math

import numpy as np

from open_eeg_synth.artifacts.base import ContinuousArtifact, Remedy, TruthRecord
from open_eeg_synth.artifacts.registry import register

FIELDS = {
    "left": {"Fp1": 1.0, "F7": 0.8, "F3": 0.45, "T3": 0.3},
    "right": {"Fp2": 1.0, "F8": 0.8, "F4": 0.45, "T4": 0.3},
}


@register
class SweatDrift(ContinuousArtifact):
    """A low-frequency OU wander representing a sweat/galvanic battery effect."""

    kind = "sweat_drift"

    def __init__(
        self,
        *,
        side: str = "random",
        rms_uv: float = 80.0,
        corner_hz: float = 0.12,
    ) -> None:
        if side not in {"left", "right", "bilateral", "random"}:
            raise ValueError("side must be left, right, bilateral or random")
        if not math.isfinite(rms_uv) or rms_uv <= 0:
            raise ValueError("rms_uv must be finite and positive")
        if not math.isfinite(corner_hz) or not 0.05 <= corner_hz <= 0.3:
            raise ValueError("corner_hz must be between 0.05 and 0.3")
        self.side = side
        self.rms_uv = float(rms_uv)
        self.corner_hz = float(corner_hz)

    def _bind_continuous(self) -> None:
        side = self.side
        if side == "random":
            side = str(self.ctx.subject_rng.choice(["left", "right", "bilateral"]))
        self.bound_side = side
        field = np.zeros(self.n_ch, dtype=float)
        selected = ("left", "right") if side == "bilateral" else (side,)
        for selected_side in selected:
            for channel, gain in FIELDS[selected_side].items():
                if channel in self.ctx.channels:
                    field[self.ctx.channels.index(channel)] = max(
                        field[self.ctx.channels.index(channel)], gain
                    )
        self.field = field
        tau = 1.0 / (2.0 * np.pi * self.corner_hz)
        self.rho = float(np.exp(-1.0 / (self.fs * tau)))
        self.innovation_scale = float(np.sqrt(1.0 - self.rho**2))
        self.state = float(self.ctx.rng.standard_normal())

    def _render_block(self, t0: int, n: int) -> np.ndarray:
        innovations = self.ctx.rng.standard_normal(n)
        waveform = np.empty(n, dtype=float)
        state = self.state
        for index, innovation in enumerate(innovations):
            state = self.rho * state + self.innovation_scale * innovation
            waveform[index] = state
        self.state = state
        return np.outer(self.field, self.rms_uv * waveform)

    def _truth_record(self) -> TruthRecord:
        channels = tuple(
            channel for channel, gain in zip(self.ctx.channels, self.field, strict=True) if gain > 0
        )
        return TruthRecord(
            self.kind,
            "galvanic",
            self.bound_side,
            channels,
            0.0,
            None,
            self._peak_uv,
            (Remedy.RAISE_HIGH_PASS, Remedy.MASK_SEGMENT),
            self.layer_name,
            {
                "side": self.bound_side,
                "rms_uv": self.rms_uv,
                "corner_hz": self.corner_hz,
                "engineering_range": True,
            },
        )
