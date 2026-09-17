"""Smoothed cortical 1/f noise seen at the sensors (DESIGN §4.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from open_eeg_synth.dsp import PinkCascade
from open_eeg_synth.headmodel import HeadModel

WARM_UP_S = 20.0


@dataclass(frozen=True)
class BackgroundSpec:
    smoothing_mm: float = 20.0
    exponent: float = 1.2
    rms_uv: float = 20.0
    network_frac: float = 0.5


class Background:
    name = "brain.background"

    def __init__(
        self,
        head: HeadModel,
        fs: float,
        spec: BackgroundSpec,
        seq: np.random.SeedSequence,
        mixing: np.ndarray | None = None,
    ) -> None:
        self.spec = spec
        M = head.smoothed_mixing(spec.smoothing_mm) if mixing is None else mixing
        self.M = M * spec.rms_uv * np.sqrt(1.0 - spec.network_frac)
        self.rng = np.random.Generator(np.random.PCG64(seq))
        self.pink = PinkCascade(head.n_channels, fs, beta=spec.exponent)
        self.pink.warm_up(self.rng, WARM_UP_S)
        self.n_ch = head.n_channels

    def render(self, t0: int, n: int) -> np.ndarray:
        white = self.rng.standard_normal((n, self.n_ch)).T
        return (self.M @ self.pink.process(white)).astype(np.float32)

    def truth(self) -> list:
        return []
