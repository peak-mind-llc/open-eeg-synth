"""Amplifier / electrode white noise layer."""

from __future__ import annotations

import numpy as np


class SensorNoise:
    name = "sensor"

    def __init__(self, n_ch: int, white_uv: float, seq: np.random.SeedSequence) -> None:
        self.n_ch = int(n_ch)
        self.white_uv = float(white_uv)
        self.rng = np.random.Generator(np.random.PCG64(seq))

    def render(self, t0: int, n: int) -> np.ndarray:
        return (self.white_uv * self.rng.standard_normal((n, self.n_ch)).T).astype(np.float32)

    def truth(self) -> list:
        return []
