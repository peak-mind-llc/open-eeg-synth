"""Beat times and an ECG waveform; the HR channel now, heartbeat-bleed plug-ins later.

DESIGN §7.3.
"""

from __future__ import annotations

import numpy as np

# (offset_s, width_s, amplitude) for P, Q, R, S, T — the classic synthesizer's table
WAVES = (
    (-0.20, 0.040, 0.10),
    (-0.04, 0.018, -0.10),
    (0.00, 0.025, 1.00),
    (0.04, 0.025, -0.30),
    (0.30, 0.080, 0.30),
)


class HeartSource:
    def __init__(
        self,
        fs: float,
        seq: np.random.SeedSequence,
        *,
        bpm: float = 72.0,
        bpm_sd: float = 2.0,
        breath_bpm: float = 5.5,
        rsa_ms: float = 40.0,
        mayer: float = 0.02,
        amp_uv: float = 200.0,
    ) -> None:
        rr_rng, noise_rng = (np.random.Generator(np.random.PCG64(s)) for s in seq.spawn(2))
        self.fs, self.rr_rng, self.noise_rng = float(fs), rr_rng, noise_rng
        self.rr_mean = 60.0 / bpm
        self.rr_sd = self.rr_mean * (bpm_sd / bpm)
        self.breath_hz, self.rsa_s, self.mayer = breath_bpm / 60.0, rsa_ms / 1000.0, mayer
        self.beats: list[float] = [float(rr_rng.uniform(0.0, self.rr_mean))]
        t = np.arange(-0.5, 0.7, 1.0 / self.fs)
        template = sum(a * np.exp(-((t - o) ** 2) / (2 * w * w)) for o, w, a in WAVES)
        self.scale = amp_uv / float(template.max())

    def _extend(self, until_s: float) -> None:
        while self.beats[-1] < until_s + 0.7:
            t = self.beats[-1]
            rr = (
                self.rr_mean * (1.0 + self.mayer * np.sin(2 * np.pi * 0.1 * t))
                + self.rsa_s * np.sin(2 * np.pi * self.breath_hz * t)
                + float(self.rr_rng.normal(0.0, self.rr_sd))
            )
            self.beats.append(t + max(0.4, rr))

    def beats_between(self, a_s: float, b_s: float) -> list[float]:
        self._extend(b_s)
        return [b for b in self.beats if a_s <= b < b_s]

    def render(self, t0: int, n: int) -> np.ndarray:
        t = (t0 + np.arange(n)) / self.fs
        self._extend(t[-1])
        sig = np.zeros(n)
        for bt in self.beats:
            if bt < t[0] - 0.7 or bt > t[-1] + 0.5:
                continue
            for off, w, amp in WAVES:
                sig += amp * np.exp(-((t - (bt + off)) ** 2) / (2 * w * w))
        self.beats = [b for b in self.beats if b >= t[0] - 1.0]
        noise = 0.005 * self.noise_rng.standard_normal(n)
        return (self.scale * (sig + noise)).astype(np.float32)
