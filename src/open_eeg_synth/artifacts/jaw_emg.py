"""Jaw-tension (temporalis) EMG bursts, generated at a high rate then decimated (DESIGN §5.6)."""

from __future__ import annotations

import numpy as np

from open_eeg_synth.artifacts import patterns
from open_eeg_synth.artifacts.base import Event, EventArtifact, Remedy, RenderContext, TruthRecord
from open_eeg_synth.artifacts.registry import register
from open_eeg_synth.dsp import lowpass_decimate, raised_cosine_envelope

DEFAULT_RATES = {"eyes_open": 1 / 60, "eyes_closed": 1 / 60, "drowsy": 1 / 120}
# Temporalis centre and sigma (Task 18 / P ruling): the brief's naive anatomical offset (5 mm
# lateral, 10 mm anterior, 10 mm inferior of T3/T4) with sigma_mm=35 undershoots DESIGN §5.6's
# target topography (measured: F7~=0.62, T5~=0.12, C3~=0.05 there vs. the design's ~0.6/~0.5/~0.3).
# Retuned by least-squares search against those three targets (T3/T4 == 1.0 is automatic — the
# centre stays closest to T3/T4 than to any other electrode): centred 4 mm posterior and 7.5 mm
# inferior of T3 / T4 (no lateral offset), sigma_mm = 54.0. Resulting values on CHANNELS_19 (max
# |error| against DESIGN's ~1.0/~0.6/~0.5/~0.3 is 0.067, comfortably inside the +/-0.15 the task
# allows) — left pattern centred on LEFT_CENTRE: T3=1.000, F7=0.667, T5=0.563, C3=0.291, next
# loudest F3=0.252 (T4=0.008); right pattern centred on RIGHT_CENTRE: T4=1.000, F8=0.544,
# T6=0.542, C4=0.366, next loudest P4=0.243 (T3=0.008). The head model is not perfectly
# left/right symmetric, so the two sides' errors differ slightly but both stay well inside
# tolerance.
LEFT_CENTRE = np.array([-0.0825, -0.0169, -0.0129])
RIGHT_CENTRE = np.array([0.0850, -0.0221, -0.0100])


@register
class JawEmg(EventArtifact):
    kind = "emg"

    def __init__(
        self,
        *,
        side: str = "random",
        rate_by_state: dict[str, float] | None = None,
        min_gap_s: float = 5.0,
        oversample: int = 8,
        peak_hz: float = 80.0,
        rms_median_uv: float = 50.0,
        rms_sigma: float = 0.6,
        rms_range_uv: tuple[float, float] = (20.0, 200.0),
        dur_median_s: float = 1.2,
        dur_sigma: float = 0.5,
        dur_range_s: tuple[float, float] = (0.5, 3.0),
        sigma_mm: float = 54.0,
    ) -> None:
        if side not in ("left", "right", "both", "random"):
            raise ValueError("side must be left, right, both or random")
        super().__init__(rate_by_state=rate_by_state or dict(DEFAULT_RATES), min_gap_s=min_gap_s)
        self.side, self.oversample, self.peak_hz = side, int(oversample), float(peak_hz)
        self.rms_median_uv, self.rms_sigma = float(rms_median_uv), float(rms_sigma)
        self.rms_range_uv, self.dur_median_s = tuple(rms_range_uv), float(dur_median_s)
        self.dur_sigma, self.dur_range_s = float(dur_sigma), tuple(dur_range_s)
        self.sigma_mm = float(sigma_mm)

    def bind(self, ctx: RenderContext) -> None:
        super().bind(ctx)
        pos = ctx.electrode_pos
        self.pattern = {
            "left": patterns.analytic_focal(LEFT_CENTRE, pos, self.sigma_mm),
            "right": patterns.analytic_focal(RIGHT_CENTRE, pos, self.sigma_mm),
        }

    @staticmethod
    def burst(
        fs: float, dur_s: float, rng: np.random.Generator, oversample: int, peak_hz: float
    ) -> np.ndarray:
        fs_hi = oversample * fs
        n_hi = int(round(dur_s * fs)) * oversample
        white = rng.standard_normal(n_hi)
        f = np.fft.rfftfreq(n_hi, 1.0 / fs_hi)
        h = (f / peak_hz) / (1.0 + (f / peak_hz) ** 2)
        x = np.fft.irfft(np.fft.rfft(white) * h, n=n_hi)
        x *= raised_cosine_envelope(n_hi, fs_hi, 0.1, 0.1)
        y = lowpass_decimate(x, oversample)
        plateau = y[int(0.15 * len(y)) : int(0.85 * len(y))]
        return y / plateau.std()

    def make_event(self, onset: int) -> Event:
        rng, fs = self.ctx.rng, self.fs
        side = self.side
        if side == "random":
            side = str(rng.choice(["left", "right", "both"], p=[0.4, 0.4, 0.2]))
        dur = float(
            np.clip(rng.lognormal(np.log(self.dur_median_s), self.dur_sigma), *self.dur_range_s)
        )
        rms = float(
            np.clip(rng.lognormal(np.log(self.rms_median_uv), self.rms_sigma), *self.rms_range_uv)
        )
        sides = ("left", "right") if side == "both" else (side,)
        block = None
        for s in sides:
            w = rms * self.burst(fs, dur, rng, self.oversample, self.peak_hz)
            part = np.outer(self.pattern[s], w)
            block = part if block is None else block + part
        combined = np.max([self.pattern[s] for s in sides], axis=0)
        channels = tuple(ch for ch, p in zip(self.ctx.channels, combined, strict=True) if p >= 0.3)
        truth = TruthRecord(
            "emg",
            "jaw",
            side,
            channels,
            onset / fs,
            (onset + block.shape[1]) / fs,
            float(np.abs(block).max()),
            (Remedy.MASK_SEGMENT, Remedy.REMOVE_COMPONENT),
            self.layer_name,
            {"duration_s": round(dur, 3), "rms_uv": round(rms, 1)},
        )
        return Event(onset, block.astype(np.float32), truth)
