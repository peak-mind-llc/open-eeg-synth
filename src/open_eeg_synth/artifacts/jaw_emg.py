"""Jaw-tension (temporalis) EMG bursts, generated at a high rate then decimated (DESIGN §5.6)."""

from __future__ import annotations

import numpy as np

from open_eeg_synth.artifacts import patterns
from open_eeg_synth.artifacts.base import Event, EventArtifact, Remedy, RenderContext, TruthRecord
from open_eeg_synth.artifacts.registry import register
from open_eeg_synth.dsp import lowpass_decimate, raised_cosine_envelope
from open_eeg_synth.headmodel import load_head_model

DEFAULT_RATES = {"eyes_open": 1 / 60, "eyes_closed": 1 / 60, "drowsy": 1 / 120}
# Temporalis placement: centred 5 mm lateral, 5 mm anterior and 7.5 mm inferior of T3 (left) /
# T4 (right) on the template head (head frame: +x right, +y anterior, +z up), sigma 50 mm.
# Relative to T3 / T4, the raw Gaussian then reads T3 1.00, F7 0.72, T5 0.40, C3 0.21 on the
# left and T4 1.00, F8 0.59, T6 0.38, C4 0.29 on the right; the opposite temporal channel is
# 0.003. Template T3 = (-0.0824765, -0.0129234, -0.0053889) and T4 = (0.0850215, -0.0180840,
# -0.0024852).
LEFT_CENTRE = np.array([-0.0874765, -0.0079234, -0.0128889])
RIGHT_CENTRE = np.array([0.0900215, -0.0130840, -0.0099852])
_CENTRES = {"left": LEFT_CENTRE, "right": RIGHT_CENTRE}


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
        sigma_mm: float = 50.0,
    ) -> None:
        if side not in ("left", "right", "both", "random"):
            raise ValueError("side must be left, right, both or random")
        rates = dict(DEFAULT_RATES) if rate_by_state is None else rate_by_state  # {} = never
        super().__init__(rate_by_state=rates, min_gap_s=min_gap_s)
        self.side, self.oversample, self.peak_hz = side, int(oversample), float(peak_hz)
        self.rms_median_uv, self.rms_sigma = float(rms_median_uv), float(rms_sigma)
        self.rms_range_uv, self.dur_median_s = tuple(rms_range_uv), float(dur_median_s)
        self.dur_sigma, self.dur_range_s = float(dur_sigma), tuple(dur_range_s)
        self.sigma_mm = float(sigma_mm)

    def bind(self, ctx: RenderContext) -> None:
        super().bind(ctx)
        # Each side's map is 1.0 at its loudest channel on the full template head (T3 / T4), so
        # the drawn plateau RMS is the RMS there, and a recording without T3 / T4 keeps its
        # channels at their full-head size instead of being stretched to 1.
        template = load_head_model().electrode_pos
        self.pattern, self._full_pattern = {}, {}
        for side, centre in _CENTRES.items():
            full = patterns.analytic_focal(centre, template, self.sigma_mm)
            self.pattern[side] = patterns.analytic_focal(centre, ctx.electrode_pos, self.sigma_mm)
            self.pattern[side] /= full.max()
            self._full_pattern[side] = full / full.max()

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
        block = full = None  # full: the same event on the template head, for the truth's peak
        for s in sides:
            w = rms * self.burst(fs, dur, rng, self.oversample, self.peak_hz)
            part, full_part = np.outer(self.pattern[s], w), np.outer(self._full_pattern[s], w)
            block = part if block is None else block + part
            full = full_part if full is None else full + full_part
        combined = np.max([self.pattern[s] for s in sides], axis=0)
        channels = tuple(ch for ch, p in zip(self.ctx.channels, combined, strict=True) if p >= 0.3)
        truth = TruthRecord(
            "emg",
            "jaw",
            side,
            channels,
            onset / fs,
            (onset + block.shape[1]) / fs,
            float(np.abs(full).max()),  # the full-head peak, whatever is recorded
            (Remedy.MASK_SEGMENT, Remedy.REMOVE_COMPONENT),
            self.layer_name,
            {"duration_s": round(dur, 3), "rms_uv": round(rms, 1)},
        )
        return Event(onset, block.astype(np.float32), truth)
