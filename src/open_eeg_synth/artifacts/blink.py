"""Eye blinks: empirical scalp map x an asymmetric pulse (DESIGN §5.6)."""

from __future__ import annotations

import numpy as np

from open_eeg_synth.artifacts import patterns
from open_eeg_synth.artifacts.base import Event, EventArtifact, Remedy, RenderContext, TruthRecord
from open_eeg_synth.artifacts.registry import register
from open_eeg_synth.dsp import raised_cosine_envelope

DEFAULT_RATES = {"eyes_open": 0.25, "eyes_closed": 0.0, "drowsy": 0.0}


@register
class Blink(EventArtifact):
    kind = "blink"

    def __init__(
        self,
        *,
        rate_by_state: dict[str, float] | None = None,
        min_gap_s: float = 0.4,
        median_uv: float = 120.0,
        sigma: float = 0.3,
        peak_range_uv: tuple[float, float] = (60.0, 250.0),
        double_p: float = 0.15,
    ) -> None:
        rates = dict(DEFAULT_RATES) if rate_by_state is None else rate_by_state  # {} = never
        super().__init__(rate_by_state=rates, min_gap_s=min_gap_s)
        self.median_uv, self.sigma = float(median_uv), float(sigma)
        self.peak_range_uv, self.double_p = tuple(peak_range_uv), float(double_p)

    def bind(self, ctx: RenderContext) -> None:
        super().bind(ctx)
        # T9/T10-referenced map on the full head's scale (P28, P30): the drawn peak is the value
        # at the full head's loudest channel (Fp1/Fp2), whichever channels this case records.
        self.pattern = patterns.empirical(
            "blink", ctx.channels, rng=ctx.subject_rng, electrode_pos=ctx.electrode_pos
        )
        # truth channels: |pattern| >= 0.3 of the full-head maximum (typically Fp1, Fp2, F7, F8,
        # F3, F4 on the 10-20 set)
        self.channels = tuple(
            ch for ch, p in zip(ctx.channels, self.pattern, strict=True) if abs(p) >= 0.3
        )

    @staticmethod
    def waveform(fs: float, dur_s: float) -> np.ndarray:
        n = int(round(dur_s * fs))
        t = np.arange(n) / fs
        rise, tau = 0.35 * dur_s, 0.22 * dur_s
        w = np.where(t < rise, 0.5 * (1.0 - np.cos(np.pi * t / rise)), np.exp(-(t - rise) / tau))
        w = w * raised_cosine_envelope(n, fs, 0.0, 0.05 * dur_s)
        return w / w.max()  # unit peak on the sampled grid

    def make_event(self, onset: int) -> Event:
        rng = self.ctx.rng
        dur = float(rng.uniform(0.2, 0.4))
        peak = float(
            np.clip(rng.lognormal(np.log(self.median_uv), self.sigma), *self.peak_range_uv)
        )
        w = peak * self.waveform(self.fs, dur)
        subtype = "single"
        if rng.random() < self.double_p:
            gap = np.zeros(int(round(0.15 * self.fs)))
            w = np.concatenate([w, gap, 0.8 * peak * self.waveform(self.fs, dur)])
            subtype = "double"
        truth = TruthRecord(
            "blink",
            subtype,
            None,
            self.channels,
            onset / self.fs,
            (onset + len(w)) / self.fs,
            peak,
            (Remedy.REMOVE_COMPONENT, Remedy.MASK_SEGMENT),
            self.layer_name,
            {"duration_s": round(dur, 3)},
        )
        return Event.from_pattern(onset, self.pattern, w, truth)
