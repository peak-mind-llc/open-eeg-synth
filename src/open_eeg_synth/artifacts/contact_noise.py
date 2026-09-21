"""Intermittent crackling from unstable electrode contact."""

from __future__ import annotations

import math

import numpy as np

from open_eeg_synth.artifacts.base import Event, EventArtifact, Remedy, RenderContext, TruthRecord
from open_eeg_synth.artifacts.registry import register

DEFAULT_RATES = {"eyes_open": 1 / 90, "eyes_closed": 1 / 120, "drowsy": 1 / 90}


@register
class ContactNoise(EventArtifact):
    kind = "contact_noise"

    def __init__(
        self,
        *,
        channel: str = "random",
        rate_by_state: dict[str, float] | None = None,
        min_gap_s: float = 8.0,
        rms_median_uv: float = 55.0,
        rms_sigma: float = 0.5,
        rms_range_uv: tuple[float, float] = (20.0, 250.0),
        duration_median_s: float = 0.7,
        duration_sigma: float = 0.55,
        duration_range_s: tuple[float, float] = (0.15, 2.0),
    ) -> None:
        if not math.isfinite(rms_median_uv) or rms_median_uv <= 0:
            raise ValueError("rms_median_uv must be finite and positive")
        if not math.isfinite(duration_median_s) or duration_median_s <= 0:
            raise ValueError("duration_median_s must be finite and positive")
        rates = dict(DEFAULT_RATES) if rate_by_state is None else rate_by_state
        super().__init__(rate_by_state=rates, min_gap_s=min_gap_s)
        self.channel = channel
        self.rms_median_uv = float(rms_median_uv)
        self.rms_sigma = float(rms_sigma)
        self.rms_range_uv = tuple(float(value) for value in rms_range_uv)
        self.duration_median_s = float(duration_median_s)
        self.duration_sigma = float(duration_sigma)
        self.duration_range_s = tuple(float(value) for value in duration_range_s)

    def bind(self, ctx: RenderContext) -> None:
        super().bind(ctx)
        if self.channel == "random":
            self.bound_channel = str(ctx.subject_rng.choice(ctx.channels))
        elif self.channel in ctx.channels:
            self.bound_channel = self.channel
        else:
            raise ValueError(f"contact_noise channel {self.channel!r} is not in the recording")
        self.channel_index = ctx.channels.index(self.bound_channel)

    def make_event(self, onset: int) -> Event:
        rng = self.ctx.rng
        rms = float(
            np.clip(
                rng.lognormal(np.log(self.rms_median_uv), self.rms_sigma),
                *self.rms_range_uv,
            )
        )
        duration = float(
            np.clip(
                rng.lognormal(np.log(self.duration_median_s), self.duration_sigma),
                *self.duration_range_s,
            )
        )
        n = max(4, int(round(duration * self.fs)))
        white = rng.standard_normal(n + 1)
        waveform = np.diff(white)
        impulse_count = max(2, int(round(duration * 6)))
        positions = rng.choice(n, size=min(impulse_count, n), replace=False)
        waveform[positions] += rng.choice([-1.0, 1.0], size=len(positions)) * rng.uniform(
            4.0, 8.0, size=len(positions)
        )
        waveform -= waveform.mean()
        waveform *= rms / waveform.std()
        block = np.zeros((len(self.ctx.channels), n), dtype=np.float32)
        block[self.channel_index] = waveform
        truth = TruthRecord(
            self.kind,
            "crackle",
            None,
            (self.bound_channel,),
            onset / self.fs,
            (onset + n) / self.fs,
            float(np.max(np.abs(waveform))),
            (Remedy.MASK_SEGMENT, Remedy.MARK_BAD_CHANNEL),
            self.layer_name,
            {
                "duration_s": n / self.fs,
                "rms_uv": rms,
                "impulse_count": int(len(positions)),
                "engineering_range": True,
            },
        )
        return Event(onset, block, truth)
