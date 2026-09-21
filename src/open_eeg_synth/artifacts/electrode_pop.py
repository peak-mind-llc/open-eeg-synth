"""Abrupt channel-local electrode potential shifts with exponential recovery."""

from __future__ import annotations

import numpy as np

from open_eeg_synth.artifacts.base import Event, EventArtifact, Remedy, RenderContext, TruthRecord
from open_eeg_synth.artifacts.registry import register

DEFAULT_RATES = {"eyes_open": 1 / 180, "eyes_closed": 1 / 240, "drowsy": 1 / 180}


@register
class ElectrodePop(EventArtifact):
    """A sudden contact-potential change confined to one acquisition channel."""

    kind = "electrode_pop"

    def __init__(
        self,
        *,
        channel: str = "random",
        polarity: str = "random",
        rate_by_state: dict[str, float] | None = None,
        min_gap_s: float = 10.0,
        peak_median_uv: float = 450.0,
        peak_sigma: float = 0.5,
        peak_range_uv: tuple[float, float] = (200.0, 2000.0),
        tau_median_s: float = 0.55,
        tau_sigma: float = 0.45,
        tau_range_s: tuple[float, float] = (0.2, 2.0),
    ) -> None:
        if polarity not in {"positive", "negative", "random"}:
            raise ValueError("polarity must be positive, negative or random")
        rates = dict(DEFAULT_RATES) if rate_by_state is None else rate_by_state
        super().__init__(rate_by_state=rates, min_gap_s=min_gap_s)
        self.channel = channel
        self.polarity = polarity
        self.peak_median_uv = float(peak_median_uv)
        self.peak_sigma = float(peak_sigma)
        self.peak_range_uv = tuple(float(value) for value in peak_range_uv)
        self.tau_median_s = float(tau_median_s)
        self.tau_sigma = float(tau_sigma)
        self.tau_range_s = tuple(float(value) for value in tau_range_s)

    def bind(self, ctx: RenderContext) -> None:
        super().bind(ctx)
        if self.channel == "random":
            self.bound_channel = str(ctx.subject_rng.choice(ctx.channels))
        elif self.channel in ctx.channels:
            self.bound_channel = self.channel
        else:
            raise ValueError(
                f"electrode_pop channel {self.channel!r} is not in the recording channels"
            )
        self.channel_index = ctx.channels.index(self.bound_channel)

    @staticmethod
    def waveform(fs: float, peak_uv: float, tau_s: float) -> np.ndarray:
        duration_s = -np.log(0.01) * tau_s
        n = int(np.ceil(duration_s * fs)) + 1
        return peak_uv * np.exp(-np.arange(n) / (fs * tau_s))

    def make_event(self, onset: int) -> Event:
        rng = self.ctx.rng
        peak = float(
            np.clip(
                rng.lognormal(np.log(self.peak_median_uv), self.peak_sigma),
                *self.peak_range_uv,
            )
        )
        tau = float(
            np.clip(
                rng.lognormal(np.log(self.tau_median_s), self.tau_sigma),
                *self.tau_range_s,
            )
        )
        polarity = self.polarity
        if polarity == "random":
            polarity = "positive" if rng.random() < 0.5 else "negative"
        signed_peak = peak if polarity == "positive" else -peak
        waveform = self.waveform(self.fs, signed_peak, tau)
        block = np.zeros((len(self.ctx.channels), waveform.size), dtype=np.float32)
        block[self.channel_index] = waveform
        truth = TruthRecord(
            self.kind,
            polarity,
            None,
            (self.bound_channel,),
            onset / self.fs,
            (onset + waveform.size) / self.fs,
            peak,
            (Remedy.MASK_SEGMENT, Remedy.MARK_BAD_CHANNEL),
            self.layer_name,
            {"duration_s": waveform.size / self.fs, "tau_s": tau},
        )
        return Event(onset, block, truth)
