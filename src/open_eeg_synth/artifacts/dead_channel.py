"""A channel that records nothing but amplifier noise for a span (DESIGN §5.3, §5.6)."""

from __future__ import annotations

import math

import numpy as np

from open_eeg_synth.artifacts.base import Remedy, RenderContext, TransformArtifact, TruthRecord
from open_eeg_synth.artifacts.registry import register
from open_eeg_synth.channels import canonical_label


@register
class DeadChannel(TransformArtifact):
    kind = "dead_channel"

    def __init__(
        self,
        *,
        channel: str,
        onset_s: float = 0.0,
        offset_s: float | None = None,
        noise_uv: float = 0.3,
    ) -> None:
        if not (math.isfinite(float(onset_s)) and float(onset_s) >= 0.0):
            raise ValueError(f"dead_channel: onset_s must be >= 0, got {onset_s!r}")
        if offset_s is not None and not float(offset_s) > float(onset_s):
            raise ValueError(
                f"dead_channel: offset_s must be later than onset_s ({onset_s!r}), got {offset_s!r}"
            )
        self.channel, self.onset_s = channel, float(onset_s)
        self.offset_s, self.noise_uv = offset_s, float(noise_uv)
        self._truth: list[TruthRecord] = []
        self._rendered = 0
        self.a = 0

    def bind(self, ctx: RenderContext) -> None:
        super().bind(ctx)
        self.label = canonical_label(self.channel, ctx.channels)
        self.idx = ctx.channels.index(self.label)
        # [a, b) in samples; the truth reports these boundaries, not the requested seconds
        self.a = int(round(self.onset_s * ctx.fs))
        self.b = math.inf if self.offset_s is None else int(round(self.offset_s * ctx.fs))
        if not self.a < self.b:
            raise ValueError(
                f"dead_channel: [{self.onset_s}, {self.offset_s}) s is no whole sample at "
                f"{ctx.fs} Hz"
            )
        self._rendered = 0
        self._truth = [
            TruthRecord(
                "dead_channel",
                None,
                None,
                (self.label,),
                self.a / ctx.fs,
                None if self.offset_s is None else self.b / ctx.fs,
                0.0,
                (Remedy.MARK_BAD_CHANNEL, Remedy.INTERPOLATE),
                self.layer_name,
                {"noise_uv": self.noise_uv},
            )
        ]

    def render_transform(self, t0: int, n: int, mix: np.ndarray) -> np.ndarray:
        self._rendered = max(self._rendered, t0 + n)
        delta = np.zeros_like(mix)
        a, b = max(self.a, t0), min(self.b, t0 + n)
        if a < b:
            a, b = int(a), int(b)
            noise = self.noise_uv * self.ctx.rng.standard_normal(b - a)
            delta[self.idx, a - t0 : b - t0] = -mix[self.idx, a - t0 : b - t0] + noise
        return delta

    def truth(self) -> list[TruthRecord]:
        """The dead span, once rendering has reached its first sample."""
        return list(self._truth) if self._rendered > self.a else []
