"""Horizontal eye movements: saccades with eyes open, slow roving otherwise (DESIGN §5.6)."""

from __future__ import annotations

import numpy as np

from open_eeg_synth.artifacts import patterns
from open_eeg_synth.artifacts.base import Event, EventArtifact, Remedy, RenderContext, TruthRecord
from open_eeg_synth.artifacts.registry import register
from open_eeg_synth.dsp import raised_cosine_envelope

DEFAULT_RATES = {"eyes_open": 0.10, "eyes_closed": 0.02, "drowsy": 0.08}


@register
class EyeMovement(EventArtifact):
    kind = "eye_movement"
    jitter_pattern = "heog"

    def __init__(
        self,
        *,
        rate_by_state: dict[str, float] | None = None,
        min_gap_s: float = 0.5,
        saccade_uv: tuple[float, float] = (30.0, 100.0),
        roving_uv: tuple[float, float] = (20.0, 60.0),
    ) -> None:
        rates = dict(DEFAULT_RATES) if rate_by_state is None else rate_by_state  # {} = never
        super().__init__(rate_by_state=rates, min_gap_s=min_gap_s)
        self.saccade_uv, self.roving_uv = tuple(saccade_uv), tuple(roving_uv)

    def bind(self, ctx: RenderContext) -> None:
        super().bind(ctx)
        self.pattern = patterns.empirical(
            self.jitter_pattern, ctx.channels, rng=ctx.subject_rng, electrode_pos=ctx.electrode_pos
        )
        self.channels = tuple(
            ch for ch, p in zip(ctx.channels, self.pattern, strict=True) if abs(p) >= 0.3
        )

    def make_event(self, onset: int) -> Event:
        rng, fs = self.ctx.rng, self.fs
        # The heog map is positive at F8 and negative at F7, so sign > 0 (F8 positive) is gaze to
        # the right and sign < 0 gaze to the left.
        sign = 1 if rng.random() < 0.5 else -1
        gaze = {"sign": sign, "direction": "right" if sign > 0 else "left"}
        if self.ctx.timeline.state_at(onset / fs) == "eyes_open":
            subtype = "saccade"
            hold = float(rng.uniform(0.3, 2.0))
            amp = sign * float(rng.uniform(*self.saccade_uv))
            nr, nh, nf = int(round(0.03 * fs)), int(round(hold * fs)), int(round(0.04 * fs))
            w = (
                np.concatenate(
                    [
                        0.5 * (1 - np.cos(np.pi * np.arange(nr) / nr)),
                        np.ones(nh),
                        0.5 * (1 + np.cos(np.pi * (np.arange(nf) + 1) / nf)),
                    ]
                )
                * amp
            )
            params = {**gaze, "hold_s": round(hold, 3)}
        else:
            subtype = "slow_roving"
            f = float(rng.uniform(0.2, 0.5))
            cycles = int(rng.integers(1, 4))
            amp = sign * float(rng.uniform(*self.roving_uv))
            n = int(round(cycles / f * fs))
            t = np.arange(n) / fs
            w = amp * np.sin(2 * np.pi * f * t) * raised_cosine_envelope(n, fs, 0.2, 0.2)
            params = {**gaze, "freq_hz": round(f, 3), "cycles": cycles}
        truth = TruthRecord(
            "eye_movement",
            subtype,
            None,
            self.channels,
            onset / fs,
            (onset + len(w)) / fs,
            abs(amp),
            (Remedy.REMOVE_COMPONENT, Remedy.MASK_SEGMENT),
            self.layer_name,
            params,
        )
        return Event.from_pattern(onset, self.pattern, w, truth)
