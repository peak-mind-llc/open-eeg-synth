"""What the subject is doing over time; drives rhythm gains and artifact rates (DESIGN §4.5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateSegment:
    t0_s: float
    t1_s: float
    state: str


@dataclass(frozen=True)
class StateTimeline:
    """Contiguous segments from 0 s; the last extends indefinitely. Compares by value."""

    segments: tuple[StateSegment, ...]
    ramp_s: float = 2.0

    def __post_init__(self) -> None:
        segs = tuple(self.segments)
        if not segs or segs[0].t0_s != 0.0:
            raise ValueError("timeline must start at 0 s")
        for a, b in zip(segs, segs[1:], strict=False):
            if b.t0_s != a.t1_s:
                raise ValueError(f"segments must be contiguous: {a} -> {b}")
        if any(s.t1_s <= s.t0_s for s in segs):
            raise ValueError("every segment needs t1_s > t0_s")
        object.__setattr__(self, "segments", segs)
        object.__setattr__(self, "ramp_s", float(self.ramp_s))

    @classmethod
    def constant(cls, state: str, duration_s: float = math.inf) -> StateTimeline:
        return cls((StateSegment(0.0, duration_s, state),))

    def state_at(self, t_s: float) -> str:
        starts = [s.t0_s for s in self.segments]
        i = int(np.searchsorted(starts, t_s, side="right") - 1)
        return self.segments[max(0, i)].state

    def weights(self, t0: int, n: int, fs: float) -> dict[str, np.ndarray]:
        t = (t0 + np.arange(n)) / fs
        r = self.ramp_s

        def up(b: float) -> np.ndarray:
            if b == -math.inf:
                return np.ones(n)
            if b == math.inf:
                return np.zeros(n)
            if r <= 0:
                return (t >= b).astype(float)
            return np.clip((t - (b - r / 2.0)) / r, 0.0, 1.0)

        out: dict[str, np.ndarray] = {}
        last = len(self.segments) - 1
        for i, seg in enumerate(self.segments):
            w = up(-math.inf if i == 0 else seg.t0_s) - up(math.inf if i == last else seg.t1_s)
            out[seg.state] = out.get(seg.state, 0.0) + w
        return out

    def gain(
        self, per_state: dict[str, float], t0: int, n: int, fs: float, default: float = 1.0
    ) -> np.ndarray:
        g = np.zeros(n)
        for state, w in self.weights(t0, n, fs).items():
            g += w * per_state.get(state, default)
        return g

    def rate(self, per_state: dict[str, float], t_s: float) -> float:
        return float(per_state.get(self.state_at(t_s), 0.0))

    def to_dict(self) -> dict:
        return {
            "ramp_s": self.ramp_s,
            "segments": [{"t0_s": s.t0_s, "t1_s": s.t1_s, "state": s.state} for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> StateTimeline:
        return cls(
            [StateSegment(s["t0_s"], s["t1_s"], s["state"]) for s in d["segments"]],
            ramp_s=d.get("ramp_s", 2.0),
        )
