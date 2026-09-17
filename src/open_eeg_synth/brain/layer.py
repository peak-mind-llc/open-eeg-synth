"""The brain layer: background + network + rhythms (DESIGN §4)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from open_eeg_synth.brain.background import Background, BackgroundSpec
from open_eeg_synth.brain.network import Network, NetworkSpec, NetworkWiring
from open_eeg_synth.brain.rhythm import Rhythm, RhythmSpec
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.headmodel import HeadModel
from open_eeg_synth.seeds import stream_seed


@dataclass(frozen=True)
class BrainSpec:
    background: BackgroundSpec = BackgroundSpec()
    network: NetworkSpec = NetworkSpec()
    rhythms: tuple[RhythmSpec, ...] = ()

    def to_dict(self) -> dict:
        return {
            "background": asdict(self.background),
            "network": asdict(self.network),
            "rhythms": [r.to_dict() for r in self.rhythms],
        }

    @classmethod
    def from_dict(cls, d: dict) -> BrainSpec:
        return cls(
            BackgroundSpec(**d["background"]),
            NetworkSpec(**d["network"]),
            tuple(RhythmSpec.from_dict(r) for r in d["rhythms"]),
        )


class BrainLayer:
    """Background + network + rhythms on the full head; ``rows`` selects the recorded channels."""

    name = "brain"

    def __init__(
        self,
        rhythms: Sequence[RhythmSpec],
        spec: BrainSpec,
        head: HeadModel,
        fs: float,
        timeline: StateTimeline,
        mixing: np.ndarray,
        wiring: NetworkWiring,
        placements: dict[str, list[int]],
        f0_hz: dict[str, float],
        case_seed: int,
        condition: str,
        rows: Sequence[int] | None = None,
        f0_offsets_hz: dict[str, Sequence[float]] | None = None,
        lags_ms: dict[str, Sequence[float]] | None = None,
    ) -> None:
        """``f0_offsets_hz``/``lags_ms`` are the subject's per-patch values by rhythm name
        (``make_subject``); a rhythm missing from them draws its own from its seed."""
        f0_offsets_hz, lags_ms = f0_offsets_hz or {}, lags_ms or {}
        self.rows = None if rows is None else list(rows)
        self.parts: list = [
            Background(
                head,
                fs,
                spec.background,
                stream_seed(case_seed, f"{condition}:background"),
                mixing=mixing,
            ),
            Network(
                head,
                fs,
                spec.network,
                spec.background,
                wiring,
                stream_seed(case_seed, f"{condition}:network"),
            ),
        ]
        for r in rhythms:
            self.parts.append(
                Rhythm(
                    head,
                    fs,
                    r,
                    timeline,
                    placements[r.name],
                    f0_hz[r.name],
                    stream_seed(case_seed, f"{condition}:rhythm:{r.name}"),
                    f0_offsets_hz=f0_offsets_hz.get(r.name),
                    lags_ms=lags_ms.get(r.name),
                )
            )

    def render(self, t0: int, n: int) -> np.ndarray:
        out = self.parts[0].render(t0, n).astype(np.float32)
        for p in self.parts[1:]:
            out += p.render(t0, n)
        return out if self.rows is None else out[self.rows]

    def truth(self) -> list:
        return []
