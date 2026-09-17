"""The one signal path: layers rendered block by block and summed (DESIGN §7.1)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from open_eeg_synth.artifacts.base import TruthRecord
    from open_eeg_synth.brain.plants import PlantRecord
    from open_eeg_synth.brain.state import StateTimeline


@runtime_checkable
class Layer(Protocol):
    name: str

    def render(self, t0: int, n: int) -> np.ndarray: ...

    def truth(self) -> list[TruthRecord]: ...


@runtime_checkable
class Transform(Protocol):
    name: str

    def render_transform(self, t0: int, n: int, mix: np.ndarray) -> np.ndarray: ...

    def truth(self) -> list[TruthRecord]: ...


@dataclass
class Frame:
    t0: int
    n: int
    layers: dict[str, np.ndarray]

    @property
    def mixed(self) -> np.ndarray:
        out = np.zeros(next(iter(self.layers.values())).shape, dtype=np.float32)
        for block in self.layers.values():
            out += block
        return out


@dataclass
class Recording:
    fs: float
    channels: tuple[str, ...]
    layers: dict[str, np.ndarray]
    truth: list[TruthRecord] = field(default_factory=list)
    plants: list[PlantRecord] = field(default_factory=list)
    timeline: StateTimeline | None = None

    @property
    def mixed(self) -> np.ndarray:
        out = np.zeros(next(iter(self.layers.values())).shape, dtype=np.float32)
        for block in self.layers.values():
            out += block
        return out

    @property
    def n_samples(self) -> int:
        return int(next(iter(self.layers.values())).shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs


class Engine:
    def __init__(
        self,
        channels: Sequence[str],
        fs: float,
        layers: Sequence[Layer],
        transforms: Sequence[Transform] = (),
    ) -> None:
        self.channels = tuple(channels)
        self.fs = float(fs)
        self.layers = list(layers)
        self.transforms = list(transforms)
        if not self.layers:
            raise ValueError("Engine requires at least one layer")
        names = [lay.name for lay in self.layers] + [t.name for t in self.transforms]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate layer names: {names}")
        self._pos = 0

    @property
    def position(self) -> int:
        return self._pos

    def render(self, t0: int, n: int) -> Frame:
        if t0 != self._pos:
            raise ValueError(f"render must be contiguous: expected t0={self._pos}, got {t0}")
        if n <= 0:
            raise ValueError("n must be positive")
        n_ch = len(self.channels)
        out: dict[str, np.ndarray] = {}
        mix = np.zeros((n_ch, n), dtype=np.float32)
        for lay in self.layers:
            # np.array(..., copy the default True) — never np.asarray — so a layer that reuses its
            # own scratch buffer across calls cannot corrupt a block already stored in this Frame or
            # in an earlier one accumulated by render_all.
            block = np.array(lay.render(t0, n), dtype=np.float32)
            if block.shape != (n_ch, n):
                raise ValueError(f"layer {lay.name} returned {block.shape}, expected {(n_ch, n)}")
            out[lay.name] = block
            mix += block
        for tr in self.transforms:
            # Transforms see the running mix read-only (they may inspect other channels, e.g. a
            # dead channel), but not mutate it; the delta is copied for the same reuse reason as
            # above, which also covers a transform returning a view of mix itself (`return mix`) —
            # uncopied, that view would keep changing as later transforms add to mix.
            view = mix.view()
            view.flags.writeable = False
            delta = np.array(tr.render_transform(t0, n, view), dtype=np.float32)
            if delta.shape != (n_ch, n):
                raise ValueError(
                    f"transform {tr.name} returned {delta.shape}, expected {(n_ch, n)}"
                )
            out[tr.name] = delta
            mix += delta
        self._pos += n
        return Frame(t0, n, out)

    def render_all(self, n_samples: int, block: int = 4096) -> Recording:
        parts: dict[str, list[np.ndarray]] = {}
        t0 = 0
        while t0 < n_samples:
            fr = self.render(t0, min(block, n_samples - t0))
            for k, v in fr.layers.items():
                parts.setdefault(k, []).append(v)
            t0 += fr.n
        layers = {k: np.concatenate(v, axis=1) for k, v in parts.items()}
        return Recording(self.fs, self.channels, layers, truth=self.truth())

    def truth(self) -> list[TruthRecord]:
        out: list[TruthRecord] = []
        for lay in self.layers:
            out.extend(lay.truth())
        for tr in self.transforms:
            out.extend(tr.truth())
        return out
