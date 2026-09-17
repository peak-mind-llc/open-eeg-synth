"""The one signal path: layers rendered block by block and summed (DESIGN §7.1)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Layer(Protocol):
    name: str

    def render(self, t0: int, n: int) -> np.ndarray: ...

    def truth(self) -> list: ...


@runtime_checkable
class Transform(Protocol):
    name: str

    def render_transform(self, t0: int, n: int, mix: np.ndarray) -> np.ndarray: ...

    def truth(self) -> list: ...


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
    truth: list = field(default_factory=list)
    plants: list = field(default_factory=list)
    timeline: Any = None

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
            block = np.asarray(lay.render(t0, n), dtype=np.float32)
            if block.shape != (n_ch, n):
                raise ValueError(f"layer {lay.name} returned {block.shape}, expected {(n_ch, n)}")
            out[lay.name] = block
            mix += block
        for tr in self.transforms:
            delta = np.asarray(tr.render_transform(t0, n, mix), dtype=np.float32)
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

    def truth(self) -> list:
        out: list = []
        for lay in self.layers:
            out.extend(lay.truth())
        for tr in self.transforms:
            out.extend(tr.truth())
        return out
