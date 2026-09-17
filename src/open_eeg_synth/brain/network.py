"""Cortical patches driven partly by delayed copies of other patches (DESIGN §4.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from open_eeg_synth._canon import canonical_fields
from open_eeg_synth.brain.background import WARM_UP_S, BackgroundSpec
from open_eeg_synth.dsp import PinkCascade
from open_eeg_synth.headmodel import HeadModel


@dataclass(frozen=True)
class NetworkSpec:
    n_nodes: int = 40
    fan_in: int = 3
    coupling: float = 1.0
    velocity_m_s: float = 6.0
    synaptic_ms: float = 5.0
    width_mm: float = 15.0

    def __post_init__(self) -> None:
        canonical_fields(self)


@dataclass(frozen=True)
class NetworkWiring:
    centres: tuple[int, ...]
    sources: tuple[tuple[int, ...], ...]  # per node: the nodes it listens to
    lags: tuple[tuple[int, ...], ...]  # per node: the lag in samples for each source

    def to_dict(self) -> dict:
        return {
            "centres": list(self.centres),
            "sources": [list(s) for s in self.sources],
            "lags": [list(lg) for lg in self.lags],
        }


def wire_network(
    head: HeadModel, spec: NetworkSpec, fs: float, rng: np.random.Generator
) -> NetworkWiring:
    centres = [int(c) for c in rng.choice(head.n_sources, spec.n_nodes, replace=False)]
    sources, lags = [], []
    for j in range(spec.n_nodes):
        others = [k for k in range(spec.n_nodes) if k != j]
        srcs = [int(k) for k in rng.choice(others, spec.fan_in, replace=False)]
        row = []
        for k in srcs:
            dist = float(np.linalg.norm(head.source_pos[centres[j]] - head.source_pos[centres[k]]))
            row.append(int(round((dist / spec.velocity_m_s + spec.synaptic_ms / 1000.0) * fs)))
        sources.append(tuple(srcs))
        lags.append(tuple(row))
    return NetworkWiring(tuple(centres), tuple(sources), tuple(lags))


class Network:
    name = "brain.network"

    def __init__(
        self,
        head: HeadModel,
        fs: float,
        spec: NetworkSpec,
        background: BackgroundSpec,
        wiring: NetworkWiring,
        seq: np.random.SeedSequence,
    ) -> None:
        self.spec, self.wiring = spec, wiring
        self.maps = np.array([head.patch_map(c, spec.width_mm) for c in wiring.centres])  # (N, ch)
        self.n = spec.n_nodes
        self.rng = np.random.Generator(np.random.PCG64(seq))
        self.pink = PinkCascade(self.n, fs, beta=background.exponent)
        self.pink.warm_up(self.rng, WARM_UP_S)
        self.node_scale = 1.0 / np.sqrt(1.0 + spec.coupling**2 / spec.fan_in)
        chan_var = (self.maps**2).sum(axis=0)
        self.scale = background.rms_uv * np.sqrt(background.network_frac) / np.sqrt(chan_var.mean())
        self.maxlag = max((lg for row in wiring.lags for lg in row), default=0)
        self.hist = np.zeros((self.n, self.maxlag))
        self.w = spec.coupling / spec.fan_in

    def render_nodes(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        x = self.pink.process(self.rng.standard_normal((n, self.n)).T)
        ext = np.concatenate([self.hist, x], axis=1)
        y = x.copy()
        for j, (srcs, lags) in enumerate(zip(self.wiring.sources, self.wiring.lags, strict=True)):
            for k, lag in zip(srcs, lags, strict=True):
                y[j] += self.w * ext[k, self.maxlag - lag : self.maxlag - lag + n]
        if self.maxlag > 0:
            self.hist = ext[:, ext.shape[1] - self.maxlag :]
        return x, y * self.node_scale

    def render(self, t0: int, n: int) -> np.ndarray:
        _, y = self.render_nodes(n)
        return (self.scale * (self.maps.T @ y)).astype(np.float32)

    def truth(self) -> list:
        return []
