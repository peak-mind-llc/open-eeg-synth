"""Rhythms as mirrored cortical patches sharing a delayed driver (DESIGN §4.3)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from open_eeg_synth._canon import canonical_fields
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.channels import MIRROR
from open_eeg_synth.dsp import OU
from open_eeg_synth.headmodel import HeadModel


@dataclass(frozen=True)
class BurstGate:
    on_s: tuple[float, float] = (1.0, 3.0)
    off_s: tuple[float, float] = (8.0, 25.0)
    ramp_s: float = 0.3

    def __post_init__(self) -> None:
        canonical_fields(self)


@dataclass(frozen=True)
class RhythmSpec:
    name: str
    f0_hz: float
    amp_uv: float
    sites: tuple[str, ...] = ()
    region: str | None = None
    n_patches: int = 4
    width_mm: float = 12.0
    f_sd: float = 0.35
    f_tau_s: float = 4.0
    env_tau_s: float = 1.5
    env_sd: float = 0.5
    env_lo: float = 0.25
    env_hi: float = 1.8
    lag_ms: float = 25.0
    indep: float = 0.3
    f0_jitter_hz: float = 0.0
    state_gain: dict[str, float] = field(default_factory=dict)
    state_f0_shift_hz: dict[str, float] = field(default_factory=dict)
    hemisphere_gain: dict[str, float] = field(default_factory=dict)
    burst: BurstGate | None = None

    def __post_init__(self) -> None:
        canonical_fields(self)
        if bool(self.sites) == bool(self.region):
            raise ValueError(
                f"RhythmSpec {self.name!r} needs exactly one of sites or region; got "
                f"sites={self.sites!r} region={self.region!r}"
            )
        if self.region is not None and self.n_patches % 2 != 0:
            raise ValueError(f"region placement needs an even n_patches; got {self.n_patches}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sites"] = list(self.sites)
        d["burst"] = (
            None
            if self.burst is None
            else {
                "on_s": list(self.burst.on_s),
                "off_s": list(self.burst.off_s),
                "ramp_s": self.burst.ramp_s,
            }
        )
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RhythmSpec:
        d = dict(d)
        d["sites"] = tuple(d.get("sites", ()))
        b = d.get("burst")
        d["burst"] = (
            None if b is None else BurstGate(tuple(b["on_s"]), tuple(b["off_s"]), b["ramp_s"])
        )
        return cls(**d)


class _Oscillator:
    """One time course: OU frequency around f0, log-normal OU envelope, phase accumulator."""

    def __init__(self, fs: float, f0: float, spec: RhythmSpec, rng: np.random.Generator) -> None:
        self.fs, self.f0, self.rng = fs, f0, rng
        self.freq = OU(1, fs, mu=0.0, sigma=spec.f_sd, tau_s=spec.f_tau_s, rng=rng)
        self.lenv = OU(1, fs, mu=0.0, sigma=spec.env_sd, tau_s=spec.env_tau_s, rng=rng)
        self.phase = float(rng.uniform(0.0, 2.0 * np.pi))
        self.lo, self.hi = np.log(spec.env_lo), np.log(spec.env_hi)

    def render(self, n: int, f0_shift: np.ndarray | float = 0.0) -> np.ndarray:
        white = self.rng.standard_normal((n, 2)).T
        f = self.f0 + f0_shift + self.freq.step(white[:1])[0]
        env = np.exp(np.clip(self.lenv.step(white[1:])[0], self.lo, self.hi))
        ph = self.phase + 2.0 * np.pi * np.cumsum(f) / self.fs
        self.phase = float(ph[-1] % (2.0 * np.pi))
        return env * np.sin(ph)


class _Gate:
    """On/off gating with linear-ramp edges; transitions drawn in order (streaming-safe)."""

    def __init__(self, burst: BurstGate, fs: float, rng: np.random.Generator) -> None:
        self.b, self.fs, self.rng = burst, fs, rng
        self.nr = max(1, int(round(burst.ramp_s * fs)))
        first_on = int(round(rng.uniform(*burst.off_s) * fs))
        self.edges: list[tuple[int, float]] = [(first_on, 1.0)]  # (sample, level after edge)
        self.base = 0.0  # level before the first stored edge

    def _extend(self, until: int) -> None:
        while self.edges[-1][0] < until + self.nr:
            s, lvl = self.edges[-1]
            dur = self.rng.uniform(*(self.b.on_s if lvl == 1.0 else self.b.off_s))
            self.edges.append((s + int(round(dur * self.fs)), 0.0 if lvl == 1.0 else 1.0))

    def render(self, t0: int, n: int) -> np.ndarray:
        self._extend(t0 + n)
        k = t0 + np.arange(n)
        g = np.full(n, self.base)
        prev = self.base
        keep: list[tuple[int, float]] = []
        for s, lvl in self.edges:
            if s + self.nr < t0:
                self.base = lvl
                prev = lvl
                g[:] = lvl
                continue
            keep.append((s, lvl))
            if s - self.nr > t0 + n:
                continue
            ramp = (
                np.clip((k - s) / self.nr + 0.5, 0.0, 1.0) if self.b.ramp_s > 0 else (k >= s) * 1.0
            )
            g += (lvl - prev) * ramp
            prev = lvl
        self.edges = keep
        return g


def _orient_patches(
    head: HeadModel, centres: list[int], raw: list[np.ndarray], ref: list[int]
) -> list[np.ndarray]:
    """Orient every patch positive at its own reference channel; a mirror patch then
    takes its partner's polarity, not its own independent orientation: its sign is
    chosen so its map value at the mirror of the partner's reference channel matches
    the partner's (positive) value there. Independent per-patch orientation flips
    mirror pairs inconsistently when their local geometry differs (DESIGN §4.3)."""
    oriented = [m if m[ref[i]] >= 0 else -m for i, m in enumerate(raw)]
    paired: set[int] = set()
    for i, c in enumerate(centres):
        if i in paired:
            continue
        mirror_c = head.mirror_source(c)
        if mirror_c == c:
            continue  # on the midline: no partner to make consistent with
        j = next(
            (k for k, c2 in enumerate(centres) if k != i and k not in paired and c2 == mirror_c),
            None,
        )
        if j is None:
            continue
        paired.add(i)
        paired.add(j)
        mirror_ch = head.index(MIRROR.get(head.channels[ref[i]], head.channels[ref[i]]))
        if oriented[j][mirror_ch] < 0:
            oriented[j] = -oriented[j]
    return oriented


PATCH_F0_SD_HZ = 0.3  # spread of each patch's own centre frequency around f0 (DESIGN §4.3)


def draw_patch_params(
    spec: RhythmSpec, n_patches: int, rng: np.random.Generator
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-patch centre-frequency offsets (Hz, N(0, 0.3)) and driver lags (ms, U(0, lag_ms)).

    These describe the subject, not the passage of time: a case draws them once from
    ``subject:rhythm:<name>`` (DESIGN §8.1) so every condition's rhythm uses the same values.
    """
    offsets = tuple(float(v) for v in rng.normal(0.0, PATCH_F0_SD_HZ, n_patches))
    lags = tuple(float(v) for v in rng.uniform(0.0, spec.lag_ms, n_patches))
    return offsets, lags


class Rhythm:
    """One rhythm's patches. ``seq`` seeds the time courses (driver, own oscillators, burst gate).

    ``f0_offsets_hz`` and ``lags_ms`` are the subject's per-patch values (see
    :func:`draw_patch_params`); when omitted, a standalone rhythm draws them from a child of
    ``seq`` that no time course uses, so supplying them never reshuffles the time courses.
    """

    def __init__(
        self,
        head: HeadModel,
        fs: float,
        spec: RhythmSpec,
        timeline: StateTimeline,
        centres: list[int],
        f0_hz: float,
        seq: np.random.SeedSequence,
        *,
        f0_offsets_hz: Sequence[float] | None = None,
        lags_ms: Sequence[float] | None = None,
    ) -> None:
        self.spec, self.fs, self.timeline = spec, float(fs), timeline
        self.name = f"brain.rhythm:{spec.name}"
        # children: driver, one per patch, the burst gate, the standalone patch-parameter draw
        children = seq.spawn(len(centres) + 3)  # consumes seq: it cannot be spawned again
        rngs = [np.random.Generator(np.random.PCG64(c)) for c in children]
        gate_rng, param_rng = rngs[-2], rngs[-1]
        if f0_offsets_hz is None or lags_ms is None:
            drawn_offsets, drawn_lags = draw_patch_params(spec, len(centres), param_rng)
            f0_offsets_hz = drawn_offsets if f0_offsets_hz is None else f0_offsets_hz
            lags_ms = drawn_lags if lags_ms is None else lags_ms
        self.f0_offsets_hz = tuple(float(v) for v in f0_offsets_hz)
        self.lags_ms = tuple(float(v) for v in lags_ms)
        if not len(self.f0_offsets_hz) == len(self.lags_ms) == len(centres):
            raise ValueError(
                f"rhythm {spec.name!r}: {len(centres)} patches but "
                f"{len(self.f0_offsets_hz)} offsets and {len(self.lags_ms)} lags"
            )
        targets = [head.index(s) for s in spec.sites] if spec.sites else None
        raw = [head.patch_map(c, spec.width_mm) for c in centres]

        # Orientation reference channel per patch: its own site for site placement, its
        # own strongest channel for region placement (patches in one region can peak at
        # very different channels, so a single shared channel is a poor reference for
        # some of them). The amplitude target stays the channel with the largest summed
        # response, shared by every patch, as DESIGN §4.3 defines it.
        if targets is not None:
            ref = list(targets)
        else:
            t = int(np.argmax(np.abs(np.sum(raw, axis=0))))
            targets = [t]
            ref = [int(np.argmax(np.abs(m))) for m in raw]

        oriented = _orient_patches(head, centres, raw, ref)
        self.maps = np.array(oriented)  # (P, n_ch), un-gained
        self.scale = spec.amp_uv / np.abs(self.maps.sum(axis=0))[targets].max()

        # hemisphere_gain is applied only after the scale is fixed, so a plant that
        # quiets one hemisphere doesn't have its own effect partly cancelled by scale.
        sides = [
            "left"
            if head.source_pos[c, 0] < -0.005
            else ("right" if head.source_pos[c, 0] > 0.005 else "midline")
            for c in centres
        ]
        self.maps = np.array(
            [
                m * spec.hemisphere_gain.get(side, 1.0)
                for m, side in zip(self.maps, sides, strict=True)
            ]
        )

        self.lags = [int(round(lag / 1000.0 * self.fs)) for lag in self.lags_ms]
        self.maxlag = max(self.lags) if self.lags else 0
        self.driver = _Oscillator(fs, f0_hz, spec, rngs[0])
        self.own = [
            _Oscillator(fs, f0_hz + off, spec, rngs[i + 1])
            for i, off in enumerate(self.f0_offsets_hz)
        ]
        self.hist = self.driver.render(self.maxlag) if self.maxlag > 0 else np.zeros(0)
        self.gate = _Gate(spec.burst, fs, gate_rng) if spec.burst is not None else None
        self.w_shared, self.w_own = np.sqrt(1.0 - spec.indep), np.sqrt(spec.indep)

    def render(self, t0: int, n: int) -> np.ndarray:
        shift = self.timeline.gain(self.spec.state_f0_shift_hz, t0, n, self.fs, default=0.0)
        drv = self.driver.render(n, shift)
        ext = np.concatenate([self.hist, drv])
        out = np.zeros((self.maps.shape[1], n))
        for p, lag in enumerate(self.lags):
            shared = ext[self.maxlag - lag : self.maxlag - lag + n]
            own = self.own[p].render(n, shift)
            out += self.maps[p][:, None] * (self.w_shared * shared + self.w_own * own)[None, :]
        if self.maxlag > 0:
            self.hist = ext[-self.maxlag :]
        gain = self.timeline.gain(self.spec.state_gain, t0, n, self.fs)
        if self.gate is not None:
            gain = gain * self.gate.render(t0, n)
        return (self.scale * out * gain[None, :]).astype(np.float32)

    def truth(self) -> list:
        return []
