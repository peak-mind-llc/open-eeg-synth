"""Planted-pattern primitives: generic signal terms compiled onto the rhythm generator.

DESIGN §4.4.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, ClassVar, Protocol

from open_eeg_synth._canon import canonical_fields
from open_eeg_synth.brain.rhythm import BurstGate, RhythmSpec


@dataclass(frozen=True)
class PlantRecord:
    kind: str
    sites: tuple[str, ...]
    band_hz: tuple[float, float]
    amp_uv: float | None
    onset_s: float
    offset_s: float | None
    params: dict
    description: str

    def __post_init__(self) -> None:
        canonical_fields(self)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sites"], d["band_hz"] = list(self.sites), list(self.band_hz)
        d["params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.params.items()}
        return d


@dataclass(frozen=True)
class Modifier:
    rhythm: str
    amp_scale: float = 1.0
    f0_shift_hz: float = 0.0
    hemisphere_gain: dict[str, float] = field(default_factory=dict)
    extra_sites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        canonical_fields(self)


def apply_modifiers(
    rhythms: Sequence[RhythmSpec], modifiers: Sequence[Modifier]
) -> tuple[RhythmSpec, ...]:
    out = {r.name: r for r in rhythms}
    for m in modifiers:
        if m.rhythm not in out:
            raise KeyError(f"modifier targets unknown rhythm {m.rhythm!r}; have {sorted(out)}")
        r = out[m.rhythm]
        hg = dict(r.hemisphere_gain)
        for side, g in m.hemisphere_gain.items():
            hg[side] = hg.get(side, 1.0) * g
        out[m.rhythm] = replace(
            r,
            amp_uv=r.amp_uv * m.amp_scale,
            f0_hz=r.f0_hz + m.f0_shift_hz,
            hemisphere_gain=hg,
            sites=r.sites + tuple(s for s in m.extra_sites if s not in r.sites),
        )
    return tuple(out[r.name] for r in rhythms)


class Plant(Protocol):
    kind: ClassVar[str]

    def rhythms(self) -> tuple[RhythmSpec, ...]: ...

    def modifiers(self) -> tuple[Modifier, ...]: ...

    def record(self) -> PlantRecord: ...


PLANTS: dict[str, type] = {}


def register_plant(cls):
    PLANTS[cls.kind] = cls
    return cls


def plant_to_dict(p: Plant) -> dict[str, Any]:
    d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(p).items()}
    return {"kind": p.kind, "params": d}


def plant_from_dict(d: dict[str, Any]) -> Plant:
    cls = PLANTS[d["kind"]]
    params = dict(d["params"])
    for k, v in params.items():
        if isinstance(v, list):
            params[k] = tuple(v)
    return cls(**params)


@register_plant
@dataclass(frozen=True)
class FocalSlow:
    kind: ClassVar[str] = "focal_slow"
    site: str
    f0_hz: float = 2.5
    amp_uv: float = 45.0
    width_mm: float = 12.0
    f_sd: float = 0.4
    env_tau_s: float = 1.5
    env_sd: float = 0.6
    indep: float = 0.3
    state_gain: dict[str, float] = field(default_factory=dict)  # e.g. {"eyes_open": 0.0}

    def __post_init__(self) -> None:
        canonical_fields(self)

    def rhythms(self) -> tuple[RhythmSpec, ...]:
        return (
            RhythmSpec(
                f"plant:focal_slow:{self.site}",
                self.f0_hz,
                self.amp_uv,
                sites=(self.site,),
                width_mm=self.width_mm,
                f_sd=self.f_sd,
                env_tau_s=self.env_tau_s,
                env_sd=self.env_sd,
                lag_ms=0.0,
                indep=self.indep,
                state_gain=dict(self.state_gain),
            ),
        )

    def modifiers(self) -> tuple[Modifier, ...]:
        return ()

    def record(self) -> PlantRecord:
        return PlantRecord(
            self.kind,
            (self.site,),
            (self.f0_hz - 1.0, self.f0_hz + 1.0),
            self.amp_uv,
            0.0,
            None,
            asdict(self),
            f"a {self.f0_hz:.1f} Hz rhythm of about {self.amp_uv:.0f} uV from one "
            f"cortical patch under {self.site}, present throughout",
        )


@register_plant
@dataclass(frozen=True)
class RhythmicBursts:
    kind: ClassVar[str] = "rhythmic_bursts"
    sites: tuple[str, ...]
    f0_hz: float = 6.5
    amp_uv: float = 30.0
    width_mm: float = 15.0
    burst_s: tuple[float, float] = (1.0, 3.0)
    gap_s: tuple[float, float] = (8.0, 25.0)
    env_tau_s: float = 1.0
    env_sd: float = 0.8
    indep: float = 0.2
    state_gain: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical_fields(self)

    def rhythms(self) -> tuple[RhythmSpec, ...]:
        name = "plant:rhythmic_bursts:" + "+".join(self.sites)
        return (
            RhythmSpec(
                name,
                self.f0_hz,
                self.amp_uv,
                sites=self.sites,
                width_mm=self.width_mm,
                env_tau_s=self.env_tau_s,
                env_sd=self.env_sd,
                env_lo=0.1,
                env_hi=3.0,
                lag_ms=0.0,
                indep=self.indep,
                state_gain=dict(self.state_gain),
                burst=BurstGate(self.burst_s, self.gap_s, 0.3),
            ),
        )

    def modifiers(self) -> tuple[Modifier, ...]:
        return ()

    def record(self) -> PlantRecord:
        return PlantRecord(
            self.kind,
            self.sites,
            (self.f0_hz - 1.0, self.f0_hz + 1.0),
            self.amp_uv,
            0.0,
            None,
            asdict(self),
            f"bursts of a {self.f0_hz:.1f} Hz rhythm ({self.burst_s[0]:.0f}-"
            f"{self.burst_s[1]:.0f} s long, every {self.gap_s[0]:.0f}-{self.gap_s[1]:.0f} s)"
            f" under {', '.join(self.sites)}",
        )


@register_plant
@dataclass(frozen=True)
class LateralImbalance:
    kind: ClassVar[str] = "lateral_imbalance"
    rhythm: str
    side: str
    factor: float

    def __post_init__(self) -> None:
        canonical_fields(self)

    def rhythms(self) -> tuple[RhythmSpec, ...]:
        return ()

    def modifiers(self) -> tuple[Modifier, ...]:
        return (Modifier(self.rhythm, hemisphere_gain={self.side: self.factor}),)

    def record(self) -> PlantRecord:
        return PlantRecord(
            self.kind,
            (),
            (0.0, 0.0),
            None,
            0.0,
            None,
            asdict(self),
            f"the {self.rhythm} rhythm scaled by {self.factor:.2f} on the {self.side}",
        )


@register_plant
@dataclass(frozen=True)
class WidespreadExcess:
    kind: ClassVar[str] = "widespread_excess"
    rhythm: str
    factor: float
    extra_sites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        canonical_fields(self)

    def rhythms(self) -> tuple[RhythmSpec, ...]:
        return ()

    def modifiers(self) -> tuple[Modifier, ...]:
        return (Modifier(self.rhythm, amp_scale=self.factor, extra_sites=self.extra_sites),)

    def record(self) -> PlantRecord:
        return PlantRecord(
            self.kind,
            self.extra_sites,
            (0.0, 0.0),
            None,
            0.0,
            None,
            asdict(self),
            f"the {self.rhythm} rhythm scaled by {self.factor:.2f} everywhere it occurs"
            + (f", extended to {', '.join(self.extra_sites)}" if self.extra_sites else ""),
        )


@register_plant
@dataclass(frozen=True)
class PeakShift:
    kind: ClassVar[str] = "peak_shift"
    rhythm: str
    shift_hz: float

    def __post_init__(self) -> None:
        canonical_fields(self)

    def rhythms(self) -> tuple[RhythmSpec, ...]:
        return ()

    def modifiers(self) -> tuple[Modifier, ...]:
        return (Modifier(self.rhythm, f0_shift_hz=self.shift_hz),)

    def record(self) -> PlantRecord:
        return PlantRecord(
            self.kind,
            (),
            (0.0, 0.0),
            None,
            0.0,
            None,
            asdict(self),
            f"the {self.rhythm} rhythm's centre frequency moved by {self.shift_hz:+.1f} Hz",
        )


@register_plant
@dataclass(frozen=True)
class ReducedRhythm:
    kind: ClassVar[str] = "reduced_rhythm"
    rhythm: str
    factor: float

    def __post_init__(self) -> None:
        canonical_fields(self)

    def rhythms(self) -> tuple[RhythmSpec, ...]:
        return ()

    def modifiers(self) -> tuple[Modifier, ...]:
        return (Modifier(self.rhythm, amp_scale=self.factor),)

    def record(self) -> PlantRecord:
        return PlantRecord(
            self.kind,
            (),
            (0.0, 0.0),
            None,
            0.0,
            None,
            asdict(self),
            f"the {self.rhythm} rhythm reduced to {self.factor:.2f} of its usual amplitude",
        )
