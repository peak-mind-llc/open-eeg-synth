"""A case = one synthetic subject under one or more conditions (DESIGN §7.2)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

import open_eeg_synth.artifacts  # noqa: F401  (registers the built-in artifact kinds on import)
from open_eeg_synth._canon import canonical_fields, inf_from_json, inf_to_json, json_plain
from open_eeg_synth.brain.layer import BrainLayer, BrainSpec
from open_eeg_synth.brain.network import NetworkWiring, wire_network
from open_eeg_synth.brain.placement import placed_centres, region_centres
from open_eeg_synth.brain.rhythm import RhythmSpec, draw_patch_params
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.engine import Engine, Recording
from open_eeg_synth.headmodel import HeadModel, load_head_model
from open_eeg_synth.seeds import stream_rng, stream_seed
from open_eeg_synth.sensor import SensorNoise


@dataclass(frozen=True)
class SensorSpec:
    white_uv: float = 1.5

    def __post_init__(self) -> None:
        canonical_fields(self)


@dataclass(frozen=True)
class ArtifactSpec:
    kind: str
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical_fields(self)
        # params are the plug-in's constructor arguments, kept exactly as JSON reads them back
        object.__setattr__(self, "params", json_plain(dict(self.params)))


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    duration_s: float
    timeline: StateTimeline

    def __post_init__(self) -> None:
        canonical_fields(self)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_s": inf_to_json(self.duration_s),  # a stream's condition is unbounded
            "timeline": self.timeline.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConditionSpec:
        return cls(
            d["name"], inf_from_json(d["duration_s"]), StateTimeline.from_dict(d["timeline"])
        )


def _default_brain() -> BrainSpec:
    from open_eeg_synth.recipes import resting_brain

    return resting_brain()


@dataclass(frozen=True)
class CaseSpec:
    seed: int
    fs: float = 256.0
    channels: tuple[str, ...] = CHANNELS_19
    head_model: str = "colin27_19ch"
    perturb_head: bool = True
    brain: BrainSpec = field(default_factory=_default_brain)
    plants: tuple = ()  # Plant instances (Task 21)
    artifacts: tuple[ArtifactSpec, ...] = ()
    sensor: SensorSpec = SensorSpec()
    conditions: tuple[ConditionSpec, ...] = ()
    label: str = "synthetic"

    def __post_init__(self) -> None:
        # Equal specs must serialise identically (digest, case_id): plain Python types only.
        canonical_fields(self)
        object.__setattr__(self, "plants", tuple(self.plants))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "conditions", tuple(self.conditions))
        names = [c.name for c in self.conditions]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate condition names: {names}")

    def to_dict(self) -> dict:
        from open_eeg_synth.brain.plants import plant_to_dict

        return {
            "seed": self.seed,
            "fs": self.fs,
            "channels": list(self.channels),
            "head_model": self.head_model,
            "perturb_head": self.perturb_head,
            "brain": self.brain.to_dict(),
            "plants": [plant_to_dict(p) for p in self.plants],
            "artifacts": [{"kind": a.kind, "params": dict(a.params)} for a in self.artifacts],
            "sensor": {"white_uv": self.sensor.white_uv},
            "conditions": [c.to_dict() for c in self.conditions],
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CaseSpec:
        from open_eeg_synth.brain.plants import plant_from_dict

        return cls(
            seed=int(d["seed"]),
            fs=float(d["fs"]),
            channels=tuple(d["channels"]),
            head_model=d["head_model"],
            perturb_head=bool(d["perturb_head"]),
            brain=BrainSpec.from_dict(d["brain"]),
            plants=tuple(plant_from_dict(p) for p in d.get("plants", [])),
            artifacts=tuple(
                ArtifactSpec(a["kind"], dict(a.get("params", {}))) for a in d["artifacts"]
            ),
            sensor=SensorSpec(**d["sensor"]),
            conditions=tuple(ConditionSpec.from_dict(c) for c in d["conditions"]),
            label=d.get("label", "synthetic"),
        )

    def digest(self) -> str:
        blob = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()


def compiled_rhythms(spec: CaseSpec) -> tuple[RhythmSpec, ...]:
    """Base rhythms with the plants' modifiers applied, followed by the plants' own rhythms.

    Every per-rhythm keyed structure (subject draws, time streams, ``BrainLayer.parts``) keys on
    ``RhythmSpec.name``, so two compiled rhythms sharing a name would collide and silently render
    as one rhythm at whichever compiled last (Task 21 fix round 1) — raise instead.
    """
    from open_eeg_synth.brain.plants import apply_modifiers

    mods, extra = [], []
    for p in spec.plants:
        mods.extend(p.modifiers())
        extra.extend(p.rhythms())
    out = tuple(apply_modifiers(spec.brain.rhythms, mods)) + tuple(extra)
    dupes = sorted(n for n, c in Counter(r.name for r in out).items() if c > 1)
    if dupes:
        raise ValueError(f"duplicate compiled rhythm name(s): {dupes}")
    return out


@dataclass
class Subject:
    head: HeadModel
    mixing: np.ndarray
    placements: dict[str, list[int]]
    f0_hz: dict[str, float]
    network: NetworkWiring
    pattern_jitter: dict[str, float]
    rows: list[int]  # rows of ``head`` that the case records, in CaseSpec.channels order
    # per rhythm, per patch: own centre-frequency offset (Hz) and driver lag (ms), DESIGN §4.3
    patch_f0_offsets_hz: dict[str, list[float]] = field(default_factory=dict)
    patch_lags_ms: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "head_perturbation": self.head.perturbation,
            "placements": self.placements,
            "f0_hz": self.f0_hz,
            "patch_f0_offsets_hz": self.patch_f0_offsets_hz,
            "patch_lags_ms": self.patch_lags_ms,
            "network": self.network.to_dict(),
            "pattern_jitter": self.pattern_jitter,
        }


def make_subject(spec: CaseSpec) -> Subject:
    """Everything drawn once per subject, on the FULL head model (DESIGN §7.2).

    Patches are placed under every 10-20 site whether or not the case records it, so a
    four-channel stream and a nineteen-channel case of the same seed share one brain; the
    recorded channels are selected by ``rows`` when the brain layer renders. The background's
    smoothing matrix is computed here, once, and every condition's Background reuses it.
    """
    head = load_head_model(spec.head_model)
    if spec.perturb_head:
        head = head.perturbed(stream_rng(spec.seed, "subject:head"))
    rows = [head.index(c) for c in spec.channels]
    mixing = head.smoothed_mixing(spec.brain.background.smoothing_mm)
    placements: dict[str, list[int]] = {}
    f0: dict[str, float] = {}
    offsets: dict[str, list[float]] = {}
    lags: dict[str, list[float]] = {}
    for r in compiled_rhythms(spec):
        rng = stream_rng(spec.seed, f"subject:rhythm:{r.name}")
        placements[r.name] = (
            region_centres(head, r.region, r.n_patches, rng)
            if r.region
            else placed_centres(head, r.sites, rng)
        )
        f0[r.name] = r.f0_hz + (float(rng.normal(0.0, r.f0_jitter_hz)) if r.f0_jitter_hz else 0.0)
        off, lag = draw_patch_params(r, len(placements[r.name]), rng)
        offsets[r.name], lags[r.name] = list(off), list(lag)
    wiring = wire_network(
        head, spec.brain.network, spec.fs, stream_rng(spec.seed, "subject:network")
    )
    jitter = {
        k: float(stream_rng(spec.seed, f"subject:artifact:{k}").normal(0.0, 0.6))
        for k in sorted({a.kind for a in spec.artifacts})
    }
    return Subject(head, mixing, placements, f0, wiring, jitter, rows, offsets, lags)


def make_engine(spec: CaseSpec, subject: Subject, condition: ConditionSpec) -> Engine:
    from open_eeg_synth.artifacts.base import Occupancy, RenderContext
    from open_eeg_synth.artifacts.registry import make_artifact

    full, fs, seed = subject.head, spec.fs, spec.seed
    head = full.subset(spec.channels)  # what the amplifier records; artifacts live here
    layers: list = [
        BrainLayer(
            compiled_rhythms(spec),
            spec.brain,
            full,
            fs,
            condition.timeline,
            subject.mixing,
            subject.network,
            subject.placements,
            subject.f0_hz,
            seed,
            condition.name,
            rows=subject.rows,
            f0_offsets_hz=subject.patch_f0_offsets_hz,
            lags_ms=subject.patch_lags_ms,
        )
    ]
    transforms: list = []
    counts = Counter(a.kind for a in spec.artifacts)
    seen: Counter = Counter()
    occupancy = Occupancy()
    for a in spec.artifacts:
        art = make_artifact(a.kind, **a.params)
        # i counts instances of this kind only, so adding another kind never moves these draws
        i = seen[a.kind]
        seen[a.kind] += 1
        suffix = f"#{seen[a.kind]}" if counts[a.kind] > 1 else ""
        prefix = "transform" if art.mode == "transform" else "artifact"
        # bound before the Engine is built: an artifact's name exists only after bind()
        art.bind(
            RenderContext(
                channels=head.channels,
                fs=fs,
                electrode_pos=head.electrode_pos,
                head=head,
                timeline=condition.timeline,
                rng=stream_rng(seed, f"{condition.name}:artifact:{a.kind}:{i}"),
                subject_rng=stream_rng(seed, f"subject:artifact:{a.kind}"),
                occupancy=occupancy,
                layer_name=f"{prefix}:{a.kind}{suffix}",
            )
        )
        (transforms if art.mode == "transform" else layers).append(art)
    layers.append(
        SensorNoise(
            head.n_channels, spec.sensor.white_uv, stream_seed(seed, f"{condition.name}:sensor")
        )
    )
    return Engine(head.channels, fs, layers, transforms)


@dataclass
class Case:
    spec: CaseSpec
    case_id: str
    subject: Subject
    recordings: dict[str, Recording]


def case_id_for(spec: CaseSpec) -> str:
    h = hashlib.blake2b(f"{spec.seed}:{spec.digest()}".encode(), digest_size=4).hexdigest()
    return f"synth-{h}"


def _plant_is_silent(plant, states: set[str]) -> bool:
    """True when ``plant``'s own ``state_gain`` (``FocalSlow``/``RhythmicBursts``; absent on a
    modifier-only plant) is exactly zero in every state a condition's timeline contains, so it
    renders nothing there and its record does not belong in that condition's truth."""
    gain = getattr(plant, "state_gain", None)
    if not gain:
        return False
    return all(gain.get(s, 1.0) == 0.0 for s in states)


def make_case(spec: CaseSpec) -> Case:
    subject = make_subject(spec)
    recordings: dict[str, Recording] = {}
    for cond in spec.conditions:
        eng = make_engine(spec, subject, cond)
        rec = eng.render_all(int(round(cond.duration_s * spec.fs)))
        rec.timeline = cond.timeline
        states = {seg.state for seg in cond.timeline.segments}
        rec.plants = [p.record() for p in spec.plants if not _plant_is_silent(p, states)]
        recordings[cond.name] = rec
    return Case(spec, case_id_for(spec), subject, recordings)
